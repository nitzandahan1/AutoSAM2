# inference_MASK_BOX_metrics.py
import os, glob, csv, cv2, torch, numpy as np
from typing import Optional, Tuple, List
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from TRAIN_MASK_BOX import (
    PrompterNet, _sorted_frames_in_dir, sam2_forward_logits_and_or_box,
    LOGIT_GAIN, LOGIT_CLAMP
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------- utils: label/gt/io/metrics/vis -------------
def _pad_to_width(img_bgr: np.ndarray, width: int) -> np.ndarray:
    """Pad (or shrink) image to exactly 'width' pixels using border on the right."""
    h, w = img_bgr.shape[:2]
    if w == width:
        return img_bgr
    if w > width:
        # אם השורה רחבה מדי — נקטין פרופורציונלית לגובה הנוכחי
        return cv2.resize(img_bgr, (width, h), interpolation=cv2.INTER_AREA)
    pad_right = width - w
    return cv2.copyMakeBorder(img_bgr, 0, 0, 0, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))

def _load_label_map(path: str) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im, dtype=np.int32) if im.mode == "P" else np.array(im.convert("L"), dtype=np.int32)
    return arr

def _choose_label_from_m0(m0_arr: np.ndarray, label_id: Optional[int] = None) -> int:
    if label_id is not None:
        return int(label_id)
    vals, counts = np.unique(m0_arr, return_counts=True)
    pos = [(int(v), int(c)) for v, c in zip(vals, counts) if int(v) > 0]
    if not pos:  # binary 0/255 -> treat positive as 1
        return 1 if (m0_arr > 0).sum() > 0 else 0
    return max(pos, key=lambda vc: vc[1])[0]

def _dice_iou(pred01: np.ndarray, gt01: np.ndarray, thr: float = 0.5, eps: float = 1e-6) -> Tuple[float, float]:
    p = (pred01 >= thr).astype(np.float32)
    g = (gt01  >= 0.5).astype(np.float32)
    inter = float((p * g).sum())
    dice = (2*inter + eps) / (p.sum() + g.sum() + eps)
    union = float((p + g - p*g).sum())
    iou  = (inter + eps) / (union + eps)
    return dice, iou

def _as_u8_img(arr: np.ndarray) -> np.ndarray:
    if arr is None: return None
    if arr.dtype in (np.float32, np.float64):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)

def _mask_overlay(img_bgr, mask01, color=(0,255,0), alpha=0.45):
    m = np.clip(mask01.astype(np.float32), 0, 1)
    col = np.zeros_like(img_bgr); col[:] = color
    return (img_bgr*(1-alpha) + col*alpha*m[...,None]).astype(np.uint8)

def _draw_box(img_bgr: np.ndarray, box_xyxy, color=(0,0,255), thickness=2):
    if box_xyxy is None: return img_bgr
    x0,y0,x1,y1 = [int(v) for v in box_xyxy]
    out = img_bgr.copy()
    cv2.rectangle(out, (x0,y0), (x1,y1), color, thickness)
    return out

def _put_title(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    h = 26
    cv2.rectangle(out,(0,0),(out.shape[1],h),(0,0,0),-1)
    cv2.putText(out, text, (6,19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return out

def _hstack_resize(imgs_bgr: List[np.ndarray], h: int) -> np.ndarray:
    tiles = []
    for im in imgs_bgr:
        if im is None or im.size == 0: continue
        im = _as_u8_img(im)
        new_w = int(round(im.shape[1]*h/max(1,im.shape[0])))
        tiles.append(cv2.resize(im, (max(1,new_w), h), interpolation=cv2.INTER_AREA))
    if not tiles:
        return np.zeros((h, h, 3), dtype=np.uint8)
    return np.concatenate(tiles, axis=1)

def _save_panel(
    out_dir: str,
    stem: str,
    img_tm1_rgb: np.ndarray,
    img_t_rgb:   np.ndarray,
    m_tm1_01:    np.ndarray,
    prior01:     Optional[np.ndarray],          # ← חדש: prior של t (ב-01)
    pred01:      np.ndarray,
    feed_box_xyxy=None,
    gt01: Optional[np.ndarray]=None
):
    os.makedirs(out_dir, exist_ok=True)
    # RGB→BGR לציור
    img_t_bgr   = cv2.cvtColor(img_t_rgb,   cv2.COLOR_RGB2BGR)
    img_tm1_bgr = cv2.cvtColor(img_tm1_rgb, cv2.COLOR_RGB2BGR)

    # --- שורה עליונה: I_{t-1}, I_t, M_{t-1} (ללא BOX) ---
    # מציגים את M_{t-1} על התמונה הנוכחית I_t (כך רואים את ה-roll קדימה)
    m_tm1_overlay = _mask_overlay(img_tm1_bgr, m_tm1_01, (255, 0, 0), 0.40)  # בלי קופסה
    row1 = _hstack_resize([
        _put_title(img_tm1_bgr, "I_{t-1}"),
        _put_title(img_t_bgr, "I_t"),
        _put_title(m_tm1_overlay, "M_{t-1} on I_{t-1} "),
    ], 300)

    # --- שורה תחתונה: M_t^{prior} + BOX_t, SAM2 Pred, GT+Pred ---
    pred_overlay = _mask_overlay(img_t_bgr, pred01, (0, 255, 0), 0.45)

    tiles_row2 = []
    # 1) prior + BOX (אם prior קיים)
    if prior01 is not None:
        m_prior_overlay = _mask_overlay(img_t_bgr, prior01, (255, 0, 0), 0.40)
        m_prior_overlay_box = _draw_box(m_prior_overlay, feed_box_xyxy, (0, 0, 255), 2)
        tiles_row2.append(_put_title(m_prior_overlay_box, "M_t^{prior} + BOX_t"))
    else:
        tiles_row2.append(_put_title(np.zeros_like(img_t_bgr), "M_t^{prior} + BOX_t (N/A)"))

    # 2) SAM2 Pred (overlay)
    tiles_row2.append(_put_title(pred_overlay, "SAM2 Pred "))

    # 3) GT + Pred (אם יש GT)
    if gt01 is not None:
        gt_overlay = _mask_overlay(img_t_bgr, gt01, (0, 0, 255), 0.45)
        pred_on_gt = _mask_overlay(gt_overlay, pred01, (0, 255, 0), 0.45)
        tiles_row2.append(_put_title(pred_on_gt, "GT + Pred overlay"))
    else:
        tiles_row2.append(_put_title(np.zeros_like(img_t_bgr), "GT + Pred overlay (N/A)"))

    row2 = _hstack_resize(tiles_row2, 300)

    # --- יישור רוחב ואיחוד ---
    W = max(row1.shape[1], row2.shape[1])
    row1 = _pad_to_width(row1, W)
    row2 = _pad_to_width(row2, W)
    panel = np.vstack([row1, row2])

    panel_path = os.path.join(out_dir, f"{stem}_PANEL.jpg")
    cv2.imwrite(panel_path, panel)


def _png_path_like(fr_path: str, out_dir: str) -> str:
    name = os.path.splitext(os.path.basename(fr_path))[0] + ".png"
    return os.path.join(out_dir, name)

# ------------- main inference -------------
@torch.no_grad()
def run_sequence(
    frames_dir: str,
    m0_path: str,
    ckpt_path: str,
    out_dir: str,
    prior_thr: float = 0.35,
    box_pad_frac: float = 0.10,
    min_box: int = 16,
    label_id: Optional[int] = None,
    gt_dir: Optional[str] = None,   # ← אם קיים GT לכל הפריימים
    save_csv: bool = True
):
    os.makedirs(out_dir, exist_ok=True)
    frames = _sorted_frames_in_dir(frames_dir)
    assert len(frames) >= 2, f"need at least 2 frames in: {frames_dir}"

    # --- load models ---
    prompter = PrompterNet(in_ch=7, base=32).to(DEVICE).eval()
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    prompter.load_state_dict(ckpt["model"])
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")

    # --- read M0 + choose label ---
    m0_arr = _load_label_map(m0_path)
    chosen_label = _choose_label_from_m0(m0_arr, label_id=label_id)
    if chosen_label == 0:
        raise ValueError("M0 appears empty (no positive labels).")
    print(f"[inference] chosen label from M0 = {chosen_label}")

    m_prev = (m0_arr == chosen_label).astype(np.float32)

    # save first mask (as our t=0 pred)
    out0 = _png_path_like(frames[0], out_dir)
    Image.fromarray((m_prev * 255).astype(np.uint8)).save(out0)

    # metrics storage
    metrics_rows = []
    have_gt = gt_dir is not None and os.path.isdir(gt_dir)

    for t in range(1, len(frames)):
        f_prev, f_curr = frames[t-1], frames[t]
        stem = os.path.splitext(os.path.basename(f_curr))[0]

        img_tm1 = cv2.cvtColor(cv2.imread(f_prev), cv2.COLOR_BGR2RGB)
        img_t   = cv2.cvtColor(cv2.imread(f_curr), cv2.COLOR_BGR2RGB)

        # tensors
        x_t   = torch.from_numpy(img_t.transpose(2,0,1)).float()[None] / 255.0
        x_tm1 = torch.from_numpy(img_tm1.transpose(2,0,1)).float()[None] / 255.0
        m_tm1 = torch.from_numpy(m_prev[None, None, ...]).float()
        x_t, x_tm1, m_tm1 = x_t.to(DEVICE), x_tm1.to(DEVICE), m_tm1.to(DEVICE)

        # prior logits from prompter
        logits_t, _ = prompter(x_t, x_tm1, m_tm1)
        raw = logits_t[0, 0]
        raw = torch.nan_to_num(raw, nan=0.0, posinf=10.0, neginf=-10.0)
        prior_logits = (raw * max(1.0, LOGIT_GAIN)).clamp(-max(6.0, LOGIT_CLAMP), max(6.0, LOGIT_CLAMP))
        prior01 = torch.sigmoid(prior_logits).detach().cpu().numpy()

        # SAM2 with mask+box from PRIOR (no GT at test)
        pred01, box_xyxy = sam2_forward_logits_and_or_box(
            predictor, img_t, prior_logits,
            use_mask=True, use_box=True,
            box_from="prior", prior_thr=prior_thr,
            box_pad_frac=box_pad_frac, min_box_size=min_box, gt01=None
        )

        # panel & save
        pred_path = _png_path_like(f_curr, out_dir)
        Image.fromarray((np.clip(pred01, 0, 1) * 255).astype(np.uint8)).save(pred_path)

        # metrics (optional if gt_dir given)
        dice = iou = None
        gt01 = None
        if have_gt:
            # assume same filename in gt_dir (png) — pick chosen_label
            gt_path = os.path.join(gt_dir, os.path.basename(pred_path))
            if not os.path.exists(gt_path):
                # try with original name from DAVIS (e.g. 00001.png)
                gt_path = os.path.join(gt_dir, os.path.basename(_png_path_like(f_curr, gt_dir)))
            if os.path.exists(gt_path):
                gt_arr = _load_label_map(gt_path)
                gt01   = (gt_arr == chosen_label).astype(np.float32)
                dice, iou = _dice_iou(pred01, gt01, thr=0.5)
                metrics_rows.append({"frame": stem, "dice": dice, "iou": iou})
            else:
                print(f"[warn] GT not found for frame: {stem}")

        # VIS panel


        _save_panel(
            out_dir=os.path.join(out_dir, "panels"),
            stem=stem,
            img_tm1_rgb=img_tm1,
            img_t_rgb=img_t,
            m_tm1_01=m_prev,
            prior01=prior01,  # ← חדש
            pred01=pred01,
            feed_box_xyxy=box_xyxy,
            gt01=gt01
        )

        # roll mask
        m_prev = (pred01 >= 0.5).astype(np.float32)

    # write metrics CSV + print summary
    if have_gt and save_csv and metrics_rows:
        csv_path = os.path.join(out_dir, "metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "dice", "iou"])
            w.writeheader(); w.writerows(metrics_rows)
        mean_d = sum(r["dice"] for r in metrics_rows) / len(metrics_rows)
        mean_i = sum(r["iou"]  for r in metrics_rows) / len(metrics_rows)
        print(f"[metrics] frames={len(metrics_rows)} | mean Dice={mean_d:.4f} | mean IoU={mean_i:.4f}")
        print(f"[metrics] saved to: {csv_path}")

    print("done:", out_dir)


if __name__ == "__main__":
    run_sequence(
        frames_dir=r"C:\pythonProject\DL_project\dataset\DAVIS2017\DAVIS_train\JPEGImages\480p\bike-packing",
        m0_path=r"C:\pythonProject\DL_project\dataset\DAVIS2017\DAVIS_train\Annotations\480p\bike-packing\00000.png",
        ckpt_path=r"C:\Users\28601\PycharmProjects\DL\runs\train_prompter_sam2\ALL_TRAIN_MASK_BOX_update\best.pt",
        out_dir=r".\runs\val\sub\net_max_box_davis\bike-packing",
        prior_thr=0.35, box_pad_frac=0.10, min_box=16,
        label_id=None,
        gt_dir=r"C:\pythonProject\DL_project\dataset\DAVIS2017\DAVIS_train\Annotations\480p\bike-packing",  # אופציונלי: בשביל Dice/IoU
        save_csv=True
    )
