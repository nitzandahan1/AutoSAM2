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
    h, w = img_bgr.shape[:2]
    if w == width:
        return img_bgr
    if w > width:
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

def _draw_contour_on(img_bgr: np.ndarray, mask01: np.ndarray, color=(128,0,128), thick: int = 3):
    out = img_bgr.copy()
    mask_u8 = (np.clip(mask01,0,1)*255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, thick, lineType=cv2.LINE_AA)
    return out

def _overlay_on(img_bgr: np.ndarray, mask01: np.ndarray, fill_bgr=(0,255,0), alpha: float=0.45, draw_outline=True):
    base = img_bgr.copy()
    m = (np.clip(mask01,0,1)).astype(np.float32)
    fill = np.zeros_like(base); fill[:] = fill_bgr
    out = (base*(1-alpha) + fill*alpha*m[...,None]).astype(np.uint8)
    if draw_outline and m.sum() > 0:
        out = _draw_contour_on(out, m, (128,0,128), 3)
    return out

def _make_overlap_trip(img_bgr: np.ndarray, pred01: np.ndarray, gt01: np.ndarray) -> np.ndarray:

    base = img_bgr.copy()
    p = (pred01 >= 0.5).astype(np.uint8)
    g = (gt01   >= 0.5).astype(np.uint8)

    tp = (p & g).astype(np.uint8)
    fn = ((~p.astype(bool)) & g.astype(bool)).astype(np.uint8)
    fp = ((p.astype(bool)) & (~g.astype(bool))).astype(np.uint8)

    def paint(mask, color_bgr):
        col = np.zeros_like(base); col[:] = color_bgr
        return (base*0.55 + col*0.45*mask[...,None]).astype(np.uint8)

    over = base.copy()
    over = np.where(fn[...,None]==1, paint(fn,(0,0,255)), over)   # FN = RED
    over = np.where(fp[...,None]==1, paint(fp,(255,0,0)), over)   # FP = BLUE
    over = np.where(tp[...,None]==1, paint(tp,(0,255,0)), over)   # TP = GREEN

    union = ((p+g) > 0).astype(np.uint8)
    over = _draw_contour_on(over, union, (128,0,128), 3)

    cv2.putText(over, "TP", (12,30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(over, "FN", (12,60),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(over, "FP", (12,90),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)

    middle = _overlay_on(base, g, fill_bgr=(255,128,0))  # teal/blue-like fill in example; adjust to taste
    right  = _overlay_on(base, p, fill_bgr=(0,255,0))

    trip = _hstack_resize([
        _put_title(over,   "Overlap: TP/FN/FP"),
        _put_title(middle, "GT"),
        _put_title(right,  "Pred")
    ], 300)
    return trip

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
    gt_dir: Optional[str] = None,
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
    trip_dir = os.path.join(out_dir, "triptych")
    os.makedirs(trip_dir, exist_ok=True)

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

        # SAM2 with mask+box from PRIOR (no GT at test)
        pred01, _ = sam2_forward_logits_and_or_box(
            predictor, img_t, prior_logits,
            use_mask=True, use_box=True,
            box_from="prior", prior_thr=prior_thr,
            box_pad_frac=box_pad_frac, min_box_size=min_box, gt01=None
        )

        # save pred mask (binary PNG for downstream)
        pred_path = _png_path_like(f_curr, out_dir)
        Image.fromarray((np.clip(pred01, 0, 1) * 255).astype(np.uint8)).save(pred_path)

        # metrics + triptych vis (requires GT)
        gt01 = None
        if have_gt:
            gt_path = os.path.join(gt_dir, os.path.basename(pred_path))
            if not os.path.exists(gt_path):
                gt_path = os.path.join(gt_dir, os.path.basename(_png_path_like(f_curr, gt_dir)))
            if os.path.exists(gt_path):
                gt_arr = _load_label_map(gt_path)
                gt01   = (gt_arr == chosen_label).astype(np.float32)
                dice, iou = _dice_iou(pred01, gt01, thr=0.5)
                metrics_rows.append({"frame": stem, "dice": dice, "iou": iou})

                # === NEW: triptych (Left overlap, Mid GT, Right Pred) ===
                trip_img = _make_overlap_trip(cv2.cvtColor(img_t, cv2.COLOR_RGB2BGR), pred01, gt01)
                cv2.imwrite(os.path.join(trip_dir, f"{stem}_TRIP.jpg"), trip_img)
            else:
                print(f"[warn] GT not found for frame: {stem}")
        else:
            # no GT — still make a Pred-only thumbnail to keep flow consistent
            pred_only = _overlay_on(cv2.cvtColor(img_t, cv2.COLOR_RGB2BGR), pred01, (0,255,0))
            pred_only = _put_title(pred_only, "Pred")
            cv2.imwrite(os.path.join(trip_dir, f"{stem}_PRED.jpg"), pred_only)

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
        frames_dir=r"C:\pythonProject\DL_project\dataset\DAVIS2017\DAVIS_train\JPEGImages\480p\motocross-jump",
        m0_path=r"C:\pythonProject\DL_project\dataset\DAVIS2017\DAVIS_train\Annotations\480p\motocross-jump\00000.png",
        ckpt_path=r"C:\Users\28601\PycharmProjects\DL\runs\train_prompter_sam2\ALL_TRAIN_MASK_BOX_update\best.pt",
        out_dir=r".\preds\EXAMPLE2",
        prior_thr=0.35, box_pad_frac=0.10, min_box=16,
        label_id=None,  # אפשר להכריח label מסוים אם צריך
        gt_dir=r"C:\pythonProject\DL_project\dataset\DAVIS2017\DAVIS_train\Annotations\480p\motocross-jump",
        save_csv=True
    )
