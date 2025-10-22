
import os, glob, csv
from typing import Optional, Tuple, List

import cv2
import torch
import numpy as np
from PIL import Image

from sam2.sam2_image_predictor import SAM2ImagePredictor
from TRAIN_MASK_BOX import (
    PrompterNet, _sorted_frames_in_dir, sam2_forward_logits_and_or_box,
    LOGIT_GAIN, LOGIT_CLAMP
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import re

_BIDI_HIDDEN = re.compile(r'[\u200e\u200f\u202a-\u202e]')

def _sanitize_path(p: str) -> str:
    if not isinstance(p, str):
        return p
    p = _BIDI_HIDDEN.sub('', p)
    return os.path.normpath(p)

def _safe_exists_dir(path: str) -> str:
    ap = _sanitize_path(path)
    if not os.path.isdir(ap):
        raise FileNotFoundError(
            f"[frames_dir not found]\n  got: {path}\n  sanitized: {ap}\n"
            f"  parent exists? {os.path.exists(os.path.dirname(ap))}\n  cwd: {os.getcwd()}"
        )
    return ap

def _pad_to_width(img_bgr: np.ndarray, width: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w == width:
        return img_bgr
    if w > width:
        return cv2.resize(img_bgr, (width, h), interpolation=cv2.INTER_AREA)
    pad_right = width - w
    return cv2.copyMakeBorder(img_bgr, 0, 0, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _load_label_map(path: str) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im, dtype=np.int32) if im.mode == "P" else np.array(im.convert("L"), dtype=np.int32)
    return arr

import re

def _frame_index_from_name(name: str) -> int:

    base = os.path.basename(name)
    m = re.search(r'(\d+)(?!.*\d)', base)
    return int(m.group(1)) if m else -1

def _sorted_medical_frames(frames_dir: str) -> List[str]:

    names, _ = _list_medical_frames_single_dir(frames_dir)
    names_sorted = sorted(names, key=lambda f: (_frame_index_from_name(f), f))
    idxs = [_frame_index_from_name(f) for f in names_sorted]
    if 0 in idxs:
        k = idxs.index(0)
        names_sorted = names_sorted[k:] + names_sorted[:k]
    return [os.path.join(frames_dir, f) for f in names_sorted]


def _choose_label_from_m0(m0_arr: np.ndarray, label_id: Optional[int] = None) -> int:

    if label_id is not None:
        return int(label_id)

    vals = np.unique(m0_arr)
    if 31 in vals:
        print("Selected target_label: 31 (medical preference)")
        return 31
    if 32 in vals:
        print("Selected target_label: 32 (medical preference)")
        return 32

    vals, counts = np.unique(m0_arr, return_counts=True)
    pos = [(int(v), int(c)) for v, c in zip(vals, counts) if int(v) > 0]
    if not pos:  # binary 0/255 -> treat positive as 1
        return 1 if (m0_arr > 0).sum() > 0 else 0
    return max(pos, key=lambda vc: vc[1])[0]


def _dice_iou(pred01: np.ndarray, gt01: np.ndarray, thr: float = 0.5, eps: float = 1e-6) -> Tuple[float, float]:
    p = (pred01 >= thr).astype(np.float32)
    g = (gt01  >= 0.5).astype(np.float32)
    inter = float((p * g).sum())
    dice = (2 * inter + eps) / (p.sum() + g.sum() + eps)
    union = float((p + g - p * g).sum())
    iou = (inter + eps) / (union + eps)
    return dice, iou


def _as_u8_img(arr: np.ndarray) -> np.ndarray:
    if arr is None:
        return None
    if arr.dtype in (np.float32, np.float64):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _mask_overlay(img_bgr: np.ndarray, mask01: np.ndarray, color_bgr=(0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    base = img_bgr.copy()
    m = (mask01 >= 0.5).astype(np.uint8)
    overlay = np.zeros_like(base)
    overlay[m == 1] = color_bgr
    return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)


def _put_title(img_bgr: np.ndarray, text: str) -> np.ndarray:
    img = img_bgr.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _hstack_resize(imgs: List[np.ndarray], H: int) -> np.ndarray:
    rs = [cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H), interpolation=cv2.INTER_AREA) for im in imgs]
    return np.hstack(rs)


def _save_panel(out_dir: str,
                stem: str,
                img_tm1_rgb: np.ndarray,
                img_t_rgb: np.ndarray,
                m_tm1_01: np.ndarray,
                pred01: np.ndarray,
                feed_box_xyxy: Optional[Tuple[float, float, float, float]] = None,
                gt01: Optional[np.ndarray] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # prepare visuals
    img_tm1_bgr = cv2.cvtColor(img_tm1_rgb, cv2.COLOR_RGB2BGR)
    img_t_bgr   = cv2.cvtColor(img_t_rgb,   cv2.COLOR_RGB2BGR)

    m_prev_overlay = _mask_overlay(img_tm1_bgr, m_tm1_01, (255, 0, 0), 0.45)
    pred_overlay   = _mask_overlay(img_t_bgr, pred01,   (0, 255, 0), 0.45)

    if feed_box_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in feed_box_xyxy]
        cv2.rectangle(pred_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    row1 = _hstack_resize([
        _put_title(img_tm1_bgr, "frame t-1"),
        _put_title(img_t_bgr,   "frame t"),
        _put_title(m_prev_overlay, "prev mask overlay"),
        _put_title(pred_overlay, "SAM2 pred (overlay)")
    ], 300)

    # row 2 optional: GT overlays
    if gt01 is not None:
        gt_overlay = _mask_overlay(img_t_bgr, gt01, (0, 0, 255), 0.45)
        pred_on_gt = _mask_overlay(gt_overlay, pred01, (0, 255, 0), 0.45)
        row2 = _hstack_resize([
            _put_title(gt_overlay, "GT overlay"),
            _put_title(pred_on_gt, "GT + Pred overlay")
        ], 300)
        W = max(row1.shape[1], row2.shape[1])
        row1 = _pad_to_width(row1, W)
        row2 = _pad_to_width(row2, W)
        panel = np.vstack([row1, row2])
    else:
        panel = row1

    panel_path = os.path.join(out_dir, f"{stem}_PANEL.jpg")
    cv2.imwrite(panel_path, panel)


def _png_path_like(fr_path: str, out_dir: str) -> str:
    name = os.path.splitext(os.path.basename(fr_path))[0] + ".png"
    return os.path.join(out_dir, name)


# --------- medical helpers: SINGLE-DIR listing and GT lookup by watershed_mask ---------
def _list_medical_frames_single_dir(video_dir: str,
                                    image_extension: str = ".jpg",
                                    gt_extension: str = ".png") -> Tuple[List[str], List[str]]:

    image_exts = (".jpg", ".jpeg", ".png")
    all_files = os.listdir(video_dir)

    image_files_with_ext = sorted(
        [f for f in all_files
         if f.lower().endswith(image_exts) and "mask" not in f.lower()]
    )
    gt_files_with_ext = sorted(
        [f for f in all_files
         if f.lower().endswith(gt_extension) and "watershed_mask" in f.lower()]
    )
    return image_files_with_ext, gt_files_with_ext



def _find_gt_for_image_medical(stem: str, search_dir: str) -> Optional[str]:
    patterns = [
        os.path.join(search_dir, f"{stem}*watershed_mask*.png"),
        os.path.join(search_dir, f"{stem}*watershed_mask*.PNG"),
    ]
    candidates: List[str] = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    if not candidates:
        return None
    return sorted(candidates)[0]


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
    gt_dir: Optional[str] = None,   # optional: kept for backward-compat; if None and medical -> use frames_dir
    save_csv: bool = True,
    use_medical_listing: bool = True,  # DEFAULT True for medical single-dir case
):
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = _safe_exists_dir(frames_dir)
    m0_path    = _sanitize_path(m0_path)

    # Decide where to search GT masks (for medical listing in SAME dir)
    gt_search_dir = gt_dir if (gt_dir is not None and os.path.isdir(gt_dir)) else frames_dir

    if use_medical_listing:
        frames = _sorted_medical_frames(frames_dir)
        if len(frames) == 0:
            frames = _sorted_frames_in_dir(frames_dir)
    else:
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

    # metrics storage (we'll compute only when a matching GT exists per frame)
    metrics_rows: List[dict] = []

    for t in range(1, len(frames)):
        f_prev, f_curr = frames[t - 1], frames[t]
        stem = os.path.splitext(os.path.basename(f_curr))[0]

        # read frames
        img_prev_bgr = cv2.imread(_sanitize_path(f_prev))
        img_curr_bgr = cv2.imread(_sanitize_path(f_curr))

        if img_prev_bgr is None:
            raise FileNotFoundError(
                f"[imread fail] prev frame not readable:\n  {f_prev}\n  sanitized: {_sanitize_path(f_prev)}")
        if img_curr_bgr is None:
            raise FileNotFoundError(
                f"[imread fail] curr frame not readable:\n  {f_curr}\n  sanitized: {_sanitize_path(f_curr)}")

        img_tm1 = cv2.cvtColor(img_prev_bgr, cv2.COLOR_BGR2RGB)
        img_t = cv2.cvtColor(img_curr_bgr, cv2.COLOR_BGR2RGB)

        # tensors
        x_t   = torch.from_numpy(img_t.transpose(2, 0, 1)).float()[None] / 255.0
        x_tm1 = torch.from_numpy(img_tm1.transpose(2, 0, 1)).float()[None] / 255.0
        m_tm1 = torch.from_numpy(m_prev[None, None, ...]).float()
        x_t, x_tm1, m_tm1 = x_t.to(DEVICE), x_tm1.to(DEVICE), m_tm1.to(DEVICE)

        # prior logits from prompter
        logits_t, _ = prompter(x_t, x_tm1, m_tm1)
        raw = logits_t[0, 0]
        raw = torch.nan_to_num(raw, nan=0.0, posinf=10.0, neginf=-10.0)
        prior_logits = (raw * max(1.0, LOGIT_GAIN)).clamp(
            -max(6.0, LOGIT_CLAMP), max(6.0, LOGIT_CLAMP)
        )

        # SAM2 with mask+box from PRIOR (no GT at test)
        pred01, box_xyxy = sam2_forward_logits_and_or_box(
            predictor, img_t, prior_logits,
            use_mask=True, use_box=True,
            box_from="prior", prior_thr=prior_thr,
            box_pad_frac=box_pad_frac, min_box_size=min_box, gt01=None
        )

        # save pred mask
        pred_path = _png_path_like(f_curr, out_dir)
        Image.fromarray((np.clip(pred01, 0, 1) * 255).astype(np.uint8)).save(pred_path)

        # metrics: try to find GT in SAME dir (watershed_mask)
        dice = iou = None
        gt01 = None
        gt_path = _find_gt_for_image_medical(stem, gt_search_dir)
        if gt_path is not None and os.path.exists(gt_path):
            gt_arr = _load_label_map(gt_path)
            vals = np.unique(gt_arr)
            if 31 in vals:
                label_for_gt = 31
            elif 32 in vals:
                label_for_gt = 32
            else:
                label_for_gt = chosen_label
            gt01 = (gt_arr == label_for_gt).astype(np.float32)
            dice, iou = _dice_iou(pred01, gt01, thr=0.5)
            metrics_rows.append({"frame": stem, "dice": dice, "iou": iou})
        else:
            # No GT for this specific frame -> ok (evaluation optional)
            pass

        # VIS panel
        _save_panel(
            out_dir=os.path.join(out_dir, "panels"),
            stem=stem,
            img_tm1_rgb=img_tm1,
            img_t_rgb=img_t,
            m_tm1_01=m_prev,
            pred01=pred01,
            feed_box_xyxy=box_xyxy,
            gt01=gt01
        )

        # roll mask to next step
        m_prev = (pred01 >= 0.5).astype(np.float32)

    # write metrics CSV + print summary
    if metrics_rows:
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
    # === Example usage ===
    run_sequence(
        frames_dir=r"C:\pythonProject\DL_project\dataset\MEDICAL\video37\video37_00528",  # images + watershed_mask*.png in SAME dir
        m0_path=r"C:\pythonProject\DL_project\dataset\MEDICAL\video37\video37_00528\frame_528_endo_watershed_mask.png",  # palette PNG containing label 31/32
        ckpt_path=r"C:\Users\28601\PycharmProjects\DL\runs\train_prompter_sam2\ALL_TRAIN_MASK_BOX_update\best.pt",
        out_dir="./preds/case_05",
        prior_thr=0.35, box_pad_frac=0.10, min_box=16,
        label_id=None,
        gt_dir=None,                 # None -> search GT in SAME dir
        save_csv=True,
        use_medical_listing=True,    # single-dir flow
    )
