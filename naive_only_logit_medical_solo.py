# naive_logit_feed_inference.py (MEDICAL-ready)
# --------------------------------------------
# T=0: מזינים ל-SAM2 לוגיטים של ה-GT (לאחר בחירת label מתוך M0 והתאמת רזולוציה)
# T>0: מזינים לוגיטים של הצעד הקודם ישירות ל-SAM2
# הגודל ל-mask_input נקבע אוטומטית ע"י DRY-RUN קצר שמחזיר את צורת ה-logits הנכונה.
# שומר per-frame metrics (Dice/J/F/J&F), heatmaps ל-logits, ו-triplets ויזואליים.
# --------------------------------------------

import os, csv, glob
from typing import Optional, Tuple, List
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch

from sam2.sam2_image_predictor import SAM2ImagePredictor


VIDEO_DIR = r"C:\pythonProject\DL_project\dataset\MEDICAL\video12\video12_19900"
OUT_DIR   = r".\runs\val\naive_logit_feed__MED"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH      = os.path.join(OUT_DIR, "per_frame_metrics.csv")
HEATMAP_DIR   = os.path.join(OUT_DIR, "heatmaps"); os.makedirs(HEATMAP_DIR, exist_ok=True)
TRIPLET_DIR   = os.path.join(OUT_DIR, "triplets"); os.makedirs(TRIPLET_DIR, exist_ok=True)

LABEL_ID: Optional[int] = None

# ================== מכשיר ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if (DEVICE == "cuda") else torch.float32

# ================== helpers ==================
def _frame_index_from_name(name: str) -> int:
    base = os.path.basename(name)
    m = __import__("re").search(r'(\d+)(?!.*\d)', base)
    return int(m.group(1)) if m else -1

def list_medical_rgb_and_gt(dirpath: str) -> Tuple[List[str], List[str]]:

    names = os.listdir(dirpath)
    rgb = sorted([f for f in names
                  if f.lower().endswith((".jpg",".jpeg",".png"))
                  and "mask" not in f.lower()],
                 key=lambda f: (_frame_index_from_name(f), f))
    gt  = sorted([f for f in names
                  if f.lower().endswith(".png") and "watershed_mask" in f.lower()],
                 key=lambda f: (_frame_index_from_name(f), f))
    return rgb, gt

def find_gt_for_stem(stem: str, search_dir: str) -> Optional[str]:
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

def _load_label_map(path: str) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im, dtype=np.int32) if im.mode == "P" else np.array(im.convert("L"), dtype=np.int32)
    return arr

def choose_label_medical(m_arr: np.ndarray, override: Optional[int] = None) -> int:

    if override is not None:
        return int(override)
    vals = set(np.unique(m_arr).astype(int).tolist())
    if 31 in vals:
        print("[select] chosen label 31 (medical preference)")
        return 31
    if 32 in vals:
        print("[select] chosen label 32 (medical preference)")
        return 32
    # fallback:
    vals_np, counts = np.unique(m_arr, return_counts=True)
    pos = [(int(v), int(c)) for v, c in zip(vals_np, counts) if int(v) > 0]
    if not pos:
        return 1 if (m_arr > 0).sum() > 0 else 0
    return max(pos, key=lambda vc: vc[1])[0]

def mask_to_logits(mask_bool: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = mask_bool.astype(np.float32)
    p = np.clip(p, eps, 1.0 - eps)
    logits = np.log(p/(1.0 - p))
    return logits[None, ...].astype(np.float32)

def _infer_prompt_hw_via_dry_run(predictor: SAM2ImagePredictor) -> Tuple[int, int]:

    dummy_pt = np.array([[1.0, 1.0]], dtype=np.float32)
    dummy_lb = np.array([0], dtype=np.int32)
    _, _, logits = predictor.predict(
        point_coords=dummy_pt, point_labels=dummy_lb,
        box=None, mask_input=None, multimask_output=False
    )
    if hasattr(logits, "shape"):
        hp, wp = int(logits.shape[-2]), int(logits.shape[-1])
    else:
        arr = logits
        if hasattr(arr, "detach"): arr = arr.detach().cpu().numpy()
        if isinstance(arr, (list, tuple)): arr = np.asarray(arr)
        hp, wp = int(arr.shape[-2]), int(arr.shape[-1])
    return hp, wp

def _resize_logits(logits_1hw: np.ndarray, target_hw: Tuple[int,int]) -> np.ndarray:
    src = logits_1hw[0].astype(np.float32)
    Ht, Wt = target_hw
    resized = cv2.resize(src, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    return resized[None, ...].astype(np.float32)

def save_logits_heatmap(idx, logits, out_dir=HEATMAP_DIR):
    if logits is None: return
    arr = logits
    if hasattr(arr, "detach"): arr = arr.detach().cpu().numpy()
    if isinstance(arr, (list, tuple)): arr = np.asarray(arr)
    if arr.ndim == 3: arr = arr[0]
    plt.figure(figsize=(6,5))
    plt.imshow(arr, cmap="viridis")
    plt.colorbar(shrink=0.8)
    plt.title(f"Logits Heatmap - Frame {idx}")
    plt.savefig(os.path.join(out_dir, f"frame_{idx:05d}_logits.png"), bbox_inches="tight", dpi=150)
    plt.close()

def save_triplet(idx, image_rgb, gt_mask_bool, prev_pred_bool, pred_t_bool):
    def overlay(rgb, m_bool, rgba=(30/255,144/255,1.0,0.55)):
        if m_bool is None: return rgb
        img = rgb.astype(np.float32).copy()
        overlay_rgb = (np.array(rgba[:3])*255).astype(np.float32)
        alpha = np.float32(rgba[3])
        mask_f = m_bool.astype(np.float32)[...,None]
        img = (1 - alpha*mask_f)*img + (alpha*mask_f)*overlay_rgb
        return img.astype(np.uint8)

    fig, axes = plt.subplots(1,3, figsize=(20,6))
    axes[0].imshow(overlay(image_rgb, gt_mask_bool, (0.1,0.8,0.2,0.50))); axes[0].set_title(f"GT  (t={idx})"); axes[0].axis("off")
    axes[1].imshow(overlay(image_rgb, prev_pred_bool, (0.85,0.2,0.8,0.45))); axes[1].set_title(f"Prev Pred (t-1)"); axes[1].axis("off")
    axes[2].imshow(overlay(image_rgb, pred_t_bool, (30/255,144/255,1.0,0.55))); axes[2].set_title(f"Pred (t={idx})"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(TRIPLET_DIR, f"frame_{idx:05d}_triplet.png"), bbox_inches="tight", dpi=150)
    plt.close()

# ============ KPI ============
def _binary_boundary(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dil = cv2.dilate(m, kernel, iterations=1)
    ero = cv2.erode(m, kernel, iterations=1)
    b = (dil - ero)
    return (b > 0).astype(np.uint8)

def _boundary_f_measure(pred01: np.ndarray, gt01: np.ndarray, tol: int = 2, eps: float = 1e-6) -> float:
    pb = _binary_boundary(pred01)
    gb = _binary_boundary(gt01)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    ksz = 2*tol + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    gb_dil = cv2.dilate(gb, kernel, iterations=1)
    pb_dil = cv2.dilate(pb, kernel, iterations=1)
    tp_p = float((pb & (gb_dil > 0)).sum())
    prec = tp_p / (float(pb.sum()) + eps)
    tp_g = float((gb & (pb_dil > 0)).sum())
    rec  = tp_g / (float(gb.sum()) + eps)
    denom = (prec + rec)
    return 0.0 if denom == 0.0 else (2.0 * prec * rec) / denom

def _dice_and_iou(pred01: np.ndarray, gt01: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    p = (pred01 > 0.5).astype(np.float32)
    g = (gt01   > 0.5).astype(np.float32)
    inter = float((p*g).sum())
    dice = (2*inter + eps) / (p.sum() + g.sum() + eps)
    union = float((p + g - p*g).sum())
    iou  = (inter + eps) / (union + eps)
    return float(dice), float(iou)
# ============================================

def main():
    np.random.seed(0); torch.manual_seed(0)

    frame_files, _ = list_medical_rgb_and_gt(VIDEO_DIR)
    assert len(frame_files) >= 2, f"Need >=2 RGB frames in {VIDEO_DIR}"

    init_idx = None
    init_gt_path = None
    for k, fn_img in enumerate(frame_files):
        stem = os.path.splitext(fn_img)[0]
        p = find_gt_for_stem(stem, VIDEO_DIR)
        if p is not None:
            init_idx = k
            init_gt_path = p
            break
    if init_idx is None:
        raise RuntimeError(f"No watershed_mask GT found in {VIDEO_DIR}")

    m0_arr = _load_label_map(init_gt_path)
    chosen_label = choose_label_medical(m0_arr, override=LABEL_ID)
    if chosen_label == 0:
        raise ValueError("M0 appears empty (no positive labels).")
    print(f"[naive] chosen label from M0 = {chosen_label} | init_frame = {frame_files[init_idx]}")

    # --- predictor ---
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")
    #predictor.to(device=DEVICE, dtype=TORCH_DTYPE)

    # --- CSV header ---
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["frame_idx","dice","J","F","J&F"])

    prev_logits = None
    prev_pred_bool = None

    _acc = {"dice": [], "J": [], "F": [], "JF": []}

    for idx, fn_img in enumerate(frame_files):
        image_path = os.path.join(VIDEO_DIR, fn_img)
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        H, W = image_rgb.shape[:2]

        has_gt = False
        gt_mask_bool = None
        gt_path = None

        stem = os.path.splitext(fn_img)[0]
        gt_path = find_gt_for_stem(stem, VIDEO_DIR)
        if gt_path is not None:
            gt_arr = _load_label_map(gt_path)
            if 31 in np.unique(gt_arr):
                gt_mask_bool = (gt_arr == 31)
            elif 32 in np.unique(gt_arr):
                gt_mask_bool = (gt_arr == 32)
            else:
                gt_mask_bool = (gt_arr == chosen_label) if (chosen_label in np.unique(gt_arr)) else (gt_arr > 0)
            has_gt = True

        predictor.set_image(image_rgb)
        Hp, Wp = _infer_prompt_hw_via_dry_run(predictor)

        if idx == init_idx:
            if not has_gt:
                raise RuntimeError(f"Expected GT at init frame {fn_img}")
            init_logits = mask_to_logits(gt_mask_bool, eps=1e-4)
            init_logits = _resize_logits(init_logits, (Hp, Wp))

            masks_pred, scores, logits_curr = predictor.predict(
                point_coords=None, point_labels=None,
                box=None, mask_input=init_logits,
                multimask_output=False
            )
        elif (idx > init_idx):
            if prev_logits is None:
                print(f"[warn] missing prev_logits at frame {idx}; skipping frame.")
                continue
            masks_pred, scores, logits_curr = predictor.predict(
                point_coords=None, point_labels=None,
                box=None, mask_input=prev_logits,
                multimask_output=False
            )
        else:
            continue

        pred_mask = masks_pred[0]
        pred_bool = (pred_mask > 0) if pred_mask.dtype != bool else pred_mask
        if pred_bool.shape != (H,W):
            pred_bool = cv2.resize(pred_bool.astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST).astype(bool)

        prev_logits = logits_curr
        save_logits_heatmap(idx, logits_curr)

        if has_gt and gt_mask_bool is not None:
            pred01 = pred_bool.astype(np.float32)
            gt01   = gt_mask_bool.astype(np.float32)

            dice, J = _dice_and_iou(pred01, gt01)
            Fb      = _boundary_f_measure(pred01, gt01, tol=2)
            JF      = 0.5*(J + Fb)

            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([idx, dice, J, Fb, JF])

            _acc["dice"].append(dice); _acc["J"].append(J); _acc["F"].append(Fb); _acc["JF"].append(JF)

        save_triplet(idx, image_rgb, gt_mask_bool if has_gt else None,
                     prev_pred_bool if (idx>init_idx) else None, pred_bool)
        prev_pred_bool = pred_bool

        if (idx - init_idx) % 10 == 0 and (idx >= init_idx):
            if has_gt:
                print(f"[{idx}] Dice={dice:.4f}  J={J:.4f}  F={Fb:.4f}  J&F={JF:.4f}")
            else:
                print(f"[{idx}] (no GT)")

    if _acc["J"]:
        mean_d = float(np.mean(_acc["dice"]))
        mean_J = float(np.mean(_acc["J"]))
        mean_F = float(np.mean(_acc["F"]))
        mean_JF= float(np.mean(_acc["JF"]))
        print(f"\n[summary] frames(with GT)={len(_acc['J'])}  Dice={mean_d:.4f}  J(IoU)={mean_J:.4f}  F={mean_F:.4f}  J&F={mean_JF:.4f}")
    else:
        print("\n[summary] No frames with GT were evaluated.")

    print("Done. Results saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
