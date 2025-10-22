# --------------------------------------------------------------
# Run naive "logit feed" inference of SAM2 over a list of medical
# sequences (each is a single folder containing RGB frames + GT masks
# named with 'watershed_mask'). For each sequence:
# T=0: מחשבים לוגיטים מה־GT, מתאימים לגודל ה־prompt של SAM2, ומזינים ל־SAM2 עם BOX מה־GT (tight box עם pad=10% ו־min=16px).
# T>0: ללא קופסה; מזינים רק mask_input=prev_logits.
# Saves per-frame CSV (Dice/J/F/J&F), logits heatmaps, and triplet visuals.
# Prints per-sequence summary and overall MACRO / MICRO averages.
# --------------------------------------------------------------

import os, csv, glob, re
from typing import Optional, Tuple, List, Dict
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch

from sam2.sam2_image_predictor import SAM2ImagePredictor

# ================== CONFIG: paths (עדכני לפי המחשב שלך) ==================
MEDICAL_ROOT = r"C:\pythonProject\DL_project\dataset\MEDICAL"  # <<< עדכני ל-root של הווידאוים
OUT_ROOT     = r".\runs\val\naive_logit_BOX_feed__MED"             # שורש פלט; לכל סט תיווצר תת-תיקייה
os.makedirs(OUT_ROOT, exist_ok=True)

# אם תרצי לקבע label מסוים ידנית (למשל 31), הגדירי כאן מספר. אחרת השאירי None.
LABEL_ID: Optional[int] = None

# BOX settings (ל-T=0 בלבד)
BOX_PAD_FRAC = 0.10
MIN_BOX      = 16

# TEST LIST (כמו שסיפקת)
TEST_LIST = [
    r"video12\video12_19900", r"video28\video28_00240", r"video01\video01_14939",
    r"video28\video28_00480", r"video43\video43_00627", r"video12\video12_19660",
    r"video01\video01_16585", r"video52\video52_00320", r"video43\video43_00467",
    r"video52\video52_00160", r"video24\video24_10076", r"video12\video12_15830",
    r"video52\video52_00400", r"video26\video26_02175", r"video01\video01_15099",
    r"video35\video35_00940", r"video01\video01_16425", r"video12\video12_19580",
    r"video12\video12_19500", r"video48\video48_00801", r"video18\video18_01139",
    r"video28\video28_00400", r"video43\video43_00787", r"video25\video25_00482",
    r"video37\video37_00688", r"video48\video48_00561", r"video52\video52_00080",
    r"video24\video24_09916", r"video01\video01_15019", r"video37\video37_00848",
    r"video35\video35_01100",
]

# ================== DEVICE/DTYPE ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if (DEVICE == "cuda") else torch.float32

# ================== GLOBAL OUTPUT DIRS (updated per sequence) ==================
CSV_PATH: Optional[str] = None
HEATMAP_DIR: Optional[str] = None
TRIPLET_DIR: Optional[str] = None

def _set_output_dirs_for_sequence(seq_relpath: str) -> str:
    """
    Configure globals CSV/HEATMAP/TRIPLET for the current sequence under OUT_ROOT/seq_name.
    Returns the absolute output directory for this sequence.
    """
    global CSV_PATH, HEATMAP_DIR, TRIPLET_DIR
    safe_name = seq_relpath.replace("\\", "__").replace("/", "__")
    out_dir = os.path.join(OUT_ROOT, safe_name)
    os.makedirs(out_dir, exist_ok=True)
    CSV_PATH = os.path.join(out_dir, "per_frame_metrics.csv")
    HEATMAP_DIR = os.path.join(out_dir, "heatmaps"); os.makedirs(HEATMAP_DIR, exist_ok=True)
    TRIPLET_DIR = os.path.join(out_dir, "triplets"); os.makedirs(TRIPLET_DIR, exist_ok=True)
    return out_dir

# ================== HELPERS ==================
def _frame_index_from_name(name: str) -> int:
    """Extract last digit run from file name (if none -> -1)."""
    base = os.path.basename(name)
    m = re.search(r'(\d+)(?!.*\d)', base)
    return int(m.group(1)) if m else -1

def list_medical_rgb_and_gt(dirpath: str) -> Tuple[List[str], List[str]]:
    """
    ONE dir (medical):
    - Images: jpg/jpeg/png without 'mask' in name.
    - GT:     png that contains 'watershed_mask' in name.
    Returns two sorted lists of file names (not full paths): (rgb_list, gt_list)
    """
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
    """Find a matching GT (watershed_mask) by stem in the same folder."""
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
    """Load GT as label map (keep palette if 'P', else L-grayscale)."""
    im = Image.open(path)
    arr = np.array(im, dtype=np.int32) if im.mode == "P" else np.array(im.convert("L"), dtype=np.int32)
    return arr

def choose_label_medical(m_arr: np.ndarray, override: Optional[int] = None) -> int:
    """
    Medical preferences: override -> 31 -> 32 -> fallback to most frequent positive.
    """
    if override is not None:
        return int(override)
    vals = set(np.unique(m_arr).astype(int).tolist())
    if 31 in vals:
        print("[select] chosen label 31 (medical preference)")
        return 31
    if 32 in vals:
        print("[select] chosen label 32 (medical preference)")
        return 32
    vals_np, counts = np.unique(m_arr, return_counts=True)
    pos = [(int(v), int(c)) for v, c in zip(vals_np, counts) if int(v) > 0]
    if not pos:
        return 1 if (m_arr > 0).sum() > 0 else 0
    return max(pos, key=lambda vc: vc[1])[0]

def mask_to_logits(mask_bool: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Convert binary mask (H,W) to logits (1,H,W)."""
    p = mask_bool.astype(np.float32)
    p = np.clip(p, eps, 1.0 - eps)
    logits = np.log(p/(1.0 - p))
    return logits[None, ...].astype(np.float32)

def _infer_prompt_hw_via_dry_run(predictor: SAM2ImagePredictor) -> Tuple[int, int]:
    """
    Dry-run with a dummy point to reveal logits shape the model returns/expects.
    predictor.set_image(image) must be called before using this.
    """
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
    """Resize from (1,H,W) to (1,Ht,Wt) in logits coordinate space."""
    src = logits_1hw[0].astype(np.float32)
    Ht, Wt = target_hw
    resized = cv2.resize(src, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    return resized[None, ...].astype(np.float32)

# --- NEW (minimal): GT mask -> tight bbox (with pad & min size), used only at T=0 ---
def _mask_to_box_xyxy(mask01: np.ndarray,
                      pad_frac: float = BOX_PAD_FRAC,
                      min_size: int = MIN_BOX,
                      H: Optional[int] = None,
                      W: Optional[int] = None) -> Optional[Tuple[int,int,int,int]]:
    m = (mask01 > 0.5).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    if H is None or W is None:
        H, W = m.shape[:2]
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    py = int(round(h * pad_frac)); px = int(round(w * pad_frac))
    y0 = max(0, y0 - py); y1 = min(H - 1, y1 + py)
    x0 = max(0, x0 - px); x1 = min(W - 1, x1 + px)
    if (y1 - y0 + 1) < min_size:
        extra = (min_size - (y1 - y0 + 1))
        y0 = max(0, y0 - extra // 2); y1 = min(H - 1, y1 + (extra - extra // 2))
    if (x1 - x0 + 1) < min_size:
        extra = (min_size - (x1 - x0 + 1))
        x0 = max(0, x0 - extra // 2); x1 = min(W - 1, x1 + (extra - extra // 2))
    return (int(x0), int(y0), int(x1), int(y1))

def save_logits_heatmap(idx: int, logits, out_dir: Optional[str] = None):
    """Save logits heatmap for frame idx under current HEATMAP_DIR (unless out_dir given)."""
    if logits is None:
        return
    if out_dir is None:
        out_dir = HEATMAP_DIR
    arr = logits
    if hasattr(arr, "detach"): arr = arr.detach().cpu().numpy()
    if isinstance(arr, (list, tuple)): arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[0]
    plt.figure(figsize=(6, 5))
    plt.imshow(arr, cmap="viridis")
    plt.colorbar(shrink=0.8)
    plt.title(f"Logits Heatmap - Frame {idx}")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"frame_{idx:05d}_logits.png"), bbox_inches="tight", dpi=150)
    plt.close()

def save_triplet(idx: int, image_rgb: np.ndarray,
                 gt_mask_bool: Optional[np.ndarray],
                 prev_pred_bool: Optional[np.ndarray],
                 pred_t_bool: Optional[np.ndarray]):
    """Save 3-panel visual: GT overlay, previous prediction, current prediction."""
    def overlay(rgb, m_bool, rgba=(30/255,144/255,1.0,0.55)):
        if m_bool is None:
            return rgb
        img = rgb.astype(np.float32).copy()
        overlay_rgb = (np.array(rgba[:3])*255).astype(np.float32)
        alpha = np.float32(rgba[3])
        mask_f = m_bool.astype(np.float32)[..., None]
        img = (1 - alpha*mask_f)*img + (alpha*mask_f)*overlay_rgb
        return img.astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].imshow(overlay(image_rgb, gt_mask_bool, (0.1, 0.8, 0.2, 0.50))); axes[0].set_title(f"GT  (t={idx})"); axes[0].axis("off")
    axes[1].imshow(overlay(image_rgb, prev_pred_bool, (0.85, 0.2, 0.8, 0.45))); axes[1].set_title("Prev Pred (t-1)"); axes[1].axis("off")
    axes[2].imshow(overlay(image_rgb, pred_t_bool, (30/255,144/255,1.0,0.55))); axes[2].set_title(f"Pred (t={idx})"); axes[2].axis("off")
    plt.tight_layout()
    os.makedirs(TRIPLET_DIR, exist_ok=True)
    plt.savefig(os.path.join(TRIPLET_DIR, f"frame_{idx:05d}_triplet.png"), bbox_inches="tight", dpi=150)
    plt.close()

# ================== KPI ==================
def _binary_boundary(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
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

# ================== RUN SINGLE SEQUENCE ==================
def run_single(video_dir: str) -> Dict[str, object]:

    np.random.seed(0); torch.manual_seed(0)

    frame_files, _ = list_medical_rgb_and_gt(video_dir)
    assert len(frame_files) >= 2, f"Need >=2 RGB frames in {video_dir}"

    init_idx, init_gt_path = None, None
    for k, fn_img in enumerate(frame_files):
        stem = os.path.splitext(fn_img)[0]
        p = find_gt_for_stem(stem, video_dir)
        if p is not None:
            init_idx, init_gt_path = k, p
            break
    if init_idx is None:
        raise RuntimeError(f"No watershed_mask GT found in {video_dir}")

    m0_arr = _load_label_map(init_gt_path)
    chosen_label = choose_label_medical(m0_arr, override=LABEL_ID)
    if chosen_label == 0:
        raise ValueError("M0 appears empty (no positive labels).")
    print(f"[naive] chosen label from M0 = {chosen_label} | init_frame = {frame_files[init_idx]}")

    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")

    # CSV header
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["frame_idx","dice","J","F","J&F"])

    prev_logits, prev_pred_bool = None, None
    _acc = {"dice": [], "J": [], "F": [], "JF": []}
    frames_with_gt = 0

    for idx, fn_img in enumerate(frame_files):
        image_path = os.path.join(video_dir, fn_img)
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        H, W = image_rgb.shape[:2]

        # check if current frame has GT
        has_gt, gt_mask_bool = False, None
        stem = os.path.splitext(fn_img)[0]
        gt_path = find_gt_for_stem(stem, video_dir)
        if gt_path is not None:
            gt_arr = _load_label_map(gt_path)
            uniq = np.unique(gt_arr)
            if 31 in uniq:
                gt_mask_bool = (gt_arr == 31)
            elif 32 in uniq:
                gt_mask_bool = (gt_arr == 32)
            else:
                gt_mask_bool = (gt_arr == chosen_label) if (chosen_label in uniq) else (gt_arr > 0)
            has_gt = True

        # must set image before dry-run
        predictor.set_image(image_rgb)
        Hp, Wp = _infer_prompt_hw_via_dry_run(predictor)

        if idx == init_idx:
            if not has_gt:
                raise RuntimeError(f"Expected GT at init frame {fn_img}")
            # T=0: logits מה-GT + BOX מה-GT (tight+pad+min)
            init_logits = mask_to_logits(gt_mask_bool, eps=1e-4)
            init_logits = _resize_logits(init_logits, (Hp, Wp))
            box_xyxy = _mask_to_box_xyxy(gt_mask_bool.astype(np.float32),
                                         pad_frac=BOX_PAD_FRAC, min_size=MIN_BOX, H=H, W=W)
            masks_pred, scores, logits_curr = predictor.predict(
                point_coords=None, point_labels=None,
                box=box_xyxy,
                mask_input=init_logits,
                multimask_output=False
            )
        elif idx > init_idx:
            if prev_logits is None:
                print(f"[warn] missing prev_logits at frame {idx}; skipping frame.")
                continue
            masks_pred, scores, logits_curr = predictor.predict(
                point_coords=None, point_labels=None,
                box=None,
                mask_input=prev_logits,
                multimask_output=False
            )
        else:
            # skip frames before init_idx
            continue

        pred_mask = masks_pred[0]
        pred_bool = (pred_mask > 0) if pred_mask.dtype != bool else pred_mask
        if pred_bool.shape != (H, W):
            pred_bool = cv2.resize(pred_bool.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        # store logits for t+1
        prev_logits = logits_curr
        save_logits_heatmap(idx, logits_curr)

        # KPIs if GT exists
        if has_gt and gt_mask_bool is not None:
            frames_with_gt += 1
            pred01 = pred_bool.astype(np.float32)
            gt01   = gt_mask_bool.astype(np.float32)
            dice, J = _dice_and_iou(pred01, gt01)
            Fb      = _boundary_f_measure(pred01, gt01, tol=2)
            JF      = 0.5*(J + Fb)

            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([idx, dice, J, Fb, JF])

            _acc["dice"].append(dice); _acc["J"].append(J); _acc["F"].append(Fb); _acc["JF"].append(JF)

        # visuals
        save_triplet(idx, image_rgb,
                     gt_mask_bool if has_gt else None,
                     prev_pred_bool if (idx > init_idx) else None,
                     pred_bool)
        prev_pred_bool = pred_bool

        if (idx - init_idx) % 10 == 0 and (idx >= init_idx):
            if has_gt and gt_mask_bool is not None:
                print(f"[{idx}] Dice={dice:.4f}  J={J:.4f}  F={Fb:.4f}  J&F={JF:.4f}")
            else:
                print(f"[{idx}] (no GT)")

    if _acc["J"]:
        mean_d = float(np.mean(_acc["dice"]))
        mean_J = float(np.mean(_acc["J"]))
        mean_F = float(np.mean(_acc["F"]))
        mean_JF= float(np.mean(_acc["JF"]))
        print(f"\n[summary] frames(with GT)={frames_with_gt}  Dice={mean_d:.4f}  J(IoU)={mean_J:.4f}  F={mean_F:.4f}  J&F={mean_JF:.4f}")
        return {"dice": mean_d, "J": mean_J, "F": mean_F, "JF": mean_JF,
                "n_frames": frames_with_gt,
                "all_values": _acc}
    else:
        print("\n[summary] No frames with GT were evaluated.")
        return {"dice": None, "J": None, "F": None, "JF": None,
                "n_frames": 0, "all_values": _acc}

# ================== RUN ALL ==================
if __name__ == "__main__":
    per_seq: List[Tuple[str, Dict[str, object]]] = []
    micro_pool = {"dice": [], "J": [], "F": [], "JF": []}

    for seq in TEST_LIST:
        seq_abs = os.path.join(MEDICAL_ROOT, seq)
        out_dir = _set_output_dirs_for_sequence(seq)
        print("\n==============================================")
        print(f"=== Running sequence: {seq} ===")
        print(f"Input dir : {seq_abs}")
        print(f"Output dir: {out_dir}")

        stats = run_single(seq_abs)
        per_seq.append((seq, stats))

        # collect micro over all frames
        if stats["all_values"]["J"]:
            micro_pool["dice"].extend(stats["all_values"]["dice"])
            micro_pool["J"].extend(stats["all_values"]["J"])
            micro_pool["F"].extend(stats["all_values"]["F"])
            micro_pool["JF"].extend(stats["all_values"]["JF"])

    # Per-sequence summary print
    print("\n========== PER-SEQUENCE SUMMARY ==========")
    valid_for_macro = []
    for seq, s in per_seq:
        if s["J"] is not None:
            print(f"{seq:>25s} | Dice={s['dice']:.4f}  J={s['J']:.4f}  F={s['F']:.4f}  J&F={s['JF']:.4f}  (frames={s['n_frames']})")
            valid_for_macro.append(s)
        else:
            print(f"{seq:>25s} | No GT-evaluated frames.")

    # Macro (equal weight per sequence)
    if valid_for_macro:
        macro_d = float(np.mean([s["dice"] for s in valid_for_macro]))
        macro_J = float(np.mean([s["J"]   for s in valid_for_macro]))
        macro_F = float(np.mean([s["F"]   for s in valid_for_macro]))
        macro_JF= float(np.mean([s["JF"]  for s in valid_for_macro]))
        print("\n----- MACRO AVERAGE (per-sequence mean) -----")
        print(f"Dice={macro_d:.4f}  J={macro_J:.4f}  F={macro_F:.4f}  J&F={macro_JF:.4f}")

    # Micro (over all frames from all sequences)
    if micro_pool["J"]:
        micro_d = float(np.mean(micro_pool["dice"]))
        micro_J = float(np.mean(micro_pool["J"]))
        micro_F = float(np.mean(micro_pool["F"]))
        micro_JF= float(np.mean(micro_pool["JF"]))
        print("\n----- MICRO AVERAGE (over all frames) -----")
        print(f"Dice={micro_d:.4f}  J={micro_J:.4f}  F={micro_F:.4f}  J&F={micro_JF:.4f}")

    print("\nDone. All results saved under:", OUT_ROOT)
