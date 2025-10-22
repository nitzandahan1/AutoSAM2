# naive_logit_feed_inference.py
# --------------------------------------------
# T=0: מזינים ל-SAM2 לוגיטים של ה-GT (לאחר בחירת label מתוך M0 והתאמת רזולוציה)
# T>0: מזינים לוגיטים של הצעד הקודם ישירות ל-SAM2
# הגודל ל-mask_input נקבע אוטומטית ע"י DRY-RUN קצר שמחזיר את צורת ה-logits הנכונה.
# שומר per-frame metrics (J/F/J&F), heatmaps ל-logits, ו-triplets ויזואליים.
# --------------------------------------------

import os, csv, numpy as np, cv2
from typing import Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import generate_binary_structure, binary_erosion, binary_dilation
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ================== paths (עדכני לפי המחשב שלך) ==================
VIDEO_DIR = r"C:\pythonProject\DL_project\dataset\DAVIS2017\DAVIS_train\JPEGImages\480p\bike-packing"
GT_DIR    = r"C:\pythonProject\DL_project\dataset\DAVIS2017\DAVIS_train\Annotations\480p\bike-packing"
OUT_DIR   = r".\runs\val\naive_logit_feed__TEMP"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "per_frame_metrics.csv")
HEATMAP_DIR = os.path.join(OUT_DIR, "heatmaps"); os.makedirs(HEATMAP_DIR, exist_ok=True)

LABEL_ID: Optional[int] = None

# ================== helpers ==================
def list_images(d, exts=(".jpg",".jpeg",".png",".JPG",".JPEG",".PNG")):
    names = [n for n in os.listdir(d) if os.path.splitext(n)[1] in exts]
    names.sort()
    return names

def _load_label_map(path: str) -> np.ndarray:
    """טעינת GT כ-label map (שומר ערכים קטגוריאליים אם 'P' אחרת L-grayscale)."""
    im = Image.open(path)
    arr = np.array(im, dtype=np.int32) if im.mode == "P" else np.array(im.convert("L"), dtype=np.int32)
    return arr

def _choose_label_from_m0(m0_arr: np.ndarray, label_id: Optional[int] = None) -> int:
    """
    בחירת label מתוך M0:
    - אם label_id סופק → משתמשים בו כפי שהוא.
    - אחרת: הערך החיובי (>0) הנפוץ ביותר.
    - אם אין חיוביים (בינארי 0/255): FG=1 אם יש פיקסלים >0, אחרת 0.
    """
    if label_id is not None:
        return int(label_id)
    vals, counts = np.unique(m0_arr, return_counts=True)
    pos = [(int(v), int(c)) for v, c in zip(vals, counts) if int(v) > 0]
    if not pos:
        return 1 if (m0_arr > 0).sum() > 0 else 0
    return max(pos, key=lambda vc: vc[1])[0]

def mask_to_logits(mask_bool: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """ממיר מסכה בינארית (H,W) ללוגיטים (1,H,W)."""
    p = mask_bool.astype(np.float32)
    p = np.clip(p, eps, 1.0 - eps)
    logits = np.log(p/(1.0 - p))         # logit(p)
    return logits[None, ...].astype(np.float32)

def _infer_prompt_hw_via_dry_run(predictor) -> Tuple[int, int]:
    """
    DRY-RUN קטן עם נקודה דמיונית כדי לברר את צורת ה-logits שהמודל מחזיר/מצפה לה.
    predictor.set_image(image) חייב לרוץ לפני הקריאה כאן.
    """
    import numpy as np
    dummy_pt = np.array([[1.0, 1.0]], dtype=np.float32)      # נקודה כלשהי בתוך התמונה
    dummy_lb = np.array([0], dtype=np.int32)                 # label לא משנה
    _, _, logits = predictor.predict(
        point_coords=dummy_pt, point_labels=dummy_lb,
        box=None, mask_input=None, multimask_output=False
    )
    # logits shape: (1, Hp, Wp) או טנסור דומה
    if hasattr(logits, "shape"):
        hp, wp = int(logits.shape[-2]), int(logits.shape[-1])
    else:
        arr = logits
        if hasattr(arr, "detach"): arr = arr.detach().cpu().numpy()
        if isinstance(arr, (list, tuple)): arr = np.asarray(arr)
        hp, wp = int(arr.shape[-2]), int(arr.shape[-1])
    return hp, wp

def _resize_logits(logits_1hw: np.ndarray, target_hw: Tuple[int,int]) -> np.ndarray:
    """Resize מ-(1,H,W) ל-(1,Ht,Wt) בקואורדינטות ה-logits (לא בתמונת המקור)."""
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

def save_triplet(idx, image_rgb, gt_mask_bool, pred_t_minus_1_bool, pred_t_bool):
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
    axes[1].imshow(overlay(image_rgb, pred_t_minus_1_bool, (0.85,0.2,0.8,0.45))); axes[1].set_title(f"Prev Pred (t-1)"); axes[1].axis("off")
    axes[2].imshow(overlay(image_rgb, pred_t_bool, (30/255,144/255,1.0,0.55))); axes[2].set_title(f"Pred (t={idx})"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"frame_{idx:05d}_triplet.png"), bbox_inches="tight", dpi=150)
    plt.close()

# ============ KPI (מהקובץ שצרפת) ============
def _binary_boundary(mask01: np.ndarray) -> np.ndarray:
    """
    גבול בעובי ~פיקסל אחד (uint8 ב-{0,1}) מתוך מסכה בינארית {0,1}.
    """
    m = (mask01 > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dil = cv2.dilate(m, kernel, iterations=1)
    ero = cv2.erode(m, kernel, iterations=1)
    b = (dil - ero)
    return (b > 0).astype(np.uint8)

def _boundary_f_measure(pred01: np.ndarray, gt01: np.ndarray, tol: int = 2, eps: float = 1e-6) -> float:
    """
    Boundary F-measure עם טולרנס (כמו בקובץ ההפניה).
    pred01, gt01 ב-[0,1] (ימיר לבינארי בסף 0.5).
    """
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

    tp_p = float((pb & gb_dil).sum())
    prec = tp_p / (float(pb.sum()) + eps)

    tp_g = float((gb & pb_dil).sum())
    rec = tp_g / (float(gb.sum()) + eps)

    f = (2.0 * prec * rec) / (prec + rec + eps)
    return float(f)

def _dice_and_iou(pred01: np.ndarray, gt01: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    """
    Dice + IoU (J). הקלט ב-{0,1}.
    """
    p = (pred01 > 0.5).astype(np.float32)
    g = (gt01   > 0.5).astype(np.float32)
    inter = float((p*g).sum())
    dice = (2*inter + eps) / (p.sum() + g.sum() + eps)
    union = float((p + g - p*g).sum())
    iou  = (inter + eps) / (union + eps)
    return float(dice), float(iou)
# ============================================

#def j_f_scores(pred_bool: np.ndarray, gt_bool: np.ndarray):
    """
    (נשמרת לתאימות לאחור — לא בשימוש עוד ל-KPI)
    """
    # Jaccard (IoU)
 #   inter = np.logical_and(pred_bool, gt_bool).sum()
  #  union = np.logical_or(pred_bool, gt_bool).sum()
 #   J = 1.0 if (union == 0 and inter == 0) else (inter/union if union>0 else 0.0)
    # F (Boundary F-measure בקירוב הישן)
  #  st = generate_binary_structure(2,1)
#    try:
 #       pred_eroded = binary_erosion(pred_bool, structure=st); bp = np.logical_and(pred_bool, np.logical_not(pred_eroded))
 #   except RuntimeError:
 #       bp = np.zeros_like(pred_bool, bool)
  #  try:
  #      gt_eroded = binary_erosion(gt_bool, structure=st); bg = np.logical_and(gt_bool, np.logical_not(gt_eroded))
  #  except RuntimeError:
   #     bg = np.zeros_like(gt_bool, bool)
 #   tol=5
  #  try: bp_dil = binary_dilation(bp, structure=st, iterations=tol)
  #  except RuntimeError: bp_dil = np.zeros_like(bp, bool)
 #   try: bg_dil = binary_dilation(bg, structure=st, iterations=tol)
  #  except RuntimeError: bg_dil = np.zeros_like(bg, bool)
  #  bp_match = np.logical_and(bp, bg_dil).sum()
  #  bg_match = np.logical_and(bg, bp_dil).sum()
 ##   prec = bp_match/(bp.sum()+1e-8)
 #   rec  = bg_match/(bg.sum()+1e-8)
 #   F = 1.0 if (bp.sum()==0 and bg.sum()==0) else (0.0 if (prec+rec)==0 else (2*prec*rec)/(prec+rec))
 #   return float(J), float(F), float((J+F)/2.0)

# ================== main loop ==================
def main():
    np.random.seed(0); torch.manual_seed(0)

    frame_files = list_images(VIDEO_DIR)
    mask_files  = list_images(GT_DIR)
    assert len(frame_files)>0 and len(frame_files)==len(mask_files), "Missing frames or masks."

    # --- בחירת label מתוך M0 (כמו בקוד שלך) ---
    m0_path = os.path.join(GT_DIR, mask_files[0])
    m0_arr  = _load_label_map(m0_path)
    chosen_label = _choose_label_from_m0(m0_arr, label_id=LABEL_ID)
    if chosen_label == 0:
        raise ValueError("M0 appears empty (no positive labels).")
    print(f"[naive] chosen label from M0 = {chosen_label}")

    # טוענים SAM2 (התאם את הצ'קפוינט/הדגם לפי הזמין אצלך)
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")

    # CSV header — הוגדל כדי לכלול גם Dice וליישר לפורמט KPI
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["frame_idx","dice","J","F","J&F"])

    prev_logits = None
    prev_pred_bool = None

    # לאיסוף ממוצעים גלובליים
    _acc = {"dice": [], "J": [], "F": [], "JF": []}

    for idx, (fn_img, fn_gt) in enumerate(zip(frame_files, mask_files)):
        # image + GT (כ-label map), ואז נגזור ממנו מסכה עבור chosen_label
        image = np.array(Image.open(os.path.join(VIDEO_DIR, fn_img)).convert("RGB"))
        gt_arr = _load_label_map(os.path.join(GT_DIR, fn_gt))
        gt_mask_bool = (gt_arr == chosen_label)

        H,W = image.shape[:2]

        # חשוב: תמיד set_image לפני מציאת גודל ה-prompt
        predictor.set_image(image)

        # DRY-RUN קצר להפקת גודל ה-logits שהמודל מצפה לו (Hp, Wp)
        Hp, Wp = _infer_prompt_hw_via_dry_run(predictor)

        if idx == 0:
            # T=0 — GT→logits (1,H,W), ואז resize לגודל (Hp,Wp) שהתגלה
            init_logits = mask_to_logits(gt_mask_bool, eps=1e-4)
            init_logits = _resize_logits(init_logits, (Hp, Wp))

            masks_pred, scores, logits_curr = predictor.predict(
                point_coords=None, point_labels=None,
                box=None, mask_input=init_logits,
                multimask_output=False
            )
        else:
            # T>0 — מכניסים לוגיטים של הצעד הקודם (כבר ב-(Hp,Wp) מהצעד הקודם)
            if prev_logits is None:
                print("[warn] missing prev_logits; skipping frame.")
                continue

            masks_pred, scores, logits_curr = predictor.predict(
                point_coords=None, point_labels=None,
                box=None, mask_input=prev_logits,
                multimask_output=False
            )

        # פלט מסכה
        pred_mask = masks_pred[0]
        pred_bool = (pred_mask > 0) if pred_mask.dtype != bool else pred_mask
        if pred_bool.shape != (H,W):
            pred_bool = cv2.resize(pred_bool.astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST).astype(bool)

        # שומרים את הלוגיטים לצעד הבא (בגודל ה-prompt)
        prev_logits = logits_curr
        save_logits_heatmap(idx, logits_curr)

        # ===== KPI (חדשים) =====
        # המרות ל-{0,1} כ-float לצורך חישובי KPI
        pred01 = pred_bool.astype(np.float32)
        gt01   = gt_mask_bool.astype(np.float32)

        dice, J = _dice_and_iou(pred01, gt01)
        Fb      = _boundary_f_measure(pred01, gt01, tol=2)
        JF      = 0.5*(J + Fb)

        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([idx, dice, J, Fb, JF])

        _acc["dice"].append(dice); _acc["J"].append(J); _acc["F"].append(Fb); _acc["JF"].append(JF)
        # ========================

        save_triplet(idx, image, gt_mask_bool, prev_pred_bool if idx>0 else None, pred_bool)
        prev_pred_bool = pred_bool

        if idx % 10 == 0:
            print(f"[{idx}] Dice={dice:.4f}  J={J:.4f}  F={Fb:.4f}  J&F={JF:.4f}")

    # סיכום קצר (לא חובה לשמירה) — רק הדפסה למסך
    if _acc["J"]:
        mean_d = np.mean(_acc["dice"]); mean_J = np.mean(_acc["J"])
        mean_F = np.mean(_acc["F"]);    mean_JF = np.mean(_acc["JF"])
        print(f"\n[summary] frames={len(_acc['J'])}  Dice={mean_d:.4f}  J(IoU)={mean_J:.4f}  F={mean_F:.4f}  J&F={mean_JF:.4f}")

    print("Done. Results saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
