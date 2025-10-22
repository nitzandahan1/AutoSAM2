
import os, csv, numpy as np, cv2
from typing import Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

VIDEO_DIR = r"C:\pythonProject\DL_project\dataset\DAVIS2017\DAVIS_train\JPEGImages\480p\bike-packing"
GT_DIR    = r"C:\pythonProject\DL_project\dataset\DAVIS2017\DAVIS_train\Annotations\480p\bike-packing"
OUT_DIR   = r".\runs\val\sub\naive_logit_feed__DAVIS"
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
    im = Image.open(path)
    arr = np.array(im, dtype=np.int32) if im.mode == "P" else np.array(im.convert("L"), dtype=np.int32)
    return arr

def _choose_label_from_m0(m0_arr: np.ndarray, label_id: Optional[int] = None) -> int:
    if label_id is not None:
        return int(label_id)
    vals, counts = np.unique(m0_arr, return_counts=True)
    pos = [(int(v), int(c)) for v, c in zip(vals, counts) if int(v) > 0]
    if not pos:
        return 1 if (m0_arr > 0).sum() > 0 else 0
    return max(pos, key=lambda vc: vc[1])[0]

def mask_to_logits(mask_bool: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = mask_bool.astype(np.float32)
    p = np.clip(p, eps, 1.0 - eps)
    logits = np.log(p/(1.0 - p))
    return logits[None, ...].astype(np.float32)

def _infer_prompt_hw_via_dry_run(predictor) -> Tuple[int, int]:
    import numpy as np
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

# ---------- overlays ----------
def _overlay(rgb, mask_bool, rgba):
    if mask_bool is None:
        return rgb
    img = rgb.astype(np.float32).copy()
    overlay_rgb = (np.array(rgba[:3])*255).astype(np.float32)
    alpha = float(rgba[3])
    mask_f = mask_bool.astype(np.float32)[...,None]
    img = (1 - alpha*mask_f)*img + (alpha*mask_f)*overlay_rgb
    return img.astype(np.uint8)

def _overlay_in_order(rgb, first_mask_bool, first_rgba, second_mask_bool, second_rgba):
    out = _overlay(rgb, first_mask_bool, first_rgba)
    out = _overlay(out, second_mask_bool, second_rgba)
    return out

def _mask_only_tile(mask_bool: np.ndarray, hw: Tuple[int,int], rgba) -> np.ndarray:
    H, W = hw
    base = np.zeros((H, W, 3), dtype=np.uint8)
    return _overlay(base, mask_bool, rgba)

def save_panel_2x3(idx, img_t_minus_1, img_t, prior_bool, gt_bool, pred_bool):

    RGBA_GT   = (1.0, 0.0, 0.0, 0.45)
    RGBA_PRED = (0.10, 0.90, 0.10, 0.50)
    RGBA_PRI  = (0.15, 0.45, 1.00, 0.55)

    H, W = img_t.shape[:2]

    # base tiles
    top0 = img_t_minus_1 if img_t_minus_1 is not None else img_t
    top1 = img_t
    top2 = _overlay(img_t, prior_bool, RGBA_PRI)

    # bottom (updated): באמצע ה-GT מעל התמונה עצמה
    bot0 = _overlay(img_t, pred_bool, RGBA_PRED)                                      # SAM2 pred over image
    bot1 = _overlay(img_t, gt_bool, RGBA_GT)                                          # GT over I_t (no pred)
    bot2 = _overlay_in_order(img_t, gt_bool, RGBA_GT, pred_bool, RGBA_PRED)           # GT with SAM2 pred on top

    # draw 2×3 on black background
    fig = plt.figure(figsize=(18, 8), constrained_layout=False, facecolor='black')
    gs  = fig.add_gridspec(nrows=2, ncols=3, wspace=0.02, hspace=0.10)

    axes = [
        fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2]),
        fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]), fig.add_subplot(gs[1,2]),
    ]
    imgs  = [top0, top1, top2, bot0, bot1, bot2]
    titles= [r"$I_{t-1}$", r"$I_t$", "Prior (from t−1)",
             "SAM2 pred", "GT (overlay)", "GT + SAM2 pred"]

    for ax, im, tt in zip(axes, imgs, titles):
        ax.set_facecolor('black')
        ax.imshow(im)
        ax.set_title(tt, fontsize=12, color='white')
        ax.axis("off")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.03, wspace=0.02, hspace=0.08)
    fig.savefig(os.path.join(OUT_DIR, f"frame_{idx:05d}_panel.png"),
                bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


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
    if pb.sum() == 0 and gb.sum() == 0: return 1.0
    if pb.sum() == 0 or gb.sum() == 0:  return 0.0
    ksz = 2*tol + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    gb_dil = cv2.dilate(gb, kernel, iterations=1)
    pb_dil = cv2.dilate(pb, kernel, iterations=1)
    tp_p = float((pb & gb_dil).sum()); prec = tp_p / (float(pb.sum()) + eps)
    tp_g = float((gb & pb_dil).sum());  rec  = tp_g / (float(gb.sum()) + eps)
    f = (2.0 * prec * rec) / (prec + rec + eps)
    return float(f)

def _dice_and_iou(pred01: np.ndarray, gt01: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    p = (pred01 > 0.5).astype(np.float32)
    g = (gt01   > 0.5).astype(np.float32)
    inter = float((p*g).sum())
    dice = (2*inter + eps) / (p.sum() + g.sum() + eps)
    union = float((p + g - p*g).sum())
    iou  = (inter + eps) / (union + eps)
    return float(dice), float(iou)

# ================== main loop ==================
def main():
    np.random.seed(0); torch.manual_seed(0)

    frame_files = list_images(VIDEO_DIR)
    mask_files  = list_images(GT_DIR)
    assert len(frame_files)>0 and len(frame_files)==len(mask_files), "Missing frames or masks."

    m0_path = os.path.join(GT_DIR, mask_files[0])
    m0_arr  = _load_label_map(m0_path)
    chosen_label = _choose_label_from_m0(m0_arr, label_id=LABEL_ID)
    if chosen_label == 0:
        raise ValueError("M0 appears empty (no positive labels).")
    print(f"[naive] chosen label from M0 = {chosen_label}")

    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")

    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["frame_idx","dice","J","F","J&F"])

    prev_logits = None
    prev_pred_bool = None
    prev_image = None

    _acc = {"dice": [], "J": [], "F": [], "JF": []}

    for idx, (fn_img, fn_gt) in enumerate(zip(frame_files, mask_files)):
        image = np.array(Image.open(os.path.join(VIDEO_DIR, fn_img)).convert("RGB"))
        gt_arr = _load_label_map(os.path.join(GT_DIR, fn_gt))
        gt_mask_bool = (gt_arr == chosen_label)

        H,W = image.shape[:2]

        predictor.set_image(image)
        Hp, Wp = _infer_prompt_hw_via_dry_run(predictor)

        if idx == 0:
            init_logits = mask_to_logits(gt_mask_bool, eps=1e-4)
            init_logits = _resize_logits(init_logits, (Hp, Wp))
            masks_pred, scores, logits_curr = predictor.predict(
                point_coords=None, point_labels=None,
                box=None, mask_input=init_logits,
                multimask_output=False
            )
        else:
            masks_pred, scores, logits_curr = predictor.predict(
                point_coords=None, point_labels=None,
                box=None, mask_input=prev_logits,
                multimask_output=False
            )

        pred_mask = masks_pred[0]
        pred_bool = (pred_mask > 0) if pred_mask.dtype != bool else pred_mask
        if pred_bool.shape != (H,W):
            pred_bool = cv2.resize(pred_bool.astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST).astype(bool)

        prev_logits = logits_curr
        save_logits_heatmap(idx, logits_curr)

        pred01 = pred_bool.astype(np.float32)
        gt01   = gt_mask_bool.astype(np.float32)

        dice, J = _dice_and_iou(pred01, gt01)
        Fb      = _boundary_f_measure(pred01, gt01, tol=2)
        JF      = 0.5*(J + Fb)

        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([idx, dice, J, Fb, JF])

        _acc["dice"].append(dice); _acc["J"].append(J); _acc["F"].append(Fb); _acc["JF"].append(JF)

        save_panel_2x3(
            idx=idx,
            img_t_minus_1=prev_image,
            img_t=image,
            prior_bool=prev_pred_bool if idx>0 else None,
            gt_bool=gt_mask_bool,
            pred_bool=pred_bool
        )
        prev_pred_bool = pred_bool
        prev_image = image

        if idx % 10 == 0:
            print(f"[{idx}] Dice={dice:.4f}  J={J:.4f}  F={Fb:.4f}  J&F={JF:.4f}")

    if _acc["J"]:
        mean_d = np.mean(_acc["dice"]); mean_J = np.mean(_acc["J"])
        mean_F = np.mean(_acc["F"]);    mean_JF = np.mean(_acc["JF"])
        print(f"\n[summary] frames={len(_acc['J'])}  Dice={mean_d:.4f}  J(IoU)={mean_J:.4f}  F={mean_F:.4f}  J&F={mean_JF:.4f}")

    print("Done. Results saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
