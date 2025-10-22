
import os, csv, json, numpy as np, cv2
from typing import Optional, Tuple, List, Dict
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import generate_binary_structure, binary_erosion, binary_dilation
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

LABEL_ID: Optional[int] = None
SAVE_TRIPLETS = False
SAVE_TRIPLETS = False
SAVE_PANEL_HEATMAPS = False
SAVE_RAW_LOGITS_HEATMAP = False
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

def save_logits_heatmap(idx, logits, out_dir: str):
    if logits is None: return
    os.makedirs(out_dir, exist_ok=True)
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

def save_triplet(idx, image_rgb, gt_mask_bool, pred_t_minus_1_bool, pred_t_bool, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
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
    plt.savefig(os.path.join(out_dir, f"frame_{idx:05d}_triplet.png"), bbox_inches="tight", dpi=150)
    plt.close()

# ----------------- ויזואליזציה בסגנון inference_VAL_batch -----------------
def _mask_overlay(img_bgr, mask01, color=(0,255,0), alpha=0.45):
    if mask01 is None:
        return img_bgr.copy()
    m = np.clip(mask01.astype(np.float32), 0, 1)
    col = np.zeros_like(img_bgr); col[:] = color
    return (img_bgr*(1-alpha) + col*alpha*m[...,None]).astype(np.uint8)

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
        if im.dtype != np.uint8:
            im = np.clip(im, 0, 255).astype(np.uint8)
        new_w = int(round(im.shape[1]*h/max(1,im.shape[0])))
        tiles.append(cv2.resize(im, (max(1,new_w), h), interpolation=cv2.INTER_AREA))
    if not tiles:
        return np.zeros((h, h, 3), dtype=np.uint8)
    return np.concatenate(tiles, axis=1)

def _save_heatmap01(arr01: np.ndarray, out_path: str):
    arr01 = np.clip(arr01.astype(np.float32), 0, 1)
    u8 = (arr01 * 255.0).astype(np.uint8)
    hm = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    cv2.imwrite(out_path, hm)

def _save_panel(
    out_dir: str,
    stem: str,
    img_tm1_rgb: np.ndarray,
    img_t_rgb:   np.ndarray,
    m_tm1_01:    Optional[np.ndarray],
    pred01:      np.ndarray,
    gt01:        Optional[np.ndarray]=None
):
    os.makedirs(out_dir, exist_ok=True)
    img_t_bgr   = cv2.cvtColor(img_t_rgb,   cv2.COLOR_RGB2BGR)
    img_tm1_bgr = cv2.cvtColor(img_tm1_rgb, cv2.COLOR_RGB2BGR) if img_tm1_rgb is not None else None

    prior_overlay = _mask_overlay(img_t_bgr, m_tm1_01, (255,0,0), 0.40) if m_tm1_01 is not None else _put_title(img_t_bgr.copy(), "no prior (t=0)")
    pred_overlay  = _mask_overlay(img_t_bgr, pred01,   (0,255,0), 0.45)

    row1_imgs = [
        _put_title(img_tm1_bgr, "I_{t-1}") if img_tm1_bgr is not None else _put_title(img_t_bgr.copy(), "I_{t-1} (N/A)"),
        _put_title(img_t_bgr,   "I_t"),
        _put_title(prior_overlay, "Prior (from t-1)"),
        _put_title(pred_overlay,  "SAM2 pred (overlay)")
    ]
    row1 = _hstack_resize(row1_imgs, 300)

    if gt01 is not None:
        gt_overlay  = _mask_overlay(img_t_bgr, gt01,   (0,0,255), 0.45)
        pred_on_gt  = _mask_overlay(gt_overlay, pred01, (0,255,0), 0.45)
        row2 = _hstack_resize([
            _put_title(gt_overlay, "GT overlay"),
            _put_title(pred_on_gt, "GT + Pred overlay")
        ], 300)
        W = max(row1.shape[1], row2.shape[1])
        if row1.shape[1] < W:
            pad_right = W - row1.shape[1]
            row1 = cv2.copyMakeBorder(row1, 0,0,0,pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
        if row2.shape[1] < W:
            pad_right = W - row2.shape[1]
            row2 = cv2.copyMakeBorder(row2, 0,0,0,pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
        panel = np.vstack([row1, row2])
    else:
        panel = row1

    cv2.imwrite(os.path.join(out_dir, f"{stem}_PANEL.jpg"), panel)

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
    tp_g = float((gb & pb_dil).sum()); rec  = tp_g / (float(gb.sum()) + eps)
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

def run_one_sequence_naive(
    video_dir: str,
    gt_dir: str,
    out_dir: str,
    label_id: Optional[int] = None,
    predictor_model: str = "facebook/sam2-hiera-small",
    seed: int = 0
) -> Dict[str, float]:
    os.makedirs(out_dir, exist_ok=True)
    heatmap_dir = os.path.join(out_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    panels_dir = os.path.join(out_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)
    hm2_dir = os.path.join(panels_dir, "heatmaps")
    os.makedirs(hm2_dir, exist_ok=True)

    np.random.seed(seed); torch.manual_seed(seed)

    frame_files = list_images(video_dir)
    mask_files  = list_images(gt_dir)
    assert len(frame_files)>0 and len(frame_files)==len(mask_files), f"[{os.path.basename(video_dir)}] Missing frames or masks."

    m0_path = os.path.join(gt_dir, mask_files[0])
    m0_arr  = _load_label_map(m0_path)
    chosen_label = _choose_label_from_m0(m0_arr, label_id=label_id)
    if chosen_label == 0:
        raise ValueError(f"[{os.path.basename(video_dir)}] M0 appears empty (no positive labels).")
    print(f"[{os.path.basename(video_dir)}] chosen label from M0 = {chosen_label}")

    predictor = SAM2ImagePredictor.from_pretrained(predictor_model)

    csv_path = os.path.join(out_dir, "per_frame_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["frame_idx","dice","J","F","J&F"])

    prev_logits = None
    prev_pred_bool = None

    acc_d, acc_J, acc_F, acc_JF = [], [], [], []

    for idx, (fn_img, fn_gt) in enumerate(zip(frame_files, mask_files)):
        image = np.array(Image.open(os.path.join(video_dir, fn_img)).convert("RGB"))
        gt_arr = _load_label_map(os.path.join(gt_dir, fn_gt))
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
            try:
                _save_panel(
                    out_dir=panels_dir, stem=f"{idx:05d}",
                    img_tm1_rgb=None, img_t_rgb=image,
                    m_tm1_01=None, pred01=(masks_pred[0] > 0).astype(np.float32),
                    gt01=gt_mask_bool.astype(np.float32)
                )
            except Exception as e:
                print(f"[warn][panel t=0] {e}")
        else:
            if prev_logits is None:
                print(f"[{os.path.basename(video_dir)}] [warn] missing prev_logits; skipping frame {idx}.")
                continue
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
        if SAVE_RAW_LOGITS_HEATMAP:
            save_logits_heatmap(idx, logits_curr, out_dir=heatmap_dir)

        pred01 = pred_bool.astype(np.float32)
        gt01   = gt_mask_bool.astype(np.float32)
        dice, J = _dice_and_iou(pred01, gt01)
        Fb      = _boundary_f_measure(pred01, gt01, tol=2)
        JF      = 0.5*(J + Fb)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([idx, dice, J, Fb, JF])

        acc_d.append(dice); acc_J.append(J); acc_F.append(Fb); acc_JF.append(JF)

        # --------- VIS PANELS + HEATMAPS ---------
        try:
            img_tm1_rgb = np.array(Image.open(os.path.join(video_dir, frame_files[max(0, idx-1)])).convert("RGB")) if idx>0 else None
            _save_panel(
                out_dir=panels_dir,
                stem=f"{idx:05d}",
                img_tm1_rgb=img_tm1_rgb,
                img_t_rgb=image,
                m_tm1_01=(prev_pred_bool.astype(np.float32) if prev_pred_bool is not None else None),
                pred01=pred01,
                gt01=gt01
            )
            if SAVE_PANEL_HEATMAPS:
                # prior
                if idx > 0 and prev_logits is not None:
                    prior_arr = prev_logits
                    if hasattr(prior_arr, "detach"):  # torch.Tensor
                        prior01 = prior_arr.detach().cpu().sigmoid().numpy()
                    else:  # numpy
                        arr = prior_arr[0] if getattr(prior_arr, "ndim", 0) == 3 else prior_arr
                        prior01 = 1.0 / (1.0 + np.exp(-arr))
                    if prior01.ndim == 3: prior01 = prior01[0]
                    _save_heatmap01(np.clip(prior01, 0, 1), os.path.join(hm2_dir, f"{idx:05d}_PRIOR_HM.png"))
                # pred
                _save_heatmap01(np.clip(pred01, 0, 1), os.path.join(hm2_dir, f"{idx:05d}_PRED_HM.png"))

        except Exception as e:
            print(f"[warn][panels/heatmaps] {e}")
        # -----------------------------------------

        if SAVE_TRIPLETS:
            save_triplet(idx, image, gt_mask_bool, prev_pred_bool if idx>0 else None, pred_bool, out_dir=out_dir)

        prev_pred_bool = pred_bool

        if idx % 10 == 0:
            print(f"[{os.path.basename(video_dir)}][{idx}] Dice={dice:.4f}  J={J:.4f}  F={Fb:.4f}  J&F={JF:.4f}")

    summary = {
        "frames": len(acc_J),
        "mean_dice": float(np.mean(acc_d)) if acc_d else 0.0,
        "mean_j":    float(np.mean(acc_J)) if acc_J else 0.0,
        "mean_f":    float(np.mean(acc_F)) if acc_F else 0.0,
        "mean_jf":   float(np.mean(acc_JF)) if acc_JF else 0.0,
    }

    if acc_J:
        print(f"[{os.path.basename(video_dir)}][summary] frames={summary['frames']}  "
              f"Dice={summary['mean_dice']:.4f}  J(IoU)={summary['mean_j']:.4f}  "
              f"F={summary['mean_f']:.4f}  J&F={summary['mean_jf']:.4f}")
    return summary

def _load_sequences_from_split_json(split_json_path: str) -> Dict[str, List[str]]:
    with open(split_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    norm: Dict[str, List[str]] = {}
    for k, v in data.items():
        key = str(k).lower()
        if isinstance(v, dict) and "sequences" in v:
            norm[key] = list(v.get("sequences", []))
        elif isinstance(v, list):
            norm[key] = list(v)
        else:
            for alt in ["val","validation","valid","train","test"]:
                if alt in key:
                    if isinstance(v, dict):
                        for cand in ["list","names","seqs","items"]:
                            if cand in v and isinstance(v[cand], list):
                                norm[key] = list(v[cand])
                                break
                    break
    return norm

def _pick_subset_key(split_map: Dict[str, List[str]], prefer: List[str]) -> Optional[str]:
    for k in prefer:
        if k in split_map:
            return k
    for k in split_map.keys():
        lk = k.lower()
        if lk in prefer: return k
    return None

def run_val_batch_naive(
    data_root: str,
    split_name: str,
    res: str,
    split_json_path: str,
    out_root: str,
    label_id: Optional[int] = None,
    subset_preference: List[str] = ("val","validation","valid"),
    predictor_model: str = "facebook/sam2-hiera-small",
    max_viz_seqs: Optional[int] = None,
    seed: int = 0
):
    os.makedirs(out_root, exist_ok=True)
    split_map = _load_sequences_from_split_json(split_json_path)

    subset_key = _pick_subset_key(split_map, [s.lower() for s in subset_preference])
    if subset_key is None:
        subset_key = next(iter(split_map.keys()))
        print(f"[warn] subset not found; using '{subset_key}' from split json.")

    sequences = split_map[subset_key]
    if not sequences:
        raise RuntimeError(f"No sequences found under '{subset_key}' in {split_json_path}")

    if max_viz_seqs is not None:
        sequences = sequences[:int(max_viz_seqs)]

    frames_base = os.path.join(data_root, split_name, "JPEGImages", res)
    masks_base  = os.path.join(data_root, split_name, "Annotations", res)

    summary_csv_path = os.path.join(out_root, "val_summary.csv")
    with open(summary_csv_path, "w", newline="") as fsum:
        w = csv.writer(fsum)
        w.writerow(["sequence","frames","mean_dice","mean_j","mean_f","mean_jf"])

        for seq in sequences:
            seq_frames_dir = os.path.join(frames_base, seq)
            seq_masks_dir  = os.path.join(masks_base,  seq)
            if not (os.path.isdir(seq_frames_dir) and os.path.isdir(seq_masks_dir)):
                print(f"[skip] {seq}: missing dir(s)")
                continue

            seq_out_dir = os.path.join(out_root, seq)
            try:
                summary = run_one_sequence_naive(
                    video_dir=seq_frames_dir,
                    gt_dir=seq_masks_dir,
                    out_dir=seq_out_dir,
                    label_id=label_id,
                    predictor_model=predictor_model,
                    seed=seed
                )
                w.writerow([seq, summary["frames"], summary["mean_dice"],
                            summary["mean_j"], summary["mean_f"], summary["mean_jf"]])
                print(f"[done] {seq}: mean_jf={summary['mean_jf']:.4f}")
            except Exception as e:
                print(f"[error] {seq}: {e}")

    print(f"\n[VAL] Summary saved to: {summary_csv_path}")

# ===================== main =====================
if __name__ == "__main__":
    DATA_ROOT  = r"C:\pythonProject\DL_project\dataset\DAVIS2017"
    SPLIT      = "DAVIS_train"
    RES        = "480p"
    SPLIT_JSON = r"C:\Users\28601\PycharmProjects\DL\tfds_davis_split_480p.json"

    OUT_ROOT   = r".\runs\preds_val_davis_only_logit"
    LABEL_ID   = None
    MAX_VIZ_SEQS = None

    run_val_batch_naive(
        data_root=DATA_ROOT,
        split_name=SPLIT,
        res=RES,
        split_json_path=SPLIT_JSON,
        out_root=OUT_ROOT,
        label_id=LABEL_ID,
        subset_preference=("val","validation","valid"),
        predictor_model="facebook/sam2-hiera-small",
        max_viz_seqs=MAX_VIZ_SEQS,
        seed=0
    )
