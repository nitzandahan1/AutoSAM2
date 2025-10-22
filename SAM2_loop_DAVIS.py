import os
import json  # [SPLIT]
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt

from sam2.sam2_video_predictor import SAM2VideoPredictor

ROOT_FRAMES_DIR = r"C:\pythonProject\DL_project\dataset\DAVIS2017\DAVIS_train\JPEGImages\480p"
ROOT_GT_DIR     = r"C:\pythonProject\DL_project\dataset\DAVIS2017\DAVIS_train\Annotations\480p"

VIDEO_DIR = os.path.join(ROOT_FRAMES_DIR, "bike-packing")
GT_DIR    = os.path.join(ROOT_GT_DIR, "bike-packing")

OUT_DIR   = r"./runs/val/preds_val_davis_VIDEO/"
os.makedirs(OUT_DIR, exist_ok=True)

# ================== DEVICE / AMP ==================
if torch.cuda.is_available():
    device = torch.device("cuda")
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
else:
    device = torch.device("cpu")
    class _NoOp:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    amp_ctx = _NoOp()
print(f"[info] device: {device}")

# ================== FLAGS ==================
VIS = True
CLEAN_VIZ = True

# ================== UTILS ==================
from typing import Optional, Tuple, Dict, List
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure

def ensure_dir_label(d: str):
    os.makedirs(d, exist_ok=True)

def list_images_label(d, exts=(".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")) -> List[str]:
    names = [n for n in os.listdir(d) if os.path.splitext(n)[1] in exts]
    if not names:
        return []
    try:
        names.sort(key=lambda p: int(os.path.splitext(p)[0]))  # DAVIS 00000.png
    except ValueError:
        names.sort()
    return names

def list_images(d, exts=(".jpg",".jpeg",".png",".JPG",".JPEG",".PNG")):
    names = [n for n in os.listdir(d) if os.path.splitext(n)[1] in exts]
    try:
        names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError:
        names.sort()
    return names

def read_davis_labels(path: str) -> np.ndarray:
    im = Image.open(path)
    if im.mode == "P":
        return np.array(im, dtype=np.int32)
    if im.mode == "L":
        return np.array(im, dtype=np.uint8).astype(np.int32)
    if im.mode == "RGBA":
        im = im.convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")
    rgb = np.array(im, dtype=np.uint8)
    H, W, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    labels = np.zeros((H * W,), dtype=np.int32)
    color_to_id: Dict[Tuple[int,int,int], int] = {}
    next_id = 1
    for i, (r, g, b) in enumerate(flat):
        if r == 0 and g == 0 and b == 0:
            continue
        key = (int(r), int(g), int(b))
        if key not in color_to_id:
            color_to_id[key] = next_id
            next_id += 1
        labels[i] = color_to_id[key]
    return labels.reshape(H, W)

def box_to_intlist_label(box: np.ndarray) -> list:
    return list(map(int, np.round(box).tolist()))

def largest_component_binary_label(bin_mask: np.ndarray) -> np.ndarray:
    bin_mask = (bin_mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return bin_mask
    num, cc = cv2.connectedComponents(bin_mask, connectivity=8)
    if num <= 1:
        return bin_mask
    best, best_area = 0, -1
    for comp_id in range(1, num):
        area = int((cc == comp_id).sum())
        if area > best_area:
            best_area = area
            best = comp_id
    return (cc == best).astype(np.uint8)

def label_stats_label(labels: np.ndarray) -> List[Dict]:
    stats = []
    uniq = [int(v) for v in np.unique(labels) if int(v) > 0]
    for lid in uniq:
        m = (labels == lid)
        area = int(m.sum())
        if area == 0:
            continue
        ys, xs = np.where(m)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        stats.append({"id": lid, "area": area, "bbox": (int(x0), int(y0), int(x1), int(y1))})
    stats.sort(key=lambda d: d["area"], reverse=True)
    return stats

def box_from_mask_label(mask: np.ndarray, pad_frac: float = 0.05, min_box: int = 16) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        cx, cy = w // 2, h // 2
        half = max(min_box // 2, 1)
        return np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    x0 = max(0, x0 - int(round(bw * pad_frac)))
    y0 = max(0, y0 - int(round(bh * pad_frac)))
    x1 = min(w - 1, x1 + int(round(bw * pad_frac)))
    y1 = min(h - 1, y1 + int(round(bh * pad_frac)))
    if (x1 - x0 + 1) < min_box:
        extra = (min_box - (x1 - x0 + 1)) // 2 + 1
        x0 = max(0, x0 - extra); x1 = min(w - 1, x1 + extra)
    if (y1 - y0 + 1) < min_box:
        extra = (min_box - (y1 - y0 + 1)) // 2 + 1
        y0 = max(0, y0 - extra); y1 = min(h - 1, y1 + extra)
    return np.array([x0, y0, x1, y1], dtype=np.float32)

def colorize_labels_label(labels: np.ndarray) -> np.ndarray:
    rng = np.random.RandomState(0)
    uniq = [int(v) for v in np.unique(labels) if int(v) > 0]
    lut = {0: (0, 0, 0)}
    for v in uniq:
        lut[v] = tuple(int(x) for x in rng.randint(20, 235, size=3))
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for v, c in lut.items():
        out[labels == v] = c
    return out

def overlay_mask_on_rgb_label(rgb: np.ndarray, mask01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    rgb = rgb.copy()
    color = np.array([255, 128, 0], dtype=np.uint8)
    mask01 = (mask01 > 0).astype(np.uint8)
    if rgb.ndim != 3 or mask01.ndim != 2 or rgb.shape[:2] != mask01.shape:
        raise ValueError(f"overlay size mismatch: rgb {rgb.shape} vs mask {mask01.shape}")
    overlay = rgb.copy()
    idx = mask01.astype(bool)
    overlay[idx] = (0.5 * rgb[idx] + 0.5 * color).astype(np.uint8)
    return overlay

def show_box_for_label(box, ax, ec='lime', lw=2.0, label_text: Optional[str] = None):
    x0, y0, x1, y1 = [float(v) for v in box]
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                               edgecolor=ec, facecolor=(0, 0, 0, 0), lw=lw))
    if label_text:
        ax.text(x0, max(0, y0 - 4), label_text, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], np.maximum(neg_points[:, 1], 0), color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def sanity_check_first_frame(video_dir: str, video_dir_gt: str):
    frame_names_gt = list_images(video_dir_gt)
    if not frame_names_gt:
        raise FileNotFoundError(f"not found: {video_dir_gt}")
    frame_names = list_images(video_dir)
    if not frame_names:
        raise FileNotFoundError(f"not found: {video_dir}")
    print(f" found {len(frame_names)} images -{len(frame_names_gt)} mask.")
    return frame_names, frame_names_gt

# ================== MAIN API ==================
def select_largest_label_and_box_from_frame0(
    gt_dir: str,
    video_dir: Optional[str] = None,
    label_id_override: Optional[int] = None,
    pad_frac: float = 0.05,
    min_box: int = 16,
    vis: bool = False,
    out_dir: Optional[str] = None,
):
    frame_names_gt = list_images_label(gt_dir)
    if not frame_names_gt:
        raise FileNotFoundError(f"not found-{gt_dir}")
    gt_path = os.path.join(gt_dir, frame_names_gt[0])
    labels0 = read_davis_labels(gt_path)
    H, W = labels0.shape

    if (label_id_override is not None) and ((labels0 == int(label_id_override)).any()):
        lid = int(label_id_override)
        mask01 = (labels0 == lid).astype(np.uint8)
    else:
        stats = label_stats_label(labels0)
        if len(stats) == 0:
            mask01 = largest_component_binary_label((labels0 > 0).astype(np.uint8))
            lid = 1 if mask01.sum() > 0 else 0
        else:
            lid = stats[0]["id"]
            mask01 = (labels0 == lid).astype(np.uint8)

    box_xyxy = box_from_mask_label(mask01, pad_frac=pad_frac, min_box=min_box).astype(np.float32)
    box_xyxy = box_xyxy.astype(np.int32).tolist()
    print(f"[info] box (xyxy): {box_xyxy}")

    if vis:
        od = out_dir or OUT_DIR
        ensure_dir_label(od)
        labels_color = colorize_labels_label(labels0)
        cv2.imwrite(os.path.join(od, "labels_colorized.png"),
                    cv2.cvtColor(labels_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(od, f"mask_label_{lid}.png"),
                    (mask01 * 255).astype(np.uint8))

        fig = plt.figure(figsize=(7, 5))
        if CLEAN_VIZ:
            fig.patch.set_alpha(0.0)
        ax = plt.gca()
        if CLEAN_VIZ:
            ax.set_axis_off()
            ax.set_position([0, 0, 1, 1])
            ax.set_facecolor((0, 0, 0, 0))
        ax.imshow(labels_color)
        show_box_for_label(box_xyxy, ax, ec='lime', lw=2.5, label_text=None)
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(od, f"labels_with_box_id_{lid}.png"),
                    dpi=160, transparent=CLEAN_VIZ, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        if video_dir is not None:
            frame_names_rgb = list_images_label(video_dir)
            if frame_names_rgb:
                rgb0_path = os.path.join(video_dir, frame_names_rgb[0])
                rgb0 = cv2.cvtColor(cv2.imread(rgb0_path), cv2.COLOR_BGR2RGB)
                if rgb0.shape[:2] != (H, W):
                    mask_resized = cv2.resize(mask01.astype(np.uint8),
                                              (rgb0.shape[1], rgb0.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
                    box = np.array(box_xyxy, dtype=np.float32)
                    sx = rgb0.shape[1] / W
                    sy = rgb0.shape[0] / H
                    box = np.array([box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy], dtype=np.float32)
                else:
                    mask_resized = mask01
                    box = np.array(box_xyxy, dtype=np.float32)

                overlay = overlay_mask_on_rgb_label(rgb0, mask_resized, alpha=0.5)
                cv2.imwrite(os.path.join(od, f"rgb0_overlay_label_{lid}.png"),
                            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                fig2 = plt.figure(figsize=(7, 5))
                if CLEAN_VIZ:
                    fig2.patch.set_alpha(0.0)
                ax2 = plt.gca()
                if CLEAN_VIZ:
                    ax2.set_axis_off()
                    ax2.set_position([0, 0, 1, 1])
                    ax2.set_facecolor((0, 0, 0, 0))
                ax2.imshow(rgb0)
                show_box_for_label(box, ax2, ec='yellow', lw=2.5, label_text=None)
                fig2.tight_layout(pad=0)
                fig2.savefig(os.path.join(od, f"rgb0_with_box_id_{lid}.png"),
                             dpi=160, transparent=CLEAN_VIZ, bbox_inches=0)
                plt.close(fig2)

    return int(lid), box_xyxy, mask01, gt_path

def load_gt_binary_masks_for_label(
    gt_dir: str,
    frame_names_gt: List[str],
    label_id: int,
    target_size: Optional[Tuple[int,int]] = None,
) -> List[np.ndarray]:
    masks = []
    for name in frame_names_gt:
        gt_path = os.path.join(gt_dir, name)
        labels = read_davis_labels(gt_path)
        m01 = (labels == int(label_id)).astype(np.uint8)
        if target_size is not None:
            Wt, Ht = target_size
            m01 = cv2.resize(m01, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        masks.append(m01)
    return masks

# ========= KPI =========
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
    pb_match = (pb & (gb_dil > 0)).sum()
    gb_match = (gb & (pb_dil > 0)).sum()
    prec = pb_match / (pb.sum() + eps)
    rec  = gb_match / (gb.sum() + eps)
    denom = (prec + rec)
    return 0.0 if denom == 0.0 else (2.0 * prec * rec) / denom

def _dice_and_iou_from_bool(pred_bool: np.ndarray, gt_bool: np.ndarray) -> Tuple[float, float]:
    pred = pred_bool.astype(bool)
    gt   = gt_bool.astype(bool)
    inter = (pred & gt).sum()
    p_sum = pred.sum()
    g_sum = gt.sum()
    union = (pred | gt).sum()
    dice = (2.0 * inter) / max(1.0, (p_sum + g_sum))
    iou  = inter / max(1.0, union)
    return float(dice), float(iou)

def _to_bool_mask_2d(arr, target_hw=None):
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim == 3:
        if a.shape[0] == 1:
            a = a[0]
        elif a.shape[-1] == 1:
            a = a[..., 0]
        else:
            a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"mask must be 2D after squeeze, got {a.shape}")
    if target_hw is not None and a.shape != target_hw:
        Ht, Wt = int(target_hw[0]), int(target_hw[1])
        if Ht <= 0 or Wt <= 0:
            raise ValueError(f"invalid target size: {(Ht, Wt)}")
        a = cv2.resize(a.astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_NEAREST).astype(bool)
    return a.astype(bool)

def save_eval_triptych(frame_rgb: np.ndarray,
                       pred_mask_bool: np.ndarray,
                       gt_mask_bool: np.ndarray,
                       out_path: str,
                       clean_viz: bool = True,
                       draw_contour: bool = True,
                       add_legend: bool = True):
    import matplotlib.patches as patches
    rgba_pred  = np.array([1.0, 0.55, 0.0, 0.55])
    rgba_gt    = np.array([0.0, 0.75, 1.0, 0.45])
    rgba_tp    = np.array([0.00, 0.85, 0.00, 0.60])
    rgba_fn    = np.array([1.00, 0.00, 0.00, 0.60])
    rgba_fp    = np.array([0.00, 0.45, 1.00, 0.60])

    pred = np.asarray(pred_mask_bool).astype(bool)
    gt   = np.asarray(gt_mask_bool).astype(bool)
    H, W = pred.shape

    pred_rgba = pred.reshape(H, W, 1) * rgba_pred.reshape(1, 1, 4)
    gt_rgba   = gt.reshape(H, W, 1)   * rgba_gt.reshape(1, 1, 4)

    tp = pred & gt
    fn = (~pred) & gt
    fp = pred & (~gt)
    err_rgba = np.zeros((H, W, 4), dtype=float)
    if tp.any(): err_rgba[tp] = rgba_tp
    if fn.any(): err_rgba[fn] = rgba_fn
    if fp.any(): err_rgba[fp] = rgba_fp

    fig = plt.figure(figsize=(15, 5))
    if clean_viz:
        fig.patch.set_alpha(0.0)

    def _prep_ax(ax, left, width):
        if clean_viz:
            ax.set_axis_off()
            ax.set_position([left, 0.0, width, 1.0])
            ax.set_facecolor((0, 0, 0, 0))

    ax1 = fig.add_subplot(1, 3, 1); _prep_ax(ax1, 0.00, 0.3334)
    ax1.imshow(frame_rgb); ax1.imshow(pred_rgba)
    if draw_contour and pred.any(): ax1.contour(pred.astype(np.uint8), levels=[0.5], linewidths=1.8)

    ax2 = fig.add_subplot(1, 3, 2); _prep_ax(ax2, 0.3333, 0.3334)
    ax2.imshow(frame_rgb); ax2.imshow(gt_rgba)
    if draw_contour and gt.any(): ax2.contour(gt.astype(np.uint8), levels=[0.5], linewidths=1.8)

    ax3 = fig.add_subplot(1, 3, 3); _prep_ax(ax3, 0.6666, 0.3334)
    ax3.imshow(frame_rgb); ax3.imshow(err_rgba)
    if draw_contour and (tp.any() or fn.any() or fp.any()):
        if tp.any(): ax3.contour(tp.astype(np.uint8), levels=[0.5], linewidths=1.2)
        if fn.any(): ax3.contour(fn.astype(np.uint8), levels=[0.5], linewidths=1.2)
        if fp.any(): ax3.contour(fp.astype(np.uint8), levels=[0.5], linewidths=1.2)

    if add_legend:
        items = [("TP", rgba_tp), ("FN", rgba_fn), ("FP", rgba_fp)]
        x0, y0, dy = 10, 10, 18
        for i, (txt, col) in enumerate(items):
            rect = patches.Rectangle((x0, y0 + i*dy), 14, 14,
                                     linewidth=0.5,
                                     edgecolor=(0,0,0,0.8),
                                     facecolor=col[:3],
                                     alpha=col[3])
            ax3.add_patch(rect)
            ax3.text(x0 + 20, y0 + i*dy + 11, txt, fontsize=10,
                     color='white',
                     bbox=dict(facecolor=(0,0,0,0.4), edgecolor='none', pad=1))

    fig.savefig(out_path, dpi=130, transparent=clean_viz, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def evaluate_video_from_gt_labels(video_segments: dict,
                                  frame_names: list,
                                  frame_names_gt: list,
                                  video_dir: str,
                                  gt_dir: str,
                                  chosen_label_id: int,
                                  out_dir_vis: str = None,
                                  clean_viz: bool = True,
                                  vis_stride: int = 1,
                                  write_csv_path: str = None,
                                  tol_pixels: int = 2):
    if out_dir_vis is not None:
        os.makedirs(out_dir_vis, exist_ok=True)

    metrics_rows = []
    n_frames = min(len(frame_names), len(frame_names_gt))

    for curr_idx in range(n_frames):
        gt_path = os.path.join(gt_dir, frame_names_gt[curr_idx])
        try:
            labels_img = read_davis_labels(gt_path)
        except Exception as e:
            print(f"[warn] Could not read GT labels for frame {curr_idx} ({gt_path}): {e}. Skipping.")
            continue

        gt_mask_bool = (labels_img == int(chosen_label_id))
        H, W = gt_mask_bool.shape

        if curr_idx not in video_segments:
            print(f"[warn] No predicted mask for frame {curr_idx}. Skipping.")
            continue

        if 1 not in video_segments[curr_idx]:
            if len(video_segments[curr_idx]) == 1:
                pred_raw = next(iter(video_segments[curr_idx].values()))
            else:
                print(f"[warn] No obj_id=1 for frame {curr_idx}. Skipping.")
                continue
        else:
            pred_raw = video_segments[curr_idx][1]

        pred_bool = _to_bool_mask_2d(pred_raw, target_hw=(H, W))

        dice, iou = _dice_and_iou_from_bool(pred_bool, gt_mask_bool)
        fscore    = _boundary_f_measure(pred_bool.astype(np.float32),
                                        gt_mask_bool.astype(np.float32),
                                        tol=tol_pixels)
        jf        = 0.5 * (iou + fscore)

        metrics_rows.append({
            "frame": f"{curr_idx:05d}",
            "dice": float(dice),
            "iou":  float(iou),
            "f":    float(fscore),
            "jf":   float(jf),
        })

        if out_dir_vis is not None and (curr_idx % vis_stride == 0):
            frame_path = os.path.join(video_dir, frame_names[curr_idx])
            frame_img = cv2.imread(frame_path)
            if frame_img is None:
                frame_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                if frame_rgb.shape[:2] != (H, W):
                    frame_rgb = cv2.resize(frame_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

            out_png = os.path.join(out_dir_vis, f"eval_{curr_idx:05d}.png")
            save_eval_triptych(frame_rgb, pred_bool, gt_mask_bool, out_png,
                               clean_viz=clean_viz, draw_contour=True, add_legend=True)

    if metrics_rows and write_csv_path is not None:
        import csv
        with open(write_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "dice", "iou", "f", "jf"])
            w.writeheader(); w.writerows(metrics_rows)

    if metrics_rows:
        mean_d = sum(r["dice"] for r in metrics_rows) / len(metrics_rows)
        mean_j = sum(r["iou"]  for r in metrics_rows) / len(metrics_rows)
        mean_f = sum(r["f"]    for r in metrics_rows) / len(metrics_rows)
        mean_jf= sum(r["jf"]   for r in metrics_rows) / len(metrics_rows)
        print(f"[metrics] {os.path.basename(video_dir)} | frames={len(metrics_rows)} | "
              f"Dice={mean_d:.4f} | J(IoU)={mean_j:.4f} | F={mean_f:.4f} | J&F={mean_jf:.4f}")
        return mean_j, mean_f, mean_jf, mean_d
    else:
        print("[metrics] No frames evaluated.")
        return None, None, None, None

def _is_sequence_dir(path: str) -> bool:
    return len(list_images(path)) > 0

def load_split_sequences(json_path: str, split_name: str = "val") -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if split_name not in data:
        raise KeyError(f"split '{split_name}' not found in {json_path}")
    return list(sorted(set(data[split_name])))


if __name__ == "__main__":
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-small")

    SPLIT_JSON = r"C:\Users\28601\PycharmProjects\DL\tfds_davis_split_480p.json"
    USE_SPLIT  = True
    SPLIT_NAME = "val"

    if USE_SPLIT:
        split_seqs = load_split_sequences(SPLIT_JSON, split_name=SPLIT_NAME)
        seq_names = [
            s for s in split_seqs
            if os.path.isdir(os.path.join(ROOT_FRAMES_DIR, s)) and
               os.path.isdir(os.path.join(ROOT_GT_DIR, s))
        ]
        if not seq_names:
            raise FileNotFoundError("לא נמצאו רצפים מתוך ה-val שקיימים פיזית ב-ROOT_FRAMES_DIR/ROOT_GT_DIR")

        summary_rows = []
        for seq in seq_names:
            print(f"\n========== [SEQ] {seq} (split={SPLIT_NAME}) ==========")
            seq_frames_dir = os.path.join(ROOT_FRAMES_DIR, seq)
            seq_gt_dir     = os.path.join(ROOT_GT_DIR, seq)
            seq_out_dir    = os.path.join(OUT_DIR, seq)
            os.makedirs(seq_out_dir, exist_ok=True)

            frame_names    = list_images(seq_frames_dir)
            frame_names_gt = list_images_label(seq_gt_dir)
            if not frame_names or not frame_names_gt:
                print(f"[skip] {seq}: חסרים פריימים או GT")
                continue

            ann_frame_idx = 0
            label_id, box_xyxy, mask01, gt_path = select_largest_label_and_box_from_frame0(
                gt_dir=seq_gt_dir,
                video_dir=seq_frames_dir,
                label_id_override=None,
                pad_frac=0.05,
                min_box=16,
                vis=VIS,
                out_dir=seq_out_dir,
            )

            inference_state = predictor.init_state(video_path=seq_frames_dir)
            predictor.reset_state(inference_state)
            ann_obj_id = 1
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box_xyxy,
            )


            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            vis_frame_stride = 1
            for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
                fig = plt.figure(figsize=(6, 4))
                if CLEAN_VIZ:
                    fig.patch.set_alpha(0.0)
                ax = plt.gca()
                if CLEAN_VIZ:
                    ax.set_axis_off()
                    ax.set_position([0, 0, 1, 1])
                    ax.set_facecolor((0, 0, 0, 0))
                rgb = Image.open(os.path.join(seq_frames_dir, frame_names[out_frame_idx]))
                ax.imshow(rgb)
                for oid, omask in video_segments.get(out_frame_idx, {}).items():
                    show_mask(omask, ax, obj_id=oid)
                fig.tight_layout(pad=0)
                out_png = os.path.join(seq_out_dir, f"prop_{out_frame_idx:05d}.png")
                fig.savefig(out_png, dpi=120, transparent=CLEAN_VIZ,
                            bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            EVAL_OUT_DIR = os.path.join(seq_out_dir, "eval_vis")
            CSV_PATH     = os.path.join(seq_out_dir, "metrics.csv")
            mean_j, mean_f, mean_jf, mean_d = evaluate_video_from_gt_labels(
                video_segments=video_segments,
                frame_names=frame_names,
                frame_names_gt=frame_names_gt,
                video_dir=seq_frames_dir,
                gt_dir=seq_gt_dir,
                chosen_label_id=label_id,
                out_dir_vis=EVAL_OUT_DIR,
                clean_viz=True,
                vis_stride=1,
                write_csv_path=CSV_PATH,
                tol_pixels=2,
            )

            if all(v is not None for v in [mean_j, mean_f, mean_jf, mean_d]):
                summary_rows.append({
                    "sequence": seq,
                    "mean_j":  float(mean_j),
                    "mean_f":  float(mean_f),
                    "mean_jf": float(mean_jf),
                    "mean_d":  float(mean_d)
                })

        if summary_rows:
            import csv
            overall_csv = os.path.join(OUT_DIR, f"overall_summary_{SPLIT_NAME}.csv")
            with open(overall_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["sequence", "mean_j", "mean_f", "mean_jf", "mean_d"])
                w.writeheader(); w.writerows(summary_rows)

            overall_j  = sum(r["mean_j"]  for r in summary_rows)/len(summary_rows)
            overall_f  = sum(r["mean_f"]  for r in summary_rows)/len(summary_rows)
            overall_jf = sum(r["mean_jf"] for r in summary_rows)/len(summary_rows)
            overall_d  = sum(r["mean_d"]  for r in summary_rows)/len(summary_rows)
            print("\n========== OVERALL SUMMARY (VAL) ==========")
            for r in summary_rows:
                print(f"{r['sequence']}: J={r['mean_j']:.4f} | F={r['mean_f']:.4f} | J&F={r['mean_jf']:.4f} | Dice={r['mean_d']:.4f}")
            print(f"OVERALL on {len(summary_rows)} val videos -> J={overall_j:.4f} | F={overall_f:.4f} | J&F={overall_jf:.4f} | Dice={overall_d:.4f}")
            print(f"Saved to: {overall_csv}")
        else:
            print("[summary] no per-sequence metrics were collected (val).")

    else:
        frame_names = list_images(VIDEO_DIR)
        frame_names_gt = list_images_label(GT_DIR)
        if not frame_names or not frame_names_gt:
            raise FileNotFoundError("error.")

        ann_frame_idx = 0
        label_id, box_xyxy, mask01, gt_path = select_largest_label_and_box_from_frame0(
            gt_dir=GT_DIR,
            video_dir=VIDEO_DIR,
            label_id_override=None,
            pad_frac=0.05,
            min_box=16,
            vis=VIS,
            out_dir=OUT_DIR,
        )

        inference_state = predictor.init_state(video_path=VIDEO_DIR)
        predictor.reset_state(inference_state)
        ann_obj_id = 1
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box_xyxy,
        )

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        vis_frame_stride = 1
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            fig = plt.figure(figsize=(6, 4))
            if CLEAN_VIZ:
                fig.patch.set_alpha(0.0)
            ax = plt.gca()
            if CLEAN_VIZ:
                ax.set_axis_off()
                ax.set_position([0, 0, 1, 1])
                ax.set_facecolor((0, 0, 0, 0))
            rgb = Image.open(os.path.join(VIDEO_DIR, frame_names[out_frame_idx]))
            ax.imshow(rgb)
            for oid, omask in video_segments.get(out_frame_idx, {}).items():
                show_mask(omask, ax, obj_id=oid)
            fig.tight_layout(pad=0)
            out_png = os.path.join(OUT_DIR, f"prop_{out_frame_idx:05d}.png")
            fig.savefig(out_png, dpi=120, transparent=CLEAN_VIZ,
                        bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        EVAL_OUT_DIR = os.path.join(OUT_DIR, "eval_vis")
        CSV_PATH = os.path.join(OUT_DIR, "metrics.csv")
        mean_j, mean_f, mean_jf, mean_d = evaluate_video_from_gt_labels(
            video_segments=video_segments,
            frame_names=frame_names,
            frame_names_gt=frame_names_gt,
            video_dir=VIDEO_DIR,
            gt_dir=GT_DIR,
            chosen_label_id=label_id,
            out_dir_vis=EVAL_OUT_DIR,
            clean_viz=True,
            vis_stride=1,
            write_csv_path=CSV_PATH,
            tol_pixels=2,
        )
