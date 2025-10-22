
import re

import os
import glob
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt

from typing import Optional, Tuple, Dict, List
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure

from sam2.sam2_video_predictor import SAM2VideoPredictor
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

def _choose_label_medical(m_arr: np.ndarray, override: Optional[int] = None) -> int:

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

def overlay_mask_on_rgb_label(rgb: np.ndarray, mask01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    rgb = rgb.copy()
    color = np.array([255, 128, 0], dtype=np.uint8)
    mask01 = (mask01 > 0).astype(np.uint8)
    overlay = rgb.copy()
    idx = mask01.astype(bool)
    overlay[idx] = (0.5 * rgb[idx] + 0.5 * color).astype(np.uint8)
    return overlay
def make_jpg_sequence_for_sam2(src_dir: str, dst_dir: str) -> list:

    os.makedirs(dst_dir, exist_ok=True)
    image_exts = (".jpg", ".jpeg", ".png")
    img_names = [f for f in os.listdir(src_dir)
                 if f.lower().endswith(image_exts) and "mask" not in f.lower()]
    import re
    def _idx(name):
        m = re.search(r'(\d+)(?!.*\d)', name)
        return int(m.group(1)) if m else -1
    img_names.sort(key=lambda f: (_idx(f), f))

    out_names = []
    for i, fn in enumerate(img_names):
        src_path = os.path.join(src_dir, fn)
        bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        out_name = f"{i:05d}.jpg"
        out_path = os.path.join(dst_dir, out_name)
        cv2.imwrite(out_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        out_names.append(out_name)

    if len(out_names) == 0:
        raise RuntimeError(f"No RGB images found in {src_dir} (without 'mask' in name).")
    return out_names


def _load_label_map(path: str) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im, dtype=np.int32) if im.mode == "P" else np.array(im.convert("L"), dtype=np.int32)
    return arr
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
def _list_medical_frames_single_dir(video_dir: str) -> Tuple[List[str], List[str]]:

    image_exts = (".jpg", ".jpeg", ".png")
    all_files = os.listdir(video_dir)
    image_files = sorted([f for f in all_files
                          if f.lower().endswith(image_exts) and "mask" not in f.lower()])
    gt_files    = sorted([f for f in all_files
                          if f.lower().endswith(".png") and "watershed_mask" in f.lower()])
    return image_files, gt_files
def _binary_boundary(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dil = cv2.dilate(m, kernel, iterations=1)
    ero = cv2.erode(m, kernel, iterations=1)
    b = (dil - ero)
    return (b > 0).astype(np.uint8)
def _frame_index_from_name(name: str) -> int:
    base = os.path.basename(name)
    m = re.search(r'(\d+)(?!.*\d)', base)
    return int(m.group(1)) if m else -1
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

def _sorted_medical_frames(video_dir: str) -> List[str]:
    names, _ = _list_medical_frames_single_dir(video_dir)
    names_sorted = sorted(names, key=lambda f: (_frame_index_from_name(f), f))
    idxs = [_frame_index_from_name(f) for f in names_sorted]
    if 0 in idxs:
        k = idxs.index(0)
        names_sorted = names_sorted[k:] + names_sorted[:k]
    return names_sorted

def show_box_for_label(box, ax, ec='lime', lw=2.0, label_text: Optional[str] = None):
    x0, y0, x1, y1 = [float(v) for v in box]
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                               edgecolor=ec, facecolor=(0, 0, 0, 0), lw=lw))
    if label_text:
        ax.text(x0, max(0, y0 - 4), label_text, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
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
        a = cv2.resize(a.astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_NEAREST).astype(bool)
    return a.astype(bool)
def _dice_and_iou_from_bool(pred_bool: np.ndarray, gt_bool: np.ndarray) -> Tuple[float, float]:
    pred = pred_bool.astype(bool)
    gt   = gt_bool.astype(bool)
    inter = (pred & gt).sum()
    p_sum = pred.sum(); g_sum = gt.sum()
    union = (pred | gt).sum()
    dice = (2.0 * inter) / max(1.0, (p_sum + g_sum))
    iou  = inter / max(1.0, union)
    return float(dice), float(iou)
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
    if clean_viz: fig.patch.set_alpha(0.0)
    def _prep_ax(ax, left, width):
        if clean_viz:
            ax.set_axis_off(); ax.set_position([left, 0.0, width, 1.0]); ax.set_facecolor((0, 0, 0, 0))
    ax1 = fig.add_subplot(1, 3, 1); _prep_ax(ax1, 0.00, 0.3334); ax1.imshow(frame_rgb); ax1.imshow(pred_rgba)
    if draw_contour and pred.any(): ax1.contour(pred.astype(np.uint8), levels=[0.5], linewidths=1.8)
    ax2 = fig.add_subplot(1, 3, 2); _prep_ax(ax2, 0.3333, 0.3334); ax2.imshow(frame_rgb); ax2.imshow(gt_rgba)
    if draw_contour and gt.any(): ax2.contour(gt.astype(np.uint8), levels=[0.5], linewidths=1.8)
    ax3 = fig.add_subplot(1, 3, 3); _prep_ax(ax3, 0.6666, 0.3334); ax3.imshow(frame_rgb); ax3.imshow(err_rgba)
    if draw_contour and (tp.any() or fn.any() or fp.any()):
        if tp.any(): ax3.contour(tp.astype(np.uint8), levels=[0.5], linewidths=1.2)
        if fn.any(): ax3.contour(fn.astype(np.uint8), levels=[0.5], linewidths=1.2)
        if fp.any(): ax3.contour(fp.astype(np.uint8), levels=[0.5], linewidths=1.2)
    fig.savefig(out_path, dpi=130, transparent=clean_viz, bbox_inches='tight', pad_inches=0); plt.close(fig)


# ================== PATHS / MODE ==================
MEDICAL_MODE = True

ROOT_MEDICAL = r"C:\pythonProject\DL_project\dataset\MEDICAL"  # <<< עדכני אם צריך
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

OUT_DIR = r"./runs/val/video_loop/"
os.makedirs(OUT_DIR, exist_ok=True)

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

VIS = True
CLEAN_VIZ = True

def evaluate_video_medical(video_segments: dict,
                           frame_names: list,
                           video_dir: str,
                           gt_search_dir: str,
                           obj_id: int = 1,
                           out_dir_vis: Optional[str] = None,
                           clean_viz: bool = True,
                           vis_stride: int = 1,
                           write_csv_path: Optional[str] = None,
                           tol_pixels: int = 2):

    metrics_rows = []
    if out_dir_vis is not None: os.makedirs(out_dir_vis, exist_ok=True)

    for idx, fname in enumerate(frame_names):
        stem = os.path.splitext(fname)[0]
        # GT lookup
        gt_path = _find_gt_for_image_medical(stem, gt_search_dir)
        if gt_path is None:
            continue
        gt_arr = _load_label_map(gt_path)
        if 31 in np.unique(gt_arr): gt01 = (gt_arr == 31).astype(np.float32)
        elif 32 in np.unique(gt_arr): gt01 = (gt_arr == 32).astype(np.float32)
        else: gt01 = (gt_arr > 0).astype(np.float32)

        if idx not in video_segments or obj_id not in video_segments[idx]:
            continue
        pred_raw = video_segments[idx][obj_id]
        H, W = gt01.shape
        pred_bool = _to_bool_mask_2d(pred_raw, target_hw=(H, W))

        dice, iou = _dice_and_iou_from_bool(pred_bool, gt01 > 0.5)
        fscore    = _boundary_f_measure(pred_bool.astype(np.float32), gt01.astype(np.float32), tol=tol_pixels)
        jf        = 0.5 * (iou + fscore)

        metrics_rows.append({"frame": f"{idx:05d}", "dice": float(dice), "iou": float(iou), "f": float(fscore), "jf": float(jf)})

        if out_dir_vis is not None and (idx % vis_stride == 0):
            frame_img = cv2.cvtColor(cv2.imread(os.path.join(video_dir, fname)), cv2.COLOR_BGR2RGB)
            out_png = os.path.join(out_dir_vis, f"eval_{idx:05d}.png")
            save_eval_triptych(frame_img, pred_bool, gt01 > 0.5, out_png, clean_viz=clean_viz, draw_contour=True, add_legend=True)

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
        print(f"[metrics] {os.path.basename(video_dir)} | frames={len(metrics_rows)} | Dice={mean_d:.4f} | J(IoU)={mean_j:.4f} | F={mean_f:.4f} | J&F={mean_jf:.4f}")
        return mean_j, mean_f, mean_jf, mean_d
    else:
        print("[metrics] No frames evaluated.")
        return None, None, None, None


if __name__ == "__main__":
    if MEDICAL_MODE:
        predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-small")

        agg = []   # כל איבר: dict(name, n_frames, mean_j, mean_f, mean_jf, mean_d)

        for rel_path in TEST_LIST:
            VIDEO_DIR = os.path.join(ROOT_MEDICAL, rel_path)
            if not os.path.isdir(VIDEO_DIR):
                print(f"[skip] Not a directory: {VIDEO_DIR}")
                continue

            OUT_DIR_ONE = os.path.join(OUT_DIR, rel_path.replace("\\", "__"))
            os.makedirs(OUT_DIR_ONE, exist_ok=True)

            frame_names = _sorted_medical_frames(VIDEO_DIR)
            if len(frame_names) < 2:
                print(f"[skip] Need >=2 frames in {VIDEO_DIR}")
                continue

            ann_frame_idx = 0
            ann_frame_name = frame_names[ann_frame_idx]
            stem0 = os.path.splitext(ann_frame_name)[0]
            gt0_path = _find_gt_for_image_medical(stem0, VIDEO_DIR)
            if gt0_path is None:
                found = False
                for k, name in enumerate(frame_names):
                    p = _find_gt_for_image_medical(os.path.splitext(name)[0], VIDEO_DIR)
                    if p is not None:
                        ann_frame_idx = k
                        ann_frame_name = name
                        gt0_path = p
                        found = True
                        break
                if not found:
                    print(f"[skip] No watershed_mask GT found in {VIDEO_DIR}")
                    continue

            labels0 = _load_label_map(gt0_path)
            label_id = _choose_label_medical(labels0, override=None)
            mask01 = (labels0 == label_id).astype(np.uint8)

            box_xyxy = box_from_mask_label(mask01, pad_frac=0.05, min_box=16).astype(np.int32).tolist()

            print(f"\n========== [SEQ] {rel_path} (MEDICAL) ==========")
            print(f"[info] start frame: {ann_frame_name} (idx={ann_frame_idx})")
            print(f"[info] chosen label id: {label_id}")
            print(f"[info] box (xyxy): {box_xyxy}")

            if VIS:
                rgb0 = cv2.cvtColor(cv2.imread(os.path.join(VIDEO_DIR, ann_frame_name)), cv2.COLOR_BGR2RGB)
                overlay = overlay_mask_on_rgb_label(rgb0, mask01, alpha=0.5)
                cv2.imwrite(os.path.join(OUT_DIR_ONE, f"rgb0_overlay_label_{label_id}.png"),
                            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                fig = plt.figure(figsize=(7, 5))
                if CLEAN_VIZ: fig.patch.set_alpha(0.0)
                ax = plt.gca()
                if CLEAN_VIZ: ax.set_axis_off(); ax.set_position([0, 0, 1, 1]); ax.set_facecolor((0, 0, 0, 0))
                ax.imshow(rgb0); show_box_for_label(box_xyxy, ax, ec='yellow', lw=2.5, label_text=None)
                fig.tight_layout(pad=0)
                fig.savefig(os.path.join(OUT_DIR_ONE, f"rgb0_with_box_id_{label_id}.png"),
                            dpi=160, transparent=CLEAN_VIZ, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            TMP_JPG_DIR = os.path.join(OUT_DIR_ONE, "sam2_jpg_seq")
            jpg_seq = make_jpg_sequence_for_sam2(VIDEO_DIR, TMP_JPG_DIR)

            inference_state = predictor.init_state(video_path=TMP_JPG_DIR)
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
            if VIS:
                for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
                    fig = plt.figure(figsize=(6, 4))
                    if CLEAN_VIZ: fig.patch.set_alpha(0.0)
                    ax = plt.gca()
                    if CLEAN_VIZ: ax.set_axis_off(); ax.set_position([0, 0, 1, 1]); ax.set_facecolor((0, 0, 0, 0))
                    rgb = Image.open(os.path.join(VIDEO_DIR, frame_names[out_frame_idx]))
                    ax.imshow(rgb)
                    for oid, omask in video_segments.get(out_frame_idx, {}).items():
                        show_mask(omask, ax, obj_id=oid)
                    fig.tight_layout(pad=0)
                    out_png = os.path.join(OUT_DIR_ONE, f"prop_{out_frame_idx:05d}.png")
                    fig.savefig(out_png, dpi=120, transparent=CLEAN_VIZ, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

            EVAL_OUT_DIR = os.path.join(OUT_DIR_ONE, "eval_vis")
            CSV_PATH     = os.path.join(OUT_DIR_ONE, "metrics.csv")
            mean_j, mean_f, mean_jf, mean_d = evaluate_video_medical(
                video_segments=video_segments,
                frame_names=frame_names,
                video_dir=VIDEO_DIR,
                gt_search_dir=VIDEO_DIR,
                obj_id=1,
                out_dir_vis=EVAL_OUT_DIR,
                clean_viz=True,
                vis_stride=1,
                write_csv_path=CSV_PATH,
                tol_pixels=2,
            )

            if mean_j is not None:
                agg.append({
                    "name": rel_path,
                    "n_frames": len(os.listdir(EVAL_OUT_DIR)) if os.path.isdir(EVAL_OUT_DIR) else 0,
                    "mean_j": float(mean_j),
                    "mean_f": float(mean_f),
                    "mean_jf": float(mean_jf),
                    "mean_d": float(mean_d),
                })
            else:
                print(f"[metrics] {rel_path} | No frames evaluated (skipped).")

        if agg:
            mj  = sum(x["mean_j"]  for x in agg) / len(agg)
            mf  = sum(x["mean_f"]  for x in agg) / len(agg)
            mjf = sum(x["mean_jf"] for x in agg) / len(agg)
            md  = sum(x["mean_d"]  for x in agg) / len(agg)
            print("\n========== [OVERALL SUMMARY] ==========")
            print(f"Videos evaluated: {len(agg)}")
            print(f"Mean J(IoU) = {mj:.4f} | Mean F = {mf:.4f} | Mean J&F = {mjf:.4f} | Mean Dice = {md:.4f}")

            summary_csv = os.path.join(OUT_DIR, "summary_metrics.csv")
            import csv
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["name","n_frames","mean_j","mean_f","mean_jf","mean_d"])
                w.writeheader()
                for row in agg:
                    w.writerow(row)
                w.writerow({"name":"__OVERALL__", "n_frames":sum(x["n_frames"] for x in agg),
                            "mean_j":mj, "mean_f":mf, "mean_jf":mjf, "mean_d":md})
        else:
            print("[OVERALL] No valid videos to summarize.")
