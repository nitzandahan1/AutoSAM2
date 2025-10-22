
import os, glob, re
from typing import Optional, Tuple, List
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sam2.sam2_image_predictor import SAM2ImagePredictor
from TRAIN_MASK_BOX import (
    PrompterNet, sam2_forward_logits_and_or_box,
    LOGIT_GAIN, LOGIT_CLAMP
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEDICAL_ROOT = r"C:\pythonProject\DL_project\dataset\MEDICAL"
CKPT_PATH = r"C:\Users\28601\PycharmProjects\DL\runs\train_prompter_sam2\ALL_TRAIN_MASK_BOX__MEDICAL\best.pt"
#CKPT_PATH = r"C:\Users\28601\PycharmProjects\DL\runs\train_prompter_sam2\ALL_TRAIN_MASK_BOX_update\best.pt",

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

VIS         = True
VIS_DIR     = r"./runs/val/vis_batch_medical_NET_TRAIN_MEDICAL"
VIS_STRIDE  = 1
CLEAN_VIZ   = True
os.makedirs(VIS_DIR, exist_ok=True)

# ====== Utils (single-dir medical) ======
_BIDI = re.compile(r'[\u200e\u200f\u202a-\u202e]')
def _san(p: str) -> str:
    return os.path.normpath(_BIDI.sub('', p)) if isinstance(p, str) else p

def _exists_dir(d: str) -> str:
    d2 = _san(d)
    if not os.path.isdir(d2):
        raise FileNotFoundError(f"[frames_dir not found]\n  got: {d}\n  sanitized: {d2}\n  cwd: {os.getcwd()}")
    return d2

def _load_label_map(path: str) -> np.ndarray:
    im = Image.open(path)
    return np.array(im, dtype=np.int32) if im.mode == "P" else np.array(im.convert("L"), dtype=np.int32)

def _frame_idx_from_name(name: str) -> int:
    base = os.path.basename(name)
    m = re.search(r'(\d+)(?!.*\d)', base)
    return int(m.group(1)) if m else -1

def _list_frames_and_gts(video_dir: str) -> Tuple[List[str], List[str]]:
    exts = (".jpg", ".jpeg", ".png")
    allf = os.listdir(video_dir)
    imgs = sorted([f for f in allf if f.lower().endswith(exts) and "mask" not in f.lower()])
    gts  = sorted([f for f in allf if f.lower().endswith(".png") and "watershed_mask" in f.lower()])
    return imgs, gts

def _sorted_frames(frames_dir: str) -> List[str]:
    names, _ = _list_frames_and_gts(frames_dir)
    names_sorted = sorted(names, key=lambda f: (_frame_idx_from_name(f), f))
    idxs = [_frame_idx_from_name(f) for f in names_sorted]
    if 0 in idxs:
        k = idxs.index(0)
        names_sorted = names_sorted[k:] + names_sorted[:k]
    return [os.path.join(frames_dir, f) for f in names_sorted]

def _find_gt_watershed(stem: str, search_dir: str) -> Optional[str]:
    pats = [os.path.join(search_dir, f"{stem}*watershed_mask*.png"),
            os.path.join(search_dir, f"{stem}*watershed_mask*.PNG")]
    cand: List[str] = []
    for p in pats: cand.extend(glob.glob(p))
    return sorted(cand)[0] if cand else None

def _choose_label_31_32_only(m0_arr: np.ndarray) -> Optional[int]:
    vals = set(np.unique(m0_arr).tolist())
    if 31 in vals: return 31
    if 32 in vals: return 32
    return None

# ====== KPI ======
def _dice_iou(pred01: np.ndarray, gt01: np.ndarray, thr: float = 0.5, eps: float = 1e-6) -> Tuple[float, float]:
    p = (pred01 >= thr).astype(np.float32); g = (gt01 >= 0.5).astype(np.float32)
    inter = float((p * g).sum())
    dice  = (2*inter + eps) / (p.sum() + g.sum() + eps)
    union = float((p + g - p*g).sum())
    iou   = (inter + eps) / (union + eps)
    return dice, iou

def _binary_boundary(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    b = cv2.dilate(m, k, 1) - cv2.erode(m, k, 1)
    return (b > 0).astype(np.uint8)

def _boundary_f_measure(pred01: np.ndarray, gt01: np.ndarray, tol: int = 2, eps: float = 1e-6) -> float:
    pb = _binary_boundary(pred01); gb = _binary_boundary(gt01)
    if pb.sum() == 0 and gb.sum() == 0: return 1.0
    if pb.sum() == 0 or gb.sum() == 0:  return 0.0
    ksz = 2*tol + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    gb_dil = cv2.dilate(gb, kernel, 1)
    pb_dil = cv2.dilate(pb, kernel, 1)
    tp_p = float((pb & gb_dil).sum()); prec = tp_p / (float(pb.sum()) + eps)
    tp_g = float((gb & pb_dil).sum()); rec  = tp_g / (float(gb.sum()) + eps)
    return float((2.0 * prec * rec) / (prec + rec + eps))

# ====== CKPT helper ======
def _normalize_ckpt_path(ckpt_path):
    if isinstance(ckpt_path, (list, tuple)):
        if len(ckpt_path) == 0:
            raise ValueError("[ckpt] empty tuple/list for ckpt_path")
        ckpt_path = ckpt_path[0]
    if not isinstance(ckpt_path, (str, bytes, os.PathLike)):
        raise TypeError(f"[ckpt] ckpt_path must be path-like, got: {type(ckpt_path)}")
    return str(ckpt_path)

def _strip_module_prefix(state_dict):
    if not state_dict: return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

def load_prompter_state_dict(ckpt_path, map_location="cpu"):
    ckpt_path = _normalize_ckpt_path(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[ckpt] not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"[ckpt] unexpected checkpoint type: {type(ckpt)}")
    return _strip_module_prefix(state_dict)

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

def overlay_mask_on_rgb_label(rgb: np.ndarray, mask01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    rgb = rgb.copy()
    color = np.array([255, 128, 0], dtype=np.uint8)
    mask01 = (mask01 > 0).astype(np.uint8)
    overlay = rgb.copy()
    idx = mask01.astype(bool)
    overlay[idx] = (0.5 * rgb[idx] + 0.5 * color).astype(np.uint8)
    return overlay

def save_eval_triptych(frame_rgb: np.ndarray,
                       pred_mask_bool: np.ndarray,
                       gt_mask_bool: np.ndarray,
                       out_path: str,
                       clean_viz: bool = True,
                       draw_contour: bool = True,
                       add_legend: bool = True):
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

@torch.no_grad()
def run_sequence_metrics_only(frames_dir: str,
                              ckpt_path: str,
                              prior_thr: float = 0.35,
                              box_pad_frac: float = 0.10,
                              min_box: int = 16,
                              out_vis_dir: Optional[str] = None,
                              vis_stride: int = 1,
                              clean_viz: bool = True) -> Tuple[int, float, float, float, float]:

    frames_dir = _exists_dir(frames_dir)
    if out_vis_dir is not None: os.makedirs(out_vis_dir, exist_ok=True)

    frames = _sorted_frames(frames_dir)
    if len(frames) < 2:
        return 0, 0.0, 0.0, 0.0, 0.0

    first_stem = os.path.splitext(os.path.basename(frames[0]))[0]
    m0_path = _find_gt_watershed(first_stem, frames_dir)
    if m0_path is None:
        return 0, 0.0, 0.0, 0.0, 0.0

    m0_arr = _load_label_map(m0_path)
    chosen = _choose_label_31_32_only(m0_arr)
    if chosen is None:
        return 0, 0.0, 0.0, 0.0, 0.0

    prompter = PrompterNet(in_ch=7, base=32).to(DEVICE).eval()
    state_dict = load_prompter_state_dict(ckpt_path, map_location=DEVICE)
    prompter.load_state_dict(state_dict)
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")

    if out_vis_dir is not None:
        img0 = cv2.cvtColor(cv2.imread(_san(frames[0])), cv2.COLOR_BGR2RGB)
        m0_bin = (m0_arr == chosen).astype(np.uint8)
        overlay0 = overlay_mask_on_rgb_label(img0, m0_bin, alpha=0.5)
        cv2.imwrite(os.path.join(out_vis_dir, f"init_rgb0_overlay_label_{chosen}.png"),
                    cv2.cvtColor(overlay0, cv2.COLOR_RGB2BGR))
        box0 = box_from_mask_label(m0_bin, pad_frac=box_pad_frac, min_box=min_box).astype(np.int32)
        img0_box = img0.copy()
        x0, y0, x1, y1 = box0.tolist()
        cv2.rectangle(img0_box, (x0, y0), (x1, y1), (255, 255, 0), 2)
        cv2.imwrite(os.path.join(out_vis_dir, f"init_rgb0_box_label_{chosen}.png"),
                    cv2.cvtColor(img0_box, cv2.COLOR_RGB2BGR))

    m_prev = (m0_arr == chosen).astype(np.float32)

    rows = []
    for t in range(1, len(frames)):
        f_prev, f_curr = frames[t-1], frames[t]
        stem = os.path.splitext(os.path.basename(f_curr))[0]

        img_prev_bgr = cv2.imread(_san(f_prev)); img_curr_bgr = cv2.imread(_san(f_curr))
        if img_prev_bgr is None or img_curr_bgr is None:
            continue
        img_tm1 = cv2.cvtColor(img_prev_bgr, cv2.COLOR_BGR2RGB)
        img_t   = cv2.cvtColor(img_curr_bgr, cv2.COLOR_BGR2RGB)

        x_t   = torch.from_numpy(img_t.transpose(2,0,1)).float()[None] / 255.0
        x_tm1 = torch.from_numpy(img_tm1.transpose(2,0,1)).float()[None] / 255.0
        m_tm1 = torch.from_numpy(m_prev[None, None, ...]).float()
        x_t, x_tm1, m_tm1 = x_t.to(DEVICE), x_tm1.to(DEVICE), m_tm1.to(DEVICE)

        logits_t, _ = prompter(x_t, x_tm1, m_tm1)
        raw = logits_t[0, 0]
        raw = torch.nan_to_num(raw, nan=0.0, posinf=10.0, neginf=-10.0)
        prior_logits = (raw * max(1.0, LOGIT_GAIN)).clamp(-max(6.0, LOGIT_CLAMP), max(6.0, LOGIT_CLAMP))

        pred01, _box_xyxy = sam2_forward_logits_and_or_box(
            predictor, img_t, prior_logits,
            use_mask=True, use_box=True,
            box_from="prior", prior_thr=prior_thr,
            box_pad_frac=box_pad_frac, min_box_size=min_box, gt01=None
        )

        gt_path = _find_gt_watershed(stem, frames_dir)
        if gt_path and os.path.exists(gt_path):
            gt_arr = _load_label_map(gt_path)
            if 31 in np.unique(gt_arr): lab = 31
            elif 32 in np.unique(gt_arr): lab = 32
            else:
                m_prev = (pred01 >= 0.5).astype(np.float32)
                continue
            gt01 = (gt_arr == lab).astype(np.float32)

            d, j = _dice_iou(pred01, gt01, thr=0.5)
            f    = _boundary_f_measure(pred01, gt01, tol=2)
            jf   = 0.5 * (j + f)
            rows.append((d, j, f, jf))

            if out_vis_dir is not None and ((t % vis_stride) == 0):
                out_png = os.path.join(out_vis_dir, f"eval_{t:05d}.png")
                save_eval_triptych(img_t, (pred01 >= 0.5), (gt01 > 0.5), out_png,
                                   clean_viz=clean_viz, draw_contour=True, add_legend=True)

        m_prev = (pred01 >= 0.5).astype(np.float32)

    if not rows:
        return 0, 0.0, 0.0, 0.0, 0.0

    arr = np.array(rows, dtype=np.float32)
    mean_d, mean_j, mean_f, mean_jf = arr.mean(axis=0).tolist()
    return len(rows), float(mean_d), float(mean_j), float(mean_f), float(mean_jf)

def main():
    total_frames = 0
    sum_dice = sum_j = sum_f = sum_jf = 0.0
    n_seq_used = 0
    skipped = []

    for rel in TEST_LIST:
        seq_dir = os.path.join(MEDICAL_ROOT, rel)

        out_vis_dir = None
        if VIS:
            out_vis_dir = os.path.join(VIS_DIR, rel.replace("\\", "__"))
            os.makedirs(out_vis_dir, exist_ok=True)

        try:
            n_frames, d, j, f, jf = run_sequence_metrics_only(
                seq_dir, CKPT_PATH, prior_thr=0.35, box_pad_frac=0.10, min_box=16,
                out_vis_dir=out_vis_dir, vis_stride=VIS_STRIDE, clean_viz=CLEAN_VIZ
            )
        except Exception:
            n_frames, d, j, f, jf = 0, 0.0, 0.0, 0.0, 0.0

        if n_frames == 0:
            skipped.append(rel)
            continue

        print(f"[seq] {rel} | frames={n_frames} | Dice={d:.4f} | J(IoU)={j:.4f} | F={f:.4f} | J&F={jf:.4f}")

        total_frames += n_frames
        sum_dice     += d  * n_frames
        sum_j        += j  * n_frames
        sum_f        += f  * n_frames
        sum_jf       += jf * n_frames
        n_seq_used   += 1

    if total_frames > 0:
        mean_d_all  = sum_dice / total_frames
        mean_j_all  = sum_j    / total_frames
        mean_f_all  = sum_f    / total_frames
        mean_jf_all = sum_jf   / total_frames
        print(f"[ALL] sequences={n_seq_used} | frames={total_frames} | "
              f"Dice={mean_d_all:.4f} | J(IoU)={mean_j_all:.4f} | F={mean_f_all:.4f} | J&F={mean_jf_all:.4f}")
    else:
        print("[ALL] no frames evaluated (no first-frame 31/32 or missing GT).")

    if skipped:
        print(f"[SKIPPED] {len(skipped)} sequences (no 31/32 in first-frame GT / no frames/GT).")

if __name__ == "__main__":
    main()
