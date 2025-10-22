# ----- VIS -----
VIS_TRAIN_MAX_PER_EPOCH = 12
VIS_VAL_MAX_PER_EPOCH   = 26
VIS_EVERY_N_STEPS       = 200
SAVE_NUMPY_DUMPS        = False
PRINT_GETITEM = False

# ================================================================
# train_temporal_prompter_sam2.py  (ResBlock+ASPP+DeepSupervision + hardened I/O)
# ================================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ABSL_LOGGING_MIN_LEVEL"] = "3"
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass

import os, glob, time, random, json, csv, warnings, re, hashlib
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

try:
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True
except Exception:
    pass

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

from sam2.sam2_image_predictor import SAM2ImagePredictor

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DATA_ROOT  = r"C:\pythonProject\DL_project\dataset\DAVIS2017"
SPLIT      = "DAVIS_train"
RES        = "480p"
SPLIT_JSON = r"C:\Users\28601\PycharmProjects\DL\tfds_davis_split_480p.json"

OUT_DIR = r".\runs\train_prompter_sam2\ALL_TRAIN_ONLY_BOX"
os.makedirs(OUT_DIR, exist_ok=True)
VIZ_DIR = os.path.join(OUT_DIR, "viz");    os.makedirs(VIZ_DIR, exist_ok=True)
CSV_DIR = os.path.join(OUT_DIR, "metrics");os.makedirs(CSV_DIR, exist_ok=True)
TB_DIR  = os.path.join(OUT_DIR, "tb");     os.makedirs(TB_DIR, exist_ok=True)


FEED_MODE = "box-only"
BOX_FROM  = "prior"
PRIOR_THR = 0.35
BOX_PAD_FRAC = 0.10
MIN_BOX_SIZE = 16

EXPERIMENT_PROB_TO_LOGITS_FEED = True
RUN_SANITY_ONCE_IN_EVAL        = False
_RUN_PRETRAIN_SANITY           = False
_SANITY_ALREADY_RAN            = False
SAVE_FEEDS_DEBUG               = False

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS       = 3
BATCH_SIZE   = 1
LR           = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS  = 2

# ---------- Warmup ----------
WARMUP_STEPS        = 2000
WARMUP_USE_LOGITS   = True
WARMUP_SKIP_GATE    = True

# ---------- Transition ----------
TRANSITION_STEPS     = 2000
NEG_BIAS_MAX         = 0.0
GATE_SIGMOID_GAMMA   = 0.20
def ramp01(global_step: int, start: int, span: int) -> float:
    if global_step <= start: return 0.0
    return float(min(1.0, max(0.0, (global_step - start) / max(1, span))))

# ---------- Loss Weights ----------
LAMBDA_SAM2_EARLY = 0.7
LAMBDA_AUX_EARLY  = 0.3
LAMBDA_SAM2_LATE  = 0.85
LAMBDA_AUX_LATE   = 0.15

# ---------- Prior Params ----------
USE_LOGITS_PRIOR = True
LOGIT_GAIN       = 2.0
LOGIT_CLAMP      = 6.0
CONF_TAU         = 0.08
NEG_BIAS         = 1.0

# Deep supervision weights (u1, u2, u3)
DS_WEIGHTS = [0.20, 0.10, 0.10]

def _mask_to_box_xyxy(mask01: np.ndarray, pad_frac: float = 0.10, min_size: int = 16, H=None, W=None):
    m = (mask01 > 0.5).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if ys.size == 0: return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    if H is None or W is None: H, W = m.shape[:2]
    h = y1 - y0 + 1; w = x1 - x0 + 1
    py = int(round(h*pad_frac)); px = int(round(w*pad_frac))
    y0 = max(0, y0 - py); y1 = min(H-1, y1 + py)
    x0 = max(0, x0 - px); x1 = min(W-1, x1 + px)
    if (y1 - y0 + 1) < min_size:
        extra = (min_size - (y1 - y0 + 1))
        y0 = max(0, y0 - extra//2); y1 = min(H-1, y1 + (extra - extra//2))
    if (x1 - x0 + 1) < min_size:
        extra = (min_size - (x1 - x0 + 1))
        x0 = max(0, x0 - extra//2); x1 = min(W-1, x1 + (extra - extra//2))
    return (int(x0), int(y0), int(x1), int(y1))

# ========= safe image write helpers =========
ImageFile.LOAD_TRUNCATED_IMAGES = True
def _norm01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def safe_quantile(arr: np.ndarray, q: float) -> float:
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.5, posinf=1.0, neginf=0.0)
    a = np.clip(a, 0.0, 1.0).ravel()
    if a.size == 0: return float(q)
    return float(np.quantile(a, q))

@torch.no_grad()
def compute_conf_maps(raw_logits_b: torch.Tensor,
                      logit_gain: float, logit_clamp: float,
                      gate_b_np: np.ndarray, neg_bias: float):
    p_raw = torch.sigmoid(raw_logits_b).detach().cpu().numpy().astype(np.float32)
    conf_raw_sym = _norm01(np.abs(p_raw - 0.5) * 2.0)
    prior_logits = (raw_logits_b * logit_gain).clamp(-logit_clamp, logit_clamp)
    p_prior = torch.sigmoid(prior_logits).detach().cpu().numpy().astype(np.float32)
    conf_prior_sym = _norm01(np.abs(p_prior - 0.5) * 2.0)
    conf_prior_pos = _norm01(np.maximum(p_prior - 0.5, 0.0) * 2.0)
    gate_t = torch.from_numpy(gate_b_np).to(raw_logits_b.device, dtype=prior_logits.dtype)
    feed_logits = prior_logits * gate_t + (-neg_bias) * (1.0 - gate_t)
    p_feed = torch.sigmoid(feed_logits).detach().cpu().numpy().astype(np.float32)
    conf_feed_sym = _norm01(np.abs(p_feed - 0.5) * 2.0)
    conf_feed_pos = _norm01(np.maximum(p_feed - 0.5, 0.0) * 2.0)
    return {
        "p_raw": p_raw, "p_prior": p_prior, "p_feed": p_feed,
        "conf_raw_sym": conf_raw_sym,
        "conf_prior_sym": conf_prior_sym, "conf_prior_pos": conf_prior_pos,
        "conf_feed_sym": conf_feed_sym,   "conf_feed_pos": conf_feed_pos,
    }

def _ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)

def _safe_stem(stem: str, max_len: int = 120) -> str:
    s = re.sub(r'[^A-Za-z0-9_\-\.]+', '_', str(stem))
    if len(s) <= max_len: return s
    h = hashlib.md5(s.encode('utf-8')).hexdigest()[:8]
    return s[:max_len-9] + "_" + h

def _as_u8_img(arr: np.ndarray) -> np.ndarray:
    if arr is None: return None
    if arr.dtype in (np.float32, np.float64):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)

def _resize_if_needed(img: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= max_h and w <= max_w: return img
    scale = min(max_h / max(1, h), max_w / max(1, w))
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def _pil_save(path: str, img_u8: np.ndarray, fmt: str) -> bool:
    try:
        _ensure_dir_for(path)
        if img_u8.ndim == 2:
            im = Image.fromarray(img_u8)
        elif img_u8.ndim == 3 and img_u8.shape[2] == 3:
            im = Image.fromarray(cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB))
        else:
            if img_u8.ndim == 3 and img_u8.shape[2] == 1:
                im = Image.fromarray(img_u8[..., 0]).convert("RGB")
            else:
                im = Image.fromarray(img_u8)
        ext = fmt.lower()
        if ext in (".jpg", ".jpeg"):
            im.save(path, format="JPEG", quality=95, subsampling=1)
        elif ext == ".png":
            im.save(path, format="PNG", compress_level=3)
        else:
            im.save(path)
        return True
    except Exception as e:
        print(f"[pil-save] failed {fmt} -> {path}: {e}")
        return False

def _write_bytes(path: str, ext: str, img: np.ndarray, params=None) -> bool:
    try:
        _ensure_dir_for(path)
        ok, buf = cv2.imencode(ext, img, params or [])
        if ok:
            with open(path, "wb") as f:
                f.write(buf.tobytes())
            return True
        return _pil_save(path, img, ext)
    except Exception as e:
        print(f"[imencode] cv2 failed ({ext}) -> {path}: {e}")
        return _pil_save(path, img, ext)

def imwrite_safe(path: str, img: np.ndarray) -> bool:
    try:
        if img is None: return False
        img = _as_u8_img(img)
        if img.size == 0 or img.shape[0] <= 0 or img.shape[1] <= 0: return False
        MAX_ANY_SIDE = 6000
        h, w = img.shape[:2]
        if h > MAX_ANY_SIDE or w > MAX_ANY_SIDE:
            scale = min(MAX_ANY_SIDE / max(1, h), max(1, w))
            img = cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))), interpolation=cv2.INTER_AREA)
        root, ext = os.path.splitext(path)
        ext = (ext or ".jpg").lower()
        if ext in (".jpg", ".jpeg"):
            return _write_bytes(path, ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        elif ext == ".png":
            if _write_bytes(path, ".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]): return True
            return _write_bytes(root + ".jpg", ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            return _write_bytes(root + ".jpg", ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    except Exception as e:
        print(f"[imwrite_safe] error writing {path}: {e}")
        return _pil_save(path if path.lower().endswith(".jpg") else (os.path.splitext(path)[0] + ".jpg"),
                         _as_u8_img(img), ".jpg")

# -------------------- Sanity: mask_input responsiveness --------------------
@torch.no_grad()
def sanity_maskinput_response(predictor, img_rgb_uint8, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    predictor.set_image(img_rgb_uint8)
    feats = predictor._features["image_embed"]
    emb = feats[0] if isinstance(feats, (list, tuple)) else feats
    He, We = int(emb.shape[-2]), int(emb.shape[-1])
    H4, W4 = He*4, We*4
    m_zero = np.zeros((1,1,H4,W4), np.float32)
    out_zero = predictor.predict(point_coords=None, point_labels=None, box=None, mask_input=m_zero, multimask_output=False)
    mA = (out_zero[0][0] if isinstance(out_zero[0], (list,tuple)) else out_zero[0]).astype(np.float32)
    m_plus = np.full((1,1,H4,W4), 5.0, np.float32)
    out_plus = predictor.predict(point_coords=None, point_labels=None, box=None, mask_input=m_plus, multimask_output=False)
    mB = (out_plus[0][0] if isinstance(out_plus[0], (list,tuple)) else out_plus[0]).astype(np.float32)
    print(f"[sanity] mean(pred|zeros)={mA.mean():.4f}, mean(pred|+5)={mB.mean():.4f}")
    imwrite_safe(os.path.join(save_dir, "pred_from_zeros.jpg"), (np.clip(mA,0,1)*255).astype(np.uint8))
    imwrite_safe(os.path.join(save_dir, "pred_from_plus5.jpg"), (np.clip(mB,0,1)*255).astype(np.uint8))

def safe_torch_compile(model):
    try:
        import platform
        if platform.system().lower().startswith("win"):
            raise RuntimeError("Windows: Triton/Inductor לא יציבים, מבטל compile.")
        if getattr(torch.version, "triton", None) in (None, "0.0.0"):
            raise RuntimeError("No working Triton found for TorchInductor.")
        import triton  # noqa: F401
        return torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print(f"[warn] torch.compile disabled: {e}")
        return model

warnings.filterwarnings("ignore", category=UserWarning)
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# --------------------------- Dataset utils ---------------------------
def _sorted_frames_in_dir(d: str) -> List[str]:
    return sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))

def load_label_map(path: str) -> np.ndarray:
    im = Image.open(path)
    if im.mode == "P":
        arr = np.array(im, dtype=np.int32)
    else:
        arr = np.array(im.convert("L"), dtype=np.int32)
    return arr

def labels_in_mask(arr: np.ndarray) -> np.ndarray:
    u = np.unique(arr)
    return u[u > 0]

def area_of_label(arr: np.ndarray, label_id: int) -> int:
    return int((arr == label_id).sum())

REQUIRE_LABEL_PRESENT_IN_ALL_FRAMES = False

# --------------------------- Dataset ---------------------------
class MultiSeqDavisDataset(Dataset):
    def __init__(self, data_root: str, split_dir: str, res: str, seq_names: List[str]):
        self.data_root, self.split_dir, self.res = data_root, split_dir, res
        self.seq_names = seq_names
        self.seq_frames: List[List[str]] = []
               # ... (ללא שינוי עד סוף המחלקה)
        self.seq_masks_paths: List[List[str]] = []
        self.sample_index: List[Tuple[int,int]] = []
        self.chosen_label_per_seq: List[int] = []
        self._printed_once = set()
        for s_idx, name in enumerate(self.seq_names):
            vdir = os.path.join(data_root, split_dir, "JPEGImages", res, name)
            gdir = os.path.join(data_root, split_dir, "Annotations", res, name)
            assert os.path.isdir(vdir), f"Missing frames dir: {vdir}"
            assert os.path.isdir(gdir), f"Missing masks dir:  {gdir}"
            frames = _sorted_frames_in_dir(vdir)
            masks  = _sorted_frames_in_dir(gdir)
            assert len(frames) == len(masks) and len(frames) >= 2, f"error{name}: "
            self.seq_frames.append(frames)
            self.seq_masks_paths.append(masks)
            from collections import defaultdict
            count_map = defaultdict(int)
            for mp in masks:
                lm = load_label_map(mp)
                labs = labels_in_mask(lm)
                for L in labs:
                    if L > 0:
                        count_map[int(L)] += 1
            lm0 = load_label_map(masks[0])
            area0 = {int(L): area_of_label(lm0, int(L)) for L in labels_in_mask(lm0)}
            if (len(count_map) == 0) and (len(area0) == 0):
                raise ValueError(f"[{name}] error.")
            else:
                universe = set(count_map.keys()) | set(area0.keys())
                score = lambda L: (1 if area0.get(int(L), 0) > 0 else 0,
                                   int(count_map.get(int(L), 0)),
                                   int(area0.get(int(L), 0)))
                chosen_label = int(max(universe, key=score))
            self.chosen_label_per_seq.append(chosen_label)
            total_frames = len(masks)
            present_frames = int(count_map.get(chosen_label, 0))
            area0_chosen = int(area0.get(chosen_label, 0))
            print(f"[label-pick] seq={name} | chosen={chosen_label} | present_frames={present_frames}/{total_frames} | area@t0={area0_chosen}")
            h0,w0 = cv2.imread(frames[0], cv2.IMREAD_COLOR).shape[:2]
            lm0_dbg = load_label_map(masks[0])
            assert lm0_dbg.shape == (h0,w0), f"GT/I0 size mismatch in seq {name}: {lm0_dbg.shape} vs {(h0,w0)}"
            for t in range(1, len(frames)):
                self.sample_index.append((s_idx, t))

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_idx, t = self.sample_index[idx]
        frames = self.seq_frames[seq_idx]
        masks  = self.seq_masks_paths[seq_idx]
        label  = self.chosen_label_per_seq[seq_idx]
        img_tm1 = cv2.cvtColor(cv2.imread(frames[t-1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img_t   = cv2.cvtColor(cv2.imread(frames[t],   cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        H,W = img_t.shape[:2]
        lm_tm1 = load_label_map(masks[t-1])
        lm_t   = load_label_map(masks[t])
        assert lm_tm1.shape == (H,W) and lm_t.shape == (H,W), f"GT size mismatch at seq {self.seq_names[seq_idx]} t={t}"
        m_tm1  = (lm_tm1 == label).astype(np.uint8)
        m_t    = (lm_t   == label).astype(np.uint8)
        if PRINT_GETITEM and t == 1 and (self.seq_names[seq_idx] not in self._printed_once):
            print(f"[getitem] seq={self.seq_names[seq_idx]} t={t} has_GT_tm1={bool(m_tm1.any())} has_GT_t={bool(m_t.any())}")
            self._printed_once.add(self.seq_names[seq_idx])
        return {
            "img_tm1": torch.from_numpy(img_tm1.transpose(2,0,1)).float()/255.0,
            "img_t":   torch.from_numpy(img_t.transpose(2,0,1)).float()/255.0,
            "m_tm1":   torch.from_numpy(m_tm1[None,...]).float(),
            "m_t":     torch.from_numpy(m_t[None,...]).float(),
            "img_t_np": img_t,
            "seq_idx": torch.tensor(seq_idx, dtype=torch.long),
            "t":       torch.tensor(t, dtype=torch.long),
        }

# ------------------------- Blocks -------------------------
class SE(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(ch // r, 1), 1), nn.ReLU(True),
            nn.Conv2d(max(ch // r, 1), ch, 1), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch)
        )
        self.se = SE(out_ch)
        self.act = nn.ReLU(True)
    def forward(self, x):
        y = self.conv(x)
        y = self.se(y)
        y = y + self.proj(x)
        return self.act(y)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12)):
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch), nn.ReLU(True)
        ))
        for r in rates[1:]:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.GroupNorm(8, out_ch), nn.ReLU(True)
            ))
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(self.branches), out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch), nn.ReLU(True)
        )
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.project(x)

# ------------------------- PrompterNet -------------------------
def _center_crop_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    _, _, H, W = x.shape
    top  = max((H - h) // 2, 0)
    left = max((W - w) // 2, 0)
    return x[:, :, top:top+h, left:left+w]

def _pad_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    _, _, H, W = x.shape
    pad_h = max(h - H, 0); pad_w = max(w - W, 0)
    pl = pad_w // 2; pr = pad_w - pl; pt = pad_h // 2; pb = pad_h - pt
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (pl, pr, pt, pb), mode="reflect")
    return x

def _match_spatial(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    _, _, h, w = ref.shape
    x = _center_crop_to(x, min(x.shape[2], h), min(x.shape[3], w))
    x = _pad_to(x, h, w)
    return x

class PrompterNet(nn.Module):
    def __init__(self, in_ch=7, base=32, groups=8):
        super().__init__()
        self.down1 = ResBlock(in_ch, base, groups);    self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResBlock(base, base*2, groups);   self.pool2 = nn.MaxPool2d(2)
        self.down3 = ResBlock(base*2, base*4, groups); self.pool3 = nn.MaxPool2d(2)
        self.down4 = ResBlock(base*4, base*8, groups)
        self.aspp = ASPP(base*8, base*8)
        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = ResBlock(base*8, base*4, groups)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ResBlock(base*4, base*2, groups)
        self.up1  = nn.ConvTranspose2d(base*2, base,   2, stride=2)
        self.dec1 = ResBlock(base*2, base, groups)
        self.out_conv = nn.Conv2d(base, 1, 1)
        self.ds_head_u1 = nn.Conv2d(base,   1, 1)
        self.ds_head_u2 = nn.Conv2d(base*2, 1, 1)
        self.ds_head_u3 = nn.Conv2d(base*4, 1, 1)
    def forward(self, img_t, img_tm1, m_tm1):
        x_in = torch.cat([img_t, img_tm1, m_tm1], dim=1)
        d1 = self.down1(x_in); p1 = self.pool1(d1)
        d2 = self.down2(p1);   p2 = self.pool2(d2)
        d3 = self.down3(p2);   p3 = self.pool3(d3)
        bott = self.down4(p3)
        bott = self.aspp(bott)
        u3 = self.up3(bott); d3m = _match_spatial(d3, u3); u3 = self.dec3(torch.cat([u3, d3m], 1))
        u2 = self.up2(u3);   d2m = _match_spatial(d2, u2); u2 = self.dec2(torch.cat([u2, d2m], 1))
        u1 = self.up1(u2);   d1m = _match_spatial(d1, u1); u1 = self.dec1(torch.cat([u1, d1m], 1))
        u1 = _match_spatial(u1, x_in)
        logits_main = self.out_conv(u1)
        def _up_to(ref, x):
            return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        ds1 = self.ds_head_u1(u1)
        ds2 = _up_to(logits_main, self.ds_head_u2(u2))
        ds3 = _up_to(logits_main, self.ds_head_u3(u3))
        return logits_main, [ds1, ds2, ds3]

# ---------------------------- Losses ----------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = eps
    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        num = 2 * (probs * target).sum(dim=(1,2,3))
        den = (probs + target).sum(dim=(1,2,3)) + self.eps
        dice = 1 - (num + self.eps) / den
        return bce + dice.mean()

def dice_from_probs(probs, target, eps=1e-6):
    num = 2 * (probs * target).sum()
    den = (probs + target).sum() + eps
    return (num + eps) / (den + eps)

def iou_from_probs(probs, target, thr=0.5, eps=1e-7):
    p = (probs >= thr).float()
    inter = (p * target).sum()
    union = (p + target - p*target).sum()
    return (inter + eps) / (union + eps)

def dice_per_sample(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B = probs.shape[0]
    p = probs.view(B, -1)
    t = target.view(B, -1)
    num = 2.0 * (p * t).sum(dim=1)
    den = (p + t).sum(dim=1) + eps
    return (num + eps) / den

# --------------------- SAM2 wrappers ---------------------
@torch.no_grad()
def sam2_forward_mask_input_logits(predictor, img_rgb_uint8, mask_logits, viz_path: Optional[str]=None):
    img_np = img_rgb_uint8.detach().cpu().numpy() if isinstance(img_rgb_uint8, torch.Tensor) else img_rgb_uint8
    if img_np.dtype != np.uint8: img_np = img_np.astype(np.uint8)
    img_np = np.ascontiguousarray(img_np)
    orig_h, orig_w = img_np.shape[:2]
    predictor.set_image(img_np)
    feats = predictor._features["image_embed"]
    emb = feats[0] if isinstance(feats, (list, tuple)) else feats
    He, We = int(emb.shape[-2]), int(emb.shape[-1])
    target_h, target_w = He * 4, We * 4
    if isinstance(mask_logits, torch.Tensor):
        mlog = mask_logits.detach().cpu().numpy().astype(np.float32)
    else:
        mlog = np.asarray(mask_logits, dtype=np.float32)
    if mlog.ndim == 3:
        mlog = mlog[0]
    mlog_rs = cv2.resize(mlog, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mask_input = mlog_rs[None, None, ...].astype(np.float32)
    if viz_path is not None and SAVE_NUMPY_DUMPS:
        viz = np.tanh(mlog_rs / 6.0) * 0.5 + 0.5
        cm  = cv2.applyColorMap((viz*255).astype(np.uint8), cv2.COLORMAP_JET)
        imwrite_safe(viz_path, cm)
    out = predictor.predict(point_coords=None, point_labels=None, box=None,
                            mask_input=mask_input, multimask_output=False)
    masks = out[0] if isinstance(out, (list, tuple)) else out
    m = masks[0] if isinstance(masks, (list, tuple)) else masks
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 3: m = m[0]
    if (m.shape[0] != orig_h) or (m.shape[1] != orig_w):
        m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0)

@torch.no_grad()
def sam2_forward_mask_input_probs(predictor, img_rgb_uint8, mask_prob_01, viz_path=None):
    img_np = img_rgb_uint8.detach().cpu().numpy() if isinstance(img_rgb_uint8, torch.Tensor) else img_rgb_uint8
    if img_np.dtype != np.uint8: img_np = img_np.astype(np.uint8)
    img_np = np.ascontiguousarray(img_np)
    orig_h, orig_w = img_np.shape[:2]
    predictor.set_image(img_np)
    feats = predictor._features["image_embed"]
    emb = feats[0] if isinstance(feats, (list, tuple)) else feats
    He, We = int(emb.shape[-2]), int(emb.shape[-1])
    target_h, target_w = He * 4, We * 4
    if isinstance(mask_prob_01, torch.Tensor): mask_np = mask_prob_01.detach().cpu().numpy()
    else: mask_np = mask_prob_01
    mask_np = np.asarray(mask_np, dtype=np.float32)
    if mask_np.ndim == 3: mask_np = mask_np[0]
    mask_resized = cv2.resize(mask_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = np.clip(mask_resized, 0.0, 1.0).astype(np.float32)
    mask_input = mask_resized[None, None, ...]
    out = predictor.predict(point_coords=None, point_labels=None, box=None,
                            mask_input=mask_input, multimask_output=False)
    masks = out[0] if isinstance(out, (list, tuple)) else out
    m = masks[0] if isinstance(masks, (list, tuple)) else masks
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 3: m = m[0]
    if (m.shape[0] != orig_h) or (m.shape[1] != orig_w):
        m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0)

@torch.no_grad()
def sam2_forward_logits_and_or_box(predictor, img_rgb_uint8, mask_logits, use_mask=True, use_box=True,
                                   box_from="prior", prior_thr=0.35, box_pad_frac=0.10, min_box_size=16,
                                   gt01=None):
    img_np = img_rgb_uint8 if isinstance(img_rgb_uint8, np.ndarray) else img_rgb_uint8.cpu().numpy()
    if img_np.dtype != np.uint8: img_np = img_np.astype(np.uint8)
    H, W = img_np.shape[:2]
    predictor.set_image(img_np)

    # mask_input
    mask_input = None
    if use_mask and (mask_logits is not None):
        feats = predictor._features["image_embed"]
        emb = feats[0] if isinstance(feats, (list, tuple)) else feats
        He, We = int(emb.shape[-2]), int(emb.shape[-1])
        H4, W4 = He*4, We*4
        mlog = mask_logits.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(mask_logits) else np.asarray(mask_logits, np.float32)
        if mlog.ndim == 3: mlog = mlog[0]
        mlog_rs = cv2.resize(mlog, (W4, H4), interpolation=cv2.INTER_LINEAR)
        mask_input = mlog_rs[None, None, ...].astype(np.float32)

    # box
    box_xyxy = None
    if use_box:
        if box_from == "gt":
            if gt01 is None: raise ValueError("box_from='gt' but gt01 is None")
            box_xyxy = _mask_to_box_xyxy(gt01, pad_frac=box_pad_frac, min_size=min_box_size, H=H, W=W)
        else:
            p = torch.sigmoid(torch.from_numpy(mask_logits) if isinstance(mask_logits, np.ndarray) else mask_logits).detach().cpu().numpy()
            if p.ndim == 3: p = p[0]
            prior_bin = (p >= prior_thr).astype(np.float32)
            if prior_bin.sum() > 0:
                box_xyxy = _mask_to_box_xyxy(prior_bin, pad_frac=box_pad_frac, min_size=min_box_size, H=H, W=W)

    out = predictor.predict(point_coords=None, point_labels=None, box=box_xyxy,
                            mask_input=mask_input, multimask_output=False)
    masks = out[0] if isinstance(out, (list, tuple)) else out
    m = masks[0] if isinstance(masks, (list, tuple)) else masks
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 3: m = m[0]
    if (m.shape[0] != H) or (m.shape[1] != W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0), box_xyxy

# --------------------- VIS helpers ---------------------
def _to_u8(x01: np.ndarray) -> np.ndarray: return (np.clip(x01, 0, 1) * 255).astype(np.uint8)

def _put_title(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy(); cv2.rectangle(out,(0,0),(out.shape[1],26),(0,0,0),-1)
    cv2.putText(out, text, (6,19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return out

def _draw_box(img_bgr: np.ndarray, box_xyxy, color=(0,0,255), thickness=2):
    if box_xyxy is None: return img_bgr
    x0,y0,x1,y1 = [int(v) for v in box_xyxy]
    out = img_bgr.copy()
    cv2.rectangle(out, (x0,y0), (x1,y1), color, thickness)
    return out

def _mask_overlay(img_bgr, mask01, color=(0,255,0), alpha=0.45):
    m = np.clip(mask01.astype(np.float32), 0, 1)
    col = np.zeros_like(img_bgr); col[:] = color
    return (img_bgr*(1-alpha) + col*alpha*m[...,None]).astype(np.uint8)

def _edge_overlay(img_bgr, mask01, color=(0,0,255)):
    m = (mask01 >= 0.5).astype(np.uint8)*255
    edges = cv2.Canny(m, 50, 150)
    out = img_bgr.copy(); out[edges>0] = color
    return out

def _jet(prob01): return cv2.applyColorMap(_to_u8(prob01), cv2.COLORMAP_JET)

def _hstack_resize(imgs_bgr: List[np.ndarray], h: int) -> np.ndarray:
    MAX_TILE_W = 2000
    rs = []
    for im in imgs_bgr:
        if im is None or im.size == 0: continue
        im = _as_u8_img(im)
        if im.shape[0] == 0: continue
        new_w = int(im.shape[1]*h/max(1, im.shape[0]))
        new_w = max(1, min(new_w, MAX_TILE_W))
        rs.append(cv2.resize(im, (new_w, h), interpolation=cv2.INTER_AREA))
    if len(rs) == 0:
        return np.zeros((h, h, 3), dtype=np.uint8)
    row = np.concatenate(rs, axis=1)
    MAX_ROW_W = 6000
    if row.shape[1] > MAX_ROW_W:
        row = cv2.resize(row, (MAX_ROW_W, h), interpolation=cv2.INTER_AREA)
    return row

def _pad_to_width(img_bgr: np.ndarray, width: int) -> np.ndarray:
    if img_bgr.shape[1] == width: return img_bgr
    pad = width - img_bgr.shape[1]
    if pad < 0:
        img_bgr = cv2.resize(img_bgr, (width, img_bgr.shape[0]), interpolation=cv2.INTER_AREA)
        return img_bgr
    return cv2.copyMakeBorder(img_bgr, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0,0,0))

def save_full_vis(epoch_dir, seq_name, stem,
                  img_t_rgb, img_tm1_rgb, m_tm1_01,
                  prior01, sam2_pred01, gt01,
                  conf01=None, gate01=None, raw_logits=None,
                  feed_box_xyxy=None, extra_panels: Dict[str, np.ndarray] = None):
    seq_dir = os.path.join(epoch_dir, seq_name)
    os.makedirs(seq_dir, exist_ok=True)
    stem = _safe_stem(stem)
    img_t_bgr   = cv2.cvtColor(img_t_rgb,   cv2.COLOR_RGB2BGR)
    img_tm1_bgr = cv2.cvtColor(img_tm1_rgb, cv2.COLOR_RGB2BGR)
    prior_cm, pred_cm = _jet(prior01), _jet(sam2_pred01)
    prior_cm_box = _draw_box(prior_cm, feed_box_xyxy, (0,0,255), 2)
    prior_overlay_box = _draw_box(_mask_overlay(img_t_bgr, prior01, (255,0,0), 0.35), feed_box_xyxy, (0,0,255), 2)

    gt_u8, m_tm1_u8   = _to_u8(gt01), _to_u8(m_tm1_01)
    imwrite_safe(os.path.join(seq_dir, f"{stem}_img_t.jpg"),   img_t_bgr)
    imwrite_safe(os.path.join(seq_dir, f"{stem}_img_tm1.jpg"), img_tm1_bgr)
    imwrite_safe(os.path.join(seq_dir, f"{stem}_m_tm1_mask.jpg"), m_tm1_u8)
    imwrite_safe(os.path.join(seq_dir, f"{stem}_prior_colormap.jpg"), prior_cm_box)
    imwrite_safe(os.path.join(seq_dir, f"{stem}_prior_overlay_box.jpg"), prior_overlay_box)
    imwrite_safe(os.path.join(seq_dir, f"{stem}_pred_colormap.jpg"),  pred_cm)
    imwrite_safe(os.path.join(seq_dir, f"{stem}_gt_mask.jpg"), gt_u8)
    imwrite_safe(os.path.join(seq_dir, f"{stem}_pred_overlay.jpg"), _mask_overlay(img_t_bgr, sam2_pred01, (0,255,0)))
    imwrite_safe(os.path.join(seq_dir, f"{stem}_gt_edges_overlay.jpg"), _edge_overlay(img_t_bgr, gt01, (0,0,255)))
    imwrite_safe(os.path.join(seq_dir, f"{stem}_gt_overlay.jpg"), _mask_overlay(img_t_bgr, gt01, (0,0,255)))

    row1 = _hstack_resize([
        _put_title(img_tm1_bgr, "I_{t-1}"),
        _put_title(img_t_bgr,   "I_t"),
        _put_title(cv2.cvtColor(m_tm1_u8, cv2.COLOR_GRAY2BGR), "M_{t-1} (GT)")
    ], 256)
    row2 = _hstack_resize([
        _put_title(prior_cm_box, "prior (feed) + BOX"),
        _put_title(prior_overlay_box, "prior+box feed (overlay)"),
        _put_title(_jet(sam2_pred01),  "SAM2 pred")
    ], 256)
    panels = []
    if conf01 is not None:
        panels.append(_put_title(_jet(conf01), "confidence pos (feed)"))
    if gate01 is not None:
        gate_u8 = _to_u8(gate01); panels.append(_put_title(cv2.cvtColor(gate_u8, cv2.COLOR_GRAY2BGR), "gate"))
    if raw_logits is not None:
        lg = np.tanh(raw_logits / 6.0) * 0.5 + 0.5
        panels.append(_put_title(_jet(lg), "raw logits (tanh)"))
    if extra_panels:
        for name, m in extra_panels.items():
            panels.append(_put_title(_jet(_norm01(m)), name))
    row3_elems = [
        _put_title(_mask_overlay(img_t_bgr, sam2_pred01, (0,255,0)), "pred overlay"),
        _put_title(_edge_overlay(img_t_bgr, gt01, (0,0,255)), "GT edges"),
        _put_title(cv2.cvtColor(_to_u8(gt01), cv2.COLOR_GRAY2BGR), "GT mask")
    ] + panels
    row3 = _hstack_resize(row3_elems, 256)
    W = max(row1.shape[1], row2.shape[1], row3.shape[1])
    row1 = _pad_to_width(row1, W); row2 = _pad_to_width(row2, W); row3 = _pad_to_width(row3, W)
    try:
        panel = np.vstack([row1, row2, row3])
        panel = _resize_if_needed(panel, 6000, 6000)
        imwrite_safe(os.path.join(seq_dir, f"{stem}_PANEL.jpg"), panel)
    except Exception as e:
        print(f"[viz-panel] skip panel: {e}")
    if SAVE_NUMPY_DUMPS:
        np.save(os.path.join(seq_dir, f"{stem}_prior.npy"), prior01.astype(np.float32))
        np.save(os.path.join(seq_dir, f"{stem}_pred.npy"),  sam2_pred01.astype(np.float32))
        np.save(os.path.join(seq_dir, f"{stem}_gt.npy"),    gt01.astype(np.float32))

# ----------------------------- EVAL -----------------------------
@torch.no_grad()
def evaluate(prompter, predictor, loader_val, device, seq_names: List[str],
             writer: SummaryWriter, epoch: int):
    global _SANITY_ALREADY_RAN
    prompter.eval()
    mean_dice, mean_iou, n = 0.0, 0.0, 0
    per_seq_sum = {s: {"dice":0.0, "iou":0.0, "count":0} for s in seq_names}
    saved_vis = 0
    epoch_dir = os.path.join(VIZ_DIR, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    if RUN_SANITY_ONCE_IN_EVAL and (not _SANITY_ALREADY_RAN):
        try:
            ex = next(iter(loader_val))
            img_u8 = ex["img_t_np"][0] if not isinstance(ex["img_t_np"], torch.Tensor) else ex["img_t_np"][0].cpu().numpy()
            sanity_maskinput_response(predictor, img_u8, os.path.join(VIZ_DIR, "sanity_sam2_eval"))
            _SANITY_ALREADY_RAN = True
        except Exception as e:
            print(f"[sanity-eval warn] {e}")

    for batch in tqdm(loader_val, desc=f"Val {epoch}/{EPOCHS}", unit="it", leave=True):
        img_tm1 = batch["img_tm1"].to(device, non_blocking=True)
        img_t   = batch["img_t"].to(device, non_blocking=True)
        m_tm1   = batch["m_tm1"].to(device, non_blocking=True)
        m_t     = batch["m_t"].to(device, non_blocking=True)
        imgs_np = batch["img_t_np"]
        seq_idx = batch["seq_idx"].tolist() if isinstance(batch["seq_idx"], torch.Tensor) else batch["seq_idx"]
        t_idx   = batch["t"].tolist() if isinstance(batch["t"], torch.Tensor) else batch["t"]

        out = prompter(img_t, img_tm1, m_tm1)
        logits_t, aux_list = (out if isinstance(out, tuple) else (out, []))
        probs_t  = torch.sigmoid(logits_t)

        sam2_preds = []
        B = logits_t.shape[0]
        for b in range(B):
            img_r = imgs_np[b]
            sname = seq_names[int(seq_idx[b])]
            stem0 = f"seq-{sname}_t-{int(t_idx[b])}"
            stem = _safe_stem(stem0)

            # prior logits
            raw_logits = logits_t[b, 0]
            raw_logits = torch.nan_to_num(raw_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            prior_logits = (raw_logits * max(1.0, LOGIT_GAIN)).clamp(-max(6.0, LOGIT_CLAMP), max(6.0, LOGIT_CLAMP))
            prior01_viz = torch.sigmoid(prior_logits).detach().cpu().numpy()

            # --- feed according to FEED_MODE/BOX_FROM ---
            use_mask = (FEED_MODE in ("mask-only", "mask+box"))
            use_box  = (FEED_MODE in ("box-only", "mask+box"))
            gt_box_mask = m_t[b,0].detach().cpu().numpy() if BOX_FROM == "gt" else None
            m_sam2, feed_box = sam2_forward_logits_and_or_box(
                predictor, img_r, prior_logits,
                use_mask=use_mask, use_box=use_box,
                box_from=("gt" if BOX_FROM=="gt" else "prior"),
                prior_thr=PRIOR_THR, box_pad_frac=BOX_PAD_FRAC, min_box_size=MIN_BOX_SIZE,
                gt01=gt_box_mask
            )

            sam2_preds.append(m_sam2[None, ...])

            if saved_vis < VIS_VAL_MAX_PER_EPOCH:
                img_rgb_t   = imgs_np[b] if not isinstance(imgs_np, torch.Tensor) else imgs_np[b].cpu().numpy()
                img_rgb_tm1 = (batch["img_tm1"][b].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                pred01      = m_sam2
                gt01        = m_t[b,0].detach().cpu().numpy()
                m_tm1_01    = m_tm1[b,0].detach().cpu().numpy()
                save_full_vis(epoch_dir, sname, stem,
                              img_rgb_t, img_rgb_tm1, m_tm1_01,
                              prior01_viz, pred01, gt01,
                              conf01=None, gate01=None,
                              raw_logits=raw_logits.detach().cpu().numpy(),
                              feed_box_xyxy=feed_box, extra_panels=None)
                saved_vis += 1

        sam2_pred = torch.from_numpy(np.stack(sam2_preds, axis=0)).to(device)
        batch_dice = float(dice_from_probs(sam2_pred, m_t).item())
        batch_iou  = float(iou_from_probs(sam2_pred, m_t).item())
        mean_dice += batch_dice; mean_iou += batch_iou; n += 1

        for b in range(B):
            sname = seq_names[int(seq_idx[b])]
            dice_b = float(dice_from_probs(sam2_pred[b:b+1], m_t[b:b+1]).item())
            iou_b  = float(iou_from_probs  (sam2_pred[b:b+1], m_t[b:b+1]).item())
            per_seq_sum[sname]["dice"]  += dice_b
            per_seq_sum[sname]["iou"]   += iou_b
            per_seq_sum[sname]["count"] += 1

        if saved_vis < 2:
            with torch.no_grad():
                dice_soft = float(dice_from_probs(sam2_pred, m_t).item())
                pred_bin  = (sam2_pred>=0.5).float()
                num = 2*(pred_bin*m_t).sum(); den = (pred_bin+m_t).sum() + 1e-6
                dice_bin  = float(((num+1e-6)/den).item())
                print(f"[metrics-check] dice_soft={dice_soft:.3f} dice_bin@0.5={dice_bin:.3f}")

    mean_dice /= max(1,n)
    mean_iou  /= max(1,n)

    rows = []
    for sname, agg in per_seq_sum.items():
        cnt = max(1, agg["count"])
        d = agg["dice"] / cnt; i = agg["iou"] / cnt
        writer.add_scalar(f"val_seq/{sname}/dice", d, epoch)
        writer.add_scalar(f"val_seq/{sname}/iou",  i, epoch)
        rows.append({"sequence": sname, "dice": d, "iou": i, "count": agg["count"]})

    csv_path = os.path.join(CSV_DIR, f"val_per_sequence_epoch_{epoch:03d}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sequence","dice","iou","count"])
        w.writeheader(); w.writerows(rows)

    return mean_dice, mean_iou

def main():
    print(f"[info] device: {DEVICE}")
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split_map = json.load(f)
    train_seqs: List[str] = split_map["train"]
    val_seqs:   List[str] = split_map["val"]

    SPLIT_TRAIN = "DAVIS_train"
    SPLIT_VAL   = "DAVIS_train"

    ds_train = MultiSeqDavisDataset(DATA_ROOT, SPLIT_TRAIN, RES, train_seqs)
    ds_val   = MultiSeqDavisDataset(DATA_ROOT, SPLIT_VAL,   RES, val_seqs)

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True, persistent_workers=True, prefetch_factor=4)
    loader_val   = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=False, persistent_workers=True, prefetch_factor=4)

    prompter  = PrompterNet(in_ch=7, base=32).to(DEVICE)
    prompter  = safe_torch_compile(prompter)
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")

    if _RUN_PRETRAIN_SANITY:
        try:
            ex = next(iter(loader_val))
            img_u8 = ex["img_t_np"][0] if not isinstance(ex["img_t_np"], torch.Tensor) else ex["img_t_np"][0].cpu().numpy()
            sanity_maskinput_response(predictor, img_u8, os.path.join(VIZ_DIR, "sanity_sam2_pretrain"))
        except Exception as e:
            print(f"[sanity-pretrain warn] {e}")

    opt   = torch.optim.Adam(prompter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(DEVICE == "cuda"))

    writer = SummaryWriter(log_dir=TB_DIR)
    for k,v in {
        "USE_LOGITS_PRIOR": int(USE_LOGITS_PRIOR),
        "LOGIT_GAIN": LOGIT_GAIN, "LOGIT_CLAMP": LOGIT_CLAMP,
        "CONF_TAU": CONF_TAU, "NEG_BIAS": NEG_BIAS,
        "LR": LR, "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS,
        "WARMUP_STEPS": WARMUP_STEPS, "WARMUP_USE_LOGITS": int(WARMUP_USE_LOGITS), "WARMUP_SKIP_GATE": int(WARMUP_SKIP_GATE),
        "TRANSITION_STEPS": TRANSITION_STEPS,
        "FEED_MODE_mask1_box0": int(FEED_MODE=="mask-only"),
        "FEED_MODE_box1": int(FEED_MODE=="box-only"),
        "FEED_MODE_mask1_box1": int(FEED_MODE=="mask+box"),
        "BOX_FROM_prior1_gt0": int(BOX_FROM=="prior"),
        "PRIOR_THR": PRIOR_THR
    }.items():
        if isinstance(v, (int, float)): writer.add_scalar(f"hparams/{k}", v, 0)

    best_val_dice = -1.0
    global_step = 0

    for epoch in range(1, EPOCHS+1):
        prompter.train()
        t0 = time.time()
        running_loss = running_dice = 0.0
        saved_train = 0

        if epoch <= 2:
            lambda_sam2_cur = LAMBDA_SAM2_EARLY
            lambda_aux_cur  = LAMBDA_AUX_EARLY
            alpha_bg_cur    = 0.35
        else:
            lambda_sam2_cur = LAMBDA_SAM2_LATE
            lambda_aux_cur  = LAMBDA_AUX_LATE
            alpha_bg_cur    = 0.20

        pbar = tqdm(loader_train, desc=f"Train {epoch}/{EPOCHS}", unit="it", leave=True)
        for _step, batch in enumerate(pbar, start=1):
            img_tm1 = batch["img_tm1"].contiguous(memory_format=torch.channels_last).to(DEVICE, non_blocking=True)
            img_t   = batch["img_t"]  .contiguous(memory_format=torch.channels_last).to(DEVICE, non_blocking=True)
            m_tm1   = batch["m_tm1"].to(DEVICE, non_blocking=True)
            m_t     = batch["m_t"]  .to(DEVICE, non_blocking=True)
            imgs_np = batch["img_t_np"]

            is_warmup = (global_step < WARMUP_STEPS)

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                out = prompter(img_t, img_tm1, m_tm1)
                logits_t, aux_list = (out if isinstance(out, tuple) else (out, []))
                probs_t  = torch.sigmoid(logits_t)

                with torch.no_grad():
                    raw = logits_t[:, 0].detach()
                    prior_logits_dbg = (raw * LOGIT_GAIN).clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
                    p_dbg = torch.sigmoid(prior_logits_dbg)
                    if _step % 600 == 0:
                        sat_hi = (raw > 10).float().mean().item()
                        sat_lo = (raw < -10).float().mean().item()
                        print(f"[mask-feed] mean(p)={p_dbg.mean().item():.3f}  sat_hi>{sat_hi:.2f}  sat_lo>{sat_lo:.2f}")

                sam2_preds = []
                for b in range(logits_t.shape[0]):
                    img_r = imgs_np[b]
                    sname = ds_train.seq_names[int(batch["seq_idx"][b])]
                    stem0 = f"seq-{sname}_step-{_step}_b{b}"
                    stem  = _safe_stem(stem0)

                    raw_logits = logits_t[b, 0]
                    raw_logits = torch.nan_to_num(raw_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                    prior_logits = (raw_logits * max(1.0, LOGIT_GAIN)).clamp(-max(6.0, LOGIT_CLAMP), max(6.0, LOGIT_CLAMP))

                    # בחירה לפי FEED_MODE
                    use_mask = (FEED_MODE in ("mask-only","mask+box"))
                    use_box  = (FEED_MODE in ("box-only","mask+box"))
                    gt_box_mask = m_t[b,0].detach().cpu().numpy() if BOX_FROM=="gt" else None

                    if use_mask or use_box:
                        m_sam2, feed_box = sam2_forward_logits_and_or_box(
                            predictor, img_r, prior_logits,
                            use_mask=use_mask, use_box=use_box,
                            box_from=("gt" if BOX_FROM=="gt" else "prior"),
                            prior_thr=PRIOR_THR, box_pad_frac=BOX_PAD_FRAC, min_box_size=MIN_BOX_SIZE,
                            gt01=gt_box_mask
                        )
                    else:
                        m_sam2 = sam2_forward_mask_input_logits(predictor, img_r, prior_logits)
                        feed_box = None

                    sam2_preds.append(m_sam2[None, ...])

                    if (_step % VIS_EVERY_N_STEPS == 0) and (saved_train < VIS_TRAIN_MAX_PER_EPOCH) and b == 0:
                        try:
                            train_epoch_dir = os.path.join(VIZ_DIR, f"epoch_{epoch:03d}_TRAIN")
                            os.makedirs(train_epoch_dir, exist_ok=True)
                            img_rgb_t   = imgs_np[0] if not isinstance(imgs_np, torch.Tensor) else imgs_np[0].cpu().numpy()
                            img_rgb_tm1 = (batch["img_tm1"][0].permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
                            pred01      = m_sam2
                            gt01        = m_t[0,0].detach().cpu().numpy()
                            m_tm1_01    = m_tm1[0,0].detach().cpu().numpy()
                            prior01_viz = torch.sigmoid(prior_logits).detach().cpu().numpy()
                            save_full_vis(
                                train_epoch_dir, sname, stem,
                                img_rgb_t, img_rgb_tm1, m_tm1_01,
                                prior01_viz, pred01, gt01,
                                conf01=None, gate01=None,
                                raw_logits=raw_logits.detach().cpu().numpy(),
                                feed_box_xyxy=feed_box, extra_panels=None
                            )
                            saved_train += 1
                        except Exception as e:
                            print(f"[train-vis warn] {e}")

                sam2_pred = torch.from_numpy(np.stack(sam2_preds, axis=0)).to(DEVICE)

                eps = 1e-6
                sp = sam2_pred.clamp(eps, 1 - eps)
                sam2_logits_pseudo = torch.log(sp) - torch.log(1 - sp)

                present = (m_t.sum(dim=(1,2,3)) > 0).float()

                bce_aux_main = F.binary_cross_entropy_with_logits(
                    logits_t, m_t, reduction="none"
                ).mean(dim=(1,2,3))
                dice_aux_main = 1.0 - dice_per_sample(torch.sigmoid(logits_t), m_t)

                bce_aux_ds, dice_aux_ds = 0.0, 0.0
                for wi, aux_logits in zip(DS_WEIGHTS, aux_list):
                    bce_i = F.binary_cross_entropy_with_logits(aux_logits, m_t, reduction="none").mean(dim=(1,2,3))
                    dice_i = 1.0 - dice_per_sample(torch.sigmoid(aux_logits), m_t)
                    bce_aux_ds = bce_aux_ds + wi * bce_i
                    dice_aux_ds = dice_aux_ds + wi * dice_i

                bce_sam2_per = F.binary_cross_entropy_with_logits(
                    sam2_logits_pseudo, m_t, reduction="none"
                ).mean(dim=(1,2,3))
                dice_sam2_per = 1.0 - dice_per_sample(torch.sigmoid(sam2_logits_pseudo), m_t)

                w = torch.where(present > 0, torch.ones_like(present), torch.full_like(present, 0.35 if epoch<=2 else 0.20))
                w = w / (w.sum() + 1e-6)

                loss_aux_per  = (bce_aux_main + dice_aux_main) + (bce_aux_ds + dice_aux_ds)
                loss_sam2_per = (bce_sam2_per + dice_sam2_per)

                loss_aux  = (w * loss_aux_per).sum()
                loss_sam2 = (w * loss_sam2_per).sum()
                loss = (0.7 if epoch<=2 else 0.85) * loss_sam2 + (0.3 if epoch<=2 else 0.15) * loss_aux

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if DEVICE == "cuda":
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(prompter.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                batch_dice = float(dice_from_probs(sam2_pred, m_t).item())
                running_loss += float(loss.item()); running_dice += batch_dice

            pbar.set_postfix({"loss": f"{float(loss.item()):.4f}",
                              "dice": f"{batch_dice:.4f}",
                              "lr": f"{sched.get_last_lr()[0]:.2e}"})
            global_step += 1
            writer.add_scalar("train/loss_total", float(loss.item()), global_step)
            writer.add_scalar("train/dice_batch", batch_dice,              global_step)
            writer.add_scalar("train/lr",         sched.get_last_lr()[0],  global_step)

        sched.step()
        ntr = len(loader_train)
        avg_loss_tr = running_loss / max(1, ntr)
        avg_dice_tr = running_dice / max(1, ntr)

        val_dice, val_iou = evaluate(prompter, predictor, loader_val, DEVICE, ds_val.seq_names, writer, epoch)

        dt = time.time() - t0
        print(f"[{epoch:03d}/{EPOCHS}] train_loss={avg_loss_tr:.4f} | train_dice={avg_dice_tr:.4f} || "
              f"val_dice={val_dice:.4f} | val_iou={val_iou:.4f} | lr={sched.get_last_lr()[0]:.6f} | {dt:.1f}s")

        writer.add_scalar("train/epoch_loss", avg_loss_tr, epoch)
        writer.add_scalar("train/epoch_dice", avg_dice_tr, epoch)
        writer.add_scalar("val/dice", val_dice, epoch)
        writer.add_scalar("val/iou",  val_iou,  epoch)
        writer.add_scalar("epoch/time_sec", dt, epoch)

        ckpt = {
            "epoch": epoch,
            "model": prompter.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "val_dice": val_dice, "val_iou": val_iou,
            "hparams": {
                "USE_LOGITS_PRIOR": USE_LOGITS_PRIOR,
                "LOGIT_GAIN": LOGIT_GAIN, "LOGIT_CLAMP": LOGIT_CLAMP,
                "CONF_TAU": CONF_TAU, "NEG_BIAS": NEG_BIAS,
                "FEED_MODE": FEED_MODE, "BOX_FROM": BOX_FROM,
                "PRIOR_THR": PRIOR_THR
            }
        }
        torch.save(ckpt, os.path.join(OUT_DIR, f"ckpt_epoch{epoch:03d}.pt"))
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(ckpt, os.path.join(OUT_DIR, f"best.pt"))
            print(f"  -> Best VAL Dice updated to {best_val_dice:.4f}")

    writer.flush(); writer.close()

if __name__ == "__main__":
    main()
