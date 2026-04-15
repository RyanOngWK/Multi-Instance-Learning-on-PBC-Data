import os
import random
import torch
import numpy as np
import pickle
import threading
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple
import dataclasses
# [Include: seed_everything, ensure_dir, otsu_threshold, make_foreground_mask]
# [Include: ForegroundMaskCache class]
# [Include: TileCoordCache class]
# [Include: sample_tile_coordinates, crop_tile, save_attention_heatmap, plot_training_curves]
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    best_var = -1.0
    threshold = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var = var_between
            threshold = t
    return int(threshold)


def make_foreground_mask(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    gray = arr.mean(axis=2).astype(np.uint8)
    thresh = otsu_threshold(gray)
    mask = gray < min(thresh, 220)
    mask = mask & (arr.std(axis=2) > 5)
    return mask.astype(bool)


class ForegroundMaskCache:
    """
    Thread-safe in-process cache for foreground masks.

    Optional on-disk persistence: if `cache_dir` is given, masks are stored
    as compressed .pkl files named by a hash of the image path.  Subsequent
    runs (or worker processes that share the same filesystem) load from disk
    instead of recomputing.

    Layout of each cached entry
    ───────────────────────────
    {
        "mask":   np.ndarray[bool],   # H × W foreground mask
    }
    """

    def __init__(self, cache_dir: Optional[str] = None):
        # In-process dict, keyed by absolute image path string.
        self._cache: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ── disk helpers ──────────────────────────────────────────────────────

    def _disk_path(self, image_path: str, shape: tuple = None) -> str:
        import hashlib
        # Include shape in key so full-res and half-res don't collide
        key_str = image_path + (str(shape) if shape else "")
        key = hashlib.md5(key_str.encode()).hexdigest()
        return os.path.join(self._cache_dir, f"{key}.pkl")

    def _save_to_disk(self, image_path: str, mask: np.ndarray) -> None:
        if not self._cache_dir:
            return
        p = self._disk_path(image_path)
        try:
            payload = {
                "packed": np.packbits(mask),   # 8× smaller
                "shape": mask.shape,
            }
            with open(p, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[WARN] Could not write mask cache for {image_path}: {e}", file=sys.stderr)
    
    def _load_from_disk(self, image_path: str) -> Optional[np.ndarray]:
        if not self._cache_dir:
            return None
        p = self._disk_path(image_path)
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    payload = pickle.load(f)
                return np.unpackbits(payload["packed"]).reshape(payload["shape"]).astype(bool)
            except Exception:
                return None
        return None
    # ── public API ────────────────────────────────────────────────────────

    def get(self, image_path: str, img: Optional[Image.Image] = None) -> np.ndarray:
        """
        Return the foreground mask for *image_path*.

        Lookup order:
          1. In-process dict  (fastest)
          2. On-disk pickle   (fast, survives restarts)
          3. Recompute        (slow, happens at most once per image)

        *img* is only opened when a cache miss forces recomputation.
        If *img* is None and a miss occurs, the image is opened here.
        """
        with self._lock:
            if image_path in self._cache:
                return self._cache[image_path]

        # Check disk outside the lock to avoid holding it during I/O.
        mask = self._load_from_disk(image_path)
        if mask is not None:
            with self._lock:
                self._cache[image_path] = mask
            return mask

        # Full recomputation.
        if img is None:
            img = load_rgb_image(image_path)
            w, h = img.size
            img = img.resize((w // 3, h // 3), Image.BILINEAR)  # ← always half-res
        mask = make_foreground_mask(img)

        with self._lock:
            self._cache[image_path] = mask
        self._save_to_disk(image_path, mask)
        return mask

    def warm_up(
        self,
        image_paths: List[str],
        max_workers: int = 4,
        desc: str = "Warming mask cache",
    ) -> None:
        """
        Pre-populate the cache for all *image_paths* using a thread pool.
        I/O-bound work (PIL open + Otsu) parallelises well with threads.
        """
        missing = [p for p in image_paths if p not in self._cache]
        if not missing:
            return

        def _compute_one(path: str):
            self.get(path)  # side-effect: populates cache

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_compute_one, p): p for p in missing}
            for _ in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=False):
                pass  # errors surface on .result() — swallow silently here


class TileCoordCache:
    """
    Thread-safe cache for candidate tile coordinate lists.

    Key = (image_path, tile_size, stride, min_fg_ratio)
    Value = List[Tuple[int, int]]

    Because the list only depends on the mask (not on random sampling),
    it is stable across epochs.  Random sampling happens *after* lookup.
    """

    def __init__(self):
        self._cache: Dict[Tuple, List[Tuple[int, int]]] = {}
        self._lock = threading.Lock()

    def get(
        self,
        image_path: str,
        img: Image.Image,
        tile_size: int,
        stride: Optional[int],
        min_fg_ratio: float,
        mask_cache: ForegroundMaskCache,
    ) -> List[Tuple[int, int]]:
        key = (image_path, tile_size, stride, min_fg_ratio)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        # Compute outside the lock.
        fg_mask = mask_cache.get(image_path, img)
        coords = build_candidate_tiles(
            img=img,
            tile_size=tile_size,
            min_fg_ratio=min_fg_ratio,
            stride=stride,
            fg_mask=fg_mask,
        )
        with self._lock:
            self._cache[key] = coords
        return coords

@dataclass
class EpochMetrics:
    loss: float
    acc: float
    f1_macro: float
    f1_weighted: float = 0.0
    precision_macro: float = 0.0
    precision_weighted: float = 0.0
    recall_macro: float = 0.0
    recall_weighted: float = 0.0

def compute_multiclass_metrics(y_true, y_pred) -> EpochMetrics:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    return EpochMetrics(
        loss=0.0,
        acc=float(acc),
        f1_macro=float(f1_macro),
        f1_weighted=float(f1_weighted),
        precision_macro=float(precision_macro),
        precision_weighted=float(precision_weighted),
        recall_macro=float(recall_macro),
        recall_weighted=float(recall_weighted),
    )

_MASK_CACHE: Optional[ForegroundMaskCache] = None
_COORD_CACHE: Optional[TileCoordCache] = None

def save_attention_heatmap(
    image_path, coords, attn, tile_size, out_path,
    max_canvas_size=1024, alpha=0.45,
):
    import matplotlib.cm as cm

    img = load_rgb_image(image_path)
    w, h = img.size# for half reso
    img = img.resize((w // 3, h // 3), Image.BILINEAR) # Match the Dataset resize # for half reso
    width, height = img.size
    base_np = np.array(img).astype(np.float32) / 255.0

    # Build heatmap and a coverage mask
    heatmap  = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=bool)   # ← track sampled pixels

    attn = np.asarray(attn, dtype=np.float32).reshape(-1)
    attn = attn - attn.min()
    if attn.max() > 0:
        attn = attn / attn.max()

    for (x, y), a in zip(coords, attn):
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + tile_size), min(height, y + tile_size)
        if x1 > x0 and y1 > y0:
            heatmap[y0:y1, x0:x1]  = np.maximum(heatmap[y0:y1, x0:x1], a)
            coverage[y0:y1, x0:x1] = True              # ← mark as sampled

    # Convert to RGBA via jet
    jet        = cm.get_cmap("jet")
    heatmap_rgba = jet(heatmap).astype(np.float32)     # (H, W, 4)

    # Key fix: zero alpha everywhere NOT covered by a tile
    heatmap_rgba[~coverage, 3] = 0.0
    heatmap_rgba[ coverage, 3] = alpha

    dpi = 100
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.imshow(base_np)
    ax.imshow(heatmap_rgba)   # per-pixel alpha, no cmap kwarg needed
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()

    with Image.open(out_path) as saved:
        w, h = saved.size
        saved.resize((w // 4, h // 4), Image.BICUBIC).save(out_path)