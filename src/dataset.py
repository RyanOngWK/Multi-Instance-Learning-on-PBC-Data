import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from .utils import load_rgb_image, _MASK_CACHE, _COORD_CACHE, sample_tile_coordinates, crop_tile

class FungaiBagDataset(Dataset):
    """
    Each item is one image = one bag of tiles.
    Returns:
        patches: [T, C, H, W]
        label:   scalar long
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tile_size: int = 224,
        tiles_per_image: int = 32,
        min_fg_ratio: float = 0.30,
        stride: Optional[int] = None,
        keep_background_ratio: float = 0.0,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.tile_size = tile_size
        self.tiles_per_image = tiles_per_image
        self.min_fg_ratio = min_fg_ratio
        self.stride = stride
        self.keep_background_ratio = keep_background_ratio
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _read_tiles(self, image_path: str):
        img = load_rgb_image(image_path)
        w, h = img.size #for half res
        img = img.resize((w // 3, h // 3), Image.BILINEAR) #for half res
        # ── use caches when available ─────────────────────────────────────
        if _MASK_CACHE is not None and _COORD_CACHE is not None:
            candidates = _COORD_CACHE.get(
                image_path=image_path,
                img=img,
                tile_size=self.tile_size,
                stride=self.stride,
                min_fg_ratio=self.min_fg_ratio,
                mask_cache=_MASK_CACHE,
            )
            coords = sample_tile_coordinates(
                img=img,
                num_tiles=self.tiles_per_image,
                tile_size=self.tile_size,
                min_fg_ratio=self.min_fg_ratio,
                stride=self.stride,
                keep_background_ratio=self.keep_background_ratio,
                candidates=candidates,   # ← skip recomputation
            )
        else:
            # Fallback: original behaviour (no caching).
            coords = sample_tile_coordinates(
                img=img,
                num_tiles=self.tiles_per_image,
                tile_size=self.tile_size,
                min_fg_ratio=self.min_fg_ratio,
                stride=self.stride,
                keep_background_ratio=self.keep_background_ratio,
            )

        tiles = []
        for x, y in coords:
            tile = crop_tile(img, x, y, self.tile_size)
            if self.transform is not None:
                tile = self.transform(tile)
            else:
                tile = transforms.ToTensor()(tile)
            tiles.append(tile)

        return torch.stack(tiles, dim=0), coords

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patches, coords = self._read_tiles(row["image_path"])
        label = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return {
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "patches": patches,
            "coords": coords,
            "label": label,
        } 
    #for pcb and fungais
    # def __getitem__(self, idx):
    #     row = self.df.iloc[idx]
    #     patches, coords = self._read_tiles(row["image_path"])
    #     label = torch.tensor(row["label_vec"], dtype=torch.float32)  # float for BCELoss
    #     return {
    #         "image_id":   row["image_id"],
    #         "image_path": row["image_path"],
    #         "patches":    patches,
    #         "coords":     coords,
    #         "label":      label,
    #     }for malaria

def collate_fn(batch):
    # MIL bags can have different patch counts — pad to the longest in the batch
    max_patches = max(b["patches"].shape[0] for b in batch)
    padded, masks = [], []
    for b in batch:
        n = b["patches"].shape[0]
        pad_size = max_patches - n
        if pad_size > 0:
            padding = torch.zeros(pad_size, *b["patches"].shape[1:], dtype=b["patches"].dtype)
            padded.append(torch.cat([b["patches"], padding], dim=0))
        else:
            padded.append(b["patches"])
        # mask: 1 = real patch, 0 = padding
        masks.append(torch.cat([torch.ones(n), torch.zeros(pad_size)]))

    return {
        "patches":   torch.stack(padded, dim=0),   # (B, max_N, C, H, W)
        "masks":     torch.stack(masks,  dim=0),   # (B, max_N)
        "labels":    torch.stack([b["label"] for b in batch], dim=0),
        "image_ids": [b["image_id"]   for b in batch],
        "image_paths":[b["image_path"] for b in batch],
        "coords":    [b["coords"]     for b in batch],
    }
def load_splits_from_csv(csv_path: str, image_root: str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    df = pd.read_csv(csv_path)
    # Build class map from the label column
    all_classes = sorted(df["species_id"].unique())
    class_map = {cls: idx for idx, cls in enumerate(all_classes)}
    df["label_id"] = df["species_id"].map(class_map)
    df["label_text"] = df["species_id"]
    df["image_id"] = df["file_name"].str.replace(r'\.[^.]+$', '', regex=True)
    # Optionally remap image paths to a local root
    if image_root:
        df["image_path"] = df["file_name"].apply(lambda f: str(Path(image_root) / f))
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df   = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df  = df[df["split"] == "test"].copy().reset_index(drop=True)
    return train_df, val_df, test_df, class_map

def load_splits_from_json(
    train_json: str, val_json: str, test_json: str, image_root: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    ALL_CATEGORIES = ['difficult', 'gametocyte', 'leukocyte', 'red blood cell', 'ring', 'schizont', 'trophozoite']
    class_map = {cat: i for i, cat in enumerate(ALL_CATEGORIES)}

    def parse_json(json_path, split_name):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for item in data:
            fname = Path(item["image"]["pathname"]).name
            image_path = str(Path(image_root) / split_name / fname)
            cats_present = set(obj["category"] for obj in item["objects"])
            # Multi-hot label vector
            label_vec = [1 if cat in cats_present else 0 for cat in ALL_CATEGORIES]
            rows.append({
                "image_id": Path(fname).stem,
                "image_path": image_path,
                "label_vec": label_vec,
                "categories": list(cats_present),
                "split": split_name,
            })
        return pd.DataFrame(rows)

    train_df = parse_json(train_json, "train_cleaned")
    val_df   = parse_json(val_json,   "val_cleaned")
    test_df  = parse_json(test_json,  "test_cleaned")
    return train_df, val_df, test_df, class_map


def scan_split_directory(split_dir: str, class_map: Dict[str, int]) -> pd.DataFrame:
    rows = []
    split_root = Path(split_dir)
    if not split_root.exists():
        raise FileNotFoundError(f"Directory not found: {split_dir}")
    for class_name, label_id in class_map.items():
        class_dir = split_root / class_name
        if not class_dir.exists() or not class_dir.is_dir():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXTS:
                rows.append({
                    "image_id": path.stem,
                    "image_path": str(path),
                    "label_text": class_name,
                    "label_id": int(label_id),
                    "split": split_root.name,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No usable images found in {split_dir}")
    return df.reset_index(drop=True)
