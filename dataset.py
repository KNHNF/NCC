"""Dataset loader for the NCC composites defect challenge.

Walks Data set I/II/III (each has Original + Augmented subfolders, each with
Images/, Masks/, metadata.csv), collects every (image, mask, um_per_pixel)
triple, and holds out a validation split for local scoring against severity.py
before the hidden test set is touched.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

DATA_ROOT = Path(__file__).parent.parent / "ncc-challenge" / "data" / "Data sets"
TRAIN_SETS = ["Data set I", "Data set II", "Data set III"]
SUBSETS = ["Original data set", "Augmented data set"]


def collect_samples():
    """Return a list of dicts: image_path, mask_path, um_per_pixel."""
    samples = []
    for dset in TRAIN_SETS:
        for subset in SUBSETS:
            base = DATA_ROOT / dset / subset
            meta_path = base / "metadata.csv"
            if not meta_path.exists():
                continue
            meta = pd.read_csv(meta_path)
            for _, row in meta.iterrows():
                img_name = row["image_id"]
                mask_name = Path(img_name).stem + ".png"
                img_path = base / "Images" / img_name
                mask_path = base / "Masks" / mask_name
                if img_path.exists() and mask_path.exists():
                    samples.append({
                        "image": img_path,
                        "mask": mask_path,
                        "um_per_px": float(row["um_per_pixel"]),
                    })
    return samples


class VoidSegDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = np.array(Image.open(s["image"]).convert("L"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(s["mask"]))
        if mask.ndim == 3:
            mask = mask[..., 0]

        img_t = torch.from_numpy(img).unsqueeze(0)          # 1xHxW
        mask_t = torch.from_numpy(mask.astype(np.int64))    # HxW, values 0/1/2

        return img_t, mask_t, s["um_per_px"], str(s["image"].name)


def train_val_split(samples, val_fraction=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_val = int(len(samples) * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]


if __name__ == "__main__":
    samples = collect_samples()
    print(f"Collected {len(samples)} labelled image/mask pairs")
    if samples:
        train, val = train_val_split(samples)
        print(f"Train: {len(train)}  Val: {len(val)}")
        ds = VoidSegDataset(train[:1])
        img, mask, um, name = ds[0]
        print(f"Sample: {name}  image shape={tuple(img.shape)}  mask shape={tuple(mask.shape)}  "
              f"mask classes present={sorted(mask.unique().tolist())}  um_per_px={um}")
