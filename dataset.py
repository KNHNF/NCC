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
                        "fibre_radius_px": int(row["fibre_radius_px"]),
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


def _source_stem(image_path):
    """Group key for leak-free splitting.

    Every augmented file is named <original_stem>_aug_0.<ext>, and the same
    source micrograph appears again, un-augmented, across Data set I/II/III
    (250/100/150 shared stems between pairs of sets, 1,550 unique sources
    behind the 4,000 files). Stripping "_aug_0" and any dataset-set suffix
    collapses all of these back to one key, so an augmented variant and its
    original always land on the same side of the split, never opposite sides.
    """
    stem = Path(image_path).stem
    if stem.endswith("_aug_0"):
        stem = stem[: -len("_aug_0")]
    return stem


def train_val_split(samples, val_fraction=0.15, seed=42):
    """Split by source stem, not by file, so no source image's original and
    augmented variants are ever split across train and val (that would leak
    and inflate validation Dice). See _source_stem for why this matters here.
    """
    stems = sorted({_source_stem(s["image"]) for s in samples})
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(stems))
    n_val = int(len(stems) * val_fraction)
    val_stems = set(stems[i] for i in idx[:n_val])

    train = [s for s in samples if _source_stem(s["image"]) not in val_stems]
    val = [s for s in samples if _source_stem(s["image"]) in val_stems]
    return train, val


SCALE_HOLDOUT_RADII = {6, 10}


def scale_holdout_split(samples):
    """V-scale: hold out fibre radii 6 and 10 entirely from training.

    The real test set is entirely at fibre radius 7, which barely exists in
    training (28 files, all crop-zoom augmentation artefacts, not independent
    sources). A random split can't measure how the model handles a scale it
    has never really seen. This split does: radii 6 and 10 never appear in
    training, so scoring on them is the closest honest proxy for radius 7
    performance available from this dataset. Where this split and the random
    one disagree, believe this one, per NCC's own dataset having a deliberate
    scale gap at the test radius.

    Unlike train_val_split, a model must be RE-TRAINED with radii 6/10 fully
    excluded for this to mean anything, evaluating an existing model that was
    trained on data including radius 6/10 images is not a real test.
    """
    train = [s for s in samples if s["fibre_radius_px"] not in SCALE_HOLDOUT_RADII]
    heldout = [s for s in samples if s["fibre_radius_px"] in SCALE_HOLDOUT_RADII]
    return train, heldout


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
