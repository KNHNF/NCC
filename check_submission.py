"""Verify predicted_masks/ before pushing, matching NCC's stated requirements exactly.

    python check_submission.py

Checks (from the master reference doc, section 7):
  - exactly 32 files, one per test image, matching metadata stems
  - single channel, 256x256
  - pixel values only in {0, 1, 2}
  - no file with more than 1500 void regions (evaluation.py's auto-fail limit)

Exits non-zero if anything fails, so this can gate a push.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label

TEST_DIR = Path(__file__).parent.parent / "ncc-challenge" / "data" / "Data sets" / "Test data set"
VOID = 2
MAX_REGIONS = 1500


def main():
    pred_dir = Path("predicted_masks")
    meta = pd.read_csv(TEST_DIR / "metadata.csv")
    expected_stems = {Path(row["image_id"]).stem for _, row in meta.iterrows()}

    problems = []

    if not pred_dir.exists():
        sys.exit(f"FAIL: {pred_dir}/ does not exist. Run make_submission.py first.")

    found_files = {p.stem: p for p in pred_dir.glob("*.png")}

    missing = expected_stems - set(found_files)
    extra = set(found_files) - expected_stems
    if missing:
        problems.append(f"{len(missing)} MISSING prediction file(s): {sorted(missing)}")
    if extra:
        problems.append(f"{len(extra)} unexpected file(s), not in the real test metadata: {sorted(extra)}")

    if len(found_files) != 32:
        problems.append(f"Expected exactly 32 files, found {len(found_files)}.")

    for stem, path in sorted(found_files.items()):
        arr = np.array(Image.open(path))
        if arr.ndim != 2:
            problems.append(f"{path.name}: not single-channel (shape {arr.shape})")
            continue
        if arr.shape != (256, 256):
            problems.append(f"{path.name}: wrong dimensions {arr.shape}, expected (256, 256)")
        bad_values = set(np.unique(arr)) - {0, 1, 2}
        if bad_values:
            problems.append(f"{path.name}: contains values outside {{0,1,2}}: {bad_values}")
        n_regions = label(arr == VOID, connectivity=2).max()
        if n_regions > MAX_REGIONS:
            problems.append(f"{path.name}: {n_regions} void regions, exceeds the {MAX_REGIONS} "
                             f"auto-fail limit, likely unfiltered speckle noise, clean it up")

    print(f"Checked {len(found_files)} files against {len(expected_stems)} expected test images.")

    if problems:
        print("\nFAILED, do not push:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    print("\nAll checks passed. 32 files, correct format, correct values, no fragmentation. Safe to push.")


if __name__ == "__main__":
    main()
