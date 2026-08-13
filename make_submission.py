"""Generate the real submission: predictions for all 32 hidden test images.

Everything else so far (training, evaluate.py, the demo) has run against our
own validation split. This is the script that actually matters for scoring,
it predicts the real test set and writes predicted_masks/ in the exact
format NCC's harness expects.

    python make_submission.py                    # uses output/best_model.pth
    python make_submission.py --model best_model_scale.pth   # use the scale model instead

After this, run check_submission.py to verify before pushing.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image

TEST_DIR = Path(__file__).parent.parent / "ncc-challenge" / "data" / "Data sets" / "Test data set"
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best_model.pth")
    ap.add_argument("--out", default="predicted_masks")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet",
                      in_channels=1, classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(f"output/{args.model}", map_location=DEVICE))
    model.eval()

    meta = pd.read_csv(TEST_DIR / "metadata.csv")
    print(f"Test set: {len(meta)} images (expect 32)")

    written = []
    with torch.no_grad():
        for _, row in meta.iterrows():
            img_name = row["image_id"]
            img_path = TEST_DIR / "Images" / img_name
            if not img_path.exists():
                print(f"  MISSING SOURCE IMAGE: {img_name}, cannot predict this one")
                continue

            img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
            img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
            pred = model(img_t).argmax(dim=1).cpu().squeeze(0).numpy().astype(np.uint8)

            stem = Path(img_name).stem
            out_path = out_dir / f"{stem}.png"
            Image.fromarray(pred, mode="L").save(out_path)
            written.append(out_path.name)

    print(f"\nWrote {len(written)} prediction files to {out_dir}/")
    if len(written) != 32:
        print(f"WARNING: expected 32 files, wrote {len(written)}. "
              f"A missing file scores as an automatic false pass under NCC's harness. "
              f"Do not submit until this says 32.")
    else:
        print("32/32, matches the real test set. Run check_submission.py next.")


if __name__ == "__main__":
    main()
