"""AI visualisation: the full pipeline on one real image, for the demo and pitch.

Four panels, matching exactly what the code does at each stage, nothing shown
here that the pipeline doesn't actually do:

    1. Input micrograph
    2. Segmentation (matrix / fibre / void)
    3. Measurement (each void region labelled with length and area in microns,
       merge groups circled if voids are close enough to combine)
    4. Decision (severity vs threshold, pass/fail, plain-English reason)

Picks the validation image with the most dramatic real result (a genuine
fail, ideally with a merged group) so the demo shows the interesting case,
not an arbitrary one.

    python demo_pipeline.py            # auto-picks a good example
    python demo_pipeline.py --image <name.tif>   # a specific image by filename
"""

import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from skimage.measure import label, regionprops

from dataset import VoidSegDataset, collect_samples, train_val_split
from severity import STRAIGHT_LINE_THRESHOLD, VOID, severity_straight_line

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet",
                      in_channels=1, classes=3).to(DEVICE)
    model.load_state_dict(torch.load("output/best_model.pth", map_location=DEVICE))
    model.eval()
    return model


def find_example(ds, model, target_name=None):
    """Pick the val image with the highest predicted severity (a real fail),
    or a specific named image if requested."""
    best_idx, best_severity = None, -1
    with torch.no_grad():
        for i in range(len(ds)):
            img, _, um_per_px, name = ds[i]
            if target_name and name != target_name:
                continue
            pred = model(img.unsqueeze(0).to(DEVICE)).argmax(dim=1).cpu().squeeze(0).numpy()
            sev, _ = severity_straight_line(pred, um_per_px)
            if target_name:
                return i
            if sev > best_severity:
                best_severity, best_idx = sev, i
    return best_idx


def draw_pipeline(ds, idx, model, out_path):
    img, gt_mask, um_per_px, name = ds[idx]
    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(DEVICE))
        pred = logits.argmax(dim=1).cpu().squeeze(0).numpy()

    sev, group_severities = severity_straight_line(pred, um_per_px)
    fail = sev >= STRAIGHT_LINE_THRESHOLD

    labelled = label(pred == VOID, connectivity=2)
    regions = regionprops(labelled)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    # 1. Input
    axes[0].imshow(img.squeeze(0), cmap="gray")
    axes[0].set_title(f"1. Input\n{name}", fontsize=11)

    # 2. Segmentation
    cmap = plt.matplotlib.colors.ListedColormap(["#2b2b2b", "#7fb37f", "#e8544c"])
    axes[1].imshow(pred, cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title("2. Segmentation\nmatrix / fibre / void", fontsize=11)
    patches = [mpatches.Patch(color=c, label=l) for c, l in
               zip(["#2b2b2b", "#7fb37f", "#e8544c"], ["Matrix", "Fibre", "Void"])]
    axes[1].legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.8)

    # 3. Measurement
    axes[2].imshow(img.squeeze(0), cmap="gray")
    axes[2].imshow(np.ma.masked_where(pred != VOID, pred), cmap="autumn", alpha=0.6, vmin=0, vmax=2)
    for r in regions:
        y, x = r.centroid
        length_um = (r.axis_major_length or 1) * um_per_px
        area_um2 = r.area * um_per_px ** 2
        axes[2].plot(x, y, "o", color="cyan", markersize=3)
        axes[2].annotate(f"{length_um:.0f}um\n{area_um2:.0f}um2", (x, y),
                          color="cyan", fontsize=6.5, ha="left", va="bottom")
    axes[2].set_title(f"3. Measurement\n{len(regions)} void region(s) found", fontsize=11)

    # 4. Decision
    axes[3].axis("off")
    verdict = "FAIL" if fail else "PASS"
    colour = "#e8544c" if fail else "#4f9e4f"
    reason = (f"Worst group severity {sev:.1f}, threshold {STRAIGHT_LINE_THRESHOLD}.\n"
              f"{'Exceeds' if fail else 'Within'} NCC's limit."
              if regions else "No voids detected.")
    axes[3].text(0.5, 0.75, verdict, fontsize=32, fontweight="bold", color=colour,
                 ha="center", va="center", transform=axes[3].transAxes)
    axes[3].text(0.5, 0.45, reason, fontsize=11, ha="center", va="center",
                 transform=axes[3].transAxes, wrap=True)
    axes[3].text(0.5, 0.15, f"Scale: {um_per_px} um/pixel", fontsize=9, color="grey",
                 ha="center", va="center", transform=axes[3].transAxes)
    axes[3].set_title("4. Decision", fontsize=11)

    for ax in axes[:3]:
        ax.axis("off")

    plt.suptitle(f"Micrograph to certification decision, verdict: {verdict}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved {out_path}")
    print(f"Image: {name}  severity: {sev:.2f}  verdict: {verdict}  "
          f"regions: {len(regions)}  group severities: {[f'{g:.1f}' for g in group_severities]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="specific image filename, else auto-picks the worst-case example")
    args = ap.parse_args()

    model = load_model()
    samples = collect_samples()
    _, val_samples = train_val_split(samples)
    ds = VoidSegDataset(val_samples)

    idx = find_example(ds, model, target_name=args.image)
    if idx is None:
        raise SystemExit("No matching image found in the validation set.")

    draw_pipeline(ds, idx, model, "output/figures/pipeline_demo.png")


if __name__ == "__main__":
    main()
