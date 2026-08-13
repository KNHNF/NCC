"""Score the trained model against NCC's real severity/pass-fail criteria.

Loads best_model.pth, predicts masks for the (leak-free) validation split,
scores them with severity.py's straight-line/additive formula (evaluation.py's,
treated as authoritative per the commit-message and NCC-slide evidence, see
../05-Day1-Challenge-Briefs.md), reports Dice/F2/final score, and saves a
confusion matrix figure.

    python evaluate.py
"""

import numpy as np
import segmentation_models_pytorch as smp
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from dataset import VoidSegDataset, collect_samples, train_val_split
from severity import (
    STRAIGHT_LINE_THRESHOLD,
    GEODESIC_THRESHOLD_UM,
    severity_geodesic,
    severity_straight_line,
)

VOID = 2
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dice_void(pred_mask, gt_mask):
    p, g = (pred_mask == VOID), (gt_mask == VOID)
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else 2.0 * (p & g).sum() / denom


def f2_score(tp, fp, fn):
    denom = 5 * tp + 4 * fn + fp
    return 0.0 if denom == 0 else 5 * tp / denom


def main():
    print(f"Device: {DEVICE}")

    model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet",
                      in_channels=1, classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("output/best_model.pth", map_location=DEVICE))
    model.eval()

    samples = collect_samples()
    _, val_samples = train_val_split(samples)
    print(f"Validation set: {len(val_samples)} images (leak-free split)")

    ds = VoidSegDataset(val_samples)

    dices, dices_void_only = [], []
    tp = fp = fn = tn = 0
    rows = []

    with torch.no_grad():
        for i in range(len(ds)):
            img, gt_mask, um_per_px, name = ds[i]
            logits = model(img.unsqueeze(0).to(DEVICE))
            pred_mask = logits.argmax(dim=1).cpu().squeeze(0).numpy()
            gt_mask = gt_mask.numpy()

            d = dice_void(pred_mask, gt_mask)
            dices.append(d)
            if (gt_mask == VOID).any():
                dices_void_only.append(d)

            # authoritative: evaluation.py's straight-line/additive, threshold 25
            sev_pred, _ = severity_straight_line(pred_mask, um_per_px)
            sev_gt, _ = severity_straight_line(gt_mask, um_per_px)
            pred_fail = sev_pred >= STRAIGHT_LINE_THRESHOLD
            gt_fail = sev_gt >= STRAIGHT_LINE_THRESHOLD

            # reference only: Drive script's geodesic/multiplicative, threshold 60
            sev_pred_geo, _ = severity_geodesic(pred_mask, um_per_px)
            sev_gt_geo, _ = severity_geodesic(gt_mask, um_per_px)

            if pred_fail and gt_fail:
                tp += 1
            elif pred_fail and not gt_fail:
                fp += 1
            elif not pred_fail and gt_fail:
                fn += 1
            else:
                tn += 1

            rows.append({
                "image": name, "dice_void": round(d, 4),
                "severity_pred_additive": round(sev_pred, 2),
                "severity_gt_additive": round(sev_gt, 2),
                "pred": "Fail" if pred_fail else "Pass",
                "truth": "Fail" if gt_fail else "Pass",
                "severity_pred_geodesic": round(sev_pred_geo, 2),
                "severity_gt_geodesic": round(sev_gt_geo, 2),
            })

    mean_dice = float(np.mean(dices_void_only)) if dices_void_only else float("nan")
    f2 = f2_score(tp, fp, fn)
    final_score = f2 * min(1.0, mean_dice / 0.8) if dices_void_only else 0.0

    print(f"\nImages scored: {len(val_samples)}  (void-containing, counted for Dice: {len(dices_void_only)})")
    print(f"TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    print(f"Dice_void (mean, void-containing images): {mean_dice:.4f}")
    print(f"F2 (evaluation.py, additive, threshold {STRAIGHT_LINE_THRESHOLD}): {f2:.4f}")
    print(f"Final score: {final_score:.4f}")

    # confusion matrix figure, cell contents named and explained, not just numbers
    cm = np.array([[tn, fp], [fn, tp]])
    cell_info = [
        [("True Negative (TN)", "Correctly passed, good part accepted"),
         ("False Positive (FP)", "False alarm, good part flagged as failed")],
        [("False Negative (FN)", "Missed defect, bad part passed as good"),
         ("True Positive (TP)", "Defect correctly caught")],
    ]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted: Pass", "Predicted: Fail"], fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: Pass", "Actual: Fail"], fontsize=11)
    ax.set_title(f"Validation confusion matrix, {len(val_samples)} images\n"
                 f"(evaluation.py formula, threshold {STRAIGHT_LINE_THRESHOLD})", fontsize=12)

    for r in range(2):
        for c in range(2):
            name, meaning = cell_info[r][c]
            colour = "white" if cm[r, c] > cm.max() / 2 else "black"
            ax.text(c, r, f"{name}\n{cm[r, c]}\n{meaning}", ha="center", va="center",
                     color=colour, fontsize=9.5, linespacing=1.6)

    plt.colorbar(im, label="count")
    plt.tight_layout()
    plt.savefig("output/figures/confusion_matrix.png", bbox_inches="tight", dpi=150)
    print("\nSaved output/figures/confusion_matrix.png")

    import pandas as pd
    pd.DataFrame(rows).to_csv("output/validation_scores.csv", index=False)
    print("Saved output/validation_scores.csv (both formulas, per-image)")

    n_agree = sum(1 for r in rows
                  if (r["severity_pred_additive"] >= STRAIGHT_LINE_THRESHOLD)
                  == (r["severity_pred_geodesic"] >= GEODESIC_THRESHOLD_UM))
    print(f"\nThe two scoring formulas agree on {n_agree}/{len(rows)} predictions "
          f"({100 * n_agree / len(rows):.0f}%), for context on how much the threshold "
          f"discrepancy actually matters on real data.")


if __name__ == "__main__":
    main()
