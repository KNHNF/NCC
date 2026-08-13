# Scores, for comparing approaches

All numbers below are on the same leak-free validation split (616 images, grouped by source image so no augmented copy leaks across train/val), scored against NCC's authoritative formula (`evaluation.py`, additive, threshold 25 microns). Reproduce with `python evaluate.py`.

## Karan's model: U-Net (resnet18, ImageNet pretrained)

| Metric | Value | Meaning |
|---|---|---|
| Dice_void | 0.78 | Pixel overlap on void class, void-containing images only. Gate is 0.8, above it earns nothing extra. |
| TP / FP / FN / TN | 182 / 9 / 16 / 409 | Out of 616 images. |
| Precision (of predicted fails) | 95% | 182 of 191 predicted fails were real. |
| Recall (of real fails) | 92% | 182 of 198 real fails were caught. |
| F2 | 0.93 | Recall-weighted, missed defects cost 4x a false alarm. This is the real differentiator. |
| Final score | 0.90 | F2 x min(1, Dice/0.8), the number NCC's harness actually reports. |
| V-scale generalisation | 0.82 Dice | Trained with fibre radii 6 and 10 fully excluded, tested on those, the honest proxy for the real test set's unseen radius-7 scale. Close to the main result, model isn't just memorising scale. |
| Two-formula agreement | 98% | How often NCC's two conflicting scoring scripts (additive/25 vs multiplicative/60) actually agree on our real predictions. |

## Akilesh's approach: [fill in once his numbers land]

| Metric | Value |
|---|---|
| Dice_void | |
| TP / FP / FN / TN | |
| F2 | |
| Final score | |
| Method | (classical thresholding / other, describe) |

## How to fill in a new row

Run `python evaluate.py --model <checkpoint>.pth` (or the classical script's own scoring, matched to `severity.py`'s formula) and copy the printed TP/FP/FN/TN, Dice, F2, and final score straight into the table above. Keep every number on the same validation split so the comparison is fair, not just similar-looking numbers from different data.
