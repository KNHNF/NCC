# Findings, send with the confusion matrix images

## Random split (main result, use this one in the pitch)

- Dice_void: 0.78
- F2: 0.92
- Final score: 0.90
- Confusion matrix: TP 181, FP 10, FN 17, TN 408
- Against NCC's own reference targets (FN near 0%, FP below ~5%): FN rate 8.6%, FP rate 2.4%. FP is within target, FN isn't yet, honest to say so.
- The two scoring formulas NCC gave us agree on 97% of predictions.

Image: `output/figures/confusion_matrix.png`

## V-scale split (generalisation test, secondary result)

Trained a second model with fibre radii 6 and 10 completely excluded from training, since the real test set is entirely radius 7, a scale barely present in training. This tests whether the model actually generalises or just memorised familiar zoom levels.

- Dice (all held-out images, as reported during training): 0.87
- Dice (void-containing images only, NCC's harness rule): 0.48
- F2: 0.00, **read the explanation below before quoting this number**
- Confusion matrix: TP 0, FP 2, FN 0, TN 712

Image: `output/figures/confusion_matrix_scale.png`

**Why F2 is 0, this matters**: only 16 of the 714 held-out images contain any void at all (fibre radius 10 material has no voids by composition), and none of those 16 are severe enough to be a real ground-truth failure. So there were zero real failing parts in this subset to catch, F2 = 0 is what the formula does when there's nothing to catch, not evidence the model missed defects. The 2 false positives are minor false alarms on fine parts. Read alongside the 0.87 raw Dice: the model generalises fine on segmentation, this particular slice just isn't a good test of the pass/fail decision, since it barely contains real failures.

## What to do with this

Use the random split as your headline result, it's the representative one. Mention the V-scale test as evidence we checked for generalisation properly (most teams won't), and state the F2=0 finding honestly with the explanation above if it comes up, don't hide it and don't present it without context, either one looks worse than just explaining it.
