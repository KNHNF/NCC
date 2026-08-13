# Update for Akilesh, and what I need from you

## What's done on my side

Fixed the train/val split bug (was leaking augmented copies of the same source image across train and validation, inflating Dice). Retrained on the fixed split and scored the result against NCC's real severity formula, not just Dice.

**Real results** (U-Net, resnet18 encoder, ImageNet pretrained, leak-free validation split of 616 images):

- Dice_void (void-containing images only): **0.784**
- Confusion matrix: TP 185, FP 11, FN 13, TN 407
- F2: **0.936**
- Final score (F2 x min(1, Dice/0.8)): **0.917**
- The two scoring formulas NCC gave us (GitHub's additive/25, Drive's geodesic/60) agree on 97% of predictions on real data, quantified this properly, not just flagged as a discrepancy.

Code: `dataset.py` (leak-free split by source stem, not by file), `severity.py` (both formulas, verified against NCC's real scripts), `evaluate.py` (runs the trained model against the validation set, produces the numbers above plus a labelled confusion matrix). All committed.

## What I need from you

You're doing this as a Python script, not a training run, a different approach from mine. Good, that's a real ablation, not a problem, the published literature runs exactly this comparison (deep learning versus classical image processing) and it strengthens our Technical Feasibility section either way the numbers land.

Please send back:

1. **What method are you using**, thresholding, edge detection, something else, and why you picked it.
2. **What results you're getting**, ideally scored the same way I did: Dice_void, and a confusion matrix (TP/FP/FN/TN) against NCC's severity formula (additive, threshold 25), on the same validation split if possible, so the comparison is fair. If you haven't scored against the real severity formula yet, `severity.py` in this repo has both formulas already verified, use it rather than writing your own.
3. **Where it does well and where it struggles**, even a rough sense, that's the actual interesting finding for the pitch, not just a final number.

We're presenting both approaches honestly in the pitch, whichever wins on real data is what we lead with, and the comparison itself is a talking point either way.
