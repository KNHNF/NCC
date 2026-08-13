# NCC, AI Hackathon composites defect challenge

Independent analysis for the National Composites Centre (NCCUK) challenge at the UWE Bristol / Tech West England Advocates / Rootcause.ai AI Hackathon, 13-14 Aug 2026. Team Nexus. This repo is my own working copy, built alongside the team's official submission, not a replacement for it.

## The task

Segment carbon fibre composite micrographs into matrix, fibre and void, then call each specimen pass or fail on defect severity, matching NCC's real certification criteria.

## Results

U-Net (resnet18 encoder, ImageNet pretrained), scored on a leak-free validation split of 616 images, against NCC's authoritative scoring formula (`evaluation.py`, additive, threshold 25):

| Metric | Value |
|---|---|
| Dice_void (void-containing images only) | 0.78 |
| F2 | 0.93 |
| Final score | 0.90 |
| Agreement between NCC's two conflicting scoring formulas | 98% |
| V-scale held-out (fibre radii 6, 10 excluded from training) | Dice 0.82, close to the random-split result, model generalises across scale |

See `output/validation_scores.csv` for the full per-image breakdown under both formulas.

## Why two severity formulas are implemented

NCC supplied two scoring scripts that disagree: `evaluation.py` on GitHub (additive severity, threshold 25, straight-line void length) and `score_submission.py` in their Drive folder (multiplicative severity, threshold 60, geodesic void length). Evidence points to `evaluation.py` being authoritative (its own commit message says it was updated to match the real judge, and NCC's own slides state the same formula and threshold), but `severity.py` implements both and `test_severity.py` verifies each one directly against NCC's real scripts on a synthetic case, so the choice of which formula to trust doesn't depend on a hand-rolled reimplementation being correct.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy scipy pandas pillow scikit-image torch segmentation-models-pytorch matplotlib scikit-learn gradio
```

Requires the challenge repo and NCC's Drive files cloned as siblings to this folder (`../ncc-challenge` and `../ncc-official-drive`), since `test_severity.py` imports NCC's real scripts directly rather than reimplementing them, and `dataset.py` reads training data from `../ncc-challenge/data`.

## Structure

- `severity.py` - both severity formulas, verified against NCC's real scripts. The part that matters most, everything downstream depends on this being correct.
- `test_severity.py` - cross-checks `severity.py` on a synthetic mask. Run after touching either formula, before trusting any number.
- `dataset.py` - data loader, plus two splits: a leak-free random split (grouped by source image, since augmented copies of the same image must never land on both sides), and a V-scale split (holds out fibre radii 6 and 10 entirely, the honest proxy for the real test set's radius-7 scale).
- `train.py` - CLI training script (`--split random` or `--split scale`).
- `01_UNet_Training.ipynb` - the same training pipeline, structured for Kaggle (GPU), with a toggle for random vs V-scale.
- `evaluate.py` - scores a trained checkpoint against the real validation set: Dice, F2, final score, a labelled confusion matrix, and the two-formula agreement rate.
- `demo_pipeline.py` - generates the four-panel AI visualisation (input, segmentation, measurement, decision) for a real pass and a real fail example.
- `gradio_app.py` - upload any micrograph, get a live segmentation and pass/fail call.
- `make_submission.py` - runs the trained model on the real 32 hidden test images, writes `predicted_masks/`.
- `check_submission.py` - verifies the submission format (32 files, correct dimensions, values, no fragmentation) before anything gets submitted.
- `WHAT-I-DID-SIMPLE.md` - plain-English explanation of the scoring work, no jargon.
- `TEAM-NEXT-STEPS.md`, `akilesh-update-and-ask.md` - team coordination notes.
- `output/` - checkpoints (gitignored, too large for git), metrics, and figures.

## Status

Segmentation trained and verified, real scores computed against NCC's actual formula, V-scale generalisation checked, AI visualisation and live demo built, submission generated and format-checked. Open item: how to actually deliver the submission to NCC, direct push access to their repo was denied, this needs a direct answer from a mentor, not resolved by this repo alone.
