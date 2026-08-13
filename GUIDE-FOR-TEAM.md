# Guide to this repo, for the team

Repo: `github.com/KNHNF/NCC`. This is Karan's working repo for the NCC challenge, scoring, model training, and the demo. Not the official submission channel, that's still an open question with the mentors.

## If you just want the headline numbers

Dice 0.78, F2 0.93, final score 0.90, on real validation data, and the model generalises well to an unseen fibre scale (0.82 Dice on a held-out scale test). Full detail in `README.md`.

## If you want to understand what was built, in plain English

Read `WHAT-I-DID-SIMPLE.md` first, no code, no jargon, just what each piece does and why.

## If you're building the pitch deck

You don't need this repo directly, the content's already pulled out into `Pitch-Materials/` in the main hackathon project folder: `presentation-script-10min.md` (the full script), `findings-summary.md` (numbers with "so what" for each), and the figures in `karan-ncc-analysis/output/figures/` (the pipeline demo images, confusion matrix, training curve).

## If you're touching the code

- Don't rewrite the severity scoring, `severity.py` is already verified against NCC's real scripts. If it needs changing, rerun `test_severity.py` afterward, don't trust a change until it passes.
- Data loading and splits live in `dataset.py`. Two splits exist: the normal one (leak-free, grouped by source image) and the V-scale one (tests generalisation to an unseen fibre size). Use the normal one for iteration, the V-scale one before trusting a final number.
- `evaluate.py` is how any trained model gets scored properly, Dice, F2, final score, and a real confusion matrix, against NCC's actual formula, not just raw accuracy.
- `make_submission.py` then `check_submission.py` is the sequence for generating and verifying the real test-set predictions. Always run the checker before anything gets submitted anywhere.

## If you want to run the live demo

`python gradio_app.py` from inside `karan-ncc-analysis`, upload a micrograph, get a segmentation and pass/fail call back live. No setup beyond the pip installs in the README.

## What's still open

- How to actually deliver the submission to NCC, direct push to their repo is denied (403), needs a mentor's answer.
- Akilesh's own approach and numbers, not yet folded in for the two-approach comparison in the pitch.
- Final choice between the random-split model and the V-scale model for the actual submission, once both are fully scored.
