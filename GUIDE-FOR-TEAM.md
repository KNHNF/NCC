# Guide to this repo, for the team

Repo: `github.com/KNHNF/NCC`. This is Karan's working repo for the NCC challenge, scoring, model training, and the demo. Not the official submission channel, that's still an open question with the mentors.

## WhatsApp share (use this, not the GitHub clone alone)

Zip and send: `../Pitch-Materials/Nexus-team-pack/`

That folder is the self-contained Karan input for the group: `KARAN-WORK.pdf`, demo notes, scores template, pitch outline/script, and open issues. Feed the PDF or markdown to ChatGPT/Claude, add Akilesh + Venkata inputs, then generate the 10-minute PPT. Demo video comes separately; do not wait for it to start the deck.

This GitHub clone can lag the laptop (uncommitted demo docs / latest `gradio_app.py`). `demo-images/` is gitignored. Local demo URL is Karan's laptop only: http://127.0.0.1:7860 (or 7861).

## If you just want the headline numbers

Dice 0.78, F2 0.93, final score 0.90, on real validation data, and the model generalises well to an unseen fibre scale (0.82 Dice on a held-out scale test). Full detail in `README.md`. Pitch materials also quote Dice 0.784 / F2 0.936 / final 0.917; do not mix the two sets on one slide. See `Nexus-team-pack/KARAN-WORK.md`.

## If you want to understand what was built, in plain English

Read `WHAT-I-DID-SIMPLE.md` first, no code, no jargon, just what each piece does and why. The "still to do" section there is stale; use `Nexus-team-pack/OPEN-ISSUES.md` instead.

## If you're building the pitch deck

Use `../Pitch-Materials/Nexus-team-pack/` (WhatsApp zip target), not this repo alone: `KARAN-WORK.pdf`, `presentation-script-10min.md`, `findings-summary.md`, `slide-outline.md`. Laptop figures still live in `karan-ncc-analysis/output/figures/` (pipeline demo images, confusion matrix, training curve); they are not inside the zip.

## If you're touching the code

- Don't rewrite the severity scoring, `severity.py` is already verified against NCC's real scripts. If it needs changing, rerun `test_severity.py` afterward, don't trust a change until it passes.
- Data loading and splits live in `dataset.py`. Two splits exist: the normal one (leak-free, grouped by source image) and the V-scale one (tests generalisation to an unseen fibre size). Use the normal one for iteration, the V-scale one before trusting a final number.
- `evaluate.py` is how any trained model gets scored properly, Dice, F2, final score, and a real confusion matrix, against NCC's actual formula, not just raw accuracy.
- `make_submission.py` then `check_submission.py` is the sequence for generating and verifying the real test-set predictions. Always run the checker before anything gets submitted anywhere.

## If you want to run the live demo

`python gradio_app.py` from inside `karan-ncc-analysis`, then open http://127.0.0.1:7860. Click a micrograph, then Analyse.

Read [DEMO-FOR-TEAM.md](DEMO-FOR-TEAM.md) before anyone records: what FAIL/PASS mean, CPU vs GPU (this laptop is CPU-only), and the two hold-out images. Recording click order and spoken lines: [DEMO-RECORDING-PLAN.md](DEMO-RECORDING-PLAN.md).

## What's still open

Full list: `../Pitch-Materials/Nexus-team-pack/OPEN-ISSUES.md`.

- How to actually deliver the submission to NCC, direct push to their repo is denied (403), needs a mentor's answer.
- Akilesh's own approach and numbers, not yet folded in for the two-approach comparison in the pitch.
- Final choice between the random-split model and the V-scale model for the actual submission, once both are fully scored. `best_model_scale.pth` was not on this laptop when last checked.
- 10-minute PPT not built yet; start from the Nexus-team-pack, do not wait for the demo video.
