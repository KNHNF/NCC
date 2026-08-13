# Next steps for the team

## What just happened

First real training run: Dice_void 0.876, clears the 0.8 gate. Don't fully trust this number yet, the split that produced it was random by file, not by source image, so an augmented copy of a micrograph could sit in validation while its original sat in training (1,550 unique source images behind the 4,000 files, confirmed by counting stems). That leaks and inflates the score. Fixed in `dataset.py` and the notebook, split is now by source stem, so an image and all its augmented variants always stay on the same side. Retrain and get a new Dice number before trusting anything downstream of it.

## For Akilesh, on the model side

1. Rerun `01_UNet_Training.ipynb` on Kaggle with the fixed split, get a real Dice_void number.
2. If you want a genuine second data point rather than a repeat of the same run, use a different seed (change `np.random.default_rng(42)` to `np.random.default_rng(7)`) or swap `encoder_name='resnet18'` to `'resnet34'`. Either is fine, just pick one so the comparison means something.
3. Once training's done, download `best_model.pth` and `unet_metrics.pkl` from the Output tab and get them back into `karan-ncc-analysis/output/`.
4. Resolve the outstanding scoring question before anyone spends time calibrating a threshold: is `evaluation.py` (additive, threshold 25) the only real scorer, or does the Drive's `score_submission.py` also run. There's real evidence for the first (the GitHub commit message says it was updated to match the real judge, and NCC's own slide 2.2 states the additive formula and 25 threshold directly). Ask the mentor directly, don't guess.

## For everyone, on process

- Run `test_severity.py` again any time `severity.py` changes, it cross-checks both formulas against NCC's actual scripts on a synthetic case. Don't trust a severity number that hasn't passed this.
- Before generating the real test-set submission, run the pre-submission checklist from the master reference doc: all 32 files present (checked programmatically, not by eye), single-channel PNG, values only 0/1/2, 256x256, filenames match the metadata stems exactly, branch is `submission/nexus` not `main`.
- If Claude Code (or Cursor) writes code for the severity or scoring logic, verify it against `test_severity.py` before trusting it, don't take generated code on faith for anything that feeds the final score.

## Useful Claude Code tools worth using directly

If you're running Claude Code yourself (not just Cursor), a few things worth knowing about:

- **`/code-review`** reviews a diff or branch for correctness bugs before you trust it, useful right before generating the real test-set submission, catches mistakes a quick glance would miss.
- **Council of five** (ask Claude to "use council of five" or similar) runs five independent review passes on a genuinely uncertain decision, then synthesises where they agree and disagree. Used this already to rank the pitch's "big picture" angles, worth reusing for any other close call where a single opinion (yours, mine, or one AI pass) isn't enough to trust, e.g. picking between two model architectures if the choice isn't obvious.
- **`/critique <role>`** is the lighter, faster version, one adversarial voice (e.g. "review this as an NCC engineer") rather than five independent ones. Good for a quick gut check before council-of-five's heavier version.
- Ask it to verify its own output against a known-good reference before trusting anything score-relevant, same discipline as `test_severity.py`, don't let generated code go straight into the pipeline unchecked.
