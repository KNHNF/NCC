# Live demo, for the team

How to run it, what it is doing, and what you can honestly say on camera.

## How to run

From `karan-ncc-analysis`:

```bash
.\.venv\Scripts\activate
python gradio_app.py
```

Open http://127.0.0.1:7860 (hard refresh if the tab was already open).

Click a micrograph, then **Analyse**. Do not rely on upload-auto-run, it is off on purpose.

## What it does on click (live inference)

1. Load the micrograph the same way training did (PIL grayscale, 256 x 256).
2. Run the trained U-Net (`output/best_model.pth`).
3. Score voids with NCC's additive formula: severity = length + 0.5 * sqrt(area), merge voids within 40 microns.
4. **FAIL / rejected** if the worst group is **25 or higher**. **PASS / accepted** otherwise.

That is live. It is not a pre-saved screenshot and not a cached mask.

## CPU or GPU?

| Stage | Where | Device |
|---|---|---|
| Training | Kaggle / Colab | GPU |
| This laptop demo | `torch 2.13.0+cpu`, `cuda_available False` | **CPU** |

A 256 x 256 image is small enough that CPU inference is fine (a couple of seconds). If asked on camera: "running live on CPU." Never imply this laptop is using a GPU.

The header line on the app states the device.

## The two recording images

Matched by grayscale pixel hash to NCC's labelled set, then checked against `dataset.train_val_split` (seed 42, leak-free by source stem).

| On camera | Demo file (PNG, for browser preview) | Original NCC file | Where | um/px | Split |
|---|---|---|---|---|---|
| First click | `demo-images/G02_3_5120_7168_aug_0_scale1.0.png` | `G02_3_5120_7168_aug_0.tif` | Data set III / Augmented / Images | 1.0 | **VAL** |
| Second click | `demo-images/2_3_2_R_cut_128_14848_scale1.33.png` | `2_3_2_R_cut_128_14848.tif` | Data set I / Original / Images | 1.33 | **VAL** |

Source stems: `G02_3_5120_7168` and `2_3_2_R_cut_128_14848`. Both stems are in the validation hold-out, so neither the original nor the `_aug_0` variant was in training.

**Safe to say:** "these two micrographs were held out of training."

**Not safe to say:** "hidden test set" (that is the separate 32-image test folder). Do not say they were cherry-picked under fake names. The filenames are the real NCC ids, with `scale1.0` / `scale1.33` added so the app can read microns per pixel.

Expected live scores: first image about **24.2 PASS**, second about **388 FAIL**.

## What FAIL and PASS mean

- **FAIL** = the part would be **rejected**. Worst defect group scored 25 or higher. It does not meet NCC's certification rule.
- **PASS** = the part would be **accepted**. Every group is below 25, or there are no voids.

## Recording click order

See [DEMO-RECORDING-PLAN.md](DEMO-RECORDING-PLAN.md). Short version: first thumbnail, Analyse, talk through PASS; second thumbnail, Analyse, talk through FAIL.

## If both images fail

Wrong scale. Severity scales linearly with microns per pixel. The first image at 1.0 scores 24.2 (pass). At the old default 1.33 it scores 32.2 (fail). Use the named thumbnails, or keep `scale1.0` / `scale1.33` in the filename.
