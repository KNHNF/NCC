# Live demo, for the team

How to run it, what it is doing, and what you can honestly say on camera.

## How to run

From `karan-ncc-analysis`:

```bash
.\.venv\Scripts\activate
python gradio_app.py
```

Open http://127.0.0.1:7860 (hard refresh if the tab was already open). Record in Microsoft Edge fullscreen (F11). No public Gradio link is needed.

Upload from `demo-images`, pick any PNG, then **Analyse**. Or click **Pick a random hold-out**, then Analyse. Do not rely on upload-auto-run, it is off on purpose. There is no thumbnail table on screen before you submit.

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

## The hold-out folder

`demo-images/` is a small specimen tray of real NCC labelled micrographs. All stems are in the leak-free validation hold-out (`dataset.train_val_split`, seed 42). None of them were in training. Filenames are the real NCC ids plus `_scale1.0` or `_scale1.33` so microns per pixel can be read from the name (NCC metadata: Data set I is often 1.33, some Data set III images are 1.0).

**Safe to say:** "these are hold-out micrographs, held out of training. Different magnifications, so microns per pixel is 1.0 or 1.33 depending on the photo, that comes from NCC's own metadata. I'll pick one at random."

**Not safe to say:** "hidden test set" (that is the separate 32-image test folder). Do not say they were cherry-picked under fake names. Do not mention a scale table or Examples row; that UI is gone.

Expected live scores if those two come up: `G02_3_5120_7168_aug_0_scale1.0.png` about **24.2 PASS**, `2_3_2_R_cut_128_14848_scale1.33.png` about **388 FAIL**. Other files in the folder also mix clear PASS (under 25, with visible voids) and clear FAIL (well over 25), plus one near the threshold.

## What FAIL and PASS mean

- **FAIL** = the part would be **rejected**. Worst defect group scored 25 or higher. It does not meet NCC's certification rule.
- **PASS** = the part would be **accepted**. Every group is below 25, or there are no voids.

## Recording click order

See [DEMO-RECORDING-PLAN.md](DEMO-RECORDING-PLAN.md). Short version: Edge fullscreen, upload from `demo-images` (or Pick a random hold-out), Analyse, talk through PASS or FAIL.

## If both images fail

Wrong scale. Severity scales linearly with microns per pixel. A 1.0 image at 1.0 scores 24.2 (pass). At the old default 1.33 it scores 32.2 (fail). Keep `scale1.0` / `scale1.33` in the filename, or use Pick a random hold-out (it fills scale too). Filename scale still wins over the number box.
