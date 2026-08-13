# Demo recording plan

Live Gradio demo for Team Nexus, NCCUK composites challenge. Local URL: http://127.0.0.1:7860 (or 7861 if 7860 is already taken).

Hard refresh the tab before you start. Do not use an old tab.

This file also lives in the hackathon Pitch-Materials folder. This copy is the one teammates get when they clone `github.com/KNHNF/NCC`.

## What you must say if asked

**Live, not a screenshot.** "This is running live inference now. On click it runs the U-Net, then NCC's severity formula. It is not a pre-saved screenshot or a cached result."

**CPU, not GPU, on this laptop.** "Training was on GPU, Kaggle or Colab. This laptop demo uses PyTorch 2.13 CPU-only, CUDA is not available here, so inference is on CPU. That is fine for a 256 by 256 micrograph. Say running live on CPU. Never imply a GPU is in use on this laptop."

**These two images were held out of training.** Both source stems landed in the leak-free validation split (seed 42, grouped by source image). Safe to say: "these two micrographs were held out of training." Do not call them the hidden test set. They are from NCC's labelled set, validation hold-out.

## Click order (about 45-60 seconds)

1. Show the header: how pass/fail is decided, threshold 25.
2. Click the first thumbnail: `G02_3_5120_7168_aug_0_scale1.0.png`. Scale should read 1.00.
3. One line: "Real NCC micrograph, held out of training. Live inference on CPU now."
4. Click **Analyse**. Wait. Expect **PASS**, severity about **24.2**, 0.8 under the limit of 25.
5. "PASS means this part would be accepted. It meets NCC's certification rule."
6. Click the second thumbnail: `2_3_2_R_cut_128_14848_scale1.33.png`. Scale should read 1.33.
7. Click **Analyse**. Wait. Expect **FAIL**, severity about **388**.
8. "FAIL means this part would be rejected. It does not meet the certification rule."

Do not say PASS or FAIL before Analyse finishes. Let the live result land on camera.

## If something looks wrong

- Scale still on 1.33 for the first image: click Analyse anyway, the filename scale wins. Or click the thumbnail again.
- Left preview blank: ignore it, the three-panel plot is the picture that matters.
- Both fail, or first image ~32: stop, that is the old wrong-scale bug. Refresh, use the named thumbnails, not a random TIFF.

## One-liners (memorise these)

- Live: "U-Net plus severity, running live on click, not a saved screenshot."
- Device: "Trained on GPU. This demo is live on CPU."
- Origin: "Two real NCC labelled micrographs, held out of training."
- PASS: "24.2, under 25, accepted."
- FAIL: "388, over 25, rejected."
