# Demo recording plan

Live Gradio demo for Team Nexus, NCCUK composites challenge. Local URL: http://127.0.0.1:7860 (or 7861 if 7860 is already taken).

Record in **Microsoft Edge fullscreen** (F11) on this laptop. No public Gradio link is needed.

Hard refresh the tab before you start. Do not use an old tab.

`demo-images/` is a small hold-out tray (gitignored, laptop only). Real NCC ids plus `_scale1.0` / `_scale1.33`. All stems are validation hold-out (seed 42), not the hidden 32-image test set. Pick any file; do not present them as two staged rows.

## What you must say if asked

**Live, not a screenshot.** "This is running live inference now. On click it runs the U-Net, then NCC's severity formula. It is not a pre-saved screenshot or a cached result."

**CPU, not GPU, on this laptop.** "Training was on GPU, Kaggle or Colab. This laptop demo uses PyTorch 2.13 CPU-only, CUDA is not available here, so inference is on CPU. That is fine for a 256 by 256 micrograph. Say running live on CPU. Never imply a GPU is in use on this laptop."

**Hold-out micrographs.** Safe to say: "These are hold-out micrographs, different magnifications so microns per pixel is 1.0 or 1.33 depending on the photo, that comes from NCC's own metadata. I'll pick one at random." Do not call them the hidden test set. Do not mention a scale table or Examples row.

## Click order (Edge fullscreen, about 45-60 seconds)

1. Open Edge, go to http://127.0.0.1:7860, press F11.
2. Show the header: how pass/fail is decided, FAIL means rejected at 25 or higher, PASS means accepted below 25, device line says CPU.
3. Spoken: "These are hold-out micrographs, different magnifications so microns per pixel is 1.0 or 1.33 depending on the photo, that comes from NCC's own metadata. I'll pick one at random."
4. Either click **Pick a random hold-out**, or click upload, browse to `karan-ncc-analysis\demo-images`, pick any PNG.
5. Click **Analyse**. Wait. Do not say PASS or FAIL before the live result lands.
6. If PASS: "PASS means this part would be accepted. It meets NCC's certification rule."
7. If FAIL: "FAIL means this part would be rejected. It does not meet the certification rule."
8. Optional: pick another at random, Analyse again, so camera sees both an accept and a reject.

Do not say PASS or FAIL before Analyse finishes.

## If something looks wrong

- Scale box still on 1.33 after a `scale1.0` file: click Analyse anyway, the filename scale wins. Or click Pick a random hold-out again.
- Left preview blank: ignore it, the three-panel plot is the picture that matters.
- Both fail, or a known near-pass scores ~32: stop, that is the old wrong-scale bug. Refresh, upload from `demo-images` or use Pick a random hold-out. Do not upload a random TIFF.

## One-liners (memorise these)

- Live: "U-Net plus severity, running live on click, not a saved screenshot."
- Device: "Trained on GPU. This demo is live on CPU."
- Origin: "Hold-out NCC micrographs, held out of training."
- PASS: "Under 25, accepted."
- FAIL: "25 or over, rejected."
