"""Upload a micrograph, get a pass/fail decision, live.

No Docker, no hosting, this is exactly what a two-day demo needs: run it,
get a shareable link for the duration of the session, done.

    python gradio_app.py

Then open the local URL it prints, or share the public link it also prints
(share=True below) for the pitch if you want it accessible off your laptop.
"""

import gradio as gr
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from skimage.measure import label, regionprops

from severity import STRAIGHT_LINE_THRESHOLD, VOID, severity_straight_line

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_UM_PER_PX = 1.33  # Data set I's common scale, a reasonable default for an unlabelled upload

MATRIX_C, FIBRE_C, VOID_C = "#2b2b2b", "#8fbf8f", "#ff5b4d"
FAIL_COLOUR, PASS_COLOUR = "#c0392b", "#2e7d32"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet",
                           in_channels=1, classes=3).to(DEVICE)
        _model.load_state_dict(torch.load("output/best_model.pth", map_location=DEVICE))
        _model.eval()
    return _model


def predict(image, um_per_px):
    if image is None:
        return None, "Upload a micrograph first."

    img = np.array(Image.fromarray(image).convert("L").resize((256, 256)), dtype=np.float32) / 255.0
    img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = get_model()(img_t).argmax(dim=1).cpu().squeeze(0).numpy()

    sev, group_severities = severity_straight_line(pred, um_per_px)
    fail = sev >= STRAIGHT_LINE_THRESHOLD
    verdict = "FAIL" if fail else "PASS"
    colour = FAIL_COLOUR if fail else PASS_COLOUR

    labelled = label(pred == VOID, connectivity=2)
    regions = regionprops(labelled)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.patch.set_facecolor("white")

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input", fontsize=13, fontweight="bold")

    cmap = plt.matplotlib.colors.ListedColormap([MATRIX_C, FIBRE_C, VOID_C])
    axes[1].imshow(pred, cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title("AI segmentation", fontsize=13, fontweight="bold")
    patches = [mpatches.Patch(color=c, label=l) for c, l in
               zip([MATRIX_C, FIBRE_C, VOID_C], ["Matrix", "Fibre", "Void"])]
    axes[1].legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.95)

    axes[2].axis("off")
    reason = (f"Worst defect group scored {sev:.1f}\nNCC's limit is {STRAIGHT_LINE_THRESHOLD}\n"
              f"{'Exceeds' if fail else 'Within'} the limit" if regions
              else "No voids detected")
    axes[2].text(0.5, 0.7, verdict, fontsize=40, fontweight="bold", color=colour,
                 ha="center", va="center", transform=axes[2].transAxes)
    axes[2].text(0.5, 0.35, reason, fontsize=12, ha="center", va="center",
                 transform=axes[2].transAxes, linespacing=1.8)
    axes[2].set_title("Decision", fontsize=13, fontweight="bold")

    for ax in axes[:2]:
        ax.axis("off")
    plt.tight_layout()

    summary = (f"**Verdict: {verdict}**\n\n"
               f"Severity: {sev:.1f} (threshold {STRAIGHT_LINE_THRESHOLD})\n\n"
               f"Void regions found: {len(regions)}\n\n"
               f"Group severities: {[round(g, 1) for g in group_severities]}")

    return fig, summary


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label="Upload a CFRP micrograph"),
        gr.Number(value=DEFAULT_UM_PER_PX, label="Microns per pixel (scale)",
                   info="Real submissions carry this in metadata.csv, set manually for an arbitrary upload"),
    ],
    outputs=[
        gr.Plot(label="Pipeline result"),
        gr.Markdown(label="Summary"),
    ],
    title="NCC composites defect detection, Team Nexus",
    description="Upload a micrograph, the model segments it, measures any voids, "
                "and gives a pass/fail call against NCC's own severity threshold.",
)

if __name__ == "__main__":
    demo.launch(share=True)
