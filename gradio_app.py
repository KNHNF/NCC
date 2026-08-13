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

    badge_bg = "#fdecea" if fail else "#e8f5e9"
    groups_str = ", ".join(f"{g:.1f}" for g in group_severities) if group_severities else "none"
    summary = f"""
<div style="border-radius:12px;border:1px solid #e0e0e0;padding:20px;background:{badge_bg};">
  <div style="font-size:34px;font-weight:800;color:{colour};letter-spacing:1px;">{verdict}</div>
  <div style="height:1px;background:#00000014;margin:12px 0;"></div>
  <table style="width:100%;font-size:15px;color:#2b2b2b;border-collapse:collapse;">
    <tr><td style="padding:4px 0;color:#666;">Severity score</td>
        <td style="padding:4px 0;text-align:right;font-weight:600;">{sev:.1f}</td></tr>
    <tr><td style="padding:4px 0;color:#666;">NCC's fail threshold</td>
        <td style="padding:4px 0;text-align:right;font-weight:600;">{STRAIGHT_LINE_THRESHOLD}</td></tr>
    <tr><td style="padding:4px 0;color:#666;">Void regions found</td>
        <td style="padding:4px 0;text-align:right;font-weight:600;">{len(regions)}</td></tr>
    <tr><td style="padding:4px 0;color:#666;">Group severities</td>
        <td style="padding:4px 0;text-align:right;font-weight:600;">{groups_str}</td></tr>
  </table>
</div>
"""
    return fig, summary


THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#f7f9fc",
    body_background_fill_dark="#f7f9fc",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_shadow="0 2px 10px rgba(20,40,80,0.06)",
    block_radius="14px",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
)

CSS = """
#header {text-align: center; padding: 8px 0 4px 0;}
#header h1 {font-size: 26px; font-weight: 800; margin-bottom: 2px; color: #0f172a;}
#header p {color: #475569; font-size: 15px;}
footer {visibility: hidden}
"""

with gr.Blocks(theme=THEME, css=CSS, title="NCC Composites Defect Detection") as demo:
    gr.HTML("""
    <div id="header">
      <h1>NCC Composites Defect Detection</h1>
      <p>Team Nexus &middot; upload a CFRP micrograph, the model segments it, measures every void,
      and calls pass or fail against NCC's own certification threshold.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(label="CFRP micrograph", height=300)
            scale_in = gr.Number(
                value=DEFAULT_UM_PER_PX, label="Microns per pixel (scale)",
                info="Real submissions carry this in metadata.csv, set manually for an arbitrary upload")
            run_btn = gr.Button("Analyse", variant="primary", size="lg")
        with gr.Column(scale=2):
            plot_out = gr.Plot(label="Segmentation and measurement")
            summary_out = gr.HTML(label="Decision")

    run_btn.click(fn=predict, inputs=[image_in, scale_in], outputs=[plot_out, summary_out])
    image_in.change(fn=predict, inputs=[image_in, scale_in], outputs=[plot_out, summary_out])

if __name__ == "__main__":
    demo.launch(share=True)
