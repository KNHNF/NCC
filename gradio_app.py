"""Upload a micrograph, get a pass/fail decision, live.

No Docker, no hosting, this is exactly what a two-day demo needs: run it,
get a shareable link for the duration of the session, done.

    python gradio_app.py

Then open the local URL it prints, or share the public link it also prints
(share=True below) for the pitch if you want it accessible off your laptop.
"""

import random
import re
from pathlib import Path

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
DEVICE_LABEL = (
    f"GPU ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda"
    else "CPU (this laptop's PyTorch build has no CUDA)"
)
DEFAULT_UM_PER_PX = 1.33  # Data set I's common scale, a reasonable default for an unlabelled upload
MODEL_SIZE = 256

DEMO_DIR = Path(__file__).parent / "demo-images"
_SCALE_IN_NAME = re.compile(r"scale(\d+(?:\.\d+)?)", re.IGNORECASE)

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


def infer_scale_from_path(path):
    if not path:
        return None
    text = str(path)
    m = _SCALE_IN_NAME.search(Path(text).name) or _SCALE_IN_NAME.search(text)
    return float(m.group(1)) if m else None


def list_demo_pngs():
    if not DEMO_DIR.is_dir():
        return []
    return sorted(p for p in DEMO_DIR.glob("*.png") if p.is_file())


def infer_scale_from_demo_pixels(image_path):
    """If Gradio renamed the upload, match pixels against demo-images."""
    if not image_path:
        return None
    try:
        uploaded = np.array(Image.open(image_path).convert("L"))
    except OSError:
        return None
    for demo_path in list_demo_pngs():
        try:
            demo = np.array(Image.open(demo_path).convert("L"))
        except OSError:
            continue
        if demo.shape == uploaded.shape and np.array_equal(demo, uploaded):
            return infer_scale_from_path(demo_path)
    return None


def infer_scale(image_path):
    inferred = infer_scale_from_path(image_path)
    if inferred is not None:
        return inferred
    return infer_scale_from_demo_pixels(image_path)


def update_scale_from_filename(image_path, current_um):
    """Prefill microns/pixel from the filename, or from a demo-images pixel match."""
    inferred = infer_scale(image_path)
    return inferred if inferred is not None else current_um


def resolve_scale(image_path, um_per_px):
    """Filename (or demo pixel match) wins; otherwise the number box."""
    inferred = infer_scale(image_path)
    if inferred is not None:
        return inferred
    if um_per_px is not None and float(um_per_px) > 0:
        return float(um_per_px)
    return DEFAULT_UM_PER_PX


def pick_random_holdout():
    pngs = list_demo_pngs()
    if not pngs:
        return None, DEFAULT_UM_PER_PX
    chosen = random.choice(pngs)
    scale = infer_scale_from_path(chosen) or DEFAULT_UM_PER_PX
    return str(chosen.resolve()), scale


def load_micrograph(image_path, um_per_px):
    """Same load path as VoidSegDataset: PIL grayscale, divide by 255.

    Training images are already 256x256, those are left untouched. Any other
    size is resized to 256x256 and um_per_px is scaled by the same factor,
    because severity is linear in microns-per-pixel.
    """
    pil = Image.open(image_path).convert("L")
    orig_w, orig_h = pil.size
    note = None
    if (orig_w, orig_h) != (MODEL_SIZE, MODEL_SIZE):
        um_per_px = um_per_px * (orig_w / MODEL_SIZE)
        pil = pil.resize((MODEL_SIZE, MODEL_SIZE), Image.BILINEAR)
        note = (f"Image was {orig_w}x{orig_h}, resized to {MODEL_SIZE}x{MODEL_SIZE}. "
                f"Scale adjusted to {um_per_px:.3f} um/px.")
        if orig_w != orig_h:
            note += " Source was not square, so geometry is slightly distorted."
    img = np.array(pil, dtype=np.float32) / 255.0
    return img, um_per_px, note


def predict(image_path, um_per_px):
    if not image_path:
        return None, "Upload a micrograph first."

    um_per_px = resolve_scale(image_path, um_per_px)
    img, um_per_px, resize_note = load_micrograph(image_path, um_per_px)
    img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = get_model()(img_t).argmax(dim=1).cpu().squeeze(0).numpy()

    sev, group_severities = severity_straight_line(pred, um_per_px)
    fail = sev >= STRAIGHT_LINE_THRESHOLD
    verdict = "FAIL" if fail else "PASS"
    colour = FAIL_COLOUR if fail else PASS_COLOUR

    labelled = label(pred == VOID, connectivity=2)
    regions = regionprops(labelled)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    fig.patch.set_facecolor("white")

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input", fontsize=16, fontweight="bold", color="#0f172a")

    cmap = plt.matplotlib.colors.ListedColormap([MATRIX_C, FIBRE_C, VOID_C])
    axes[1].imshow(pred, cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title("AI segmentation", fontsize=16, fontweight="bold", color="#0f172a")
    patches = [mpatches.Patch(color=c, label=l) for c, l in
               zip([MATRIX_C, FIBRE_C, VOID_C], ["Matrix", "Fibre", "Void"])]
    axes[1].legend(handles=patches, loc="lower right", fontsize=12, framealpha=0.95,
                   edgecolor="#0f172a", labelcolor="#0f172a")

    axes[2].axis("off")
    reason = (f"Worst defect group scored {sev:.1f}\nNCC's limit is {STRAIGHT_LINE_THRESHOLD}\n"
              f"{'Exceeds' if fail else 'Within'} the limit" if regions
              else "No voids detected")
    axes[2].text(0.5, 0.72, verdict, fontsize=48, fontweight="bold", color=colour,
                 ha="center", va="center", transform=axes[2].transAxes)
    axes[2].text(0.5, 0.34, reason, fontsize=16, color="#0f172a", ha="center", va="center",
                 transform=axes[2].transAxes, linespacing=1.7)
    axes[2].set_title("Decision", fontsize=16, fontweight="bold", color="#0f172a")

    for ax in axes[:2]:
        ax.axis("off")
    plt.tight_layout()

    badge_bg = "#fdecea" if fail else "#e8f5e9"
    groups_str = ", ".join(f"{g:.1f}" for g in group_severities) if group_severities else "none"

    margin = sev - STRAIGHT_LINE_THRESHOLD
    if not regions:
        meaning = "No voids detected, so this part would be accepted."
        plain = ("PASS means the part meets NCC's certification rule. "
                 "Nothing to reject, so it is accepted by default.")
    elif fail:
        meaning = "FAIL means this part would be REJECTED: it does not meet NCC's certification rule."
        plain = (f"Worst defect group scored {sev:.1f}, which is {margin:.1f} points "
                 f"OVER the fail threshold of {STRAIGHT_LINE_THRESHOLD}. "
                 f"A score of {STRAIGHT_LINE_THRESHOLD} or higher is a fail.")
    else:
        meaning = "PASS means this part would be ACCEPTED: it meets NCC's certification rule."
        plain = (f"Worst defect group scored {sev:.1f}, which is {abs(margin):.1f} points "
                 f"UNDER the fail threshold of {STRAIGHT_LINE_THRESHOLD}. "
                 f"Every group must stay below {STRAIGHT_LINE_THRESHOLD}.")

    bar_max = max(STRAIGHT_LINE_THRESHOLD * 1.8, sev * 1.1, 1)
    threshold_pct = min(100, STRAIGHT_LINE_THRESHOLD / bar_max * 100)
    severity_pct = min(100, sev / bar_max * 100)

    extra_note = f"<div class='note'>{resize_note}</div>" if resize_note else ""

    summary = f"""
<div id="decision-card" style="border-radius:12px;border:1px solid #e0e0e0;padding:20px;background:{badge_bg};">
  <div class="verdict" style="font-size:34px;font-weight:800;color:{colour};letter-spacing:1px;">{verdict}</div>
  <div class="meaning">{meaning}</div>
  <div class="plain">{plain}</div>

  <div style="margin-top:16px;">
    <div class="bar-scale" style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;">
      <span>0</span><span>Fail threshold ({STRAIGHT_LINE_THRESHOLD})</span><span>{bar_max:.0f}+</span>
    </div>
    <div style="position:relative;height:14px;background:#e2e8f0;border-radius:7px;overflow:visible;">
      <div style="position:absolute;left:0;top:0;height:14px;width:{severity_pct}%;
                  background:{colour};border-radius:7px;"></div>
      <div style="position:absolute;left:{threshold_pct}%;top:-3px;height:20px;width:2px;background:#0f172a;"></div>
    </div>
    <div class="bar-caption" style="font-size:11.5px;margin-top:3px;">
      solid bar = this part's severity &nbsp;&middot;&nbsp; black line = the pass/fail cutoff
    </div>
  </div>

  <div style="height:1px;background:#00000022;margin:16px 0;"></div>
  <table>
    <tr><td class="lbl">Severity score</td><td class="val">{sev:.1f}</td></tr>
    <tr><td class="lbl">NCC's fail threshold</td><td class="val">{STRAIGHT_LINE_THRESHOLD}</td></tr>
    <tr><td class="lbl">Microns per pixel used</td><td class="val">{um_per_px:.2f}</td></tr>
    <tr><td class="lbl">Void regions found</td><td class="val">{len(regions)}</td></tr>
    <tr><td class="lbl">Group severities</td><td class="val">{groups_str}</td></tr>
  </table>
  {extra_note}
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
    body_text_color="#0f172a",
    body_text_color_subdued="#0f172a",
    block_background_fill="#ffffff",
    block_label_text_color="#0f172a",
    block_title_text_color="#0f172a",
    block_info_text_color="#334155",
    block_border_width="1px",
    block_shadow="0 2px 10px rgba(20,40,80,0.06)",
    block_radius="14px",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
)

CSS = """
#header {text-align: center; padding: 8px 0 4px 0;}
#header h1 {font-size: 26px; font-weight: 800; margin-bottom: 2px; color: #0f172a !important;}
#header p {color: #0f172a !important; font-size: 15px;}
#device-line {color: #0f172a !important; font-size: 13.5px; font-weight: 700; margin-top: 4px;}
#rule-box, #rule-box * {color: #0f172a !important;}
#recording-hint {
  border-radius: 12px; border: 1px solid #c7d2fe; background: #eef2ff;
  padding: 12px 16px; margin: 8px 0 4px 0; font-size: 15px;
  color: #0f172a !important; font-weight: 700; line-height: 1.45;
}
footer {visibility: hidden}

.gradio-container, .gradio-container label, .gradio-container span,
.gradio-container p, .gradio-container td, .gradio-container th,
.gradio-container .info, .gradio-container [data-testid="block-info"],
.gradio-container .block-info, .gradio-container .svelte-1gfkn6j {
  color: #0f172a !important; opacity: 1 !important;
}
.gradio-container table th {
  background: #e2e8f0 !important; color: #0f172a !important; font-weight: 700 !important;
}
.gradio-container table td {
  background: #ffffff !important; color: #0f172a !important; font-weight: 600 !important;
}

#decision-card, #decision-card * {opacity: 1 !important;}
#decision-card .verdict {color: inherit;}
#decision-card .meaning {
  font-size: 14.5px; font-weight: 700; color: #0f172a !important;
  margin-top: 6px; line-height: 1.45;
}
#decision-card .plain {
  font-size: 14px; color: #0f172a !important;
  margin-top: 6px; line-height: 1.5;
}
#decision-card .bar-scale,
#decision-card .bar-caption {
  color: #0f172a !important; font-weight: 600;
}
#decision-card table {
  width: 100%; font-size: 15px; border-collapse: collapse; color: #0f172a !important;
}
#decision-card td.lbl {
  padding: 5px 0; color: #0f172a !important; font-weight: 600;
}
#decision-card td.val {
  padding: 5px 0; text-align: right; color: #0f172a !important;
  font-weight: 800 !important;
}
#decision-card .note {
  margin-top: 10px; font-size: 12.5px; color: #0f172a !important; line-height: 1.4;
}
"""

with gr.Blocks(title="NCC Composites Defect Detection") as demo:
    gr.HTML(f"""
    <div id="header">
      <h1>NCC Composites Defect Detection</h1>
      <p>Team Nexus &middot; upload a CFRP micrograph, the model segments it, measures every void,
      and calls pass or fail against NCC's own certification threshold.</p>
      <div id="device-line">Inference runs on {DEVICE_LABEL}</div>
    </div>
    """)

    gr.HTML(f"""
    <div id="rule-box" style="border-radius:12px;border:1px solid #dbeafe;background:#eff6ff;
                padding:14px 18px;margin-bottom:4px;font-size:13.5px;color:#0f172a;line-height:1.6;">
      <b>How the accept / reject call is made</b><br>
      Every void the model finds is measured for length and area. Voids within 40 microns of each
      other are treated as one connected defect, since a crack can propagate through the gap between them.
      Each group's severity is scored as <b>length + 0.5 &times; &radic;area</b> (microns).
      <br>
      <b>FAIL</b> means the part would be <b>rejected</b>: worst group scored
      <b>{STRAIGHT_LINE_THRESHOLD} or higher</b>, so it does not meet NCC's certification rule.
      <b>PASS</b> means the part would be <b>accepted</b>: every group scored below
      {STRAIGHT_LINE_THRESHOLD}, or no voids were found.
    </div>
    """)

    gr.HTML("""
    <div id="recording-hint">
      Upload from demo-images, pick any file (or use Pick a random hold-out), then Analyse.
      Inference runs live on this laptop (CPU), not from a saved screenshot.
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(
                label="CFRP micrograph", type="filepath", height=300,
                sources=["upload"], format="png")
            scale_in = gr.Number(
                value=DEFAULT_UM_PER_PX, label="Microns per pixel (optional)",
                info="Taken from the filename when it contains scale1.0 or scale1.33. Only needed if the name has no scale.")
            random_btn = gr.Button("Pick a random hold-out", variant="secondary")
            run_btn = gr.Button("Analyse", variant="primary", size="lg")
        with gr.Column(scale=2):
            plot_out = gr.Plot(label="Segmentation and measurement")
            summary_out = gr.HTML(label="Decision", elem_id="decision-html")

    random_btn.click(fn=pick_random_holdout, outputs=[image_in, scale_in])
    run_btn.click(fn=predict, inputs=[image_in, scale_in], outputs=[plot_out, summary_out])
    image_in.change(fn=update_scale_from_filename, inputs=[image_in, scale_in], outputs=[scale_in])

if __name__ == "__main__":
    demo.launch(
        share=True,
        theme=THEME,
        css=CSS,
        ssr_mode=False,
        allowed_paths=[str(DEMO_DIR)] if DEMO_DIR.exists() else None,
    )
