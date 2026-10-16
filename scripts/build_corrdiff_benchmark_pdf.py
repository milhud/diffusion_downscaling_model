"""Companion PDF for corrdiff_benchmark_slides.pptx -- same content/order/captions,
for quick viewing without opening PowerPoint/Slides. No LibreOffice on this system,
so this is built directly from the same source images via matplotlib rather than a
true pptx->pdf conversion.
"""
import glob
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

REPO = "/gpfsm/dnb33/hpmille1/diffusion_downscaling_model"
FIG = f"{REPO}/event_benchmark_output/figures"
CORR = f"{REPO}/results/corrdiff_comparison"
SBS = f"{REPO}/results/corrdiff_comparison/side_by_side"
OUT = "/gpfsm/dnb33/tdirs/login/discover32.121613.hpmille1/claude-337314102/-gpfsm-dnb33-hpmille1-diffusion-downscaling-model/f0ce2534-90ec-47a6-8a75-3e70f04cce09/scratchpad/deck/corrdiff_benchmark_slides.pdf"

PAGE_W, PAGE_H = 13.333, 7.5
FG = "#11141A"


def new_page():
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor="white")
    return fig


def image_page(pdf, path, caption=None):
    fig = new_page()
    top = 0.88 if caption else 0.95
    img = mpimg.imread(path)
    ax = fig.add_axes([0.04, 0.03, 0.92, top - 0.03])
    ax.imshow(img)
    ax.axis("off")
    if caption:
        fig.text(0.04, 0.93, caption, fontsize=15, fontweight="bold", color=FG, ha="left", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def title_page(pdf, title, subtitle):
    fig = new_page()
    fig.text(0.06, 0.58, title, fontsize=32, fontweight="bold", color=FG, ha="left", va="center")
    fig.text(0.06, 0.48, subtitle, fontsize=15, color="#5B6470", ha="left", va="center")
    pdf.savefig(fig)
    plt.close(fig)


def bar_chart_page(pdf, title, categories, values, color):
    fig = new_page()
    fig.text(0.06, 0.93, title, fontsize=17, fontweight="bold", color=FG, ha="left", va="top")
    ax = fig.add_axes([0.08, 0.1, 0.87, 0.74])
    x = range(len(categories))
    bars = ax.bar(x, values, color=color, width=0.55)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:g}", ha="center", va="bottom", fontsize=12, color=FG)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, fontsize=12, color=FG)
    ax.set_ylabel("GPU-hours", fontsize=12, color="#3A414A")
    ax.tick_params(colors="#3A414A")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D8DCE2")
    ax.yaxis.grid(True, color="#D8DCE2")
    ax.set_axisbelow(True)
    pdf.savefig(fig)
    plt.close(fig)


with PdfPages(OUT) as pdf:
    title_page(pdf, "ERA5→CONUS404 Latent CorrDiff",
               "Frequency-domain detail · NVIDIA CorrDiff comparison · Anderson Creek Fire case study")

    image_page(pdf, f"{FIG}/t2_spectrum_detail.png",
               "T2 power spectrum (log-log) — raw view, all methods look nearly identical")
    image_page(pdf, f"{FIG}/t2_spectrum_linear_freq.png",
               "Same data, linear frequency axis — right panel = pred/truth power ratio")
    image_page(pdf, f"{FIG}/t2_spectrum_ratio_zoom.png",
               "Zoomed pred/truth ratio, 4–128 km — where the methods actually separate")

    image_page(pdf, f"{CORR}/ours_vs_nvidia_corrdiff_rmse.png",
               "Ours vs. NVIDIA CorrDiff — mean RMSE, 6 matched held-out dates")
    image_page(pdf, f"{CORR}/nvidia_corrdiff_power_spectrum.png",
               "NVIDIA CorrDiff power spectrum, same 6 dates")

    bar_chart_page(pdf, "NVIDIA CorrDiff (official physicsnemo code) — training time, one clean pass",
                    ["Regression\n(5M steps)", "Diffusion\n(5M steps)"], [318.6, 476.1], "#76B900")
    bar_chart_page(pdf, "Our latent model (DRN + VAE + Diffusion) — training time, clean path only",
                    ["DRN", "VAE", "Diffusion\n(epoch 106/300)"], [326, 149, 582], "#2A9D8F")

    for sp in sorted(glob.glob(f"{SBS}/sidebyside_*.png")):
        date_str = os.path.basename(sp).replace("sidebyside_", "").replace(".png", "")
        image_page(pdf, sp, f"NVIDIA CorrDiff vs. ours, {date_str} — same date, different (unrecorded) crops")

    stage_captions = {
        "start": "Anderson Creek Fire — day before ignition (2016-03-21)",
        "peak": "Anderson Creek Fire — day of maximum spread (2016-03-23)",
        "end": "Anderson Creek Fire — near-full containment (2016-03-29)",
    }
    for stage in ["start", "peak", "end"]:
        image_page(pdf, f"{FIG}/anderson_creek_fire_weather_{stage}.png", stage_captions[stage])

    image_page(pdf, f"{FIG}/anderson_creek_timeline.png", "Fire-weather timeline: start → peak → containment")
    image_page(pdf, f"{FIG}/anderson_creek_perimeter_overlay.png", "Peak-spread FFWI vs. the actual MTBS burn perimeter")
    image_page(pdf, f"{FIG}/anderson_creek_perimeter_overlay_zoom.png", "Zoomed on the actual burn scar (4 km native resolution)")
    image_page(pdf, f"{FIG}/anderson_creek_sensitivity.png", "Sensitivity: does the fire-weather formula amplify upstream error?")
    image_page(pdf, f"{FIG}/anderson_creek_raws_validation.png", "Independent station check (AVK ASOS, ~46 km from the fire) vs. model")

print("wrote", OUT)
