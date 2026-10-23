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


def text_page(pdf, title, lines):
    fig = new_page()
    fig.text(0.06, 0.9, title, fontsize=24, fontweight="bold", color=FG, ha="left", va="top")
    y = 0.72
    for line in lines:
        wrapped = textwrap.wrap(line, width=100)
        for j, w in enumerate(wrapped):
            fig.text(0.08, y, w, fontsize=15, color=FG, ha="left", va="top")
            y -= 0.055
        y -= 0.04
    pdf.savefig(fig)
    plt.close(fig)


with PdfPages(OUT) as pdf:
    title_page(pdf, "ERA5→CONUS404 Latent CorrDiff",
               "Frequency-domain detail, CorrDiff comparison, Anderson Creek Fire case study")

    for v in ["T2", "TD2", "U10", "V10", "PSFC"]:
        image_page(pdf, f"{CORR}/power_spectrum_sidebyside_{v}.png", f"power spectrum, {v}, corrdiff vs ours")

    image_page(pdf, f"{CORR}/ours_vs_nvidia_corrdiff_rmse.png",
               "ours vs corrdiff, T2 and precip only, mean RMSE")
    image_page(pdf, f"{CORR}/accuracy_allvars_bars.png",
               "accuracy, all variables, corrdiff vs ours")

    text_page(pdf, "CorrDiff training time, one clean pass", [
        "Regression (CorrDiffRegressionUNet, 5,000,000 steps): 7 SLURM jobs, 3 days 7 hours 39 minutes compute, about 318.6 GPU-hours on 4xA100",
        "Diffusion (EDMPrecondSuperResolution, 5,000,000 steps): 10 SLURM jobs, 4 days 23 hours 2 minutes compute, about 476.1 GPU-hours",
        "Total: 17 jobs, 8 days 6 hours 40 minutes compute, about 794.7 GPU-hours",
        "Wall clock span: 8 days 18 hours 11 minutes (longer than compute time because of two multi hour SLURM queue waits between resubmissions)",
    ])
    text_page(pdf, "Our training time (DRN + VAE + diffusion), clean path only", [
        "DRN: 17 jobs, 3 days 9 hours 29 minutes compute, about 326 GPU-hours",
        "VAE: 4 jobs, 1 day 13 hours 17 minutes compute, about 149 GPU-hours",
        "Diffusion: 16 jobs, 6 days 1 hour 37 minutes compute, about 582 GPU-hours (only reached epoch 106 of a planned 300)",
        "Total: 37 jobs, 11 days 0 hours 23 minutes compute, about 1,058 GPU-hours",
        "This is more total GPU-hours than CorrDiff's 794.7, even on the clean path, because our pipeline has three stages (DRN, then VAE, then diffusion) against CorrDiff's two (regression, then diffusion)",
    ])
    text_page(pdf, "Wall clock: training time vs inference time point in different directions", [
        "Training (previous two pages): ours uses more total GPU hours (about 1,058 vs CorrDiff's 794.7), because our pipeline has three trained stages against CorrDiff's two",
        "Inference, live example just run: ours took 1.03s for 4 diffusion steps and 4.25s for 16 steps (4 ensemble members, one A100, 256x256 patch)",
        "CorrDiff's own logged inference timing (job 58824533): 64.6s total for 4 steps and 4 members on a 448x448 crop, about 4.04s per batch element",
        "At matched step count (4 steps), ours was about 1s vs CorrDiff's about 4s per element, but the patches are not the same size, so part of that gap is resolution, not purely the architecture",
        "Training cost and inference cost are two separate questions here and they do not agree: more trained stages cost more to train, but the smaller compressed latent space costs less to run per inference call",
    ])

    for sp in sorted(glob.glob(f"{SBS}/sidebyside_*.png")):
        date_str = os.path.basename(sp).replace("sidebyside_", "").replace(".png", "")
        image_page(pdf, sp, f"corrdiff vs ours, {date_str}, same date, different map locations")

    stage_captions = {
        "start": "Anderson Creek Fire, day before ignition (2016-03-21)",
        "peak": "Anderson Creek Fire, day of maximum spread (2016-03-23)",
        "end": "Anderson Creek Fire, near-full containment (2016-03-29)",
    }
    for stage in ["start", "peak", "end"]:
        image_page(pdf, f"{FIG}/anderson_creek_fire_weather_{stage}.png", stage_captions[stage])

    image_page(pdf, f"{FIG}/anderson_creek_timeline.png", "Fire-weather timeline: start → peak → containment")
    image_page(pdf, f"{FIG}/anderson_creek_perimeter_overlay.png", "Peak-spread FFWI vs. the actual MTBS burn perimeter")
    image_page(pdf, f"{FIG}/anderson_creek_perimeter_overlay_zoom.png", "Zoomed on the actual burn scar (4 km native resolution)")
    image_page(pdf, f"{FIG}/anderson_creek_sensitivity.png", "Sensitivity: does the fire-weather formula amplify upstream error?")
    image_page(pdf, f"{FIG}/anderson_creek_raws_validation.png", "Independent station check (AVK ASOS, ~46 km from the fire) vs. model")

print("wrote", OUT)
