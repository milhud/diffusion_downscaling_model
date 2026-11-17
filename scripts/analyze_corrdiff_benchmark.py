"""
Accuracy benchmark: CorrDiff (full pipeline, both stages trained to
completion) vs our in-house Latent CorrDiff, on held-out test-year (2018-2020)
samples.

Reads external/corrdiff/physicsnemo_src/examples/weather/corrdiff/corrdiff_output.nc
(written by generate.py) and computes RMSE + power spectra per variable,
printing a summary comparable to our own model's numbers from
scripts/run_generic_baselines.py (same dates, held-out years, but NOT the same
exact crop location -- physicsnemo's dataset adapter picks a random crop per
sample with no recorded lat/lon, so this is a statistical comparison on
matched dates/years, not a pixel-identical one).

Usage: python scripts/analyze_corrdiff_benchmark.py
"""
from pathlib import Path

import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import power_spectrum_2d

NC_PATH = "external/corrdiff/physicsnemo_src/examples/weather/corrdiff/corrdiff_output.nc"
OUT_DIR = Path("results/corrdiff_comparison")
PLOT_VARS = ["T2", "PREC_ACC_NC"]
VAR_TO_ERA5 = {"T2": "t2m", "PREC_ACC_NC": "tp"}

# Our model's generic-baseline numbers restricted to the 3 test-year (2018-2020)
# samples among the original 8 (the other 5 are 2015-2017 val years, not used
# by CorrDiff's generation config which only loads test years) -- see
# scripts/run_generic_baselines.py's console output from job 58691071.
OURS_TEST_YEAR_SAMPLES = {
    "T2":          {"era5": [1.31, 1.86, 0.81], "drn": [0.98, 0.91, 0.97], "pipeline": [0.85, 0.89, 1.01]},
    "PREC_ACC_NC": {"era5": [0.47, 4.81, 3.65], "drn": [0.27, 4.91, 3.28], "pipeline": [0.34, 5.46, 3.55]},
}


def main():
    f = nc.Dataset(NC_PATH)
    times = [str(t) for t in f["time"][:]]
    print(f"CorrDiff generation output: {len(times)} samples -- {times}")

    BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#86847c"

    corrdiff_rmse = {v: {"era5": [], "pipeline": []} for v in PLOT_VARS}

    for var in PLOT_VARS:
        truth = f["truth"][var][:]           # (time, H, W)
        pred = f["prediction"][var][:]       # (ensemble, time, H, W)
        era5 = f["input"][VAR_TO_ERA5[var]][:]  # (time, H, W)
        pred_mean = pred.mean(axis=0)        # (time, H, W)

        for t in range(truth.shape[0]):
            rmse_era5 = float(np.sqrt(np.mean((era5[t] - truth[t]) ** 2)))
            rmse_pred = float(np.sqrt(np.mean((pred_mean[t] - truth[t]) ** 2)))
            corrdiff_rmse[var]["era5"].append(rmse_era5)
            corrdiff_rmse[var]["pipeline"].append(rmse_pred)

    print("\n=== CorrDiff (full pipeline, trained to completion) ===")
    for var in PLOT_VARS:
        era5_vals = corrdiff_rmse[var]["era5"]
        pipe_vals = corrdiff_rmse[var]["pipeline"]
        print(f"{var}: per-sample RMSE  ERA5={['%.2f' % v for v in era5_vals]}  "
              f"CorrDiff={['%.2f' % v for v in pipe_vals]}")
        print(f"{var}: mean RMSE  ERA5={np.mean(era5_vals):.3f}  CorrDiff={np.mean(pipe_vals):.3f}")

    print("\n=== Ours (Latent CorrDiff, DRN+VAE+diffusion), same 3 test-year dates ===")
    for var in PLOT_VARS:
        o = OURS_TEST_YEAR_SAMPLES[var]
        print(f"{var}: mean RMSE  ERA5={np.mean(o['era5']):.3f}  DRN={np.mean(o['drn']):.3f}  "
              f"Ours(pipeline)={np.mean(o['pipeline']):.3f}")

    # --- Comparison bar chart ---
    fig, axes = plt.subplots(1, len(PLOT_VARS), figsize=(11, 5), dpi=140)
    for i, var in enumerate(PLOT_VARS):
        ax = axes[i]
        unit = "K" if var == "T2" else "mm"
        o = OURS_TEST_YEAR_SAMPLES[var]
        labels = ["ERA5\n(no downscale)", "Ours\n(Latent CorrDiff)", "CorrDiff"]
        vals = [
            float(np.mean(o["era5"]) + np.mean(corrdiff_rmse[var]["era5"])) / 2,  # shared ERA5 baseline, averaged
            float(np.mean(o["pipeline"])),
            float(np.mean(corrdiff_rmse[var]["pipeline"])),
        ]
        colors = [AQUA, BLUE, ORANGE]
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_title(f"{var} -- mean RMSE vs CONUS404 target\n(matched dates, test years 2018-2020)", fontsize=10)
        ax.set_ylabel(f"RMSE ({unit})")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Ours vs CorrDiff", fontsize=12, fontweight="bold", y=1.03)
    fig.text(0.5, -0.03,
              "Caveat: same held-out dates, but NOT the same exact crop location (CorrDiff's generation\n"
              "picks a random CONUS crop per sample with no recorded coordinates) -- a statistical, not pixel-identical, comparison.",
              ha="center", fontsize=8.5, color=GREY)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "ours_vs_nvidia_corrdiff_rmse.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {OUT_DIR}/ours_vs_nvidia_corrdiff_rmse.png")

    # --- Power spectrum comparison (CorrDiff's own samples, averaged over its 6 times) ---
    fig, axes = plt.subplots(1, len(PLOT_VARS), figsize=(13, 5.5), dpi=140)
    for i, var in enumerate(PLOT_VARS):
        ax = axes[i]
        truth = f["truth"][var][:]
        pred_mean = f["prediction"][var][:].mean(axis=0)
        era5 = f["input"][VAR_TO_ERA5[var]][:]

        wl = None
        pw_truth_all, pw_pred_all, pw_era5_all = [], [], []
        for t in range(truth.shape[0]):
            wl, pw_t = power_spectrum_2d(truth[t])
            _, pw_p = power_spectrum_2d(pred_mean[t])
            _, pw_e = power_spectrum_2d(era5[t])
            pw_truth_all.append(pw_t)
            pw_pred_all.append(pw_p)
            pw_era5_all.append(pw_e)
        pw_truth = np.mean(pw_truth_all, axis=0)
        pw_pred = np.mean(pw_pred_all, axis=0)
        pw_era5 = np.mean(pw_era5_all, axis=0)

        ax.loglog(wl, pw_truth, color="black", linewidth=2, label="CONUS404 truth")
        ax.loglog(wl, pw_era5, color=AQUA, linewidth=1.5, linestyle="--", label="ERA5 (no downscale)")
        ax.loglog(wl, pw_pred, color=ORANGE, linewidth=1.5, label="CorrDiff")
        ax.set_title(f"Power spectrum (mean over {truth.shape[0]} samples)\n{var}", fontsize=10)
        ax.set_xlabel("wavelength (km)")
        ax.set_ylabel("power")
        ax.invert_xaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.suptitle("CorrDiff power spectrum vs CONUS404 truth", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "nvidia_corrdiff_power_spectrum.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/nvidia_corrdiff_power_spectrum.png")


if __name__ == "__main__":
    main()
