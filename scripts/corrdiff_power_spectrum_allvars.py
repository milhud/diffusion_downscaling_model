"""Power spectrum for CorrDiff's own output, all 5 non-precip base variables
(T2, TD2, U10, V10, PSFC), mean over its 6 generation dates. Precip excluded for the
same reason as the main benchmark's all-variable spectrum sheet: its spiky,
intermittent spatial structure puts it on a wildly different power scale that
distorts a shared-layout sheet.

Reads external/corrdiff/physicsnemo_src/examples/weather/corrdiff/corrdiff_output.nc
(CorrDiff's own truth/input/prediction at its own, unrecorded crop location per date --
same ground truth source as everywhere else in this repo, see
results/corrdiff_comparison/README.md). CPU only, no GPU rerun needed.

Writes results/corrdiff_comparison/corrdiff_power_spectrum_allvars.png: one sheet,
CONUS404 truth vs ERA5 input vs CorrDiff pipeline, each panel with a "graph in graph"
inset zoomed to the wavelength range where ERA5 and the pipeline separate most.

Usage: python scripts/corrdiff_power_spectrum_allvars.py
"""
from pathlib import Path

import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import power_spectrum_2d

NC_PATH = "external/corrdiff/physicsnemo_src/examples/weather/corrdiff/corrdiff_output.nc"
OUT_PATH = Path("results/corrdiff_comparison/corrdiff_power_spectrum_allvars.png")
VARS = ["T2", "TD2", "U10", "V10", "PSFC"]
UNITS = {"T2": "K", "TD2": "K", "U10": "m/s", "V10": "m/s", "PSFC": "Pa"}
VAR_TO_ERA5 = {"T2": "t2m", "TD2": "d2m", "U10": "u10", "V10": "v10", "PSFC": "sp"}
COL = {"CONUS404": "k", "ERA5 interp": "#888888", "Pipeline": "#264653"}


def main():
    f = nc.Dataset(NC_PATH)
    n_dates = f["time"].shape[0]

    series = {}
    for v in VARS:
        truth = f["truth"][v][:]
        era5 = f["input"][VAR_TO_ERA5[v]][:]
        pred = f["prediction"][v][0]   # ensemble member 0, the full pipeline, not an averaged mean

        wl = None
        pw_truth, pw_era5, pw_pipe = [], [], []
        for t in range(n_dates):
            wl, p_t = power_spectrum_2d(truth[t])
            _, p_e = power_spectrum_2d(era5[t])
            _, p_p = power_spectrum_2d(pred[t])
            pw_truth.append(p_t); pw_era5.append(p_e); pw_pipe.append(p_p)
        series[v] = dict(wl=wl, CONUS404=np.mean(pw_truth, axis=0),
                         **{"ERA5 interp": np.mean(pw_era5, axis=0), "Pipeline": np.mean(pw_pipe, axis=0)})

    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, v in zip(axs.flat, VARS):
        wl = series[v]["wl"]
        for meth, c in COL.items():
            ax.loglog(wl, series[v][meth], color=c, label=meth)
        ax.invert_xaxis()
        ax.set_title(f"{v} [{UNITS[v]}]", fontsize=11)
        ax.set_xlabel("wavelength (km)"); ax.set_ylabel("power")

        ratio_era5 = series[v]["ERA5 interp"] / series[v]["CONUS404"]
        ratio_pipe = series[v]["Pipeline"] / series[v]["CONUS404"]
        spread = np.abs(ratio_era5 - ratio_pipe)
        search_lo, search_hi = len(wl) // 5, len(wl) - len(wl) // 10
        if spread[search_lo:search_hi].max() > 0:
            peak_idx = search_lo + int(np.argmax(spread[search_lo:search_hi]))
        else:
            peak_idx = len(wl) // 2
        lo = max(0, peak_idx - len(wl) // 6)
        hi = min(len(wl), peak_idx + len(wl) // 6)
        inset = ax.inset_axes([0.08, 0.08, 0.42, 0.38])
        for meth, c in COL.items():
            inset.loglog(wl[lo:hi], series[v][meth][lo:hi], color=c)
        inset.invert_xaxis()
        inset.set_title(f"{wl[hi-1]:.0f}-{wl[lo]:.0f} km", fontsize=7)
        inset.tick_params(labelsize=6)
    axs.flat[0].legend(fontsize=9)
    axs.flat[-1].axis("off")
    fig.suptitle(f"CorrDiff power spectrum, all variables except precip ({n_dates} dates, its own crop per date)",
                 fontsize=14)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=140)
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
