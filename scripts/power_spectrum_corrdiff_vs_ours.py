"""Power spectrum, one slide per variable, CorrDiff on the left and ours on the right,
for the 5 non-precip base variables (T2, TD2, U10, V10, PSFC). Precip excluded: its
spiky, intermittent spatial structure puts it on a power scale orders of magnitude
noisier than the other variables.

Left panel: CorrDiff's own output (CONUS404 truth, ERA5 interp, CorrDiff pipeline),
from external/corrdiff/physicsnemo_src/.../corrdiff_output.nc, its own 6 generation
dates, its own unrecorded crop per date.
Right panel: ours (CONUS404 truth, ERA5 interp, DRN, pipeline), from the main 10-event
test-year benchmark's cached predictions, 3 land-dominated event windows.
Not the same dates or the same crop, see results/corrdiff_comparison/README.md for
why (CorrDiff's generate.py does not save which crop it used, so dates are the only
thing that can be matched, and even those are a different sample here from the
main benchmark's). Each panel has its own "graph in graph" inset zoomed to the
wavelength range where ERA5 and the pipeline separate most.

Usage: python scripts/power_spectrum_corrdiff_vs_ours.py
"""
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import power_spectrum_2d
from src.evaluation.event_analyze import load_event, event_fields, spectrum, DAT, UNITS

NC_PATH = "external/corrdiff/physicsnemo_src/examples/weather/corrdiff/corrdiff_output.nc"
OUT_DIR = Path("results/corrdiff_comparison")
VARS = ["T2", "TD2", "U10", "V10", "PSFC"]
VAR_TO_ERA5 = {"T2": "t2m", "TD2": "d2m", "U10": "u10", "V10": "v10", "PSFC": "sp"}

COL_CD = {"CONUS404": "k", "ERA5 interp": "#888888", "Pipeline": "#264653"}
COL_OURS = {"CONUS404": "k", "ERA5 interp": "#888888", "DRN": "#F4A261", "Pipeline": "#264653"}


def corrdiff_series():
    f = nc.Dataset(NC_PATH)
    n_dates = f["time"].shape[0]
    series = {}
    for v in VARS:
        truth = f["truth"][v][:]
        era5 = f["input"][VAR_TO_ERA5[v]][:]
        pred = f["prediction"][v][0]
        wl = None
        pw_truth, pw_era5, pw_pipe = [], [], []
        for t in range(n_dates):
            wl, p_t = power_spectrum_2d(truth[t])
            _, p_e = power_spectrum_2d(era5[t])
            _, p_p = power_spectrum_2d(pred[t])
            pw_truth.append(p_t); pw_era5.append(p_e); pw_pipe.append(p_p)
        series[v] = dict(wl=wl, CONUS404=np.mean(pw_truth, axis=0),
                         **{"ERA5 interp": np.mean(pw_era5, axis=0), "Pipeline": np.mean(pw_pipe, axis=0)})
    return series, n_dates


def ours_series():
    cat = json.load(open(DAT / "event_catalog.json"))
    cat = [e for e in cat if (DAT / f"pred_{e['name']}.npz").exists() and not e["name"].startswith("AndersonCreek")]
    S = {v: {m: [] for m in ["CONUS404", "ERA5 interp", "DRN", "Pipeline"]} for v in VARS}
    n = 0
    for ev in cat:
        d = load_event(ev["name"]); land = d["land"].astype(bool)
        if land.mean() < 0.85:
            continue
        n += 1
        di, tr, m = event_fields(d)
        for v in VARS:
            S[v]["CONUS404"].append(spectrum(tr[v]))
            S[v]["ERA5 interp"].append(spectrum(m["ERA5 interp"][v]))
            S[v]["DRN"].append(spectrum(m["DRN"][v]))
            S[v]["Pipeline"].append(spectrum(m["Ens"][v][0]))
    k = np.arange(1, len(S[VARS[0]]["CONUS404"][0]))
    wl = 2048.0 / k
    series = {v: dict(wl=wl, **{meth: np.mean(S[v][meth], axis=0)[1:] for meth in S[v]}) for v in VARS}
    return series, n


def plot_panel(ax, inset_ax, wl, series, colors):
    for meth, c in colors.items():
        ax.loglog(wl, series[meth], color=c, label=meth)
    ax.invert_xaxis()

    ratio_era5 = series["ERA5 interp"] / series["CONUS404"]
    ratio_pipe = series["Pipeline"] / series["CONUS404"]
    spread = np.abs(ratio_era5 - ratio_pipe)
    search_lo, search_hi = len(wl) // 5, len(wl) - len(wl) // 10
    if spread[search_lo:search_hi].max() > 0:
        peak_idx = search_lo + int(np.argmax(spread[search_lo:search_hi]))
    else:
        peak_idx = len(wl) // 2
    lo = max(0, peak_idx - len(wl) // 6)
    hi = min(len(wl), peak_idx + len(wl) // 6)
    for meth, c in colors.items():
        inset_ax.loglog(wl[lo:hi], series[meth][lo:hi], color=c)
    inset_ax.invert_xaxis()
    inset_ax.set_title(f"{wl[hi-1]:.0f}-{wl[lo]:.0f} km", fontsize=8)
    inset_ax.tick_params(labelsize=7)


def main():
    cd_series, n_dates = corrdiff_series()
    ours_s, n_ours = ours_series()

    for v in VARS:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        plot_panel(axs[0], axs[0].inset_axes([0.08, 0.08, 0.42, 0.38]), cd_series[v]["wl"], cd_series[v], COL_CD)
        axs[0].set_title(f"CorrDiff, {v} [{UNITS[v]}], {n_dates} dates", fontsize=12)
        axs[0].set_xlabel("wavelength (km)"); axs[0].set_ylabel("power")
        axs[0].legend(fontsize=9)

        plot_panel(axs[1], axs[1].inset_axes([0.08, 0.08, 0.42, 0.38]), ours_s[v]["wl"], ours_s[v], COL_OURS)
        axs[1].set_title(f"Ours, {v} [{UNITS[v]}], {n_ours} event windows", fontsize=12)
        axs[1].set_xlabel("wavelength (km)"); axs[1].set_ylabel("power")
        axs[1].legend(fontsize=9)

        fig.suptitle(f"power spectrum, {v}, corrdiff vs ours (not the same dates or crop)", fontsize=13)
        fig.tight_layout()
        out = OUT_DIR / f"power_spectrum_sidebyside_{v}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
