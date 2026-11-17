"""One sheet, all 6 variables, CorrDiff vs ours accuracy, as a bar chart.

Reads results/corrdiff_comparison/allvars_rmse.csv (from
scripts/corrdiff_vs_ours_sidebyside.py) and writes
results/corrdiff_comparison/accuracy_allvars_bars.png.

CorrDiff's map patch and ours are different (CorrDiff's generate.py does not
save which lat/lon it ran on), so the two ERA5 baselines are not the same patch and
are shown as separate bars rather than averaged together.
"""
import csv
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("results/corrdiff_comparison/allvars_rmse.csv")
OUT_PATH = Path("results/corrdiff_comparison/accuracy_allvars_bars.png")
UNITS = {"T2": "K", "TD2": "K", "U10": "m/s", "V10": "m/s", "PSFC": "Pa", "PREC_ACC_NC": "mm/day"}
VARS = ["T2", "TD2", "U10", "V10", "PSFC", "PREC_ACC_NC"]


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for k in ["corrdiff_era5_rmse", "corrdiff_pipeline_rmse", "ours_era5_rmse", "ours_pipeline_rmse"]:
            agg[r["var"]][k].append(float(r[k]))
    mean = {v: {k: sum(vals) / len(vals) for k, vals in agg[v].items()} for v in VARS}

    labels = ["corrdiff patch: era5", "corrdiff patch: pipeline", "our patch: era5", "our patch: pipeline"]
    colors = ["#A9A9A9", "#eb6834", "#C9C9E8", "#2a78d6"]
    keys = ["corrdiff_era5_rmse", "corrdiff_pipeline_rmse", "ours_era5_rmse", "ours_pipeline_rmse"]

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, v in zip(axs.flat, VARS):
        vals = [mean[v][k] for k in keys]
        bars = ax.bar(range(4), vals, color=colors)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_title(f"{v} [{UNITS[v]}]", fontsize=12)
        ax.set_ylabel("RMSE")
    fig.suptitle("accuracy, all variables, mean over the 6 matched dates", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=140)
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
