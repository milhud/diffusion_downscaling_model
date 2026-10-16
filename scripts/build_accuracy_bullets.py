"""One sheet, all 6 variables, CorrDiff vs ours accuracy, text bullets not a bar chart.

Reads results/corrdiff_comparison/allvars_rmse.csv (from
scripts/corrdiff_vs_ours_sidebyside.py) and writes
results/corrdiff_comparison/accuracy_allvars_bullets.png.

CorrDiff's crop location and ours are different (unrecorded for CorrDiff), so the two
ERA5 baselines are not the same number and are listed separately rather than averaged
together -- averaging two different locations' baselines would misrepresent both.
"""
import csv
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = Path("results/corrdiff_comparison/allvars_rmse.csv")
OUT_PATH = Path("results/corrdiff_comparison/accuracy_allvars_bullets.png")
UNITS = {"T2": "K", "TD2": "K", "U10": "m/s", "V10": "m/s", "PSFC": "Pa", "PREC_ACC_NC": "mm/day"}
VARS = ["T2", "TD2", "U10", "V10", "PSFC", "PREC_ACC_NC"]


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for k in ["corrdiff_era5_rmse", "corrdiff_pipeline_rmse", "ours_era5_rmse", "ours_pipeline_rmse"]:
            agg[r["var"]][k].append(float(r[k]))
    mean = {v: {k: sum(vals) / len(vals) for k, vals in agg[v].items()} for v in VARS}

    fig = plt.figure(figsize=(12, 9), facecolor="white")
    fig.text(0.05, 0.96, "accuracy, all variables, mean over the 6 matched dates", fontsize=18,
              fontweight="bold", color="#11141A")
    fig.text(0.05, 0.92, "corrdiff and our crop locations differ for each date, so the two era5 baselines below "
             "are not the same patch, do not compare them to each other directly", fontsize=10.5, color="#5B6470")

    y = 0.86
    for v in VARS:
        u = UNITS[v]
        m = mean[v]
        fig.text(0.05, y, f"{v} [{u}]", fontsize=14, fontweight="bold", color="#11141A")
        y -= 0.035
        fig.text(0.08, y, f"corrdiff crop: era5 {m['corrdiff_era5_rmse']:.2f}, pipeline {m['corrdiff_pipeline_rmse']:.2f}",
                  fontsize=11.5, color="#2A2F3A")
        y -= 0.03
        fig.text(0.08, y, f"our crop: era5 {m['ours_era5_rmse']:.2f}, pipeline {m['ours_pipeline_rmse']:.2f}",
                  fontsize=11.5, color="#2A2F3A")
        y -= 0.045

    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
