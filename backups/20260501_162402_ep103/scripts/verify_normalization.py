"""Verify normalization statistics are correct.

Checks:
  - Per-variable mean/std from norm_stats.npz
  - Normalized training data has mean~0, std~1
  - Pretransforms produce reasonable ranges
  - Histograms: raw vs normalized per variable

Usage:
    python scripts/verify_normalization.py [--cache_dir cached_data]
"""

import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ERA5_VARS, CONUS404_VARS, PRETRANSFORMS, VARIABLE_NAMES
from src.preprocessing.normalization import NormalizationStats, apply_pretransform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="cached_data")
    parser.add_argument("--stats_path", default="norm_stats.npz")
    parser.add_argument("--output", default="scripts/verify_plots")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    stats = NormalizationStats()
    stats.load(args.stats_path)

    print("=== Normalization Statistics ===")
    print(f"\nERA5 variables ({len(ERA5_VARS)}):")
    for i, var in enumerate(ERA5_VARS):
        m = stats.era5_mean[0, i].item()
        s = stats.era5_std[0, i].item()
        pre = PRETRANSFORMS.get(var, "none")
        print(f"  {var:>6s}: mean={m:12.4f}, std={s:12.4f}, pretransform={pre}")

    print(f"\nCONUS404 variables ({len(CONUS404_VARS)}):")
    for i, var in enumerate(CONUS404_VARS):
        m = stats.conus_mean[0, i].item()
        s = stats.conus_std[0, i].item()
        pre = PRETRANSFORMS.get(var, "none")
        print(f"  {var:>15s}: mean={m:12.4f}, std={s:12.4f}, pretransform={pre}")

    # Verify on actual data
    print("\n=== Verification on Training Data ===")
    year = 1990
    era5_path = Path(args.cache_dir) / f"era5_{year}.npy"
    conus_path = Path(args.cache_dir) / f"conus_{year}.npy"

    if not era5_path.exists():
        print(f"  SKIP: {era5_path} not found")
        return

    era5 = np.load(era5_path, mmap_mode="r")
    conus = np.load(conus_path, mmap_mode="r")

    # Sample 10 random days
    rng = np.random.default_rng(42)
    valid_days = [d for d in range(era5.shape[0]) if not np.all(np.isnan(conus[d]))]
    sample_days = rng.choice(valid_days, min(10, len(valid_days)), replace=False)

    fig, axes = plt.subplots(2, max(len(ERA5_VARS), len(CONUS404_VARS)),
                             figsize=(4 * max(len(ERA5_VARS), len(CONUS404_VARS)), 8))
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    for i, var in enumerate(ERA5_VARS):
        if i >= era5.shape[1]:
            break
        raw_vals = []
        norm_vals = []
        for d in sample_days:
            raw = era5[d, i].flatten()
            raw = raw[~np.isnan(raw)]
            raw_vals.append(raw[::100])
            transformed = apply_pretransform(raw[::100], var)
            m = stats.era5_mean[0, i].item()
            s = stats.era5_std[0, i].item()
            norm_vals.append((transformed - m) / s)

        raw_all = np.concatenate(raw_vals)
        norm_all = np.concatenate(norm_vals)

        if i < axes.shape[1]:
            axes[0, i].hist(raw_all, bins=50, alpha=0.7, density=True)
            axes[0, i].set_title(f"ERA5 {var} (raw)")
            axes[1, i].hist(norm_all, bins=50, alpha=0.7, density=True, color="orange")
            axes[1, i].set_title(f"ERA5 {var} (norm)")
            axes[1, i].axvline(0, color="k", linestyle="--", alpha=0.5)

        print(f"  ERA5 {var}: normalized mean={norm_all.mean():.4f}, std={norm_all.std():.4f}")

    plt.suptitle(f"Normalization Verification (year={year})")
    plt.tight_layout()
    fig.savefig(f"{args.output}/verify_normalization.png", dpi=100)
    plt.close(fig)
    print(f"\n  Saved {args.output}/verify_normalization.png")


if __name__ == "__main__":
    main()
