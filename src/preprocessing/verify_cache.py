"""Quick integrity check for cached data files.

Checks file existence, shapes, dtype, and cross-variable correlations.

Usage:
    python -m src.preprocessing.verify_cache [--cache_dir cached_data]
"""

import argparse
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ERA5_VARS, CONUS404_VARS, TRAIN, VARIABLE_NAMES


def check_integrity(cache_dir):
    """Check all cached files exist with correct shapes and dtype."""
    cache = Path(cache_dir)
    print("=== Cache Integrity Check ===")
    all_years = TRAIN["train_years"] + TRAIN["val_years"] + TRAIN["test_years"]
    n_era5 = len(ERA5_VARS)
    n_conus = len(CONUS404_VARS)

    missing = []
    for year in all_years:
        era5_f = cache / f"era5_{year}.npy"
        conus_f = cache / f"conus_{year}.npy"
        if not era5_f.exists():
            missing.append(str(era5_f))
        if not conus_f.exists():
            missing.append(str(conus_f))

    if missing:
        print(f"  MISSING {len(missing)} files:")
        for m in missing[:10]:
            print(f"    {m}")
        return False

    # Check shapes for a sample year
    year = all_years[len(all_years) // 2]
    era5 = np.load(cache / f"era5_{year}.npy", mmap_mode="r")
    conus = np.load(cache / f"conus_{year}.npy", mmap_mode="r")
    print(f"  Sample year {year}:")
    print(f"    ERA5:  shape={era5.shape}, dtype={era5.dtype}")
    print(f"    CONUS: shape={conus.shape}, dtype={conus.dtype}")
    print(f"    Expected: ({365 or 366}, {n_era5}, H, W) and ({365 or 366}, {n_conus}, H, W)")

    # Check static fields
    static_f = cache / "static_fields.npy"
    if static_f.exists():
        static = np.load(static_f)
        print(f"    Static: shape={static.shape}, dtype={static.dtype}")
    else:
        print("    Static: MISSING")
        return False

    print(f"  All {len(all_years) * 2 + 1} files present")
    return True


def check_spot(cache_dir, year=2000, day=100):
    """Spot-check: verify data is not all zeros or NaN."""
    print(f"\n=== Spot Check: year={year}, day={day} ===")
    cache = Path(cache_dir)
    era5 = np.load(cache / f"era5_{year}.npy", mmap_mode="r")
    conus = np.load(cache / f"conus_{year}.npy", mmap_mode="r")

    if day >= era5.shape[0]:
        print(f"  Day {day} out of range (max={era5.shape[0]-1})")
        return False

    era5_day = np.array(era5[day])
    conus_day = np.array(conus[day])

    print(f"  ERA5 day {day}: all_zero={np.all(era5_day == 0)}, all_nan={np.all(np.isnan(era5_day))}")
    print(f"  CONUS day {day}: all_zero={np.all(conus_day == 0)}, all_nan={np.all(np.isnan(conus_day))}")

    for i, var in enumerate(ERA5_VARS):
        if i >= era5_day.shape[0]:
            break
        v = era5_day[i]
        print(f"    ERA5 {var}: min={np.nanmin(v):.2f}, max={np.nanmax(v):.2f}, "
              f"nan_frac={np.isnan(v).mean():.4f}")

    for i, var in enumerate(CONUS404_VARS):
        if i >= conus_day.shape[0]:
            break
        v = conus_day[i]
        print(f"    CONUS {var}: min={np.nanmin(v):.2f}, max={np.nanmax(v):.2f}, "
              f"nan_frac={np.isnan(v).mean():.4f}")

    return True


def check_correlations(cache_dir, year=2000, day=100):
    """Cross-variable correlation check — physically related vars should correlate."""
    print(f"\n=== Cross-Variable Correlation: year={year}, day={day} ===")
    cache = Path(cache_dir)
    conus = np.load(cache / f"conus_{year}.npy", mmap_mode="r")

    if day >= conus.shape[0] or np.all(np.isnan(conus[day])):
        print("  SKIP: day is NaN or out of range")
        return True

    conus_day = np.array(conus[day])
    var_names = CONUS404_VARS

    # Check T2 - TD2 correlation (dewpoint <= temperature)
    if "T2" in var_names and "TD2" in var_names:
        t2_idx = var_names.index("T2")
        td2_idx = var_names.index("TD2")
        t2 = conus_day[t2_idx].flatten()
        td2 = conus_day[td2_idx].flatten()
        mask = ~(np.isnan(t2) | np.isnan(td2))
        if mask.sum() > 100:
            corr = np.corrcoef(t2[mask], td2[mask])[0, 1]
            frac_td2_gt_t2 = (td2[mask] > t2[mask]).mean()
            print(f"  T2 vs TD2: corr={corr:.4f}, frac(TD2>T2)={frac_td2_gt_t2:.4f}")
            if frac_td2_gt_t2 > 0.05:
                print("  WARNING: >5% of pixels have dewpoint > temperature")

    # Check U10 vs V10 (should be relatively independent)
    if "U10" in var_names and "V10" in var_names:
        u_idx = var_names.index("U10")
        v_idx = var_names.index("V10")
        u = conus_day[u_idx].flatten()
        v = conus_day[v_idx].flatten()
        mask = ~(np.isnan(u) | np.isnan(v))
        if mask.sum() > 100:
            corr = np.corrcoef(u[mask], v[mask])[0, 1]
            print(f"  U10 vs V10: corr={corr:.4f} (expect low)")

    return True


def plot_cache_overview(cache_dir, year=2000, day=100, output="scripts/verify_plots"):
    """Generate visual overview plots of cached data."""
    Path(output).mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir)

    era5_path = cache / f"era5_{year}.npy"
    conus_path = cache / f"conus_{year}.npy"
    if not era5_path.exists():
        print("  SKIP: cached data not found")
        return

    era5 = np.load(era5_path, mmap_mode="r")
    conus = np.load(conus_path, mmap_mode="r")
    if day >= era5.shape[0] or np.all(np.isnan(conus[day])):
        print("  SKIP: day is NaN or out of range")
        return

    # Plot all ERA5 variables
    n_era5 = min(len(ERA5_VARS), era5.shape[1])
    fig, axes = plt.subplots(1, n_era5, figsize=(5 * n_era5, 4))
    if n_era5 == 1:
        axes = [axes]
    for i in range(n_era5):
        data = era5[day, i]
        im = axes[i].imshow(data, origin="lower", aspect="auto")
        axes[i].set_title(f"ERA5: {ERA5_VARS[i]}")
        plt.colorbar(im, ax=axes[i], shrink=0.7)
    plt.suptitle(f"ERA5 Cache -- year={year}, day={day}")
    plt.tight_layout()
    fig.savefig(f"{output}/cache_era5_{year}_d{day}.png", dpi=100)
    plt.close(fig)

    # Plot all CONUS404 variables
    n_conus = min(len(CONUS404_VARS), conus.shape[1])
    fig, axes = plt.subplots(1, n_conus, figsize=(5 * n_conus, 4))
    if n_conus == 1:
        axes = [axes]
    for i in range(n_conus):
        data = conus[day, i]
        im = axes[i].imshow(data, origin="lower", aspect="auto")
        name = VARIABLE_NAMES.get(CONUS404_VARS[i], CONUS404_VARS[i])
        axes[i].set_title(f"CONUS: {name}")
        plt.colorbar(im, ax=axes[i], shrink=0.7)
    plt.suptitle(f"CONUS404 Cache -- year={year}, day={day}")
    plt.tight_layout()
    fig.savefig(f"{output}/cache_conus_{year}_d{day}.png", dpi=100)
    plt.close(fig)

    # Plot static fields
    static_path = cache / "static_fields.npy"
    if static_path.exists():
        static = np.load(static_path)
        field_names = ["terrain", "orog_var", "lat", "lon", "LAI", "land_mask"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for i, (ax, name) in enumerate(zip(axes.flatten(), field_names)):
            if i < static.shape[0]:
                im = ax.imshow(static[i], origin="lower", aspect="auto")
                ax.set_title(name)
                plt.colorbar(im, ax=ax, shrink=0.7)
        plt.suptitle("Static Fields (z-scored)")
        plt.tight_layout()
        fig.savefig(f"{output}/cache_static_fields.png", dpi=100)
        plt.close(fig)

    # Correlation scatter plots (if multi-var)
    if n_conus > 1:
        conus_day = np.array(conus[day])
        pairs = []
        if "T2" in CONUS404_VARS and "TD2" in CONUS404_VARS:
            pairs.append(("T2", "TD2"))
        if "U10" in CONUS404_VARS and "V10" in CONUS404_VARS:
            pairs.append(("U10", "V10"))
        if "T2" in CONUS404_VARS and "Q2" in CONUS404_VARS:
            pairs.append(("T2", "Q2"))

        if pairs:
            fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
            if len(pairs) == 1:
                axes = [axes]
            for ax, (v1, v2) in zip(axes, pairs):
                i1, i2 = CONUS404_VARS.index(v1), CONUS404_VARS.index(v2)
                d1, d2 = conus_day[i1].flatten(), conus_day[i2].flatten()
                mask = ~(np.isnan(d1) | np.isnan(d2))
                ax.scatter(d1[mask][::50], d2[mask][::50], s=1, alpha=0.3)
                ax.set_xlabel(v1); ax.set_ylabel(v2)
                corr = np.corrcoef(d1[mask], d2[mask])[0, 1]
                ax.set_title(f"{v1} vs {v2} (r={corr:.3f})")
            plt.suptitle(f"Cross-Variable Correlations -- year={year}, day={day}")
            plt.tight_layout()
            fig.savefig(f"{output}/cache_correlations_{year}_d{day}.png", dpi=100)
            plt.close(fig)

    print(f"  Saved verification plots to {output}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="cached_data")
    parser.add_argument("--output", default="scripts/verify_plots")
    args = parser.parse_args()

    ok = check_integrity(args.cache_dir)
    if ok:
        check_spot(args.cache_dir)
        check_correlations(args.cache_dir)
        plot_cache_overview(args.cache_dir, output=args.output)


if __name__ == "__main__":
    main()
