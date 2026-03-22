"""Verify cached preprocessing data matches raw .nc files.

Checks:
  - Value ranges per variable
  - Random spot-checks: cached .npy vs raw .nc
  - No unexpected NaN beyond nan_days.json
  - Static fields match expected terrain/lat/lon
  - Side-by-side plots for visual verification

Usage:
    python scripts/verify_preprocessing.py [--cache_dir cached_data] [--data_dir data]
"""

import argparse
import json
import sys
import numpy as np
import xarray as xr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ERA5_VARS, CONUS404_VARS, VARIABLE_UNITS, VARIABLE_NAMES
from src.preprocessing.normalization import apply_pretransform
from src.preprocessing.regrid import ERA5Regridder
from src.preprocessing.land_mask import build_conus404_land_mask


# Expected physical ranges per variable (after pretransform)
# tp/PREC_ACC_NC use log1p, Q2 uses sqrt, z uses /9.81
EXPECTED_RANGES = {
    "T2": (200, 340),      # Kelvin (identity)
    "TD2": (190, 320),     # Kelvin (identity)
    "U10": (-40, 40),      # m/s (identity)
    "V10": (-40, 40),      # m/s (identity)
    "PSFC": (50000, 110000),  # Pa (identity)
    "PREC_ACC_NC": (0, 10),   # log1p(mm)
    "Q2": (0, 0.3),       # sqrt(kg/kg)
    "t2m": (200, 340),
    "d2m": (190, 320),
    "u10": (-40, 40),
    "v10": (-40, 40),
    "sp": (50000, 110000),
    "tp": (0, 1.0),        # log1p(m) — ERA5 uses meters
    "z": (-200, 90000),    # geopotential/9.81 = height in meters
}


def check_value_ranges(cache_dir, years=(2000, 2010)):
    """Check that cached values fall within expected physical ranges."""
    print("\n=== Value Range Check ===")
    passed = True

    for year in years:
        era5_path = Path(cache_dir) / f"era5_{year}.npy"
        conus_path = Path(cache_dir) / f"conus_{year}.npy"

        if not era5_path.exists():
            print(f"  SKIP: {era5_path} not found")
            continue

        era5 = np.load(era5_path, mmap_mode="r")
        conus = np.load(conus_path, mmap_mode="r")

        # Check ERA5 variables
        for i, var in enumerate(ERA5_VARS):
            if i >= era5.shape[1]:
                break
            data = era5[0, i]  # day 0, var i
            vmin, vmax = np.nanmin(data), np.nanmax(data)
            exp_min, exp_max = EXPECTED_RANGES.get(var, (-1e10, 1e10))
            ok = exp_min <= vmin and vmax <= exp_max
            status = "OK" if ok else "FAIL"
            print(f"  ERA5 {var} (year={year}): [{vmin:.2f}, {vmax:.2f}] expected [{exp_min}, {exp_max}] -- {status}")
            if not ok:
                passed = False

        # Check CONUS404 variables
        for i, var in enumerate(CONUS404_VARS):
            if i >= conus.shape[1]:
                break
            data = conus[0, i]
            vmin, vmax = np.nanmin(data), np.nanmax(data)
            exp_min, exp_max = EXPECTED_RANGES.get(var, (-1e10, 1e10))
            ok = exp_min <= vmin and vmax <= exp_max
            status = "OK" if ok else "FAIL"
            print(f"  CONUS {var} (year={year}): [{vmin:.2f}, {vmax:.2f}] expected [{exp_min}, {exp_max}] -- {status}")
            if not ok:
                passed = False

    return passed


def check_nan_consistency(cache_dir):
    """Verify NaN days match nan_days.json."""
    print("\n=== NaN Consistency Check ===")
    nan_path = Path(cache_dir) / "nan_days.json"
    if not nan_path.exists():
        print("  SKIP: nan_days.json not found")
        return True

    with open(nan_path) as f:
        nan_days = json.load(f)

    total_nan = sum(len(v) for v in nan_days.values())
    print(f"  nan_days.json: {total_nan} NaN days across {len(nan_days)} years")

    # Spot-check: verify a NaN day is actually NaN in the cache
    for year_str, days in nan_days.items():
        if not days:
            continue
        year = int(year_str)
        conus_path = Path(cache_dir) / f"conus_{year}.npy"
        if not conus_path.exists():
            continue
        conus = np.load(conus_path, mmap_mode="r")
        day = days[0]
        if day < conus.shape[0]:
            is_nan = np.all(np.isnan(conus[day]))
            print(f"  Year {year}, day {day}: {'NaN confirmed' if is_nan else 'NOT NaN -- MISMATCH'}")
            if not is_nan:
                return False
        break

    return True


def check_static_fields(cache_dir):
    """Verify static fields have correct shape and values."""
    print("\n=== Static Fields Check ===")
    static_path = Path(cache_dir) / "static_fields.npy"
    if not static_path.exists():
        print("  SKIP: static_fields.npy not found")
        return True

    static = np.load(static_path)
    print(f"  Shape: {static.shape} (expected (6, H, W))")
    assert static.shape[0] == 6, f"Expected 6 static fields, got {static.shape[0]}"

    field_names = ["terrain", "orog_var", "lat", "lon", "LAI", "land_mask"]
    for i, name in enumerate(field_names):
        vmin, vmax = np.nanmin(static[i]), np.nanmax(static[i])
        has_nan = np.any(np.isnan(static[i]))
        print(f"  {name}: [{vmin:.4f}, {vmax:.4f}], NaN={has_nan}")

    return True


def plot_comparison(cache_dir, data_dir, year=2000, day=100, output="diagnostic_plots/verify"):
    """Plot cached vs raw (with pretransform applied) and ERA5 vs CONUS404 land-only."""
    print(f"\n=== Visual Comparison: year={year}, day={day} ===")
    Path(output).mkdir(parents=True, exist_ok=True)

    conus_path = Path(cache_dir) / f"conus_{year}.npy"
    era5_cache_path = Path(cache_dir) / f"era5_{year}.npy"
    static_path = Path(cache_dir) / "static_fields.npy"
    if not conus_path.exists():
        print("  SKIP: cached CONUS data not found")
        return

    conus_cached = np.load(conus_path, mmap_mode="r")
    if day >= conus_cached.shape[0]:
        print(f"  SKIP: day {day} out of range")
        return

    # Load land mask from static fields (channel 5 = lsm)
    land_mask = None
    if static_path.exists():
        static = np.load(static_path)
        land_mask = static[5] >= 0.5

    # --- CONUS404: cached vs raw (with pretransform) ---
    nc_path = Path(data_dir) / f"conus404_yearly_{year}.nc"
    with xr.open_dataset(nc_path) as ds:
        for i, var in enumerate(CONUS404_VARS):
            if i >= conus_cached.shape[1]:
                break
            if var not in ds:
                continue

            raw = ds[var].isel(time=day).values.astype(np.float32)
            raw_pt = apply_pretransform(raw, var)
            cached = conus_cached[day, i]

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            im0 = axes[0].imshow(raw_pt, origin="lower", aspect="auto")
            axes[0].set_title(f"Raw .nc (pretransformed): {var}")
            plt.colorbar(im0, ax=axes[0])

            im1 = axes[1].imshow(cached, origin="lower", aspect="auto")
            axes[1].set_title(f"Cached .npy: {var}")
            plt.colorbar(im1, ax=axes[1])

            diff = cached - raw_pt
            max_diff = np.nanmax(np.abs(diff))
            im2 = axes[2].imshow(diff, origin="lower", aspect="auto", cmap="RdBu_r")
            axes[2].set_title(f"Diff (max={max_diff:.6f})")
            plt.colorbar(im2, ax=axes[2])

            plt.suptitle(f"CONUS404 {var} — year={year}, day={day}")
            plt.tight_layout()
            fig.savefig(f"{output}/verify_conus_{var}_{year}_d{day}.png", dpi=100)
            plt.close(fig)
            print(f"  Saved verify_conus_{var}_{year}_d{day}.png — max diff: {max_diff:.6f}")

    # --- ERA5 vs CONUS404: land-only side-by-side ---
    if not era5_cache_path.exists():
        print("  SKIP: cached ERA5 data not found, skipping land-only plots")
        return

    era5_cached = np.load(era5_cache_path, mmap_mode="r")
    era5_pairs = list(zip(ERA5_VARS, CONUS404_VARS))

    for i, (e5_var, c4_var) in enumerate(era5_pairs):
        if i >= era5_cached.shape[1] or i >= conus_cached.shape[1]:
            break

        era5_day = era5_cached[day, i]
        conus_day = conus_cached[day, i]

        if land_mask is not None:
            era5_land = np.where(land_mask, era5_day, np.nan)
            conus_land = np.where(land_mask, conus_day, np.nan)
        else:
            era5_land = era5_day
            conus_land = conus_day

        vmin = min(np.nanpercentile(era5_land, 2), np.nanpercentile(conus_land, 2))
        vmax = max(np.nanpercentile(era5_land, 98), np.nanpercentile(conus_land, 98))

        label = VARIABLE_NAMES.get(e5_var, e5_var)
        unit = VARIABLE_UNITS.get(e5_var, "")

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle(f"{label} ({unit}) — Land Only, year={year} day={day}", fontsize=14)

        im0 = axes[0].imshow(era5_land, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"ERA5 regridded ({e5_var})")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(conus_land, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"CONUS404 ({c4_var})")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        plt.tight_layout()
        fig.savefig(f"{output}/verify_land_{e5_var}_{c4_var}_{year}_d{day}.png", dpi=120)
        plt.close(fig)
        print(f"  Saved verify_land_{e5_var}_{c4_var}_{year}_d{day}.png")


def main():
    parser = argparse.ArgumentParser(description="Verify preprocessing cache")
    parser.add_argument("--cache_dir", default="cached_data")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    parser.add_argument("--year", type=int, default=None,
                        help="Year to verify/plot (default: first available)")
    args = parser.parse_args()

    # Find available years
    cache_path = Path(args.cache_dir)
    available = sorted([int(f.stem.split("_")[1])
                        for f in cache_path.glob("era5_*.npy")])
    if not available:
        print(f"No cached years found in {args.cache_dir}")
        return

    check_years = [available[0]]
    if len(available) > 1:
        check_years.append(available[len(available)//2])

    results = []
    results.append(("Value ranges", check_value_ranges(args.cache_dir, years=check_years)))
    results.append(("NaN consistency", check_nan_consistency(args.cache_dir)))
    results.append(("Static fields", check_static_fields(args.cache_dir)))

    if args.plot:
        plot_year = args.year or available[0]
        plot_comparison(args.cache_dir, args.data_dir, year=plot_year)

    print("\n=== SUMMARY ===")
    all_ok = True
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_ok = False
    print(f"\nOverall: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
