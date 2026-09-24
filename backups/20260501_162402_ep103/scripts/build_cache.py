#!/usr/bin/env python
"""Pre-regrid ERA5 data and cache all variables to .npy files.

This script regrids ERA5 from native (111x235) to the CONUS404 grid
(1015x1367), applies pretransforms, and saves per-year .npy files for
both ERA5 (regridded) and CONUS404 (raw). Also builds static fields
and scans for NaN days.

All output is float32 to preserve precision.

Usage:
    python scripts/build_cache.py --output_dir /path/to/cache
    python scripts/build_cache.py --output_dir /path/to/cache --years 1980 1985
    python scripts/build_cache.py --output_dir /path/to/cache --workers 4

The output_dir can be on any filesystem with enough space.
Estimated size: ~5 GB per year for 6 ERA5 + 6 CONUS404 variables.
Total for 41 years: ~200 GB.
"""

import argparse
import json
import time
import sys
from pathlib import Path
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ERA5_VARS, CONUS404_VARS, VARIABLE_PAIRS,
)
from src.preprocessing.regrid import ERA5Regridder
from src.preprocessing.normalization import apply_pretransform
from src.preprocessing.land_mask import build_conus404_land_mask, get_valid_patch_origins


def _is_leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _month_for_day(day_of_year, leap=False):
    days_in_month = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cum = 0
    for m, d in enumerate(days_in_month):
        cum += d
        if day_of_year <= cum:
            return m
    return 11


def build_static_fields(conus_ds, era5_ds, conus_lat, conus_lon, regridder):
    """Build 6 static field channels, z-scored."""
    from scipy.ndimage import uniform_filter

    # Terrain height
    z = conus_ds["Z"].isel(time=0, bottom_top_stag=0).values
    z_norm = (z - z.mean()) / (z.std() + 1e-8)

    # Orographic variance
    z_mean = uniform_filter(z, size=5)
    z_sq_mean = uniform_filter(z ** 2, size=5)
    orog_var = np.sqrt(np.maximum(z_sq_mean - z_mean ** 2, 0))
    orog_var_norm = (orog_var - orog_var.mean()) / (orog_var.std() + 1e-8)

    # Lat/lon
    lat_norm = (conus_lat - conus_lat.mean()) / (conus_lat.std() + 1e-8)
    lon_norm = (conus_lon - conus_lon.mean()) / (conus_lon.std() + 1e-8)

    # LAI — try ERA5 first, regrid to CONUS grid
    if "lai_lv" in era5_ds:
        lai_raw = era5_ds["lai_lv"].isel(time=0, valid_time=0).values.astype(np.float32)
        lai = regridder.regrid(lai_raw)
    elif "LAI" in conus_ds:
        lai = conus_ds["LAI"].isel(time=0).values
    else:
        lai = np.zeros_like(z)
    lai = np.nan_to_num(lai, nan=0.0)
    lai_norm = (lai - lai.mean()) / (lai.std() + 1e-8)

    # Land-sea mask
    lsm = build_conus404_land_mask(conus_lat, conus_lon, era5_ds).astype(np.float32)

    static = np.stack(
        [z_norm, orog_var_norm, lat_norm, lon_norm, lai_norm, lsm], axis=0
    ).astype(np.float32)
    return static


def process_year(year, data_dir, output_dir, regridder, era5_vars, conus_vars):
    """Process one year: regrid ERA5, extract CONUS404, save both."""
    leap = _is_leap(year)
    n_days = 366 if leap else 365
    n_era5 = len(era5_vars)
    n_conus = len(conus_vars)

    era5_path = data_dir / f"era5_{year}.nc"
    conus_path = data_dir / f"conus404_yearly_{year}.nc"

    era5_out = np.zeros((n_days, n_era5, 1015, 1367), dtype=np.float32)
    conus_out = np.zeros((n_days, n_conus, 1015, 1367), dtype=np.float32)
    nan_days = []

    era5_ds = xr.open_dataset(era5_path)
    conus_ds = xr.open_dataset(conus_path)

    for d in range(n_days):
        day_1indexed = d + 1
        month_idx = _month_for_day(day_1indexed, leap)

        # Check for NaN day in CONUS404
        test_val = conus_ds[conus_vars[0]].isel(time=d).values
        if np.all(np.isnan(test_val)):
            nan_days.append(d)
            era5_out[d] = np.nan
            conus_out[d] = np.nan
            continue

        # ERA5: read, pretransform, regrid
        era5_fields = []
        for var in era5_vars:
            try:
                raw = era5_ds[var].isel(time=month_idx, valid_time=d).values.astype(np.float32)
                raw = apply_pretransform(raw, var)
                era5_fields.append(raw)
            except (IndexError, ValueError):
                era5_fields.append(np.full((111, 235), np.nan, dtype=np.float32))

        era5_stack = np.stack(era5_fields, axis=0)  # (n_era5, 111, 235)
        era5_regridded = regridder.regrid_batch(era5_stack)  # (n_era5, 1015, 1367)
        era5_out[d] = era5_regridded

        # CONUS404: read and pretransform
        for ci, var in enumerate(conus_vars):
            raw = conus_ds[var].isel(time=d).values.astype(np.float32)
            conus_out[d, ci] = apply_pretransform(raw, var)

        if (d + 1) % 50 == 0 or d == 0:
            print(f"  {year} day {d+1}/{n_days}")

    era5_ds.close()
    conus_ds.close()

    # Save as float32 numpy
    np.save(output_dir / f"era5_{year}.npy", era5_out)
    np.save(output_dir / f"conus_{year}.npy", conus_out)

    return nan_days


def main():
    parser = argparse.ArgumentParser(description="Build regridded cache for training")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save cached .npy files (can be any filesystem)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory with raw .nc symlinks")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Specific years to process (default: all 1980-2020)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    years = args.years or list(range(1980, 2021))
    era5_vars = ERA5_VARS
    conus_vars = CONUS404_VARS

    print("=" * 60)
    print("CACHE BUILDER")
    print(f"  ERA5 vars:   {era5_vars}")
    print(f"  CONUS vars:  {conus_vars}")
    print(f"  Years:       {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"  Output:      {output_dir}")
    print(f"  Precision:   float32")
    n_vars = len(era5_vars) + len(conus_vars)
    est_per_year = n_vars * 366 * 1015 * 1367 * 4 / 1e9
    print(f"  Est. size:   ~{est_per_year:.1f} GB/year, ~{est_per_year * len(years):.0f} GB total")
    print("=" * 60)

    # Build regridder
    print("\nInitializing regridder...")
    with xr.open_dataset(data_dir / "era5_1980.nc") as ds:
        era5_lat = ds["latitude"].values
        era5_lon = ds["longitude"].values
        era5_ds_for_static = ds  # keep open for static fields

    with xr.open_dataset(data_dir / "conus404_yearly_1980.nc") as ds:
        conus_lat = ds["lat"].values
        conus_lon = ds["lon"].values

    regridder = ERA5Regridder(era5_lat, era5_lon, conus_lat, conus_lon)
    print("Regridder ready.\n")

    # Build static fields (skip if already exists from another job)
    static_path = output_dir / "static_fields.npy"
    if static_path.exists():
        print(f"Static fields already exist at {static_path}, skipping.")
    else:
        print("Building static fields...")
        era5_ds_static = xr.open_dataset(data_dir / "era5_1980.nc")
        conus_ds_static = xr.open_dataset(data_dir / "conus404_yearly_1980.nc")
        static = build_static_fields(conus_ds_static, era5_ds_static, conus_lat, conus_lon, regridder)
        np.save(static_path, static)
        print(f"  Saved static_fields.npy {static.shape}")
        era5_ds_static.close()
        conus_ds_static.close()

    # Process each year
    all_nan_days = {}
    for i, year in enumerate(years):
        t0 = time.time()
        print(f"\n[{i+1}/{len(years)}] Processing {year}...")
        nan_days = process_year(year, data_dir, output_dir, regridder, era5_vars, conus_vars)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s. ", end="")
        if nan_days:
            all_nan_days[year] = nan_days
            print(f"{len(nan_days)} NaN days.")
        else:
            print("No NaN days.")

        # Print size
        era5_size = (output_dir / f"era5_{year}.npy").stat().st_size / 1e9
        conus_size = (output_dir / f"conus_{year}.npy").stat().st_size / 1e9
        print(f"  era5_{year}.npy: {era5_size:.2f} GB, conus_{year}.npy: {conus_size:.2f} GB")

    # Save NaN days (merge with existing if another job wrote some)
    nan_days_path = output_dir / "nan_days.json"
    if nan_days_path.exists():
        with open(nan_days_path) as f:
            existing = json.load(f)
        for k, v in all_nan_days.items():
            existing[str(k)] = v
        all_nan_days = existing
    with open(nan_days_path, "w") as f:
        json.dump(all_nan_days, f)
    print(f"\nSaved nan_days.json ({sum(len(v) for v in all_nan_days.values())} total NaN days)")

    print("\n" + "=" * 60)
    print("CACHE BUILD COMPLETE")
    print(f"  Output: {output_dir}")
    print(f"  Files:  {len(years)*2 + 2} (.npy + static + nan_days.json)")
    print("=" * 60)


if __name__ == "__main__":
    main()
