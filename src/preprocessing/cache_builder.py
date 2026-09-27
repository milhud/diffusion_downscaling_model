"""Pre-compute regridded ERA5 + static fields and CONUS404 targets.

Saves per-year numpy files so the training dataloader can skip
netCDF I/O and regridding at runtime.

Output structure:
    cached_data/static_fields.npy    — (6, 1015, 1367) float32
    cached_data/era5_{year}.npy      — (N_days, 1015, 1367) float32  (regridded t2m)
    cached_data/conus_{year}.npy     — (N_days, 1015, 1367) float32  (T2)

Usage:
    python preprocess_cache.py --year 1980          # single year
    python preprocess_cache.py --year 1980 --year 1981  # multiple
    python preprocess_cache.py --all                # all years 1980-2020
"""

import argparse
import time
import numpy as np
import xarray as xr
from pathlib import Path

from src.preprocessing.regrid import ERA5Regridder
from src.preprocessing.normalization import apply_pretransform
from config import ERA5_VARS, CONUS404_VARS, TRAIN


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _month_for_day(day_of_year: int, leap: bool = False) -> int:
    days_in_month = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cum = 0
    for m, d in enumerate(days_in_month):
        cum += d
        if day_of_year <= cum:
            return m
    return 11


def build_static_fields(conus_ds, cache_dir):
    """Compute and save static fields (once)."""
    out_path = cache_dir / "static_fields.npy"
    if out_path.exists():
        print(f"  [static] Already cached: {out_path}")
        return

    from scipy.ndimage import uniform_filter

    lat = conus_ds["lat"].values
    lon = conus_ds["lon"].values
    lat_norm = (lat - lat.mean()) / (lat.std() + 1e-8)
    lon_norm = (lon - lon.mean()) / (lon.std() + 1e-8)

    z = conus_ds["Z"].isel(time=0, bottom_top_stag=0).values
    z_norm = (z - z.mean()) / (z.std() + 1e-8)

    lai = conus_ds["LAI"].isel(time=0).values
    lai = np.nan_to_num(lai, nan=0.0)
    lai_norm = (lai - lai.mean()) / (lai.std() + 1e-8)

    z_mean = uniform_filter(z, size=5)
    z_sq_mean = uniform_filter(z ** 2, size=5)
    orog_var = np.sqrt(np.maximum(z_sq_mean - z_mean ** 2, 0))
    orog_var_norm = (orog_var - orog_var.mean()) / (orog_var.std() + 1e-8)

    lsm = (z > 0).astype(np.float32)

    static = np.stack([z_norm, orog_var_norm, lat_norm, lon_norm, lai_norm, lsm],
                      axis=0).astype(np.float32)
    np.save(out_path, static)
    print(f"  [static] Saved {static.shape} -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def process_year(year, data_dir, cache_dir, regridder):
    """Pre-compute regridded ERA5 and raw CONUS404 for one year."""
    era5_path = cache_dir / f"era5_{year}.npy"
    conus_path = cache_dir / f"conus_{year}.npy"

    if era5_path.exists() and conus_path.exists():
        print(f"  [{year}] Already cached, skipping")
        return

    leap = _is_leap(year)
    n_days = 366 if leap else 365

    t0 = time.time()

    # Load ERA5 and CONUS404 for this year
    era5_ds = xr.open_dataset(data_dir / f"era5_{year}.nc")
    conus_ds = xr.open_dataset(data_dir / f"conus404_yearly_{year}.nc")

    # Pre-allocate output arrays
    H, W = 1015, 1367
    n_era5 = len(ERA5_VARS)
    n_conus = len(CONUS404_VARS)
    era5_out = np.empty((n_days, n_era5, H, W), dtype=np.float32)
    conus_out = np.empty((n_days, n_conus, H, W), dtype=np.float32)

    for day_idx in range(n_days):
        month_idx = _month_for_day(day_idx + 1, leap)

        # ERA5: read, pretransform, regrid
        era5_channels = []
        for var in ERA5_VARS:
            raw = era5_ds[var].isel(time=month_idx, valid_time=day_idx).values.astype(np.float32)
            raw = apply_pretransform(raw, var)
            regridded = regridder.regrid(raw)
            era5_channels.append(regridded)
        era5_out[day_idx] = np.stack(era5_channels, axis=0)

        # CONUS404: read, pretransform
        conus_channels = []
        for var in CONUS404_VARS:
            raw = conus_ds[var].isel(time=day_idx).values.astype(np.float32)
            raw = apply_pretransform(raw, var)
            conus_channels.append(raw)
        conus_out[day_idx] = np.stack(conus_channels, axis=0)

        if (day_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (day_idx + 1) / elapsed
            eta = (n_days - day_idx - 1) / rate
            print(f"  [{year}] Day {day_idx+1}/{n_days} "
                  f"({elapsed:.0f}s elapsed, {eta:.0f}s remaining)")

    era5_ds.close()
    conus_ds.close()

    # Save
    np.save(era5_path, era5_out)
    np.save(conus_path, conus_out)

    elapsed = time.time() - t0
    era5_mb = era5_path.stat().st_size / 1e6
    conus_mb = conus_path.stat().st_size / 1e6
    print(f"  [{year}] Done in {elapsed:.0f}s — "
          f"ERA5: {era5_mb:.0f} MB, CONUS: {conus_mb:.0f} MB")


def main():
    parser = argparse.ArgumentParser(description="Cache regridded data for training")
    parser.add_argument("--year", type=int, action="append",
                        help="Year(s) to process (repeatable)")
    parser.add_argument("--all", action="store_true",
                        help="Process all years 1980-2020")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="cached_data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        years = list(range(1980, 2021))
    elif args.year:
        years = args.year
    else:
        parser.error("Specify --year YYYY or --all")

    print(f"Preprocessing {len(years)} year(s) -> {cache_dir}/")
    print(f"  ERA5 vars: {ERA5_VARS}")
    print(f"  CONUS404 vars: {CONUS404_VARS}")

    # Build regridder (needs grid coords from any year's files)
    era5_ds = xr.open_dataset(data_dir / "era5_1980.nc")
    conus_ds = xr.open_dataset(data_dir / "conus404_yearly_1980.nc")
    era5_lat = era5_ds["latitude"].values
    era5_lon = era5_ds["longitude"].values
    conus_lat = conus_ds["lat"].values
    conus_lon = conus_ds["lon"].values

    print("Building regridder...")
    regridder = ERA5Regridder(era5_lat, era5_lon, conus_lat, conus_lon)

    # Static fields (only needs one CONUS404 file)
    build_static_fields(conus_ds, cache_dir)
    era5_ds.close()
    conus_ds.close()

    # Process each year
    for year in years:
        process_year(year, data_dir, cache_dir, regridder)

    print("\nDone!")


if __name__ == "__main__":
    main()
