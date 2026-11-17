"""Download ERA5 daily CONUS fields from the public ARCO-ERA5 mirror (no credentials needed).

Writes era5_{year}.nc in the layout preprocess_cache.py expects:
dims (time=12 months, valid_time=days in year, latitude, longitude), NaN outside each day's month.
Instant variables (t2m, d2m, u10, v10, sp) are the mean of hours 00-23; tp is the sum of hours 01-24 (metres).
"""
import argparse
import calendar
import time
from pathlib import Path

import numpy as np
import xarray as xr

ZARR = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VARS = {
    "t2m": ("2m_temperature", "K", "2 metre temperature"),
    "d2m": ("2m_dewpoint_temperature", "K", "2 metre dewpoint temperature"),
    "u10": ("10m_u_component_of_wind", "m s**-1", "10 metre U wind component"),
    "v10": ("10m_v_component_of_wind", "m s**-1", "10 metre V wind component"),
    "sp": ("surface_pressure", "Pa", "Surface pressure"),
    "tp": ("total_precipitation", "m", "Total precipitation"),
}
LAT = slice(51.5, 24.0)
LON = slice(235.0, 293.5)  # ARCO uses 0-360; -125..-66.5 degE


def load_month(ds, year, month, tries=6):
    start = np.datetime64(f"{year}-{month:02d}-01T00")
    end = np.datetime64(f"{year + (month == 12)}-{1 if month == 12 else month + 1:02d}-01T00")
    for attempt in range(tries):
        try:
            sub = ds[[v[0] for v in VARS.values()]].sel(
                latitude=LAT, longitude=LON, time=slice(start, end)
            )
            return sub.load()
        except Exception as e:
            print(f"  retry {attempt + 1}/{tries} after {type(e).__name__}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"failed to load {year}-{month:02d}")


def build_year(year, months):
    ds = xr.open_zarr(ZARR, chunks=None)
    ndays = 366 if calendar.isleap(year) else 365
    out = {k: np.full((12, ndays, 111, 235), np.nan, np.float32) for k in VARS}
    lat = lon = None
    doy = 0
    for m in range(1, 13):
        nd = calendar.monthrange(year, m)[1]
        if m in months:
            t0 = time.time()
            sub = load_month(ds, year, m)
            assert sub.sizes["time"] == nd * 24 + 1, sub.sizes
            lat, lon = sub.latitude.values, sub.longitude.values - 360.0
            for k, (name, _, _) in VARS.items():
                a = sub[name].values
                if k == "tp":
                    d = a[1:].reshape(nd, 24, *a.shape[1:]).sum(1, dtype=np.float64)
                else:
                    d = a[:-1].reshape(nd, 24, *a.shape[1:]).mean(1, dtype=np.float64)
                out[k][m - 1, doy:doy + nd] = d
            print(f"{year}-{m:02d} done in {time.time() - t0:.0f}s", flush=True)
        doy += nd
    days = np.arange(f"{year}-01-01", f"{year + 1}-01-01", dtype="datetime64[D]").astype("datetime64[ns]")
    dims = ("time", "valid_time", "latitude", "longitude")
    result = xr.Dataset(
        {k: (dims, out[k], {"units": VARS[k][1], "long_name": VARS[k][2]}) for k in VARS},
        coords={"valid_time": days, "latitude": lat, "longitude": lon},
        attrs={"source": "ARCO-ERA5 (ECMWF ERA5 hourly 0.25deg), daily aggregated", "year": year},
    )
    return result


def load_lsm():
    ds = xr.open_zarr(ZARR, chunks=None)
    a = ds["land_sea_mask"].sel(time="1980-01-01T00", latitude=LAT, longitude=LON).load().values
    assert a.shape == (111, 235) and not np.isnan(a).any()
    return a.astype(np.float32)


def add_lsm(path, lsm):
    """Append a static land-sea mask in the same (time, valid_time, lat, lon) month-slot layout, via a temp copy."""
    import shutil
    import netCDF4

    tmp = path.with_name(f".{path.name}.lsm.tmp")
    shutil.copy2(path, tmp)
    with netCDF4.Dataset(tmp, "a") as nc:
        if "lsm" in nc.variables:
            tmp.unlink()
            return False
        v = nc.createVariable("lsm", "f4", ("time", "valid_time", "latitude", "longitude"), zlib=True,
                              complevel=5, chunksizes=(1, 46, 14, 30), fill_value=np.float32("nan"))
        v.units, v.long_name = "(0 - 1)", "Land-sea mask"
        days = np.arange(f"{path.stem[-4:]}-01-01", f"{int(path.stem[-4:]) + 1}-01-01", dtype="datetime64[D]")
        months = days.astype("datetime64[M]").astype(int) % 12
        for m in range(12):
            idx = np.where(months == m)[0]
            block = np.full((len(days), 111, 235), np.nan, np.float32)
            block[idx] = lsm
            v[m, :, :, :] = block
    tmp.rename(path)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("year", type=int, nargs="?")
    p.add_argument("--out", default="/discover/nobackup/hpmille1/data/era5_data")
    p.add_argument("--months", type=int, nargs="*", default=list(range(1, 13)))
    p.add_argument("--add-lsm", action="store_true", help="append lsm to all existing era5_YYYY.nc in --out")
    args = p.parse_args()

    if args.add_lsm:
        lsm = load_lsm()
        for f in sorted(Path(args.out).glob("era5_[0-9][0-9][0-9][0-9].nc")):
            print(f.name, "added" if add_lsm(f, lsm) else "already has lsm", flush=True)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"era5_{args.year}.nc"
    if final.exists() and len(args.months) == 12:
        print(f"{final} exists, skipping")
        return
    ds = build_year(args.year, set(args.months))
    tmp = out / f".era5_{args.year}.nc.tmp"
    enc = {k: {"zlib": True, "complevel": 5, "dtype": "float32", "_FillValue": np.float32("nan"),
               "chunksizes": (1, 46, 14, 30)} for k in VARS}
    ds.to_netcdf(tmp, encoding=enc)
    dest = final if len(args.months) == 12 else out / f"era5_{args.year}_partial.nc"
    tmp.rename(dest)
    if dest == final:
        add_lsm(final, load_lsm())
    print("wrote", dest)


if __name__ == "__main__":
    main()
