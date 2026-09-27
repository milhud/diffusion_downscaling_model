"""Event catalog for case-study benchmarking (test years 2018-2020 only).

Each event = (name, date, center lat/lon, kind). Windows are 512x512 px (2x2 tiles of
256) on the 4 km CONUS404 grid, clipped to the domain. Day index into the cache is
(date - Jan 1) in days, since the cache stores one daily-mean field per day.
"""
import datetime as dt
import json
from pathlib import Path
import numpy as np

CACHE_DIR = "/discover/nobackup/sduan/.data"
NC_2018 = "/discover/nobackup/hpmille1/final_data/conus404_yearly_2018.nc"
WIN = 512

EVENTS = [
    # name, date, lat, lon, kind
    ("Florence",        "2018-09-14", 34.2, -77.8, "tropical_cyclone"),
    ("Michael",         "2018-10-10", 30.3, -85.5, "tropical_cyclone"),
    ("CampFire_wind",   "2018-11-08", 39.8, -121.6, "fire_weather"),
    ("PolarVortex",     "2019-01-30", 41.9, -87.6, "cold_outbreak"),
    ("BombCyclone",     "2019-03-13", 39.0, -104.5, "extratropical_storm"),
    ("Dorian",          "2019-09-06", 34.5, -76.5, "tropical_cyclone"),
    ("IowaDerecho",     "2020-08-10", 42.0, -93.0, "convective_wind"),
    ("Laura",           "2020-08-27", 30.4, -93.2, "tropical_cyclone"),
    ("LaborDay_heat",   "2020-09-06", 36.5, -119.3, "fire_weather"),
    ("Sally",           "2020-09-16", 30.6, -87.6, "tropical_cyclone"),
]


def build_catalog(nc_path=NC_2018):
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    lat = ds["lat"].values
    lon = ds["lon"].values
    H, W = lat.shape
    out = []
    for name, date, la, lo, kind in EVENTS:
        d = dt.date.fromisoformat(date)
        dist = (lat - la) ** 2 + ((lon - lo) * np.cos(np.deg2rad(la))) ** 2
        cy, cx = np.unravel_index(np.argmin(dist), dist.shape)
        y0 = int(np.clip(cy - WIN // 2, 0, H - WIN))
        x0 = int(np.clip(cx - WIN // 2, 0, W - WIN))
        out.append(dict(name=name, date=date, year=d.year,
                        day_index=(d - dt.date(d.year, 1, 1)).days,
                        lat=la, lon=lo, kind=kind, cy=int(cy), cx=int(cx),
                        y0=y0, x0=x0, size=WIN))
    return out, lat, lon


if __name__ == "__main__":
    cat, lat, lon = build_catalog()
    out = Path("event_benchmark_output/data")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "grid_latlon.npz", lat=lat, lon=lon)
    (out / "event_catalog.json").write_text(json.dumps(cat, indent=2))
    for e in cat:
        print(e["name"], e["date"], "idx", e["day_index"], "win", e["y0"], e["x0"])
