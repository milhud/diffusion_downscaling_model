"""Plot ERA5 (regridded) and CONUS404 side-by-side, both masked to land only."""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import VARIABLE_PAIRS, VARIABLE_NAMES, VARIABLE_UNITS
from src.preprocessing.regrid import ERA5Regridder
from src.preprocessing.normalization import apply_pretransform
from src.preprocessing.land_mask import build_conus404_land_mask

data_dir = Path("data")
out_dir = Path("diagnostic_plots")
out_dir.mkdir(exist_ok=True)

# Load grids
era5_ds = xr.open_dataset(data_dir / "era5_1980.nc")
conus_ds = xr.open_dataset(data_dir / "conus404_yearly_1980.nc")

era5_lat = era5_ds["latitude"].values
era5_lon = era5_ds["longitude"].values
conus_lat = conus_ds["lat"].values
conus_lon = conus_ds["lon"].values

# Build regridder and land mask
print("Building regridder...")
regridder = ERA5Regridder(era5_lat, era5_lon, conus_lat, conus_lon)
land_mask = build_conus404_land_mask(conus_lat, conus_lon, era5_ds)
print(f"Land mask: {land_mask.sum()}/{land_mask.size} pixels")

# Read day 0 data for all variables
for e5_var, c4_var in VARIABLE_PAIRS.items():
    print(f"Plotting {e5_var} / {c4_var}...")

    # ERA5: read, pretransform, regrid
    raw_era5 = era5_ds[e5_var].isel(time=0, valid_time=0).values.astype(np.float32)
    era5_val = apply_pretransform(raw_era5, e5_var)
    era5_rg = regridder.regrid(era5_val)

    # CONUS404: read, pretransform
    raw_conus = conus_ds[c4_var].isel(time=0).values.astype(np.float32)
    conus_val = apply_pretransform(raw_conus, c4_var)

    # Mask to land only
    era5_land = np.where(land_mask, era5_rg, np.nan)
    conus_land = np.where(land_mask, conus_val, np.nan)

    # Shared color range from both
    vmin = min(np.nanpercentile(era5_land, 2), np.nanpercentile(conus_land, 2))
    vmax = max(np.nanpercentile(era5_land, 98), np.nanpercentile(conus_land, 98))

    label = VARIABLE_NAMES.get(e5_var, e5_var)
    unit = VARIABLE_UNITS.get(e5_var, "")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"{label} ({unit}) — Land Only", fontsize=14)

    im0 = axes[0].imshow(era5_land, origin="lower", aspect="auto",
                          vmin=vmin, vmax=vmax)
    axes[0].set_title(f"ERA5 regridded ({e5_var})")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(conus_land, origin="lower", aspect="auto",
                          vmin=vmin, vmax=vmax)
    axes[1].set_title(f"CONUS404 ({c4_var})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(out_dir / f"land_only_{e5_var}_{c4_var}.png", dpi=120)
    plt.close()
    print(f"  Saved land_only_{e5_var}_{c4_var}.png")

era5_ds.close()
conus_ds.close()
print("Done.")
