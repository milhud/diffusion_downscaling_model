"""Diagnose data loading speed and verify regridding with plots."""
import time
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ERA5_VARS, CONUS404_VARS, VARIABLE_PAIRS
from src.preprocessing.regrid import ERA5Regridder
from src.preprocessing.normalization import apply_pretransform
from src.preprocessing.land_mask import build_conus404_land_mask

data_dir = Path("data")
out_dir = Path("diagnostic_plots")
out_dir.mkdir(exist_ok=True)

# --- 1. Time each step of on-the-fly loading ---
print("=" * 60)
print("TIMING BREAKDOWN: single sample, 7 variables")
print("=" * 60)

# Open datasets
t0 = time.time()
era5_ds = xr.open_dataset(data_dir / "era5_1980.nc")
t_open_era5 = time.time() - t0

t0 = time.time()
conus_ds = xr.open_dataset(data_dir / "conus404_yearly_1980.nc")
t_open_conus = time.time() - t0

print(f"Open ERA5 netCDF:   {t_open_era5:.3f}s")
print(f"Open CONUS404 netCDF: {t_open_conus:.3f}s")
print("(These are cached after first open per worker)")

# Read ERA5 data for one day (7 vars)
t0 = time.time()
era5_day = {}
for var in ERA5_VARS:
    raw = era5_ds[var].isel(time=0, valid_time=0).values
    era5_day[var] = apply_pretransform(raw.astype(np.float32), var)
era5_stack = np.stack([era5_day[v] for v in ERA5_VARS], axis=0)
t_read_era5 = time.time() - t0
print(f"Read 7 ERA5 vars:   {t_read_era5:.3f}s  shape={era5_stack.shape}")

# Read ERA5 data for one day (1 var only)
t0 = time.time()
raw = era5_ds["t2m"].isel(time=0, valid_time=0).values.astype(np.float32)
t_read_era5_1 = time.time() - t0
print(f"Read 1 ERA5 var:    {t_read_era5_1:.3f}s")

# Build regridder
era5_lat = era5_ds["latitude"].values
era5_lon = era5_ds["longitude"].values
conus_lat = conus_ds["lat"].values
conus_lon = conus_ds["lon"].values

t0 = time.time()
regridder = ERA5Regridder(era5_lat, era5_lon, conus_lat, conus_lon)
t_regrid_init = time.time() - t0
print(f"Init regridder:     {t_regrid_init:.3f}s (one-time)")

# Regrid 7 vars
t0 = time.time()
era5_regridded = regridder.regrid_batch(era5_stack)
t_regrid_7 = time.time() - t0
print(f"Regrid 7 vars:      {t_regrid_7:.3f}s  shape={era5_regridded.shape}")

# Regrid 1 var
t0 = time.time()
era5_regridded_1 = regridder.regrid_batch(era5_stack[:1])
t_regrid_1 = time.time() - t0
print(f"Regrid 1 var:       {t_regrid_1:.3f}s")

# Read CONUS404 data (7 vars)
t0 = time.time()
conus_day = {}
for var in CONUS404_VARS:
    raw = conus_ds[var].isel(time=0).values
    conus_day[var] = apply_pretransform(raw.astype(np.float32), var)
conus_stack = np.stack([conus_day[v] for v in CONUS404_VARS], axis=0)
t_read_conus = time.time() - t0
print(f"Read 7 CONUS vars:  {t_read_conus:.3f}s  shape={conus_stack.shape}")

# Read CONUS404 data (1 var)
t0 = time.time()
raw = conus_ds["T2"].isel(time=0).values.astype(np.float32)
t_read_conus_1 = time.time() - t0
print(f"Read 1 CONUS var:   {t_read_conus_1:.3f}s")

print()
total_7 = t_read_era5 + t_regrid_7 + t_read_conus
total_1 = t_read_era5_1 + t_regrid_1 + t_read_conus_1
print(f"TOTAL per sample (7 vars): {total_7:.3f}s")
print(f"TOTAL per sample (1 var):  {total_1:.3f}s")
print(f"Ratio 7var/1var:           {total_7/total_1:.1f}x")
print(f"Estimated batch (bs=8):    {total_7 * 8 / 4:.1f}s (with 4 workers)")
print(f"Estimated epoch (25k samples): {total_7 * 25000 / 4 / 3600:.1f}h (with 4 workers)")

# --- 2. Regridding verification plots ---
print("\n" + "=" * 60)
print("GENERATING REGRIDDING VERIFICATION PLOTS")
print("=" * 60)

# Build land mask
land_mask = build_conus404_land_mask(conus_lat, conus_lon, era5_ds)

# Plot each variable: ERA5 native, ERA5 regridded, CONUS404, difference
for i, (e5_var, c4_var) in enumerate(VARIABLE_PAIRS.items()):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{e5_var} (ERA5) → {c4_var} (CONUS404)", fontsize=14)

    # ERA5 native
    era5_native = era5_day[e5_var]
    im0 = axes[0, 0].imshow(era5_native, origin="lower", aspect="auto")
    axes[0, 0].set_title(f"ERA5 native ({era5_native.shape})")
    plt.colorbar(im0, ax=axes[0, 0])

    # ERA5 regridded
    era5_rg = era5_regridded[i]
    im1 = axes[0, 1].imshow(era5_rg, origin="lower", aspect="auto")
    axes[0, 1].set_title(f"ERA5 regridded ({era5_rg.shape})")
    plt.colorbar(im1, ax=axes[0, 1])

    # CONUS404
    conus_val = conus_stack[i]
    im2 = axes[1, 0].imshow(conus_val, origin="lower", aspect="auto")
    axes[1, 0].set_title(f"CONUS404 ({conus_val.shape})")
    plt.colorbar(im2, ax=axes[1, 0])

    # Difference (ERA5 regridded - CONUS404) masked to land
    diff = era5_rg - conus_val
    diff_masked = np.where(land_mask, diff, np.nan)
    vmax = np.nanpercentile(np.abs(diff_masked), 95)
    im3 = axes[1, 1].imshow(diff_masked, origin="lower", aspect="auto",
                              cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title(f"Difference (regridded - target), land only")
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(out_dir / f"regrid_{e5_var}_{c4_var}.png", dpi=100)
    plt.close()
    print(f"  Saved regrid_{e5_var}_{c4_var}.png")

# Land mask plot
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].imshow(land_mask, origin="lower", aspect="auto", cmap="Greens")
axes[0].set_title(f"Land mask ({land_mask.shape}), {land_mask.sum()}/{land_mask.size} land pixels")

# Overlay ERA5 regridded T2m with land mask boundary
t2m_rg = era5_regridded[0]
t2m_masked = np.where(land_mask, t2m_rg, np.nan)
axes[1].imshow(t2m_masked, origin="lower", aspect="auto")
axes[1].set_title("ERA5 T2m regridded (land only)")

plt.tight_layout()
plt.savefig(out_dir / "land_mask_verification.png", dpi=100)
plt.close()
print("  Saved land_mask_verification.png")

era5_ds.close()
conus_ds.close()
print("\nDone! Plots in diagnostic_plots/")
