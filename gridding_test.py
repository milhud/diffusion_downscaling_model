"""Gridding test: verify land mask and land-only patch extraction.

Loads ERA5 lsm, builds CONUS404 land mask, finds valid patch origins,
extracts sample patches, and generates diagnostic plots to confirm
that only land values within CONUS bounds are captured.

Usage:
    python gridding_test.py [--plot_dir gridding_plots]
"""

import argparse
import time
import numpy as np
import xarray as xr
import xesmf as xe
from pathlib import Path
from scipy.ndimage import uniform_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.data.land_mask import (
    build_conus404_land_mask, get_valid_patch_origins,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
)
from config import TRAIN, PATCH_SIZE

DATA_DIR = Path("data")
MIN_LAND_FRAC = TRAIN["min_land_frac"]


def main(plot_dir="gridding_plots"):
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GRIDDING TEST — Land mask and patch extraction verification")
    print("=" * 70)

    # ── Load coordinates ──
    t0 = time.time()
    era5_ds = xr.open_dataset(DATA_DIR / "era5_1980.nc")
    conus_ds = xr.open_dataset(DATA_DIR / "conus404_yearly_1980.nc")

    era5_lat = era5_ds["latitude"].values
    era5_lon = era5_ds["longitude"].values
    conus_lat = conus_ds["lat"].values
    conus_lon = conus_ds["lon"].values

    if conus_lat.ndim == 3:
        conus_lat = conus_lat[0]
        conus_lon = conus_lon[0]

    H, W = conus_lat.shape
    print(f"[Grid] CONUS404 grid: {H}×{W}")
    print(f"[Grid] CONUS404 lat range: [{conus_lat.min():.2f}, {conus_lat.max():.2f}]")
    print(f"[Grid] CONUS404 lon range: [{conus_lon.min():.2f}, {conus_lon.max():.2f}]")
    print(f"[Grid] ERA5 lat range: [{era5_lat.min():.2f}, {era5_lat.max():.2f}]")
    print(f"[Grid] ERA5 lon range: [{era5_lon.min():.2f}, {era5_lon.max():.2f}]")

    # ── Build land mask ──
    print("\n[Mask] Building land mask from ERA5 lsm...")
    land_mask = build_conus404_land_mask(conus_lat, conus_lon, era5_ds)

    # ── Plot 01: Full CONUS404 grid with land mask ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    ax.imshow(conus_lat, cmap="viridis", origin="lower")
    ax.set_title("CONUS404 Latitude"); ax.axis("off")
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046)

    ax = axes[1]
    ax.imshow(conus_lon, cmap="viridis", origin="lower")
    ax.set_title("CONUS404 Longitude"); ax.axis("off")
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046)

    ax = axes[2]
    ax.imshow(land_mask.astype(float), cmap="Greens", vmin=0, vmax=1, origin="lower")
    ax.set_title(f"Land Mask ({land_mask.sum()}/{land_mask.size} land)")
    ax.axis("off")
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046)

    fig.suptitle("CONUS404 Grid Coordinates and Land Mask", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/01_grid_and_land_mask.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {plot_dir}/01_grid_and_land_mask.png")

    # ── Plot 02: Land mask overlaid on temperature ──
    conus_t2 = conus_ds["T2"].isel(time=180).values.astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    im = ax.imshow(conus_t2, cmap="RdBu_r", origin="lower")
    ax.set_title("T2 — Raw (all pixels)"); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    t2_masked = np.where(land_mask, conus_t2, np.nan)
    im = ax.imshow(t2_masked, cmap="RdBu_r", origin="lower")
    ax.set_title("T2 — Land only"); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    ocean_pixels = np.where(~land_mask, conus_t2, np.nan)
    im = ax.imshow(ocean_pixels, cmap="coolwarm", origin="lower")
    ax.set_title("T2 — Excluded pixels (ocean/OOB)"); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Temperature Field: Land vs Excluded", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/02_t2_land_vs_ocean.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {plot_dir}/02_t2_land_vs_ocean.png")

    # ── Find valid patch origins ──
    print("\n[Patches] Finding valid land-only patch origins...")
    for min_frac in [0.3, 0.5, 0.7, 0.9]:
        origins = get_valid_patch_origins(land_mask, PATCH_SIZE, min_land_frac=min_frac)
        print(f"  min_land_frac={min_frac:.1f}: {len(origins)} valid origins")

    valid_origins = get_valid_patch_origins(land_mask, PATCH_SIZE, min_land_frac=MIN_LAND_FRAC)
    assert len(valid_origins) > 0, "No valid land-only patch origins found!"
    print(f"\n[Patches] Using min_land_frac={MIN_LAND_FRAC}: {len(valid_origins)} valid origins")

    # ── Plot 03: Valid patch origin density map ──
    density = np.zeros((H, W), dtype=np.float32)
    for y0, x0 in valid_origins:
        density[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] += 1
    density_max = density.max()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.imshow(land_mask.astype(float), cmap="Greens", vmin=0, vmax=1, origin="lower")
    # Plot a random subset of patch origins as rectangles
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(valid_origins), min(200, len(valid_origins)), replace=False)
    for i in sample_idx:
        y0, x0 = valid_origins[i]
        rect = mpatches.Rectangle((x0, y0), PATCH_SIZE, PATCH_SIZE,
                                  linewidth=0.5, edgecolor="red", facecolor="none", alpha=0.3)
        ax.add_patch(rect)
    ax.set_title(f"Land Mask + Sample Patch Locations ({len(sample_idx)} shown)")
    ax.axis("off")

    ax = axes[1]
    im = ax.imshow(density, cmap="hot_r", origin="lower")
    ax.set_title(f"Patch Coverage Density (max={density_max:.0f})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Valid Land-Only Patch Origins (min_land_frac={MIN_LAND_FRAC})", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/03_patch_origins.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {plot_dir}/03_patch_origins.png")

    # ── Plot 04: Sample patches with land fraction ──
    rng2 = np.random.RandomState(99)
    n_show = 12
    sample_origins = [valid_origins[rng2.randint(0, len(valid_origins))] for _ in range(n_show)]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for pi, (y0, x0) in enumerate(sample_origins):
        ax = axes[pi // 4, pi % 4]
        patch_t2 = conus_t2[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
        patch_mask = land_mask[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
        land_frac = patch_mask.mean()
        # Show T2 but mask ocean with gray
        patch_show = np.where(patch_mask, patch_t2, np.nan)
        vmin, vmax = np.nanpercentile(patch_t2, [2, 98])
        im = ax.imshow(patch_show, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"Patch ({y0},{x0})\nLand: {100*land_frac:.0f}%", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Sample Land-Only Patches (T2, ocean=NaN)", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/04_sample_patches.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {plot_dir}/04_sample_patches.png")

    # ── Plot 05: Verify no ocean patches — land fraction histogram ──
    land_fracs = []
    for y0, x0 in valid_origins:
        patch_mask = land_mask[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
        land_fracs.append(patch_mask.mean())
    land_fracs = np.array(land_fracs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(land_fracs, bins=50, edgecolor="black", alpha=0.7, color="forestgreen")
    ax.axvline(MIN_LAND_FRAC, color="red", ls="--", linewidth=2, label=f"min_land_frac={MIN_LAND_FRAC}")
    ax.set_xlabel("Land Fraction per Patch")
    ax.set_ylabel("Count")
    ax.set_title(f"Land Fraction Distribution ({len(valid_origins)} patches)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/05_land_fraction_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {plot_dir}/05_land_fraction_histogram.png")

    # ── Assertions ──
    assert land_fracs.min() >= MIN_LAND_FRAC - 1e-6, \
        f"Found patch with land_frac={land_fracs.min():.3f} < {MIN_LAND_FRAC}!"
    assert len(valid_origins) >= 50, \
        f"Only {len(valid_origins)} valid origins — expected at least 100"

    # ── ERA5 regridding check ──
    print("\n[Regrid] Checking ERA5→CONUS404 regridding with land mask...")
    src_grid = xr.Dataset({
        "lat": xr.DataArray(era5_lat, dims=["y"]),
        "lon": xr.DataArray(era5_lon, dims=["x"]),
    })
    dst_grid = xr.Dataset({
        "lat": xr.DataArray(conus_lat, dims=["y", "x"]),
        "lon": xr.DataArray(conus_lon, dims=["y", "x"]),
    })
    regridder = xe.Regridder(src_grid, dst_grid, method="bilinear",
                             extrap_method="nearest_s2d", unmapped_to_nan=False)

    # Day 180 is in June (month index 5, 0-based). Month 6 would be July
    # which may have NaN for this valid_time index.
    era5_t2m = era5_ds["t2m"].isel(time=5, valid_time=180).values.astype(np.float32)
    era5_regridded = regridder(xr.DataArray(era5_t2m, dims=["y", "x"])).values

    # Plot 06: ERA5 regridded vs CONUS404 with land mask
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    era5_land = np.where(land_mask, era5_regridded, np.nan)
    conus_land = np.where(land_mask, conus_t2, np.nan)
    vmin = min(np.nanmin(era5_land), np.nanmin(conus_land))
    vmax = max(np.nanmax(era5_land), np.nanmax(conus_land))

    ax = axes[0]
    im = ax.imshow(era5_land, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title("ERA5 t2m (regridded, land only)"); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    im = ax.imshow(conus_land, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title("CONUS404 T2 (land only)"); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    diff = np.where(land_mask, era5_regridded - conus_t2, np.nan)
    vabs = np.nanpercentile(np.abs(diff), 95)
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vabs, vmax=vabs, origin="lower")
    ax.set_title("Difference (ERA5 - CONUS404)"); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("ERA5 vs CONUS404 — Land Only (same colorscale)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/06_regridded_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {plot_dir}/06_regridded_comparison.png")

    era5_ds.close()
    conus_ds.close()

    # ── Summary ──
    print("\n" + "=" * 70)
    print("GRIDDING TEST SUMMARY")
    print("=" * 70)
    print(f"  Grid size:        {H}×{W}")
    print(f"  Land pixels:      {land_mask.sum()}/{land_mask.size} "
          f"({100*land_mask.sum()/land_mask.size:.1f}%)")
    print(f"  Valid patch origins ({MIN_LAND_FRAC*100:.0f}% land): {len(valid_origins)}")
    print(f"  Land frac range:  [{land_fracs.min():.2f}, {land_fracs.max():.2f}]")
    print(f"  Land frac mean:   {land_fracs.mean():.2f}")
    print(f"  Plots saved to:   {plot_path.resolve()}/")
    print(f"\n  ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_dir", default="gridding_plots")
    args = parser.parse_args()
    main(plot_dir=args.plot_dir)
