"""Land mask utilities — ERA5 lsm projected onto CONUS404 grid.

Builds a boolean mask (True = land within CONUS bounds) by interpolating
ERA5's fractional land-sea mask onto the CONUS404 curvilinear grid.
Ocean, Great Lakes, and pixels outside US bounds are masked out.
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# US CONUS bounds (same as config.py)
LAT_MIN, LAT_MAX = 24, 50
LON_MIN, LON_MAX = -125, -66

# Module-level cache
_CACHE = {}


def get_era5_land_mask(era5_ds):
    """Extract ERA5 land-sea mask (lsm >= 0.5 = land), trimmed to CONUS bounds.

    Returns:
        land_bool: 2D boolean array (True = land), or None if lsm missing
        lat_e: 1D latitude array
        lon_e: 1D longitude array
    """
    if "lsm" not in era5_ds:
        return None, None, None
    lsm_da = era5_ds["lsm"]
    for dim in ["time", "valid_time"]:
        if dim in lsm_da.dims:
            lsm_da = lsm_da.isel({dim: 0})
    lsm_sub = lsm_da.sel(
        latitude=slice(LAT_MAX, LAT_MIN),
        longitude=slice(LON_MIN, LON_MAX),
    )
    lat_e = lsm_sub["latitude"].values
    lon_e = lsm_sub["longitude"].values
    land_bool = lsm_sub.values >= 0.5
    return land_bool, lat_e, lon_e


def build_conus404_land_mask(conus_lat2d, conus_lon2d, era5_ds):
    """Build a land mask for CONUS404 by nearest-neighbor interpolation of ERA5 lsm.

    Also masks pixels outside US CONUS bounds (lat 24-50, lon -125 to -66).
    Result is cached by grid shape.

    Args:
        conus_lat2d: (H, W) 2D latitude array
        conus_lon2d: (H, W) 2D longitude array
        era5_ds: xarray Dataset with 'lsm' variable

    Returns:
        2D boolean array (True = land within CONUS bounds)
    """
    shape_key = conus_lat2d.shape
    if shape_key in _CACHE:
        return _CACHE[shape_key]

    land_bool, lat_e, lon_e = get_era5_land_mask(era5_ds)
    if land_bool is None:
        print("[LandMask] WARNING: ERA5 lsm not found; using all-land fallback")
        mask = np.ones(conus_lat2d.shape, dtype=bool)
        _CACHE[shape_key] = mask
        return mask

    # ERA5 latitude is descending (north→south); flip for interpolator
    if lat_e[0] > lat_e[-1]:
        lat_e = lat_e[::-1]
        land_bool = land_bool[::-1, :]

    interp = RegularGridInterpolator(
        (lat_e, lon_e),
        land_bool.astype(np.float32),
        method="nearest",
        bounds_error=False,
        fill_value=0.0,  # outside ERA5 domain → not land
    )

    result = interp(np.column_stack([conus_lat2d.ravel(), conus_lon2d.ravel()]))
    mask = (result >= 0.5).reshape(conus_lat2d.shape)

    # Also restrict to CONUS bounds
    bounds_mask = (
        (conus_lat2d >= LAT_MIN) & (conus_lat2d <= LAT_MAX)
        & (conus_lon2d >= LON_MIN) & (conus_lon2d <= LON_MAX)
    )
    mask = mask & bounds_mask

    excluded = int((~mask).sum())
    total = mask.size
    print(f"[LandMask] CONUS404 land mask: {mask.sum()}/{total} pixels "
          f"({100 * mask.sum() / total:.1f}% land within CONUS bounds), "
          f"{excluded} excluded")
    _CACHE[shape_key] = mask
    return mask


def get_valid_patch_origins(land_mask, patch_size=256, min_land_frac=0.5):
    """Find all (y0, x0) patch origins where land fraction >= min_land_frac.

    Uses a sliding-window land fraction computed via integral images for speed.

    Returns:
        List of (y0, x0) tuples
    """
    H, W = land_mask.shape
    ps = patch_size
    if H < ps or W < ps:
        return []

    # Integral image for fast patch-sum computation
    integral = np.cumsum(np.cumsum(land_mask.astype(np.float64), axis=0), axis=1)
    patch_area = ps * ps

    origins = []
    for y0 in range(0, H - ps + 1, ps // 4):  # stride ps//4 for density
        for x0 in range(0, W - ps + 1, ps // 4):
            y1, x1 = y0 + ps, x0 + ps
            # Sum of land pixels in patch via integral image
            s = integral[y1 - 1, x1 - 1]
            if y0 > 0:
                s -= integral[y0 - 1, x1 - 1]
            if x0 > 0:
                s -= integral[y1 - 1, x0 - 1]
            if y0 > 0 and x0 > 0:
                s += integral[y0 - 1, x0 - 1]
            if s / patch_area >= min_land_frac:
                origins.append((y0, x0))

    return origins
