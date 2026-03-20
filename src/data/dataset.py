"""Dataset and DataModule for ERA5→CONUS404 downscaling."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from pathlib import Path
from typing import Optional, List, Tuple

from .normalization import (
    ERA5_VARS as DEFAULT_ERA5_VARS,
    CONUS404_VARS as DEFAULT_CONUS404_VARS,
    NormalizationStats,
    apply_pretransform, PRETRANSFORMS,
)
from .regrid import ERA5Regridder
from .land_mask import build_conus404_land_mask, get_valid_patch_origins


def _month_for_day(day_of_year: int, leap: bool = False) -> int:
    """Return 0-indexed month for a 1-indexed day of year."""
    days_in_month = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cum = 0
    for m, d in enumerate(days_in_month):
        cum += d
        if day_of_year <= cum:
            return m
    return 11


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


class DownscalingDataset(Dataset):
    """Lazy-loading dataset that yields paired (ERA5, CONUS404) patches.

    Each sample is one day, one random 256×256 patch from land-only regions
    of the CONUS404 grid, with the corresponding regridded ERA5 data.
    """

    def __init__(
        self,
        data_dir: str,
        years: list,
        norm_stats: NormalizationStats,
        patch_size: int = 256,
        patches_per_day: int = 4,
        regridder: Optional[ERA5Regridder] = None,
        conus_lat: Optional[np.ndarray] = None,
        conus_lon: Optional[np.ndarray] = None,
        land_mask: Optional[np.ndarray] = None,
        valid_origins: Optional[List[Tuple[int, int]]] = None,
        era5_vars: Optional[list] = None,
        conus_vars: Optional[list] = None,
    ):
        self.data_dir = Path(data_dir)
        self.years = years
        self.norm_stats = norm_stats
        self.patch_size = patch_size
        self.patches_per_day = patches_per_day
        self.regridder = regridder
        self.conus_lat = conus_lat
        self.conus_lon = conus_lon
        self.land_mask = land_mask
        self.valid_origins = valid_origins
        self.era5_vars = era5_vars or DEFAULT_ERA5_VARS
        self.conus_vars = conus_vars or DEFAULT_CONUS404_VARS

        # Build index: (year, day_index) for all valid days
        self.index = []
        for y in years:
            n_days = 366 if _is_leap(y) else 365
            for d in range(n_days):
                for _ in range(patches_per_day):
                    self.index.append((y, d))

        # Static fields (computed lazily)
        self._static_fields = None

    def _get_static_fields(self, conus_ds: xr.Dataset) -> np.ndarray:
        """Build static fields: terrain_height, lat_norm, lon_norm, lsm-proxy, lai-proxy, elevation-variance.

        Returns (6, 1015, 1367) array.
        """
        if self._static_fields is not None:
            return self._static_fields

        lat = self.conus_lat  # (1015, 1367)
        lon = self.conus_lon
        # Normalize to [-1, 1]
        lat_norm = (lat - lat.mean()) / (lat.std() + 1e-8)
        lon_norm = (lon - lon.mean()) / (lon.std() + 1e-8)

        # Terrain height from Z variable (level 0, first time step)
        z = conus_ds["Z"].isel(time=0, bottom_top_stag=0).values  # (1015, 1367)
        z_norm = (z - z.mean()) / (z.std() + 1e-8)

        # LAI from first time step
        lai = conus_ds["LAI"].isel(time=0).values
        lai_norm = (lai - lai.mean()) / (lai.std() + 1e-8)

        # Orographic variance (local std of terrain in 5×5 window, simplified)
        from scipy.ndimage import uniform_filter
        z_mean = uniform_filter(z, size=5)
        z_sq_mean = uniform_filter(z ** 2, size=5)
        orog_var = np.sqrt(np.maximum(z_sq_mean - z_mean ** 2, 0))
        orog_var_norm = (orog_var - orog_var.mean()) / (orog_var.std() + 1e-8)

        # Land mask proxy (Z > some threshold indicates land; simplified)
        lsm = (z > 0).astype(np.float32)

        self._static_fields = np.stack([z_norm, orog_var_norm, lat_norm, lon_norm, lai_norm, lsm], axis=0).astype(np.float32)
        return self._static_fields

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        year, day_idx = self.index[idx]
        leap = _is_leap(year)
        month_idx = _month_for_day(day_idx + 1, leap)

        # Load ERA5 data for this day
        era5_path = self.data_dir / f"era5_{year}.nc"
        with xr.open_dataset(era5_path) as ds:
            era5_day = {}
            for var in self.era5_vars:
                raw = ds[var].isel(time=month_idx, valid_time=day_idx).values  # (111, 235)
                era5_day[var] = apply_pretransform(raw.astype(np.float32), var)

        # Stack ERA5 channels and regrid to CONUS404 grid
        era5_stack = np.stack([era5_day[v] for v in self.era5_vars], axis=0)
        if self.regridder is not None:
            era5_regridded = self.regridder.regrid_batch(era5_stack)  # (7, 1015, 1367)
        else:
            raise RuntimeError("Regridder not initialized")

        # Load CONUS404 data for this day
        conus_path = self.data_dir / f"conus404_yearly_{year}.nc"
        with xr.open_dataset(conus_path) as ds:
            conus_day = {}
            for var in self.conus_vars:
                raw = ds[var].isel(time=day_idx).values  # (1015, 1367)
                conus_day[var] = apply_pretransform(raw.astype(np.float32), var)
            static = self._get_static_fields(ds)

        # Stack CONUS404 channels
        conus_stack = np.stack([conus_day[v] for v in self.conus_vars], axis=0)

        # Concatenate ERA5 regridded + static fields for input
        era5_input = np.concatenate([era5_regridded, static], axis=0)  # (13, 1015, 1367)

        # Random patch extraction — land-only if valid_origins provided
        H, W = 1015, 1367
        ps = self.patch_size
        if self.valid_origins is not None and len(self.valid_origins) > 0:
            idx_origin = np.random.randint(0, len(self.valid_origins))
            y0, x0 = self.valid_origins[idx_origin]
        else:
            y0 = np.random.randint(0, H - ps)
            x0 = np.random.randint(0, W - ps)

        era5_patch = era5_input[:, y0:y0+ps, x0:x0+ps].copy()
        conus_patch = conus_stack[:, y0:y0+ps, x0:x0+ps].copy()

        # Fill non-land pixels with per-channel patch mean so they don't
        # distort gradients (patches are >=80% land but not always 100%)
        if self.land_mask is not None:
            patch_mask = self.land_mask[y0:y0+ps, x0:x0+ps]
            if not patch_mask.all():
                for c in range(era5_patch.shape[0]):
                    ch = era5_patch[c]
                    ch[~patch_mask] = ch[patch_mask].mean()
                for c in range(conus_patch.shape[0]):
                    ch = conus_patch[c]
                    ch[~patch_mask] = ch[patch_mask].mean()

        era5_patch = torch.from_numpy(era5_patch)
        conus_patch = torch.from_numpy(conus_patch)

        # Normalize
        era5_patch = self.norm_stats.normalize_era5(era5_patch.unsqueeze(0)).squeeze(0)
        conus_patch = self.norm_stats.normalize_conus(conus_patch.unsqueeze(0)).squeeze(0)

        return era5_patch, conus_patch


def build_dataloaders(
    data_dir: str,
    norm_stats: NormalizationStats,
    regridder: ERA5Regridder,
    conus_lat: np.ndarray,
    conus_lon: np.ndarray,
    batch_size: int = 16,
    patch_size: int = 256,
    patches_per_day: int = 4,
    num_workers: int = 4,
    train_years: list = None,
    val_years: list = None,
    land_mask: np.ndarray = None,
    valid_origins: list = None,
    era5_vars: list = None,
    conus_vars: list = None,
):
    """Build train and val DataLoaders with optional land-only patch extraction."""
    if train_years is None:
        train_years = list(range(1980, 2015))
    if val_years is None:
        val_years = list(range(2015, 2018))

    train_ds = DownscalingDataset(
        data_dir, train_years, norm_stats, patch_size,
        patches_per_day=patches_per_day,
        regridder=regridder, conus_lat=conus_lat, conus_lon=conus_lon,
        land_mask=land_mask, valid_origins=valid_origins,
        era5_vars=era5_vars, conus_vars=conus_vars,
    )
    val_ds = DownscalingDataset(
        data_dir, val_years, norm_stats, patch_size,
        patches_per_day=1, regridder=regridder,
        conus_lat=conus_lat, conus_lon=conus_lon,
        land_mask=land_mask, valid_origins=valid_origins,
        era5_vars=era5_vars, conus_vars=conus_vars,
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl
