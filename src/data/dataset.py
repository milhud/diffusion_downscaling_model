"""Dataset and DataModule for ERA5→CONUS404 downscaling.

Supports two modes:
  1. Cached (fast): loads pre-computed .npy files from cached_data/
  2. On-the-fly (slow): reads netCDF + regrids per sample (fallback)

Run preprocess_cache.py first to generate cached data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from pathlib import Path
from typing import Optional, List, Tuple

from src.preprocessing.normalization import (
    ERA5_VARS as DEFAULT_ERA5_VARS,
    CONUS404_VARS as DEFAULT_CONUS404_VARS,
    NormalizationStats,
    apply_pretransform, PRETRANSFORMS,
)
from src.preprocessing.regrid import ERA5Regridder
from src.preprocessing.land_mask import build_conus404_land_mask, get_valid_patch_origins


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


class CachedDownscalingDataset(Dataset):
    """Fast dataset that reads from pre-computed .npy cache files.

    Each sample loads a single day slice via numpy mmap, extracts a random
    land-only patch, fills non-land pixels, and normalizes.
    """

    def __init__(
        self,
        cache_dir: str,
        years: list,
        norm_stats: NormalizationStats,
        patch_size: int = 256,
        patches_per_day: int = 2,
        land_mask: Optional[np.ndarray] = None,
        valid_origins: Optional[List[Tuple[int, int]]] = None,
        era5_vars: Optional[list] = None,
        conus_vars: Optional[list] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.norm_stats = norm_stats
        self.patch_size = patch_size
        self.land_mask = land_mask
        self.valid_origins = valid_origins
        self.era5_vars = era5_vars or DEFAULT_ERA5_VARS
        self.conus_vars = conus_vars or DEFAULT_CONUS404_VARS

        # Load static fields
        self.static = np.load(self.cache_dir / "static_fields.npy")  # (6, H, W)

        # Memory-map all year files for fast random access
        self._era5_maps = {}
        self._conus_maps = {}
        for y in years:
            self._era5_maps[y] = np.load(
                self.cache_dir / f"era5_{y}.npy", mmap_mode="r")
            self._conus_maps[y] = np.load(
                self.cache_dir / f"conus_{y}.npy", mmap_mode="r")

        # Build index: (year, day_index), skipping fully-NaN days.
        # Bad days pre-computed by preprocess_cache.py → nan_days.json
        import json
        nan_days_path = self.cache_dir / "nan_days.json"
        if nan_days_path.exists():
            with open(nan_days_path) as f:
                nan_days = {int(k): set(v) for k, v in json.load(f).items()}
        else:
            nan_days = {}

        self.index = []
        skipped = 0
        for y in years:
            n_days = self._era5_maps[y].shape[0]
            bad = nan_days.get(y, set())
            for d in range(n_days):
                if d in bad:
                    skipped += 1
                    continue
                for _ in range(patches_per_day):
                    self.index.append((y, d))
        if skipped > 0:
            print(f"[Data] Skipped {skipped} fully-NaN days (from nan_days.json)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        year, day_idx = self.index[idx]

        # Read single day from mmap (very fast — no netCDF, no regrid)
        era5_day = self._era5_maps[year][day_idx].copy()    # (n_vars, H, W)
        conus_day = self._conus_maps[year][day_idx].copy()  # (n_vars, H, W)

        # Concatenate ERA5 + static fields
        era5_input = np.concatenate([era5_day, self.static], axis=0)  # (n_vars+6, H, W)

        # Random patch extraction
        ps = self.patch_size
        if self.valid_origins is not None and len(self.valid_origins) > 0:
            idx_origin = np.random.randint(0, len(self.valid_origins))
            y0, x0 = self.valid_origins[idx_origin]
        else:
            H, W = era5_input.shape[1], era5_input.shape[2]
            y0 = np.random.randint(0, H - ps)
            x0 = np.random.randint(0, W - ps)

        era5_patch = era5_input[:, y0:y0+ps, x0:x0+ps].copy()
        conus_patch = conus_day[:, y0:y0+ps, x0:x0+ps].copy()

        # Fill non-land pixels with per-channel patch mean
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

        # Normalize only ERA5 variable channels (static fields already z-scored)
        n_era5 = len(self.era5_vars)
        era5_vars_patch = era5_patch[:n_era5].unsqueeze(0)
        era5_vars_patch = self.norm_stats.normalize_era5(era5_vars_patch).squeeze(0)
        era5_patch = torch.cat([era5_vars_patch, era5_patch[n_era5:]], dim=0)

        conus_patch = self.norm_stats.normalize_conus(conus_patch.unsqueeze(0)).squeeze(0)

        return era5_patch, conus_patch


class DownscalingDataset(Dataset):
    """On-the-fly dataset that reads netCDF and regrids per sample.

    Caches opened xarray dataset handles and static fields to avoid
    re-opening files every sample. ~2-3x slower than cached .npy but
    requires no disk space for preprocessed data.
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

        # Load pre-computed NaN days (from cached_data/nan_days.json)
        import json
        nan_days_path = Path("cached_data/nan_days.json")
        if nan_days_path.exists():
            with open(nan_days_path) as f:
                nan_days = {int(k): set(v) for k, v in json.load(f).items()}
            print(f"[Data] Loaded NaN days from {nan_days_path}")
        else:
            nan_days = {}
            print("[Data] No nan_days.json found, not skipping any days")

        self.index = []
        skipped = 0
        for y in years:
            n_days = 366 if _is_leap(y) else 365
            bad = nan_days.get(y, set())
            for d in range(n_days):
                if d in bad:
                    skipped += 1
                    continue
                for _ in range(patches_per_day):
                    self.index.append((y, d))
        if skipped > 0:
            print(f"[Data] Skipped {skipped} fully-NaN days total")

        self._static_fields = None
        # Per-worker caches for opened datasets (lazy init in __getitem__)
        self._era5_cache = {}
        self._conus_cache = {}

    def _get_era5_ds(self, year: int) -> xr.Dataset:
        """Get cached xarray dataset handle for ERA5 year."""
        if year not in self._era5_cache:
            self._era5_cache[year] = xr.open_dataset(
                self.data_dir / f"era5_{year}.nc")
        return self._era5_cache[year]

    def _get_conus_ds(self, year: int) -> xr.Dataset:
        """Get cached xarray dataset handle for CONUS404 year."""
        if year not in self._conus_cache:
            self._conus_cache[year] = xr.open_dataset(
                self.data_dir / f"conus404_yearly_{year}.nc")
        return self._conus_cache[year]

    def _get_static_fields(self, conus_ds: xr.Dataset) -> np.ndarray:
        if self._static_fields is not None:
            return self._static_fields

        lat = self.conus_lat
        lon = self.conus_lon
        lat_norm = (lat - lat.mean()) / (lat.std() + 1e-8)
        lon_norm = (lon - lon.mean()) / (lon.std() + 1e-8)

        z = conus_ds["Z"].isel(time=0, bottom_top_stag=0).values
        z_norm = (z - z.mean()) / (z.std() + 1e-8)

        lai = conus_ds["LAI"].isel(time=0).values
        lai = np.nan_to_num(lai, nan=0.0)
        lai_norm = (lai - lai.mean()) / (lai.std() + 1e-8)

        from scipy.ndimage import uniform_filter
        z_mean = uniform_filter(z, size=5)
        z_sq_mean = uniform_filter(z ** 2, size=5)
        orog_var = np.sqrt(np.maximum(z_sq_mean - z_mean ** 2, 0))
        orog_var_norm = (orog_var - orog_var.mean()) / (orog_var.std() + 1e-8)

        lsm = (z > 0).astype(np.float32)

        self._static_fields = np.stack(
            [z_norm, orog_var_norm, lat_norm, lon_norm, lai_norm, lsm], axis=0
        ).astype(np.float32)
        return self._static_fields

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        year, day_idx = self.index[idx]
        leap = _is_leap(year)
        month_idx = _month_for_day(day_idx + 1, leap)

        era5_ds = self._get_era5_ds(year)
        era5_day = {}
        for var in self.era5_vars:
            raw = era5_ds[var].isel(time=month_idx, valid_time=day_idx).values
            era5_day[var] = apply_pretransform(raw.astype(np.float32), var)

        era5_stack = np.stack([era5_day[v] for v in self.era5_vars], axis=0)
        if self.regridder is not None:
            era5_regridded = self.regridder.regrid_batch(era5_stack)
        else:
            raise RuntimeError("Regridder not initialized")

        conus_ds = self._get_conus_ds(year)
        conus_day = {}
        for var in self.conus_vars:
            raw = conus_ds[var].isel(time=day_idx).values
            conus_day[var] = apply_pretransform(raw.astype(np.float32), var)
        static = self._get_static_fields(conus_ds)

        conus_stack = np.stack([conus_day[v] for v in self.conus_vars], axis=0)
        era5_input = np.concatenate([era5_regridded, static], axis=0)

        ps = self.patch_size
        if self.valid_origins is not None and len(self.valid_origins) > 0:
            idx_origin = np.random.randint(0, len(self.valid_origins))
            y0, x0 = self.valid_origins[idx_origin]
        else:
            H, W = 1015, 1367
            y0 = np.random.randint(0, H - ps)
            x0 = np.random.randint(0, W - ps)

        era5_patch = era5_input[:, y0:y0+ps, x0:x0+ps].copy()
        conus_patch = conus_stack[:, y0:y0+ps, x0:x0+ps].copy()

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

        n_era5 = len(self.era5_vars)
        era5_vars_patch = era5_patch[:n_era5].unsqueeze(0)
        era5_vars_patch = self.norm_stats.normalize_era5(era5_vars_patch).squeeze(0)
        era5_patch = torch.cat([era5_vars_patch, era5_patch[n_era5:]], dim=0)

        conus_patch = self.norm_stats.normalize_conus(conus_patch.unsqueeze(0)).squeeze(0)

        return era5_patch, conus_patch


def build_dataloaders(
    data_dir: str,
    norm_stats: NormalizationStats,
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
    cache_dir: str = None,
    # Legacy params (ignored when using cache)
    regridder: ERA5Regridder = None,
    conus_lat: np.ndarray = None,
    conus_lon: np.ndarray = None,
):
    """Build train and val DataLoaders.

    Uses CachedDownscalingDataset if cache_dir exists, otherwise falls back
    to on-the-fly DownscalingDataset.
    """
    if train_years is None:
        train_years = list(range(1980, 2015))
    if val_years is None:
        val_years = list(range(2015, 2018))

    use_cache = (cache_dir is not None
                 and Path(cache_dir).exists()
                 and (Path(cache_dir) / "static_fields.npy").exists())

    if use_cache:
        print(f"[Data] Using cached data from {cache_dir}")
        DatasetClass = CachedDownscalingDataset
        common_kwargs = dict(
            cache_dir=cache_dir,
            norm_stats=norm_stats,
            patch_size=patch_size,
            land_mask=land_mask,
            valid_origins=valid_origins,
            era5_vars=era5_vars,
            conus_vars=conus_vars,
        )
        train_ds = DatasetClass(years=train_years, patches_per_day=patches_per_day,
                                **common_kwargs)
        val_ds = DatasetClass(years=val_years, patches_per_day=1, **common_kwargs)
    else:
        print(f"[Data] WARNING: No cache found at {cache_dir}, using slow on-the-fly loading")
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
