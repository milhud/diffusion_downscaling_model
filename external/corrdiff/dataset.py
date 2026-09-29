"""CorrDiff `DownscalingDataset` adapter over our own ERA5->CONUS404 cache.

Wraps the same `cached_data/era5_{year}.npy` / `conus_{year}.npy` /
`static_fields.npy` cache used by `src.data.dataset.CachedDownscalingDataset`
(see that file + `src/preprocessing/cache_builder.py` for the on-disk format)
so CorrDiff trains on the *same* preprocessed data as our in-house Latent
CorrDiff model. Reuses `src.preprocessing.normalization.NormalizationStats`
for z-scoring rather than reimplementing it, so both models see identical
normalized inputs.

Registered with physicsnemo's `custom` dataset path as:
    dataset:
        type: <repo_root>/external/corrdiff/dataset.py::Era5Conus404Dataset

Differences from `CachedDownscalingDataset` (deliberate, for CorrDiff):
  - CorrDiff's `RandomPatching2D`/`GridPatching2D` do the patch cropping
    internally (via `training.hp.patch_shape_x/y`), so `__getitem__` here
    returns a larger random *crop* (default 768x768, see `crop_size`) rather
    than a final 256x256 training patch, giving the patcher room to sample
    diverse sub-patches per crop.
  - ERA5 is already regridded onto the CONUS404 grid at cache-build time, so
    `img_lr` is already "pre-upsampled" as CorrDiff's dataset contract
    requires -- no interpolation happens in this adapter.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Make the parent repo importable regardless of cwd (physicsnemo imports this
# file directly via importlib, not as part of a package).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.base import ChannelMetadata, DownscalingDataset  # noqa: E402

from src.preprocessing.normalization import NormalizationStats  # noqa: E402
from src.preprocessing.land_mask import get_valid_patch_origins  # noqa: E402


class Era5Conus404Dataset(DownscalingDataset):
    """ERA5 -> CONUS404 dataset, cache-backed, for CorrDiff training/generation.

    Parameters
    ----------
    cache_dir: str
        Directory with era5_{year}.npy / conus_{year}.npy / static_fields.npy
        (see src/preprocessing/cache_builder.py).
    norm_stats_path: str
        Path to norm_stats.npz (see src/preprocessing/normalization.py).
    years: list[int]
        Years to include (e.g. config.py TRAIN["train_years"]).
    input_variables: list[str]
        ERA5 variable names, e.g. config.ERA5_VARS. Must match the channel
        order the cache was built with.
    output_variables: list[str]
        CONUS404 variable names, e.g. config.CONUS404_VARS.
    invariant_variables: list[str] | None
        Names for the 6 static channels (terrain, orog_var, lat, lon, lai,
        lsm), purely for ChannelMetadata labeling -- order must match
        cache_builder.build_static_fields.
    crop_size: int
        Size of the random square crop returned per sample; CorrDiff's own
        patcher then draws `patch_num` sub-patches of `patch_shape_x/y` from
        this crop. Must be >= patch_shape_x/y used in the training config.
    min_land_frac: float
        Minimum land fraction for a crop origin to be considered valid
        (reuses the same land-mask logic as the in-house model).
    """

    _STATIC_NAMES = ["terrain", "orog_var", "lat", "lon", "lai", "lsm"]

    def __init__(
        self,
        cache_dir: str,
        norm_stats_path: str,
        years: List[int],
        input_variables: List[str],
        output_variables: List[str],
        invariant_variables: Optional[List[str]] = None,
        crop_size: int = 768,
        min_land_frac: float = 0.8,
    ):
        self.cache_dir = Path(cache_dir)
        self.years = list(years)
        self.input_variables = list(input_variables)
        self.output_variables = list(output_variables)
        self.invariant_variables = list(invariant_variables or self._STATIC_NAMES)
        self.crop_size = crop_size

        self.norm_stats = NormalizationStats()
        self.norm_stats.load(norm_stats_path)

        self.static = np.load(self.cache_dir / "static_fields.npy")  # (6, H, W)
        self._H, self._W = self.static.shape[-2:]

        self._era5_maps = {
            y: np.load(self.cache_dir / f"era5_{y}.npy", mmap_mode="r")
            for y in self.years
        }
        self._conus_maps = {
            y: np.load(self.cache_dir / f"conus_{y}.npy", mmap_mode="r")
            for y in self.years
        }

        import json

        nan_days_path = self.cache_dir / "nan_days.json"
        nan_days = {}
        if nan_days_path.exists():
            with open(nan_days_path) as f:
                nan_days = {int(k): set(v) for k, v in json.load(f).items()}

        self.index = []
        for y in self.years:
            n_days = self._era5_maps[y].shape[0]
            bad = nan_days.get(y, set())
            self.index.extend((y, d) for d in range(n_days) if d not in bad)

        # Land mask for valid crop origins: LSM is static channel index 5
        # (see cache_builder.build_static_fields channel order).
        land_mask = self.static[5] > 0.5
        self.valid_origins = get_valid_patch_origins(
            land_mask, crop_size, min_land_frac
        )
        if not self.valid_origins:
            # Fall back to unrestricted random cropping if the land mask is
            # too strict for this crop size.
            self.valid_origins = None

        self._img_shape = (crop_size, crop_size)

    def __len__(self):
        return len(self.index)

    def _origin(self, rng: np.random.Generator) -> Tuple[int, int]:
        if self.valid_origins:
            i = rng.integers(0, len(self.valid_origins))
            return self.valid_origins[i]
        y0 = rng.integers(0, self._H - self.crop_size)
        x0 = rng.integers(0, self._W - self.crop_size)
        return int(y0), int(x0)

    def __getitem__(self, idx):
        year, day_idx = self.index[idx]
        rng = np.random.default_rng()

        era5_day = self._era5_maps[year][day_idx]  # (n_in, H, W)
        conus_day = self._conus_maps[year][day_idx]  # (n_out, H, W)

        y0, x0 = self._origin(rng)
        cs = self.crop_size
        img_lr = np.concatenate(
            [
                era5_day[:, y0 : y0 + cs, x0 : x0 + cs],
                self.static[:, y0 : y0 + cs, x0 : x0 + cs],
            ],
            axis=0,
        ).astype(np.float32)
        img_clean = conus_day[:, y0 : y0 + cs, x0 : x0 + cs].astype(np.float32)

        img_lr = self.normalize_input(img_lr)
        img_clean = self.normalize_output(img_clean)

        return img_clean, img_lr

    # --- normalization: only the ERA5-var channels get z-scored; the 6
    # static channels are already normalized once at cache-build time
    # (cache_builder.build_static_fields). ---

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        import torch

        n = len(self.input_variables)
        vars_t = torch.from_numpy(x[:n]).unsqueeze(0)
        vars_t = self.norm_stats.normalize_era5(vars_t).squeeze(0).numpy()
        return np.concatenate([vars_t, x[n:]], axis=0)

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        import torch

        n = len(self.input_variables)
        vars_t = torch.from_numpy(x[:n]).unsqueeze(0)
        vars_t = self.norm_stats.denormalize_era5(vars_t).squeeze(0).numpy()
        return np.concatenate([vars_t, x[n:]], axis=0)

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        import torch

        t = torch.from_numpy(x).unsqueeze(0)
        return self.norm_stats.normalize_conus(t).squeeze(0).numpy()

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        import torch

        t = torch.from_numpy(x).unsqueeze(0)
        return self.norm_stats.denormalize_conus(t).squeeze(0).numpy()

    # --- DownscalingDataset metadata contract ---

    def longitude(self) -> np.ndarray:
        return np.full(self._img_shape, np.nan, dtype=np.float32)

    def latitude(self) -> np.ndarray:
        return np.full(self._img_shape, np.nan, dtype=np.float32)

    def input_channels(self) -> List[ChannelMetadata]:
        inputs = [ChannelMetadata(name=v) for v in self.input_variables]
        invariants = [
            ChannelMetadata(name=v, auxiliary=True) for v in self.invariant_variables
        ]
        return inputs + invariants

    def output_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=v) for v in self.output_variables]

    def time(self) -> List:
        return [f"{y}-day{d:03d}" for y, d in self.index]

    def image_shape(self) -> Tuple[int, int]:
        return self._img_shape

    def info(self) -> dict:
        return {
            "cache_dir": str(self.cache_dir),
            "years": self.years,
            "n_samples": len(self.index),
            "input_variables": self.input_variables,
            "output_variables": self.output_variables,
            "invariant_variables": self.invariant_variables,
            "crop_size": self.crop_size,
        }
