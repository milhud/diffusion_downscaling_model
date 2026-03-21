"""In-memory normalization stats and transforms for ERA5 and CONUS404 variables."""

import numpy as np
import torch
from pathlib import Path
from typing import Optional


# Variable groups and their transforms
ERA5_VARS = ["t2m", "d2m", "u10", "v10", "sp", "tp", "z"]
CONUS404_VARS = ["T2", "TD2", "U10", "V10", "PSFC", "Q2", "PREC_ACC_NC"]

# Transforms applied before z-scoring
PRETRANSFORMS = {
    "tp": "log1p",
    "PREC_ACC_NC": "log1p",
    "Q2": "sqrt",
    "z": "geopotential",  # divide by g=9.81
}
GRAVITY = 9.80665


def apply_pretransform(data: np.ndarray, var_name: str) -> np.ndarray:
    """Apply variable-specific transform before z-scoring."""
    t = PRETRANSFORMS.get(var_name)
    if t == "log1p":
        return np.log1p(np.maximum(data, 0))
    elif t == "sqrt":
        return np.sqrt(np.maximum(data, 0))
    elif t == "geopotential":
        return data / GRAVITY
    return data


def invert_pretransform(data: np.ndarray, var_name: str) -> np.ndarray:
    """Invert the pretransform."""
    t = PRETRANSFORMS.get(var_name)
    if t == "log1p":
        return np.expm1(data)
    elif t == "sqrt":
        return data ** 2
    elif t == "geopotential":
        return data * GRAVITY
    return data


class NormalizationStats:
    """Compute and store per-variable mean/std for z-score normalization.

    All stats are computed in-memory from training data and stored as tensors.
    Optionally cached to norm_stats.npz for reuse.
    """

    def __init__(self):
        self.era5_mean: Optional[torch.Tensor] = None  # (num_era5_vars,)
        self.era5_std: Optional[torch.Tensor] = None
        self.conus_mean: Optional[torch.Tensor] = None  # (num_conus_vars,)
        self.conus_std: Optional[torch.Tensor] = None

    def compute_from_data(self, era5_samples: dict, conus_samples: dict):
        """Compute stats from dicts of {var_name: flat_array_of_values}.

        Each entry should be all training values for that variable,
        already pretransformed. Uses the dict keys directly (not module-level
        ERA5_VARS/CONUS404_VARS) so it works with any variable subset.
        """
        era5_means, era5_stds = [], []
        for var in era5_samples.keys():
            vals = era5_samples[var]
            era5_means.append(float(np.nanmean(vals)))
            era5_stds.append(float(np.nanstd(vals)))
        self.era5_mean = torch.tensor(era5_means, dtype=torch.float32)
        self.era5_std = torch.tensor(era5_stds, dtype=torch.float32).clamp(min=1e-8)

        conus_means, conus_stds = [], []
        for var in conus_samples.keys():
            vals = conus_samples[var]
            conus_means.append(float(np.nanmean(vals)))
            conus_stds.append(float(np.nanstd(vals)))
        self.conus_mean = torch.tensor(conus_means, dtype=torch.float32)
        self.conus_std = torch.tensor(conus_stds, dtype=torch.float32).clamp(min=1e-8)

    def save(self, path: str):
        np.savez(
            path,
            era5_mean=self.era5_mean.numpy(),
            era5_std=self.era5_std.numpy(),
            conus_mean=self.conus_mean.numpy(),
            conus_std=self.conus_std.numpy(),
        )

    def load(self, path: str):
        d = np.load(path)
        self.era5_mean = torch.from_numpy(d["era5_mean"]).float()
        self.era5_std = torch.from_numpy(d["era5_std"]).float()
        self.conus_mean = torch.from_numpy(d["conus_mean"]).float()
        self.conus_std = torch.from_numpy(d["conus_std"]).float()

    def normalize_era5(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ERA5 tensor (B, C, H, W) with per-channel z-score."""
        mean = self.era5_mean.to(x.device).view(1, -1, 1, 1)
        std = self.era5_std.to(x.device).view(1, -1, 1, 1)
        return (x - mean) / std

    def denormalize_era5(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.era5_mean.to(x.device).view(1, -1, 1, 1)
        std = self.era5_std.to(x.device).view(1, -1, 1, 1)
        return x * std + mean

    def normalize_conus(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize CONUS404 tensor (B, C, H, W) with per-channel z-score."""
        mean = self.conus_mean.to(x.device).view(1, -1, 1, 1)
        std = self.conus_std.to(x.device).view(1, -1, 1, 1)
        return (x - mean) / std

    def denormalize_conus(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.conus_mean.to(x.device).view(1, -1, 1, 1)
        std = self.conus_std.to(x.device).view(1, -1, 1, 1)
        return x * std + mean
