"""Shared setup helpers for evaluation scripts.

Avoids opening ERA5 .nc files (symlinks are broken after sduan data move).
Extracts land mask from cached static_fields.npy instead.
"""

import numpy as np
import torch
from pathlib import Path

from config import (
    ERA5_VARS, CONUS404_VARS, IN_CH, OUT_CH, PATCH_SIZE,
    LATENT_CH, MODEL, TRAIN,
)
from src.models.drn import DRN
from src.models.vae import VAE
from src.models.diffusion_unet import DiffusionUNet
from src.models.edm import EDMSchedule
from src.training.ema import EMA
from src.preprocessing.normalization import NormalizationStats
from src.preprocessing.land_mask import get_valid_patch_origins
from src.data.dataset import build_dataloaders


def load_land_mask_from_cache(cache_dir: str = "cached_data") -> np.ndarray:
    """Extract binary land mask from static_fields.npy channel 5 (lsm)."""
    static = np.load(Path(cache_dir) / "static_fields.npy")
    return (static[5] >= 0.5)


def build_test_dataloader(
    data_dir: str = "data",
    cache_dir: str = "cached_data",
    batch_size: int = 4,
    num_workers: int = 2,
    patches_per_day: int = 1,
):
    """Build test dataloader using cached .npy files only (no ERA5 .nc needed)."""
    stats = NormalizationStats()
    stats.load("norm_stats.npz")

    land_mask = load_land_mask_from_cache(cache_dir)
    valid_origins = get_valid_patch_origins(land_mask, PATCH_SIZE, TRAIN["min_land_frac"])

    _, test_dl = build_dataloaders(
        data_dir, stats, batch_size=batch_size,
        patches_per_day=patches_per_day, num_workers=num_workers,
        train_years=TRAIN["train_years"], val_years=TRAIN["test_years"],
        land_mask=land_mask, valid_origins=valid_origins,
        era5_vars=ERA5_VARS, conus_vars=CONUS404_VARS,
        cache_dir=cache_dir,
        regridder=None, conus_lat=None, conus_lon=None,
    )
    return test_dl, stats, land_mask, valid_origins


def load_models(
    drn_checkpoint: str = "checkpoints/drn_best.pt",
    vae_checkpoint: str = "checkpoints/vae_best.pt",
    diff_checkpoint: str = "checkpoints/diffusion_best.pt",
    device: str = "cuda",
):
    """Load DRN, VAE, and diffusion model from checkpoints."""
    drn = DRN(
        in_ch=IN_CH, out_ch=OUT_CH,
        base_ch=MODEL["drn_base_ch"], ch_mults=MODEL["drn_ch_mults"],
        num_res_blocks=MODEL["drn_num_res_blocks"],
        attn_resolutions=MODEL["drn_attn_resolutions"],
    )
    drn.load_state_dict(torch.load(drn_checkpoint, map_location=device)["model_state_dict"])
    drn = drn.to(device).eval()

    vae = VAE(in_ch=OUT_CH, latent_ch=LATENT_CH, base_ch=MODEL["vae_base_ch"])
    vae.load_state_dict(torch.load(vae_checkpoint, map_location=device)["model_state_dict"])
    vae = vae.to(device).eval()

    diff_in_ch = LATENT_CH + IN_CH + LATENT_CH + 2
    diff_model = DiffusionUNet(
        in_ch=diff_in_ch, out_ch=LATENT_CH,
        base_ch=MODEL["diff_base_ch"], ch_mults=MODEL["diff_ch_mults"],
        num_res_blocks=MODEL["diff_num_res_blocks"],
        attn_resolutions=MODEL["diff_attn_resolutions"],
        time_dim=MODEL["diff_time_dim"],
    )
    ckpt = torch.load(diff_checkpoint, map_location=device)
    diff_model.load_state_dict(ckpt["model_state_dict"])
    diff_model = diff_model.to(device).eval()

    ema = EMA(diff_model, decay=TRAIN["ema_decay"])
    if "ema_state_dict" in ckpt:
        ema.load_state_dict(ckpt["ema_state_dict"])

    schedule = EDMSchedule()
    return drn, vae, diff_model, ema, schedule
