"""Entry point for training: torchrun train.py --stage [drn|vae|diffusion]"""

import argparse
import torch
import numpy as np
import xarray as xr
from pathlib import Path

from src.data.normalization import NormalizationStats, ERA5_VARS, CONUS404_VARS, apply_pretransform
from src.data.regrid import ERA5Regridder
from src.data.dataset import build_dataloaders
from src.models.drn import DRN
from src.models.vae import VAE
from src.models.diffusion_unet import DiffusionUNet
from src.training.train_drn import train_drn
from src.training.train_vae import train_vae
from src.training.train_diffusion import train_diffusion


def compute_norm_stats(data_dir: str, years: list, cache_path: str = "norm_stats.npz"):
    """Compute normalization statistics from training years."""
    cache = Path(cache_path)
    stats = NormalizationStats()
    if cache.exists():
        print(f"Loading cached norm stats from {cache}")
        stats.load(str(cache))
        return stats

    print(f"Computing normalization stats from {len(years)} years...")
    era5_accum = {v: [] for v in ERA5_VARS}
    conus_accum = {v: [] for v in CONUS404_VARS}

    for y in years:
        # Sample a subset of days per year for efficiency
        sample_days = list(range(0, 366, 30))  # ~12 days/year

        with xr.open_dataset(f"{data_dir}/era5_{y}.nc") as ds:
            for d in sample_days:
                try:
                    for m in range(12):
                        for var in ERA5_VARS:
                            vals = ds[var].isel(time=m, valid_time=d).values.astype(np.float32)
                            if not np.all(np.isnan(vals)):
                                era5_accum[var].append(apply_pretransform(vals.flatten()[::100], var))
                                break
                except (IndexError, ValueError):
                    continue

        with xr.open_dataset(f"{data_dir}/conus404_yearly_{y}.nc") as ds:
            for d in sample_days:
                try:
                    for var in CONUS404_VARS:
                        vals = ds[var].isel(time=d).values.astype(np.float32)
                        conus_accum[var].append(apply_pretransform(vals.flatten()[::100], var))
                except (IndexError, ValueError):
                    continue

    era5_samples = {v: np.concatenate(era5_accum[v]) for v in ERA5_VARS}
    conus_samples = {v: np.concatenate(conus_accum[v]) for v in CONUS404_VARS}

    stats.compute_from_data(era5_samples, conus_samples)
    stats.save(str(cache))
    print(f"Saved norm stats to {cache}")
    return stats


def setup_regridder(data_dir: str):
    """Build ERA5→CONUS404 regridder from coordinate arrays."""
    with xr.open_dataset(f"{data_dir}/era5_1980.nc") as ds:
        era5_lat = ds["latitude"].values
        era5_lon = ds["longitude"].values

    with xr.open_dataset(f"{data_dir}/conus404_yearly_1980.nc") as ds:
        conus_lat = ds["lat"].values
        conus_lon = ds["lon"].values

    print("Building ERA5→CONUS404 regridder...")
    regridder = ERA5Regridder(era5_lat, era5_lon, conus_lat, conus_lon)
    return regridder, conus_lat, conus_lon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True, choices=["drn", "vae", "diffusion"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--drn_checkpoint", type=str, default="checkpoints/drn_best.pt")
    parser.add_argument("--vae_checkpoint", type=str, default="checkpoints/vae_best.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train_years = list(range(1980, 2015))
    val_years = list(range(2015, 2018))

    stats = compute_norm_stats(args.data_dir, train_years)
    regridder, conus_lat, conus_lon = setup_regridder(args.data_dir)

    train_dl, val_dl = build_dataloaders(
        args.data_dir, stats, regridder, conus_lat, conus_lon,
        batch_size=args.batch_size, train_years=train_years, val_years=val_years,
    )

    if args.stage == "drn":
        model = DRN(in_ch=13, out_ch=7)
        epochs = args.epochs or 100
        train_drn(model, train_dl, val_dl, epochs=epochs, lr=args.lr,
                  warmup_epochs=5, device=args.device,
                  checkpoint_dir=args.checkpoint_dir)

    elif args.stage == "vae":
        drn = DRN(in_ch=13, out_ch=7)
        drn.load_state_dict(torch.load(args.drn_checkpoint)["model_state_dict"])

        vae = VAE(in_ch=7, latent_ch=8)
        epochs = args.epochs or 50
        train_vae(vae, drn, train_dl, val_dl, epochs=epochs, lr=args.lr,
                  beta_max=1e-3, beta_anneal_frac=0.3, warmup_epochs=3,
                  device=args.device, checkpoint_dir=args.checkpoint_dir)

    elif args.stage == "diffusion":
        drn = DRN(in_ch=13, out_ch=7)
        drn.load_state_dict(torch.load(args.drn_checkpoint)["model_state_dict"])

        vae = VAE(in_ch=7, latent_ch=8)
        vae.load_state_dict(torch.load(args.vae_checkpoint)["model_state_dict"])

        diff_model = DiffusionUNet(in_ch=25, out_ch=8)
        epochs = args.epochs or 100
        train_diffusion(diff_model, drn, vae, train_dl, val_dl, epochs=epochs,
                        lr=2e-4, warmup_epochs=5, device=args.device,
                        checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
