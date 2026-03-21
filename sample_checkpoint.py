"""Sample from trained DRN + VAE and generate pipeline comparison plots.

Shows: ERA5 input → CONUS404 target → DRN prediction → DRN+VAE reconstruction

Usage:
    python sample_checkpoint.py [--num_samples 5] [--device cpu]
"""

import argparse
import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    IN_CH, OUT_CH, LATENT_CH, PATCH_SIZE, MODEL,
    ERA5_VARS, CONUS404_VARS, TRAIN,
)
from src.models.drn import DRN
from src.models.vae import VAE
from src.data.normalization import NormalizationStats
from src.data.dataset import CachedDownscalingDataset
from src.data.land_mask import build_conus404_land_mask, get_valid_patch_origins

import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--cache_dir", type=str, default="cached_data")
    parser.add_argument("--plot_dir", type=str, default="sample_plots")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    plot_path = Path(args.plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load DRN
    drn = DRN(in_ch=IN_CH, out_ch=OUT_CH, base_ch=MODEL["drn_base_ch"],
              ch_mults=MODEL["drn_ch_mults"],
              num_res_blocks=MODEL["drn_num_res_blocks"],
              attn_resolutions=MODEL["drn_attn_resolutions"])
    drn_ckpt = torch.load(f"{args.checkpoint_dir}/drn_best.pt",
                          map_location=device, weights_only=True)
    drn.load_state_dict(drn_ckpt["model_state_dict"])
    drn = drn.to(device).eval()
    print(f"[DRN] Loaded epoch {drn_ckpt['epoch']+1}, val_loss={drn_ckpt['val_loss']:.6f}")

    # Load VAE
    vae = VAE(in_ch=OUT_CH, latent_ch=LATENT_CH, base_ch=MODEL["vae_base_ch"])
    vae_ckpt = torch.load(f"{args.checkpoint_dir}/vae_best.pt",
                          map_location=device, weights_only=True)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae = vae.to(device).eval()
    print(f"[VAE] Loaded epoch {vae_ckpt['epoch']+1}, val_loss={vae_ckpt['val_loss']:.6f}")

    # Norm stats + land mask
    norm = NormalizationStats()
    norm.load("norm_stats.npz")

    era5_ds = xr.open_dataset("data/era5_1980.nc")
    conus_ds = xr.open_dataset("data/conus404_yearly_1980.nc")
    land_mask = build_conus404_land_mask(
        conus_ds["lat"].values, conus_ds["lon"].values, era5_ds)
    valid_origins = get_valid_patch_origins(land_mask, PATCH_SIZE, TRAIN["min_land_frac"])
    era5_ds.close(); conus_ds.close()

    # Validation dataset
    val_ds = CachedDownscalingDataset(
        cache_dir=args.cache_dir, years=TRAIN["val_years"], norm_stats=norm,
        patch_size=PATCH_SIZE, patches_per_day=1,
        land_mask=land_mask, valid_origins=valid_origins,
        era5_vars=ERA5_VARS, conus_vars=CONUS404_VARS,
    )
    print(f"[Data] {len(val_ds)} validation samples")

    # Pick spread-out samples
    np.random.seed(42)
    indices = np.linspace(0, len(val_ds) - 1, args.num_samples, dtype=int)

    for i, idx in enumerate(indices):
        era5, conus = val_ds[idx]
        era5 = era5.unsqueeze(0).to(device)
        conus = conus.unsqueeze(0).to(device)

        with torch.no_grad():
            drn_pred = drn(era5)
            residual = conus - drn_pred
            resid_recon, _, _ = vae(residual)
            full_recon = drn_pred + resid_recon

        # Denormalize to Kelvin
        target_K = norm.denormalize_conus(conus.cpu())[0, 0].numpy()
        drn_K = norm.denormalize_conus(drn_pred.cpu())[0, 0].numpy()
        full_K = norm.denormalize_conus(full_recon.cpu())[0, 0].numpy()
        era5_norm = era5[0, 0].cpu().numpy()  # normalized, just for spatial structure

        drn_rmse = np.sqrt(((drn_K - target_K) ** 2).mean())
        full_rmse = np.sqrt(((full_K - target_K) ** 2).mean())

        year, day = val_ds.index[idx]

        # Shared scale across target/DRN/full
        vmin = np.nanpercentile(target_K, 2)
        vmax = np.nanpercentile(target_K, 98)
        err_abs = max(abs(np.nanpercentile(full_K - target_K, 2)),
                      abs(np.nanpercentile(full_K - target_K, 98)))

        fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))

        im0 = axes[0].imshow(era5_norm, cmap="RdBu_r", origin="lower")
        axes[0].set_title("ERA5 input (norm)")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(target_K, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        axes[1].set_title("CONUS404 target (K)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(drn_K, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        axes[2].set_title(f"DRN pred (RMSE={drn_rmse:.2f}K)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        im3 = axes[3].imshow(full_K, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        axes[3].set_title(f"DRN+VAE (RMSE={full_rmse:.2f}K)")
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        im4 = axes[4].imshow(full_K - target_K, cmap="bwr", vmin=-err_abs, vmax=err_abs,
                             origin="lower")
        axes[4].set_title("Error (K)")
        plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.axis("off")

        fig.suptitle(f"Sample {i+1} — year={year}, day={day}", fontsize=14, y=1.02)
        plt.tight_layout()
        save = f"{args.plot_dir}/pipeline_{i+1:02d}.png"
        fig.savefig(save, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{i+1}] DRN RMSE={drn_rmse:.2f}K, DRN+VAE RMSE={full_rmse:.2f}K -> {save}")

    print(f"\nDone — plots in {args.plot_dir}/")


if __name__ == "__main__":
    main()
