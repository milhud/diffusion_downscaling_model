"""Per-variable analysis of VAE latent space.

Visualizes what each latent channel encodes: spatial patterns,
variable correlations, and channel specialization.

Usage:
    python -m src.evaluation.latent_analysis --output_dir results/latent_analysis
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from config import (
    ERA5_VARS, CONUS404_VARS, IN_CH, OUT_CH, PATCH_SIZE,
    LATENT_CH, MODEL, TRAIN, VARIABLE_NAMES,
)
from src.models.drn import DRN
from src.models.vae import VAE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze_latent_space(drn, vae, test_dl, var_names, output_dir,
                         device="cuda", max_batches=30):
    """Analyze what each VAE latent channel encodes."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    drn.eval()
    vae.eval()

    # Collect latent statistics
    latent_means = []
    latent_stds = []
    latent_maps = []
    residual_maps = []

    with torch.no_grad():
        for batch_idx, (era5, conus) in enumerate(test_dl):
            if batch_idx >= max_batches:
                break
            era5, conus = era5.to(device), conus.to(device)
            drn_pred = drn(era5)
            residual = conus - drn_pred
            mu, logvar = vae.encode(residual)

            latent_means.append(mu.cpu())
            latent_stds.append((0.5 * logvar).exp().cpu())
            if batch_idx < 3:
                latent_maps.append(mu[0].cpu())
                residual_maps.append(residual[0].cpu())

    all_mu = torch.cat(latent_means, dim=0)  # (N, LATENT_CH, 64, 64)
    all_std = torch.cat(latent_stds, dim=0)

    # 1. Per-channel activation statistics
    fig, axes = plt.subplots(2, LATENT_CH // 2 if LATENT_CH > 1 else 1,
                             figsize=(3 * LATENT_CH, 6))
    axes = np.atleast_1d(axes).flatten()
    for ch in range(min(LATENT_CH, len(axes))):
        ch_data = all_mu[:, ch].flatten().numpy()
        axes[ch].hist(ch_data, bins=100, density=True, alpha=0.7, color="#2196F3")
        axes[ch].set_title(f"Latent ch {ch}\nmean={ch_data.mean():.3f}, std={ch_data.std():.3f}")
        axes[ch].set_xlabel("Activation")
    fig.suptitle("Per-Channel Latent Activation Distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "latent_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Spatial activation maps for a single sample
    for sample_idx in range(min(3, len(latent_maps))):
        mu_sample = latent_maps[sample_idx]  # (LATENT_CH, 64, 64)
        res_sample = residual_maps[sample_idx]  # (OUT_CH, 256, 256)

        n_cols = max(LATENT_CH, len(var_names))
        fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))

        # Top row: latent channels
        for ch in range(LATENT_CH):
            if ch < n_cols:
                im = axes[0, ch].imshow(mu_sample[ch].numpy(), cmap="RdBu_r")
                axes[0, ch].set_title(f"Latent ch {ch}")
                axes[0, ch].axis("off")
                plt.colorbar(im, ax=axes[0, ch], fraction=0.046)
        for ch in range(LATENT_CH, n_cols):
            axes[0, ch].axis("off")

        # Bottom row: residual channels
        for vi in range(len(var_names)):
            if vi < n_cols:
                im = axes[1, vi].imshow(res_sample[vi].numpy(), cmap="RdBu_r")
                axes[1, vi].set_title(f"Residual: {var_names[vi]}")
                axes[1, vi].axis("off")
                plt.colorbar(im, ax=axes[1, vi], fraction=0.046)
        for vi in range(len(var_names), n_cols):
            axes[1, vi].axis("off")

        fig.suptitle(f"Latent Channels vs Residual Fields (Sample {sample_idx})", y=1.02)
        fig.tight_layout()
        fig.savefig(out / f"latent_spatial_sample{sample_idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. Channel correlation matrix
    # Correlation between latent channels across all samples
    flat = all_mu.reshape(all_mu.shape[0], LATENT_CH, -1)  # (N, C, 64*64)
    channel_means = flat.mean(dim=(0, 2)).numpy()  # (C,)
    # Compute pairwise correlation
    corr_matrix = np.zeros((LATENT_CH, LATENT_CH))
    for i in range(LATENT_CH):
        for j in range(LATENT_CH):
            ci = flat[:, i].flatten().numpy()
            cj = flat[:, j].flatten().numpy()
            corr_matrix[i, j] = np.corrcoef(ci[:10000], cj[:10000])[0, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(LATENT_CH))
    ax.set_yticks(range(LATENT_CH))
    ax.set_xlabel("Latent Channel")
    ax.set_ylabel("Latent Channel")
    ax.set_title("Inter-Channel Correlation Matrix")
    plt.colorbar(im, ax=ax)
    for i in range(LATENT_CH):
        for j in range(LATENT_CH):
            ax.text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "latent_correlation_matrix.png", dpi=150)
    plt.close(fig)

    # 4. Mean uncertainty (std) per channel
    mean_std_per_ch = all_std.mean(dim=(0, 2, 3)).numpy()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(LATENT_CH), mean_std_per_ch, color="#FF9800")
    ax.set_xlabel("Latent Channel")
    ax.set_ylabel("Mean Posterior Std")
    ax.set_title("VAE Posterior Uncertainty by Channel")
    ax.set_xticks(range(LATENT_CH))
    fig.tight_layout()
    fig.savefig(out / "latent_uncertainty.png", dpi=150)
    plt.close(fig)

    print(f"[LatentAnalysis] Saved to {out}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/latent_analysis")
    parser.add_argument("--max_batches", type=int, default=30)
    parser.add_argument("--drn_checkpoint", default="checkpoints/drn_best.pt")
    parser.add_argument("--vae_checkpoint", default="checkpoints/vae_best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="cached_data")
    args = parser.parse_args()

    from src.preprocessing.normalization import NormalizationStats
    from src.preprocessing.land_mask import build_conus404_land_mask, get_valid_patch_origins
    from src.preprocessing.regrid import ERA5Regridder
    from src.data.dataset import build_dataloaders
    import xarray as xr

    stats = NormalizationStats()
    stats.load("norm_stats.npz")

    with xr.open_dataset(f"{args.data_dir}/era5_1980.nc") as ds:
        era5_lat, era5_lon = ds["latitude"].values, ds["longitude"].values
        land_mask = build_conus404_land_mask(
            xr.open_dataset(f"{args.data_dir}/conus404_yearly_1980.nc")["lat"].values,
            xr.open_dataset(f"{args.data_dir}/conus404_yearly_1980.nc")["lon"].values, ds)
    with xr.open_dataset(f"{args.data_dir}/conus404_yearly_1980.nc") as ds:
        conus_lat, conus_lon = ds["lat"].values, ds["lon"].values

    regridder = ERA5Regridder(era5_lat, era5_lon, conus_lat, conus_lon)
    valid_origins = get_valid_patch_origins(land_mask, PATCH_SIZE, TRAIN["min_land_frac"])

    _, test_dl = build_dataloaders(
        args.data_dir, stats, batch_size=4, patches_per_day=1, num_workers=2,
        train_years=TRAIN["train_years"], val_years=TRAIN["test_years"],
        land_mask=land_mask, valid_origins=valid_origins,
        era5_vars=ERA5_VARS, conus_vars=CONUS404_VARS,
        cache_dir=args.cache_dir, regridder=regridder,
        conus_lat=conus_lat, conus_lon=conus_lon)

    drn = DRN(in_ch=IN_CH, out_ch=OUT_CH, base_ch=MODEL["drn_base_ch"],
              ch_mults=MODEL["drn_ch_mults"], num_res_blocks=MODEL["drn_num_res_blocks"],
              attn_resolutions=MODEL["drn_attn_resolutions"])
    drn.load_state_dict(torch.load(args.drn_checkpoint)["model_state_dict"])

    vae = VAE(in_ch=OUT_CH, latent_ch=LATENT_CH, base_ch=MODEL["vae_base_ch"])
    vae.load_state_dict(torch.load(args.vae_checkpoint)["model_state_dict"])

    analyze_latent_space(drn, vae, test_dl, CONUS404_VARS, args.output_dir,
                         device=args.device, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
