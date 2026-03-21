"""Sample from trained DRN + VAE checkpoints and generate diagnostic plots.

Loads cached data, runs inference on a few validation samples, and saves
comparison plots showing ERA5 input → DRN prediction → CONUS404 target,
plus VAE encode/decode quality.

Usage:
    python sample_checkpoint.py [--num_samples 6] [--device cuda]
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
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


def load_models(checkpoint_dir, device):
    """Load DRN and VAE from checkpoints."""
    drn = DRN(in_ch=IN_CH, out_ch=OUT_CH, base_ch=MODEL["drn_base_ch"],
              ch_mults=MODEL["drn_ch_mults"],
              num_res_blocks=MODEL["drn_num_res_blocks"],
              attn_resolutions=MODEL["drn_attn_resolutions"])

    drn_ckpt = torch.load(f"{checkpoint_dir}/drn_best.pt", map_location=device, weights_only=True)
    drn.load_state_dict(drn_ckpt["model_state_dict"])
    drn = drn.to(device).eval()
    print(f"[DRN] Loaded epoch {drn_ckpt['epoch']+1}, val_loss={drn_ckpt['val_loss']:.6f}")

    vae = VAE(in_ch=OUT_CH, latent_ch=LATENT_CH, base_ch=MODEL["vae_base_ch"])
    vae_ckpt = torch.load(f"{checkpoint_dir}/vae_best.pt", map_location=device, weights_only=True)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae = vae.to(device).eval()
    print(f"[VAE] Loaded epoch {vae_ckpt['epoch']+1}, val_loss={vae_ckpt['val_loss']:.6f}")

    return drn, vae


def plot_comparison(panels, titles, save_path, suptitle="", share_groups=None, cmap="RdBu_r"):
    """Plot panels with shared colorbar scales."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    panels_np = [p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p for p in panels]

    if share_groups is not None:
        groups = {}
        for i, g in enumerate(share_groups):
            groups.setdefault(g, []).append(i)
        group_ranges = {}
        for g, idxs in groups.items():
            vals = np.concatenate([panels_np[i].flatten() for i in idxs])
            vals = vals[np.isfinite(vals)]
            group_ranges[g] = np.nanpercentile(vals, [2, 98]) if len(vals) > 0 else (0, 1)
    else:
        group_ranges = None

    for i, (ax, p, title) in enumerate(zip(axes, panels_np, titles)):
        if group_ranges is not None:
            vmin, vmax = group_ranges[share_groups[i]]
        else:
            finite = p[np.isfinite(p)]
            vmin, vmax = (np.nanpercentile(finite, [2, 98]) if len(finite) > 0 else (0, 1))
        im = ax.imshow(p, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=6)
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
        print("[WARN] CUDA not available, using CPU")

    # Load models
    drn, vae = load_models(args.checkpoint_dir, device)

    # Load norm stats
    norm = NormalizationStats()
    norm.load("norm_stats.npz")

    # Build land mask
    era5_ds = xr.open_dataset("data/era5_1980.nc")
    conus_ds = xr.open_dataset("data/conus404_yearly_1980.nc")
    conus_lat = conus_ds["lat"].values
    conus_lon = conus_ds["lon"].values
    land_mask = build_conus404_land_mask(conus_lat, conus_lon, era5_ds)
    valid_origins = get_valid_patch_origins(land_mask, PATCH_SIZE, TRAIN["min_land_frac"])
    era5_ds.close()
    conus_ds.close()

    # Load validation dataset
    val_ds = CachedDownscalingDataset(
        cache_dir=args.cache_dir,
        years=TRAIN["val_years"],
        norm_stats=norm,
        patch_size=PATCH_SIZE,
        patches_per_day=1,
        land_mask=land_mask,
        valid_origins=valid_origins,
        era5_vars=ERA5_VARS,
        conus_vars=CONUS404_VARS,
    )
    print(f"[Data] {len(val_ds)} validation samples")

    # Sample diverse indices (spread across the dataset)
    np.random.seed(42)
    indices = np.linspace(0, len(val_ds) - 1, args.num_samples, dtype=int)

    for i, idx in enumerate(indices):
        era5, conus = val_ds[idx]
        era5 = era5.unsqueeze(0).to(device)
        conus = conus.unsqueeze(0).to(device)

        with torch.no_grad():
            # DRN prediction
            drn_pred = drn(era5)
            drn_residual = conus - drn_pred

            # VAE encode/decode the target
            recon, mu, logvar = vae(conus)
            vae_residual = conus - recon

            # VAE encode/decode the DRN residual
            drn_resid_recon, mu_r, logvar_r = vae(drn_residual)

        # Denormalize for display
        conus_denorm = norm.denormalize_conus(conus.cpu())[0, 0]
        drn_denorm = norm.denormalize_conus(drn_pred.cpu())[0, 0]
        recon_denorm = norm.denormalize_conus(recon.cpu())[0, 0]

        year, day = val_ds.index[idx]
        title_prefix = f"Sample {i+1} (year={year}, day={day})"

        # Plot 1: DRN — ERA5 input (ch0) / target / prediction / error
        drn_error = drn_denorm - conus_denorm
        rmse = drn_error.pow(2).mean().sqrt().item()
        plot_comparison(
            [era5[0, 0].cpu(), conus_denorm, drn_denorm, drn_error],
            ["ERA5 input (t2m)", "CONUS404 target", f"DRN pred (RMSE={rmse:.2f}K)", "Error (K)"],
            f"{args.plot_dir}/sample_{i+1:02d}_drn.png",
            suptitle=f"{title_prefix} — DRN",
            share_groups=[0, 0, 0, 1],
        )

        # Plot 2: VAE — target / VAE reconstruction / VAE error
        vae_error = recon_denorm - conus_denorm
        vae_rmse = vae_error.pow(2).mean().sqrt().item()
        plot_comparison(
            [conus_denorm, recon_denorm, vae_error],
            ["Target", f"VAE recon (RMSE={vae_rmse:.2f}K)", "VAE error (K)"],
            f"{args.plot_dir}/sample_{i+1:02d}_vae.png",
            suptitle=f"{title_prefix} — VAE reconstruction",
            share_groups=[0, 0, 1],
        )

        # Plot 3: Full pipeline — target / DRN / DRN residual / VAE(residual)
        plot_comparison(
            [conus[0, 0].cpu(), drn_pred[0, 0].cpu(),
             drn_residual[0, 0].cpu(), drn_resid_recon[0, 0].cpu()],
            ["Target (norm)", "DRN (norm)", "Residual", "VAE(residual)"],
            f"{args.plot_dir}/sample_{i+1:02d}_pipeline.png",
            suptitle=f"{title_prefix} — Pipeline",
            share_groups=[0, 0, 1, 1],
        )

    # Summary plot: DRN RMSE across all samples
    print(f"\nAll plots saved to {args.plot_dir}/")


if __name__ == "__main__":
    main()
