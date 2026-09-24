"""Ablation: Compare VAE compression ratios (LDM-2, LDM-4, LDM-8).

Trains three VAE variants with different spatial compression factors,
then evaluates reconstruction quality and downstream diffusion performance.

Usage:
    python -m src.evaluation.ablation_compression --ratio 2   # LDM-2: 128x128 latent
    python -m src.evaluation.ablation_compression --ratio 4   # LDM-4: 64x64 latent (default)
    python -m src.evaluation.ablation_compression --ratio 8   # LDM-8: 32x32 latent
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from config import IN_CH, OUT_CH, PATCH_SIZE, MODEL, TRAIN, ERA5_VARS, CONUS404_VARS
from src.models.drn import DRN
from src.models.vae import VAE
from src.training.train_vae import train_vae

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RATIO_CONFIG = {
    2: {"latent_h": 128, "latent_w": 128, "latent_ch": 4, "label": "LDM-2"},
    4: {"latent_h": 64, "latent_w": 64, "latent_ch": 4, "label": "LDM-4"},
    8: {"latent_h": 32, "latent_w": 32, "latent_ch": 8, "label": "LDM-8"},
}


def run_compression_ablation(ratio, drn, train_dl, val_dl, device, checkpoint_dir, plot_dir):
    """Train a VAE with the specified compression ratio and evaluate."""
    cfg = RATIO_CONFIG[ratio]
    ckpt_dir = Path(checkpoint_dir) / f"ldm_{ratio}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    # Determine number of downsample levels from ratio
    import math
    num_levels = int(math.log2(ratio))

    vae = VAE(in_ch=OUT_CH, latent_ch=cfg["latent_ch"], base_ch=MODEL["vae_base_ch"])
    print(f"[Ablation LDM-{ratio}] Latent: {cfg['latent_h']}x{cfg['latent_w']}x{cfg['latent_ch']}")
    print(f"[Ablation LDM-{ratio}] VAE params: {sum(p.numel() for p in vae.parameters()):,}")

    vae = train_vae(
        vae, drn, train_dl, val_dl,
        epochs=TRAIN["vae_epochs"],
        lr=TRAIN["vae_lr"],
        beta_max=TRAIN["vae_beta_max"],
        beta_anneal_frac=TRAIN["vae_beta_anneal_frac"],
        warmup_epochs=TRAIN["vae_warmup_epochs"],
        device=device,
        checkpoint_dir=str(ckpt_dir),
    )

    # Evaluate reconstruction quality
    vae.eval()
    drn.eval()
    recon_errors = []

    with torch.no_grad():
        for era5, conus in val_dl:
            era5, conus = era5.to(device), conus.to(device)
            drn_pred = drn(era5)
            residual = conus - drn_pred
            mu, logvar = vae.encode(residual)
            z = mu  # deterministic for eval
            recon = vae.decode(z)
            full_pred = drn_pred + recon
            rmse = torch.sqrt(((full_pred - conus) ** 2).mean()).item()
            recon_errors.append(rmse)

    mean_rmse = np.mean(recon_errors)
    print(f"[Ablation LDM-{ratio}] Mean DRN+VAE RMSE: {mean_rmse:.4f}")

    # Save results
    np.savez(plot_path / f"ablation_ldm{ratio}.npz",
             ratio=ratio, mean_rmse=mean_rmse,
             recon_errors=np.array(recon_errors),
             latent_h=cfg["latent_h"], latent_w=cfg["latent_w"],
             latent_ch=cfg["latent_ch"])

    return mean_rmse


def plot_compression_comparison(plot_dir):
    """Compare all compression ratios (run after all three are done)."""
    plot_path = Path(plot_dir)
    ratios, rmses, labels = [], [], []

    for r in [2, 4, 8]:
        f = plot_path / f"ablation_ldm{r}.npz"
        if f.exists():
            data = np.load(f)
            ratios.append(r)
            rmses.append(float(data["mean_rmse"]))
            labels.append(f"LDM-{r}\n({int(data['latent_h'])}x{int(data['latent_w'])})")

    if len(ratios) < 2:
        print("Need at least 2 ratios to compare. Run more ablations first.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, rmses, color=["#4CAF50", "#2196F3", "#FF9800"])
    ax.set_ylabel("DRN+VAE RMSE (normalized)")
    ax.set_title("VAE Compression Ratio Ablation")
    for i, v in enumerate(rmses):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(plot_path / "ablation_compression_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {plot_path / 'ablation_compression_comparison.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=int, required=True, choices=[2, 4, 8])
    parser.add_argument("--drn_checkpoint", default="checkpoints/drn_best.pt")
    parser.add_argument("--checkpoint_dir", default="checkpoints/ablation_compression")
    parser.add_argument("--plot_dir", default="train_plots/ablation_compression")
    parser.add_argument("--compare_only", action="store_true",
                        help="Only plot comparison (skip training)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="cached_data")
    args = parser.parse_args()

    if args.compare_only:
        plot_compression_comparison(args.plot_dir)
        return

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

    train_dl, val_dl = build_dataloaders(
        args.data_dir, stats, batch_size=TRAIN["batch_size"],
        patches_per_day=TRAIN["patches_per_day"], num_workers=2,
        train_years=TRAIN["train_years"], val_years=TRAIN["val_years"],
        land_mask=land_mask, valid_origins=valid_origins,
        era5_vars=ERA5_VARS, conus_vars=CONUS404_VARS,
        cache_dir=args.cache_dir, regridder=regridder,
        conus_lat=conus_lat, conus_lon=conus_lon)

    drn = DRN(in_ch=IN_CH, out_ch=OUT_CH, base_ch=MODEL["drn_base_ch"],
              ch_mults=MODEL["drn_ch_mults"],
              num_res_blocks=MODEL["drn_num_res_blocks"],
              attn_resolutions=MODEL["drn_attn_resolutions"])
    drn.load_state_dict(torch.load(args.drn_checkpoint)["model_state_dict"])
    drn = drn.to(args.device).eval()

    run_compression_ablation(args.ratio, drn, train_dl, val_dl,
                             args.device, args.checkpoint_dir, args.plot_dir)


if __name__ == "__main__":
    main()
