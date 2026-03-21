"""Ablation: Train pixel-space CorrDiff (no VAE) for comparison.

Trains the diffusion model directly on 256x256 residuals instead of 64x64 latents.
This isolates the benefit of latent-space compression.

Usage:
    python -m src.evaluation.ablation_pixel --checkpoint_dir checkpoints/ablation_pixel
"""

import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from config import (
    ERA5_VARS, CONUS404_VARS, IN_CH, OUT_CH, PATCH_SIZE,
    MODEL, TRAIN, LATENT_H,
)
from src.models.drn import DRN
from src.models.diffusion_unet import DiffusionUNet
from src.models.edm import EDMSchedule, edm_training_loss, heun_sampler
from src.training.ema import EMA
from src.evaluation.plots import plot_loss_curves, radial_power_spectrum

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _make_pos_embedding(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy, xx], dim=0)


def _build_pixel_cond(era5, drn_pred, pos_emb, p_uncond=0.1):
    """Build conditioning in pixel space (no VAE encoding)."""
    B = era5.shape[0]
    pos = pos_emb.unsqueeze(0).expand(B, -1, -1, -1)
    cond = torch.cat([era5, drn_pred, pos], dim=1)
    if p_uncond > 0 and torch.rand(1).item() < p_uncond:
        cond = torch.zeros_like(cond)
    return cond


def train_pixel_diffusion(
    diff_model, drn, train_dl, val_dl,
    epochs=50, lr=2e-4, warmup_epochs=5,
    ema_decay=0.9999, p_uncond=0.1,
    device="cuda", checkpoint_dir="checkpoints/ablation_pixel",
    plot_dir="train_plots/ablation_pixel",
):
    """Train diffusion model in pixel space (256x256) on DRN residuals."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    drn = drn.to(device).eval()
    diff_model = diff_model.to(device)
    schedule = EDMSchedule()
    ema = EMA(diff_model, decay=ema_decay)
    optimizer = torch.optim.AdamW(diff_model.parameters(), lr=lr, weight_decay=1e-6)

    pos_emb = _make_pos_embedding(PATCH_SIZE, PATCH_SIZE, device)

    train_losses = []
    val_losses = []
    best_val = float("inf")
    timings = []

    for epoch in range(1, epochs + 1):
        diff_model.train()
        epoch_loss = []
        t0 = time.time()

        for step, (era5, conus) in enumerate(train_dl):
            era5, conus = era5.to(device), conus.to(device)
            with torch.no_grad():
                drn_pred = drn(era5)
            residual = conus - drn_pred  # (B, OUT_CH, 256, 256)

            cond = _build_pixel_cond(era5, drn_pred, pos_emb, p_uncond)
            loss = edm_training_loss(diff_model, residual, cond, schedule)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.0)
            optimizer.step()
            ema.update()

            epoch_loss.append(loss.item())
            train_losses.append(loss.item())

        epoch_time = time.time() - t0
        timings.append(epoch_time)

        # Validation
        diff_model.eval()
        val_loss_acc = []
        with torch.no_grad():
            for era5, conus in val_dl:
                era5, conus = era5.to(device), conus.to(device)
                drn_pred = drn(era5)
                residual = conus - drn_pred
                cond = _build_pixel_cond(era5, drn_pred, pos_emb, p_uncond=0)
                loss = edm_training_loss(diff_model, residual, cond, schedule)
                val_loss_acc.append(loss.item())

        val_loss = np.mean(val_loss_acc)
        val_losses.append(val_loss)

        print(f"  [PixelDiff] Epoch {epoch}, Train: {np.mean(epoch_loss):.4f}, "
              f"Val: {val_loss:.4f}, Time: {epoch_time:.0f}s")

        # Save checkpoints
        state = {
            "epoch": epoch,
            "model_state_dict": diff_model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(state, ckpt_dir / "pixel_diff_latest.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, ckpt_dir / "pixel_diff_best.pt")

        # Plot loss curves
        plot_loss_curves(
            {"Pixel Diff Train": train_losses},
            str(plot_path / "pixel_diff_loss.png"),
            smooth_window=50,
        )

    # Save timing data
    np.savez(plot_path / "pixel_diff_timings.npz",
             epoch_times=np.array(timings),
             train_losses=np.array(train_losses),
             val_losses=np.array(val_losses))

    return diff_model, ema, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drn_checkpoint", default="checkpoints/drn_best.pt")
    parser.add_argument("--checkpoint_dir", default="checkpoints/ablation_pixel")
    parser.add_argument("--plot_dir", default="train_plots/ablation_pixel")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="cached_data")
    args = parser.parse_args()

    from src.preprocessing.normalization import NormalizationStats
    from src.preprocessing.land_mask import build_conus404_land_mask, get_valid_patch_origins
    from src.preprocessing.regrid import ERA5Regridder
    from src.data.dataset import build_dataloaders
    import xarray as xr

    # Setup data
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

    # Load DRN
    drn = DRN(in_ch=IN_CH, out_ch=OUT_CH, base_ch=MODEL["drn_base_ch"],
              ch_mults=MODEL["drn_ch_mults"],
              num_res_blocks=MODEL["drn_num_res_blocks"],
              attn_resolutions=MODEL["drn_attn_resolutions"])
    drn.load_state_dict(torch.load(args.drn_checkpoint)["model_state_dict"])

    # Pixel-space diffusion: input is OUT_CH (residual) conditioned on IN_CH + OUT_CH + 2 (pos)
    pixel_diff_in_ch = OUT_CH + IN_CH + OUT_CH + 2
    diff_model = DiffusionUNet(
        in_ch=pixel_diff_in_ch, out_ch=OUT_CH,
        base_ch=MODEL["diff_base_ch"],
        ch_mults=MODEL["diff_ch_mults"],
        num_res_blocks=MODEL["diff_num_res_blocks"],
        attn_resolutions=MODEL["diff_attn_resolutions"],
        time_dim=MODEL["diff_time_dim"],
    )
    print(f"[PixelDiff] Params: {sum(p.numel() for p in diff_model.parameters()):,}")
    print(f"[PixelDiff] Input channels: {pixel_diff_in_ch} (residual {OUT_CH} + cond {IN_CH}+{OUT_CH}+2)")

    train_pixel_diffusion(
        diff_model, drn, train_dl, val_dl,
        epochs=args.epochs, device=args.device,
        checkpoint_dir=args.checkpoint_dir, plot_dir=args.plot_dir)


if __name__ == "__main__":
    main()
