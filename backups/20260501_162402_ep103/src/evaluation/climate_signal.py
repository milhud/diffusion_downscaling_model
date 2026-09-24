"""Climate change signal preservation test.

Compares early-period (1980-1989) vs late-period (2011-2020) climatologies
to verify the model preserves warming/moistening trends.

Usage:
    python -m src.evaluation.climate_signal --output_dir results/climate_signal
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from config import (
    ERA5_VARS, CONUS404_VARS, IN_CH, OUT_CH, PATCH_SIZE,
    LATENT_CH, MODEL, TRAIN, VARIABLE_NAMES, VARIABLE_UNITS,
)
from src.models.drn import DRN
from src.models.vae import VAE
from src.models.diffusion_unet import DiffusionUNet
from src.models.edm import EDMSchedule
from src.training.ema import EMA
from src.inference.pipeline import run_pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_period_climatology(
    drn, vae, diff_model, ema, schedule,
    years, data_dir, cache_dir, stats, land_mask, valid_origins,
    num_samples=4, num_steps=16, device="cuda", samples_per_year=30,
):
    """Compute mean fields for a set of years."""
    from src.data.dataset import build_dataloaders
    from src.preprocessing.regrid import ERA5Regridder
    import xarray as xr

    with xr.open_dataset(f"{data_dir}/era5_1980.nc") as ds:
        era5_lat, era5_lon = ds["latitude"].values, ds["longitude"].values
    with xr.open_dataset(f"{data_dir}/conus404_yearly_1980.nc") as ds:
        conus_lat, conus_lon = ds["lat"].values, ds["lon"].values
    regridder = ERA5Regridder(era5_lat, era5_lon, conus_lat, conus_lon)

    _, dl = build_dataloaders(
        data_dir, stats, batch_size=4, patches_per_day=1, num_workers=2,
        train_years=TRAIN["train_years"], val_years=years,
        land_mask=land_mask, valid_origins=valid_origins,
        era5_vars=ERA5_VARS, conus_vars=CONUS404_VARS,
        cache_dir=cache_dir, regridder=regridder,
        conus_lat=conus_lat, conus_lon=conus_lon)

    target_acc = []
    pred_acc = []
    drn_acc = []

    drn.eval()
    vae.eval()
    diff_model.eval()

    max_batches = samples_per_year * len(years) // 4
    with torch.no_grad():
        for batch_idx, (era5, conus) in enumerate(dl):
            if batch_idx >= max_batches:
                break
            era5, conus = era5.to(device), conus.to(device)

            with ema.apply():
                drn_pred, samples = run_pipeline(
                    era5, drn, vae, diff_model, schedule,
                    num_steps=num_steps, num_samples=num_samples, device=device)

            target_acc.append(conus.cpu().numpy())
            pred_acc.append(samples.mean(dim=1).cpu().numpy())  # ensemble mean
            drn_acc.append(drn_pred.cpu().numpy())

    target_mean = np.concatenate(target_acc, axis=0).mean(axis=0)  # (OUT_CH, H, W)
    pred_mean = np.concatenate(pred_acc, axis=0).mean(axis=0)
    drn_mean = np.concatenate(drn_acc, axis=0).mean(axis=0)

    return target_mean, pred_mean, drn_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/climate_signal")
    parser.add_argument("--early_years", type=int, nargs="+", default=list(range(1980, 1990)))
    parser.add_argument("--late_years", type=int, nargs="+", default=list(range(2011, 2021)))
    parser.add_argument("--samples_per_year", type=int, default=30)
    parser.add_argument("--drn_checkpoint", default="checkpoints/drn_best.pt")
    parser.add_argument("--vae_checkpoint", default="checkpoints/vae_best.pt")
    parser.add_argument("--diff_checkpoint", default="checkpoints/diffusion_best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="cached_data")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from src.preprocessing.normalization import NormalizationStats
    from src.preprocessing.land_mask import build_conus404_land_mask, get_valid_patch_origins
    import xarray as xr

    stats = NormalizationStats()
    stats.load("norm_stats.npz")

    with xr.open_dataset(f"{args.data_dir}/era5_1980.nc") as ds:
        land_mask = build_conus404_land_mask(
            xr.open_dataset(f"{args.data_dir}/conus404_yearly_1980.nc")["lat"].values,
            xr.open_dataset(f"{args.data_dir}/conus404_yearly_1980.nc")["lon"].values, ds)
    valid_origins = get_valid_patch_origins(land_mask, PATCH_SIZE, TRAIN["min_land_frac"])

    # Load models
    drn = DRN(in_ch=IN_CH, out_ch=OUT_CH, base_ch=MODEL["drn_base_ch"],
              ch_mults=MODEL["drn_ch_mults"], num_res_blocks=MODEL["drn_num_res_blocks"],
              attn_resolutions=MODEL["drn_attn_resolutions"])
    drn.load_state_dict(torch.load(args.drn_checkpoint)["model_state_dict"])

    vae = VAE(in_ch=OUT_CH, latent_ch=LATENT_CH, base_ch=MODEL["vae_base_ch"])
    vae.load_state_dict(torch.load(args.vae_checkpoint)["model_state_dict"])

    diff_in_ch = LATENT_CH + IN_CH + LATENT_CH + 2
    diff_model = DiffusionUNet(
        in_ch=diff_in_ch, out_ch=LATENT_CH, base_ch=MODEL["diff_base_ch"],
        ch_mults=MODEL["diff_ch_mults"], num_res_blocks=MODEL["diff_num_res_blocks"],
        attn_resolutions=MODEL["diff_attn_resolutions"], time_dim=MODEL["diff_time_dim"])
    ckpt = torch.load(args.diff_checkpoint)
    diff_model.load_state_dict(ckpt["model_state_dict"])

    ema = EMA(diff_model, decay=TRAIN["ema_decay"])
    if "ema_state_dict" in ckpt:
        ema.load_state_dict(ckpt["ema_state_dict"])

    schedule = EDMSchedule()

    print(f"Computing early period climatology ({args.early_years[0]}-{args.early_years[-1]})...")
    early_tgt, early_pred, early_drn = compute_period_climatology(
        drn, vae, diff_model, ema, schedule,
        args.early_years, args.data_dir, args.cache_dir, stats,
        land_mask, valid_origins, device=args.device,
        samples_per_year=args.samples_per_year)

    print(f"Computing late period climatology ({args.late_years[0]}-{args.late_years[-1]})...")
    late_tgt, late_pred, late_drn = compute_period_climatology(
        drn, vae, diff_model, ema, schedule,
        args.late_years, args.data_dir, args.cache_dir, stats,
        land_mask, valid_origins, device=args.device,
        samples_per_year=args.samples_per_year)

    # Compute change signals
    delta_tgt = late_tgt - early_tgt
    delta_pred = late_pred - early_pred
    delta_drn = late_drn - early_drn

    # Plot per variable
    var_names = CONUS404_VARS
    for vi, vname in enumerate(var_names):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        vmin = min(delta_tgt[vi].min(), delta_pred[vi].min(), delta_drn[vi].min())
        vmax = max(delta_tgt[vi].max(), delta_pred[vi].max(), delta_drn[vi].max())
        abs_max = max(abs(vmin), abs(vmax))

        for ax, data, title in zip(axes,
                                    [delta_tgt[vi], delta_drn[vi], delta_pred[vi]],
                                    ["Target (CONUS404)", "DRN", "DRN+Diffusion"]):
            im = ax.imshow(data, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
            ax.set_title(f"{title}\nmean={data.mean():.4f}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

        unit = VARIABLE_UNITS.get(vname, "")
        fig.suptitle(f"Climate Signal: {VARIABLE_NAMES.get(vname, vname)} "
                     f"({args.late_years[0]}-{args.late_years[-1]} minus "
                     f"{args.early_years[0]}-{args.early_years[-1]}) [{unit}]", y=1.02)
        fig.tight_layout()
        fig.savefig(out / f"climate_signal_{vname}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Summary
    with open(out / "climate_signal_summary.txt", "w") as f:
        f.write(f"Climate Signal Preservation Test\n")
        f.write(f"Early: {args.early_years[0]}-{args.early_years[-1]}\n")
        f.write(f"Late: {args.late_years[0]}-{args.late_years[-1]}\n\n")
        f.write(f"{'Variable':<15} {'Target':>10} {'DRN':>10} {'DRN+Diff':>10}\n")
        f.write("-" * 50 + "\n")
        for vi, vname in enumerate(var_names):
            f.write(f"{vname:<15} {delta_tgt[vi].mean():>10.4f} "
                    f"{delta_drn[vi].mean():>10.4f} {delta_pred[vi].mean():>10.4f}\n")

    print(f"[ClimateSignal] Saved to {out}/")


if __name__ == "__main__":
    main()
