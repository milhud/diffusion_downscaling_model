"""Ablation: Denoising step count vs quality.

Tests diffusion sampling quality at 2, 4, 8, 16, 32, 64 denoising steps
to find the minimum steps needed for acceptable quality.

Usage:
    python -m src.evaluation.step_ablation --output_dir results/step_ablation
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from config import (
    ERA5_VARS, CONUS404_VARS, IN_CH, OUT_CH, PATCH_SIZE,
    LATENT_CH, MODEL, TRAIN,
)
from src.models.drn import DRN
from src.models.vae import VAE
from src.models.diffusion_unet import DiffusionUNet
from src.models.edm import EDMSchedule, heun_sampler
from src.training.ema import EMA
from src.evaluation.metrics import crps_ensemble, rmse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _make_pos_embedding(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy, xx], dim=0)


def run_step_ablation(
    drn, vae, diff_model, ema, schedule,
    test_dl, output_dir, step_counts=(2, 4, 8, 16, 32, 64),
    num_ensemble=8, guidance_scale=0.2, device="cuda", max_batches=20,
):
    """Evaluate quality at different denoising step counts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    drn.eval()
    vae.eval()
    diff_model.eval()
    pos_emb = _make_pos_embedding(64, 64, device)

    results = {s: {"rmse": [], "crps": []} for s in step_counts}

    n_batches = min(len(test_dl), max_batches)
    print(f"[StepAblation] Testing steps: {step_counts}, {n_batches} batches")

    with torch.no_grad():
        for batch_idx, (era5, conus) in enumerate(test_dl):
            if batch_idx >= max_batches:
                break
            era5, conus = era5.to(device), conus.to(device)
            B = era5.shape[0]

            drn_pred = drn(era5)
            era5_down = F.interpolate(era5, size=(64, 64), mode="bilinear", align_corners=False)
            mu_drn, _ = vae.encode(drn_pred)
            pos = pos_emb.unsqueeze(0).expand(B, -1, -1, -1)
            cond = torch.cat([era5_down, mu_drn, pos], dim=1)

            for ns in step_counts:
                samples = []
                with ema.apply():
                    for _ in range(num_ensemble):
                        z = heun_sampler(diff_model, schedule, cond,
                                         shape=(B, LATENT_CH, 64, 64),
                                         num_steps=ns, guidance_scale=guidance_scale)
                        r = vae.decode(z)
                        samples.append(drn_pred + r)

                ensemble = torch.stack(samples, dim=1)  # (B, M, C, H, W)
                ens_mean = ensemble.mean(dim=1)

                for b in range(B):
                    r = rmse(ens_mean[b, 0].cpu().numpy(), conus[b, 0].cpu().numpy())
                    results[ns]["rmse"].append(r)

                    ens_flat = ensemble[b, :, 0].cpu().numpy().reshape(num_ensemble, -1).T
                    tgt_flat = conus[b, 0].cpu().numpy().flatten()
                    c = crps_ensemble(tgt_flat, ens_flat)
                    results[ns]["crps"].append(c)

            if (batch_idx + 1) % 5 == 0:
                print(f"  [{batch_idx+1}/{n_batches}]")

    # Aggregate
    step_list = sorted(results.keys())
    mean_rmse = [np.mean(results[s]["rmse"]) for s in step_list]
    mean_crps = [np.mean(results[s]["crps"]) for s in step_list]

    print("\n  Steps | RMSE   | CRPS")
    print("  ------+--------+------")
    for s, r, c in zip(step_list, mean_rmse, mean_crps):
        print(f"  {s:>5} | {r:.4f} | {c:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(step_list, mean_rmse, "bo-", lw=2, markersize=8)
    ax1.set_xlabel("Denoising Steps")
    ax1.set_ylabel("RMSE")
    ax1.set_title("RMSE vs Denoising Steps")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)

    ax2.plot(step_list, mean_crps, "ro-", lw=2, markersize=8)
    ax2.set_xlabel("Denoising Steps")
    ax2.set_ylabel("CRPS")
    ax2.set_title("CRPS vs Denoising Steps")
    ax2.set_xscale("log", base=2)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Denoising Step Count Ablation", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "step_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(out / "step_ablation_data.npz",
             steps=np.array(step_list),
             rmse=np.array(mean_rmse),
             crps=np.array(mean_crps))

    print(f"\nSaved to {out}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/step_ablation")
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--num_ensemble", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--drn_checkpoint", default="checkpoints/drn_best.pt")
    parser.add_argument("--vae_checkpoint", default="checkpoints/vae_best.pt")
    parser.add_argument("--diff_checkpoint", default="checkpoints/diffusion_best.pt")
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

    run_step_ablation(drn, vae, diff_model, ema, schedule, test_dl, args.output_dir,
                      step_counts=args.steps, num_ensemble=args.num_ensemble,
                      max_batches=args.max_batches, device=args.device)


if __name__ == "__main__":
    main()
