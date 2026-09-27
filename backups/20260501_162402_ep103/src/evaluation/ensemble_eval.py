"""32-member ensemble evaluation on test set.

Computes CRPS, rank histograms, spread-skill ratio, Q-Q plots,
and per-variable metrics across the full test set.

Usage:
    python -m src.evaluation.ensemble_eval \
        --num_members 32 \
        --num_steps 32 \
        --output_dir results/ensemble
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
from src.evaluation.metrics import (
    crps_ensemble, rmse, mae, spread_skill_ratio,
    rank_histogram, qq_quantiles, power_spectrum_2d,
    per_variable_metrics,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_ensemble(
    drn, vae, diff_model, ema, schedule,
    test_dl, var_names, output_dir,
    num_members=32, num_steps=32, guidance_scale=0.2,
    device="cuda",
):
    """Run full ensemble evaluation on test set."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_crps = {v: [] for v in var_names}
    all_rmse = {v: [] for v in var_names}
    all_mae = {v: [] for v in var_names}
    all_ssr = {v: [] for v in var_names}
    all_rank_counts = {v: np.zeros(num_members + 1) for v in var_names}
    all_qq_pred = {v: [] for v in var_names}
    all_qq_tgt = {v: [] for v in var_names}
    all_spectra_pred = {v: [] for v in var_names}
    all_spectra_tgt = {v: [] for v in var_names}

    drn.eval()
    vae.eval()
    diff_model.eval()

    n_batches = len(test_dl)
    print(f"[Ensemble] Evaluating {n_batches} batches x {num_members} members")

    with torch.no_grad():
        for batch_idx, (era5, conus) in enumerate(test_dl):
            era5, conus = era5.to(device), conus.to(device)

            with ema.apply():
                drn_pred, samples = run_pipeline(
                    era5, drn, vae, diff_model, schedule,
                    num_steps=num_steps, guidance_scale=guidance_scale,
                    num_samples=num_members, device=device,
                )

            # samples: (B, num_members, OUT_CH, H, W)
            # conus: (B, OUT_CH, H, W)
            B = conus.shape[0]

            for b in range(B):
                target = conus[b].cpu().numpy()       # (OUT_CH, H, W)
                ensemble = samples[b].cpu().numpy()   # (num_members, OUT_CH, H, W)

                for vi, vname in enumerate(var_names):
                    tgt_v = target[vi]                          # (H, W)
                    ens_v = ensemble[:, vi, :, :]               # (M, H, W)

                    # CRPS
                    c = crps_ensemble(tgt_v.flatten(), ens_v.reshape(num_members, -1).T)
                    all_crps[vname].append(c)

                    # RMSE and MAE of ensemble mean
                    ens_mean = ens_v.mean(axis=0)
                    all_rmse[vname].append(rmse(ens_mean, tgt_v))
                    all_mae[vname].append(mae(ens_mean, tgt_v))

                    # Spread-skill ratio
                    ssr = spread_skill_ratio(ens_v, tgt_v)
                    all_ssr[vname].append(ssr)

                    # Rank histogram (subsample spatially for efficiency)
                    step = max(1, tgt_v.shape[0] // 32)
                    sub_tgt = tgt_v[::step, ::step].flatten()
                    sub_ens = ens_v[:, ::step, ::step].reshape(num_members, -1).T
                    rh = rank_histogram(sub_ens.T, sub_tgt, num_bins=num_members + 1)
                    all_rank_counts[vname] += rh

                    # Q-Q quantiles
                    qq_p, qq_t = qq_quantiles(ens_mean, tgt_v, n_quantiles=100)
                    all_qq_pred[vname].append(qq_p)
                    all_qq_tgt[vname].append(qq_t)

                    # Power spectra
                    k_p, ps_p = power_spectrum_2d(ens_mean)
                    k_t, ps_t = power_spectrum_2d(tgt_v)
                    all_spectra_pred[vname].append(ps_p)
                    all_spectra_tgt[vname].append(ps_t)

            if (batch_idx + 1) % 10 == 0:
                print(f"  [{batch_idx+1}/{n_batches}] batches done")

    # Aggregate and save
    results = {}
    for vname in var_names:
        results[vname] = {
            "crps": np.mean(all_crps[vname]),
            "rmse": np.mean(all_rmse[vname]),
            "mae": np.mean(all_mae[vname]),
            "ssr": np.mean(all_ssr[vname]),
        }
        print(f"  {vname}: CRPS={results[vname]['crps']:.4f}, "
              f"RMSE={results[vname]['rmse']:.4f}, "
              f"MAE={results[vname]['mae']:.4f}, "
              f"SSR={results[vname]['ssr']:.3f}")

    # Save raw results
    np.savez(out / "ensemble_results.npz", **{
        f"{v}_crps": np.array(all_crps[v]) for v in var_names
    }, **{
        f"{v}_rmse": np.array(all_rmse[v]) for v in var_names
    }, **{
        f"{v}_ssr": np.array(all_ssr[v]) for v in var_names
    })

    # Plot rank histograms
    n_vars = len(var_names)
    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 3.5))
    if n_vars == 1:
        axes = [axes]
    for ax, vname in zip(axes, var_names):
        counts = all_rank_counts[vname]
        counts = counts / counts.sum()
        ax.bar(range(len(counts)), counts, color="#2196F3", alpha=0.8)
        ax.axhline(1.0 / len(counts), color="red", ls="--", lw=1, label="Uniform")
        ax.set_title(f"{VARIABLE_NAMES.get(vname, vname)}")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
    fig.suptitle(f"Rank Histograms ({num_members}-member ensemble)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "rank_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot Q-Q plots
    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 4))
    if n_vars == 1:
        axes = [axes]
    for ax, vname in zip(axes, var_names):
        qq_p = np.mean(all_qq_pred[vname], axis=0)
        qq_t = np.mean(all_qq_tgt[vname], axis=0)
        ax.plot(qq_t, qq_p, "b.", markersize=3)
        lims = [min(qq_t.min(), qq_p.min()), max(qq_t.max(), qq_p.max())]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlabel("Target quantiles")
        ax.set_ylabel("Prediction quantiles")
        ax.set_title(f"{VARIABLE_NAMES.get(vname, vname)}")
        ax.set_aspect("equal")
    fig.suptitle("Q-Q Plots (Ensemble Mean vs Target)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "qq_plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot power spectra
    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 4))
    if n_vars == 1:
        axes = [axes]
    for ax, vname in zip(axes, var_names):
        ps_p = np.mean(all_spectra_pred[vname], axis=0)
        ps_t = np.mean(all_spectra_tgt[vname], axis=0)
        k, _ = power_spectrum_2d(np.zeros((PATCH_SIZE, PATCH_SIZE)))
        ax.loglog(k[:len(ps_t)], ps_t, "g-", label="Target", lw=1.5)
        ax.loglog(k[:len(ps_p)], ps_p, "r-", label="Ens. Mean", lw=1.5)
        ax.set_xlabel("Wavenumber k")
        ax.set_ylabel("Power")
        ax.set_title(f"{VARIABLE_NAMES.get(vname, vname)}")
        ax.legend(fontsize=8)
    fig.suptitle("Power Spectra Comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "power_spectra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary table
    with open(out / "summary.txt", "w") as f:
        f.write(f"Ensemble Evaluation Summary ({num_members} members, {num_steps} steps)\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Variable':<20} {'CRPS':>8} {'RMSE':>8} {'MAE':>8} {'SSR':>8}\n")
        f.write("-" * 70 + "\n")
        for vname in var_names:
            r = results[vname]
            f.write(f"{vname:<20} {r['crps']:>8.4f} {r['rmse']:>8.4f} "
                    f"{r['mae']:>8.4f} {r['ssr']:>8.3f}\n")

    print(f"\n[Ensemble] Results saved to {out}/")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_members", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=0.2)
    parser.add_argument("--output_dir", default="results/ensemble")
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

    # Use test years
    _, test_dl = build_dataloaders(
        args.data_dir, stats, batch_size=4,
        patches_per_day=1, num_workers=2,
        train_years=TRAIN["train_years"], val_years=TRAIN["test_years"],
        land_mask=land_mask, valid_origins=valid_origins,
        era5_vars=ERA5_VARS, conus_vars=CONUS404_VARS,
        cache_dir=args.cache_dir, regridder=regridder,
        conus_lat=conus_lat, conus_lon=conus_lon)

    # Load models
    drn = DRN(in_ch=IN_CH, out_ch=OUT_CH, base_ch=MODEL["drn_base_ch"],
              ch_mults=MODEL["drn_ch_mults"],
              num_res_blocks=MODEL["drn_num_res_blocks"],
              attn_resolutions=MODEL["drn_attn_resolutions"])
    drn.load_state_dict(torch.load(args.drn_checkpoint)["model_state_dict"])

    vae = VAE(in_ch=OUT_CH, latent_ch=LATENT_CH, base_ch=MODEL["vae_base_ch"])
    vae.load_state_dict(torch.load(args.vae_checkpoint)["model_state_dict"])

    diff_in_ch = LATENT_CH + IN_CH + LATENT_CH + 2
    diff_model = DiffusionUNet(
        in_ch=diff_in_ch, out_ch=LATENT_CH,
        base_ch=MODEL["diff_base_ch"], ch_mults=MODEL["diff_ch_mults"],
        num_res_blocks=MODEL["diff_num_res_blocks"],
        attn_resolutions=MODEL["diff_attn_resolutions"],
        time_dim=MODEL["diff_time_dim"])
    ckpt = torch.load(args.diff_checkpoint)
    diff_model.load_state_dict(ckpt["model_state_dict"])

    ema = EMA(diff_model, decay=TRAIN["ema_decay"])
    if "ema_state_dict" in ckpt:
        ema.load_state_dict(ckpt["ema_state_dict"])

    schedule = EDMSchedule()

    evaluate_ensemble(
        drn, vae, diff_model, ema, schedule,
        test_dl, CONUS404_VARS, args.output_dir,
        num_members=args.num_members, num_steps=args.num_steps,
        guidance_scale=args.guidance_scale, device=args.device)


if __name__ == "__main__":
    main()
