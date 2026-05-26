"""Power spectra analysis at each pipeline stage.

Generates a single plot showing how variance is progressively recovered:
ERA5 interpolation -> DRN -> DRN+VAE -> DRN+Diffusion -> CONUS404 target

Usage:
    python -m src.evaluation.stage_spectra --output_dir results/spectra
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from config import (
    ERA5_VARS, CONUS404_VARS, IN_CH, OUT_CH, PATCH_SIZE,
    LATENT_CH, MODEL, TRAIN, VARIABLE_NAMES,
)
from src.models.drn import DRN
from src.models.vae import VAE
from src.models.diffusion_unet import DiffusionUNet
from src.models.edm import EDMSchedule, heun_sampler
from src.training.ema import EMA
from src.evaluation.metrics import power_spectrum_2d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _make_pos_embedding(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy, xx], dim=0)


def compute_stage_spectra(
    drn, vae, diff_model, ema, schedule,
    test_dl, var_names, output_dir,
    num_steps=32, num_ensemble=4, guidance_scale=0.2,
    device="cuda", max_batches=50,
):
    """Compute mean power spectra at each pipeline stage."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    drn.eval()
    vae.eval()
    diff_model.eval()

    stages = ["ERA5 Interp", "DRN", "DRN+VAE", "DRN+Diff", "Target"]
    spectra = {s: {v: [] for v in var_names} for s in stages}

    pos_emb = _make_pos_embedding(64, 64, device)

    n_batches = min(len(test_dl), max_batches)
    print(f"[Spectra] Computing spectra over {n_batches} batches")

    with torch.no_grad():
        for batch_idx, (era5, conus) in enumerate(test_dl):
            if batch_idx >= max_batches:
                break
            era5, conus = era5.to(device), conus.to(device)

            # Stage: ERA5 interpolation (just the ERA5 dynamic vars, not static)
            n_era5 = len(ERA5_VARS)
            era5_dynamic = era5[:, :n_era5]

            # Stage: DRN
            drn_pred = drn(era5)

            # Stage: DRN+VAE (deterministic reconstruction)
            residual_true = conus - drn_pred
            mu, _ = vae.encode(residual_true)
            vae_recon = vae.decode(mu)
            drn_vae_pred = drn_pred + vae_recon

            # Stage: DRN+Diff (stochastic, average over ensemble)
            era5_down = F.interpolate(era5, size=(64, 64), mode="bilinear", align_corners=False)
            mu_drn, _ = vae.encode(drn_pred)
            B = era5.shape[0]
            pos = pos_emb.unsqueeze(0).expand(B, -1, -1, -1)
            cond = torch.cat([era5_down, mu_drn, pos], dim=1)

            diff_preds = []
            with ema.apply():
                for _ in range(num_ensemble):
                    z_sample = heun_sampler(
                        diff_model, schedule, cond,
                        shape=(B, LATENT_CH, 64, 64),
                        num_steps=num_steps, guidance_scale=guidance_scale)
                    r_sample = vae.decode(z_sample)
                    diff_preds.append(drn_pred + r_sample)
            diff_mean = torch.stack(diff_preds).mean(dim=0)

            # Compute spectra for each sample in batch
            for b in range(B):
                for vi, vname in enumerate(var_names):
                    if vi < n_era5:
                        _, ps = power_spectrum_2d(era5_dynamic[b, vi].cpu().numpy())
                        spectra["ERA5 Interp"][vname].append(ps)

                    _, ps = power_spectrum_2d(drn_pred[b, vi].cpu().numpy())
                    spectra["DRN"][vname].append(ps)

                    _, ps = power_spectrum_2d(drn_vae_pred[b, vi].cpu().numpy())
                    spectra["DRN+VAE"][vname].append(ps)

                    _, ps = power_spectrum_2d(diff_mean[b, vi].cpu().numpy())
                    spectra["DRN+Diff"][vname].append(ps)

                    _, ps = power_spectrum_2d(conus[b, vi].cpu().numpy())
                    spectra["Target"][vname].append(ps)

            if (batch_idx + 1) % 10 == 0:
                print(f"  [{batch_idx+1}/{n_batches}]")

    # Get wavenumber axis
    k, _ = power_spectrum_2d(np.zeros((PATCH_SIZE, PATCH_SIZE)))

    # Plot
    n_vars = len(var_names)
    colors = {"ERA5 Interp": "gray", "DRN": "black", "DRN+VAE": "blue",
              "DRN+Diff": "red", "Target": "green"}
    linewidths = {"ERA5 Interp": 1, "DRN": 1.5, "DRN+VAE": 1.5,
                  "DRN+Diff": 2, "Target": 2}

    ncols = 3
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = np.array(axes).flatten()
    # hide unused panels
    for extra_ax in axes_flat[n_vars:]:
        extra_ax.set_visible(False)

    for ax, vname in zip(axes_flat[:n_vars], var_names):
        for stage in stages:
            if len(spectra[stage][vname]) == 0:
                continue
            mean_ps = np.mean(spectra[stage][vname], axis=0)
            ax.loglog(k[:len(mean_ps)], mean_ps,
                      color=colors[stage], lw=linewidths[stage],
                      label=stage, alpha=0.9)
        ax.set_xlabel("Wavenumber k", fontsize=10)
        ax.set_ylabel("Power", fontsize=10)
        ax.set_title(VARIABLE_NAMES.get(vname, vname), fontsize=14, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Power Spectra at Each Pipeline Stage", y=1.02, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "stage_spectra.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Spectra] Saved to {out / 'stage_spectra.png'}")

    # Save raw data
    np.savez(out / "stage_spectra_data.npz", wavenumber=k, **{
        f"{stage}_{vname}": np.mean(spectra[stage][vname], axis=0)
        for stage in stages for vname in var_names
        if len(spectra[stage][vname]) > 0
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/spectra")
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--num_ensemble", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--drn_checkpoint", default="checkpoints/drn_best.pt")
    parser.add_argument("--vae_checkpoint", default="checkpoints/vae_best.pt")
    parser.add_argument("--diff_checkpoint", default="checkpoints/diffusion_best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="cached_data")
    args = parser.parse_args()

    from src.evaluation._eval_setup import build_test_dataloader, load_models

    test_dl, stats, land_mask, valid_origins = build_test_dataloader(
        data_dir=args.data_dir, cache_dir=args.cache_dir,
        batch_size=4, num_workers=2, patches_per_day=1)

    drn, vae, diff_model, ema, schedule = load_models(
        args.drn_checkpoint, args.vae_checkpoint, args.diff_checkpoint, args.device)

    compute_stage_spectra(
        drn, vae, diff_model, ema, schedule, test_dl, CONUS404_VARS, args.output_dir,
        num_steps=args.num_steps, num_ensemble=args.num_ensemble,
        max_batches=args.max_batches, device=args.device)


if __name__ == "__main__":
    main()
