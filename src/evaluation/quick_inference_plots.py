"""Quick inference visualization: ERA5 | Target | DRN | Diff mean | Spread | Error.

Grabs N random test patches, runs 4-member ensemble with 16 steps.
One figure per sample, one row per variable.

Usage:
    python -m src.evaluation.quick_inference_plots --num_samples 6 --output_dir results/inference_plots
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from config import CONUS404_VARS, ERA5_VARS, VARIABLE_NAMES, VARIABLE_UNITS, OUT_CH, IN_CH
from src.inference.pipeline import run_pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CMAPS = {
    "T2":           "RdBu_r",
    "TD2":          "BrBG",
    "U10":          "PuOr",
    "V10":          "PuOr",
    "PSFC":         "viridis",
    "PREC_ACC_NC":  "Blues",
}


def _denorm_channel(arr, mean, std):
    return arr * std + mean


def plot_sample(era5, target, drn_pred, ensemble, var_names, stats, sample_idx, out_dir,
                lat=None, lon=None):
    """One figure: rows=vars, cols=ERA5|Target|DRN|Diff mean|Error."""
    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, 5, figsize=(19, 3.5 * n_vars))
    if n_vars == 1:
        axes = axes[None, :]

    col_titles = ["ERA5 (interp)", "Target", "DRN", "Ensemble mean", "Error (Ens−Target)"]

    ens_mean = ensemble.mean(axis=0)   # (C, H, W)

    for vi, v in enumerate(var_names):
        si = CONUS404_VARS.index(v) if v in CONUS404_VARS else vi
        c_mean = float(stats.conus_mean[si])
        c_std  = float(stats.conus_std[si])
        e_mean = float(stats.era5_mean[si]) if si < len(stats.era5_mean) else c_mean
        e_std  = float(stats.era5_std[si])  if si < len(stats.era5_std)  else c_std

        era5_phys  = _denorm_channel(era5[vi],      e_mean, e_std)
        tgt_phys   = _denorm_channel(target[vi],    c_mean, c_std)
        drn_phys   = _denorm_channel(drn_pred[vi],  c_mean, c_std)
        diff_phys  = _denorm_channel(ens_mean[vi],  c_mean, c_std)
        error_phys = diff_phys - tgt_phys

        unit = VARIABLE_UNITS.get(v, "")
        cmap = CMAPS.get(v, "viridis")

        panels = [era5_phys, tgt_phys, drn_phys, diff_phys, error_phys]
        err_abs = np.nanpercentile(np.abs(error_phys), 95)

        H, W = tgt_phys.shape
        extent = [0, W, 0, H]
        if lat is not None and lon is not None:
            extent = [lon[0], lon[-1], lat[0], lat[-1]]

        for ci, (ax, data) in enumerate(zip(axes[vi], panels)):
            if ci == 4:  # error: diverging, 95th pct clip
                vmax = max(err_abs, 1e-6)
                im = ax.imshow(data, origin="lower", cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax, extent=extent, aspect="auto")
            else:
                vmin = np.nanpercentile(tgt_phys, 2)
                vmax = np.nanpercentile(tgt_phys, 98)
                im = ax.imshow(data, origin="lower", cmap=cmap,
                               vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
            if lat is not None and lon is not None:
                ax.set_xlabel("Lon", fontsize=7)
                if ci == 0:
                    ax.set_ylabel(f"{VARIABLE_NAMES.get(v, v)}\nLat", fontsize=8)
                else:
                    ax.set_yticks([])
                ax.tick_params(labelsize=6)
            else:
                ax.axis("off")
                if ci == 0:
                    ax.set_ylabel(VARIABLE_NAMES.get(v, v), fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=unit)
            if vi == 0:
                ax.set_title(col_titles[ci], fontsize=10, fontweight="bold")

    fig.suptitle(f"Sample {sample_idx+1}: ERA5→CONUS404 Downscaling (4-member ensemble, 16 steps)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    path = out_dir / f"sample_{sample_idx+1:02d}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples",    type=int,   default=6)
    parser.add_argument("--num_members",    type=int,   default=4)
    parser.add_argument("--num_steps",      type=int,   default=16)
    parser.add_argument("--output_dir",     default="results/inference_plots")
    parser.add_argument("--drn_checkpoint",  default="checkpoints/drn_best.pt")
    parser.add_argument("--vae_checkpoint",  default="checkpoints/vae_best.pt")
    parser.add_argument("--diff_checkpoint", default="checkpoints/diffusion_best.pt")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--data_dir",        default="data")
    parser.add_argument("--cache_dir",       default="/discover/nobackup/sduan/.data")
    parser.add_argument("--scan",            type=int,   default=0,
                        help="Scan this many patches and keep --num_samples with lowest RMSE. 0=sequential.")
    parser.add_argument("--vars",            nargs="+",  default=None,
                        help="Subset of variables to plot, e.g. --vars T2 U10 PREC_ACC_NC")
    args = parser.parse_args()

    from src.evaluation._eval_setup import build_test_dataloader, load_models
    import torch.nn.functional as F

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    test_dl, stats, _, _ = build_test_dataloader(
        data_dir=args.data_dir, cache_dir=args.cache_dir,
        batch_size=1, num_workers=2, patches_per_day=1)

    drn, vae, diff_model, ema, schedule = load_models(
        args.drn_checkpoint, args.vae_checkpoint, args.diff_checkpoint, args.device)

    plot_vars = args.vars if args.vars else CONUS404_VARS
    var_indices = [CONUS404_VARS.index(v) for v in plot_vars]

    n_scan = args.scan if args.scan > 0 else args.num_samples
    candidates = []  # list of (rmse, era5_up_np, conus_np, drn_np, ensemble_np)

    with torch.no_grad(), ema.apply():
        for i, (era5, conus) in enumerate(test_dl):
            if i >= n_scan:
                break
            era5 = era5.to(args.device)
            conus = conus.to(args.device)

            drn_pred, samples = run_pipeline(
                era5, drn, vae, diff_model, schedule,
                num_steps=args.num_steps, num_samples=args.num_members,
                device=args.device)

            era5_up = F.interpolate(era5[:, :OUT_CH], (256, 256), mode="bilinear", align_corners=False)

            ens_mean = samples[0].mean(dim=0)  # (C, H, W)
            patch_rmse = float(torch.sqrt(((ens_mean - conus[0]) ** 2).mean()).cpu())

            candidates.append((
                patch_rmse,
                era5_up[0].cpu().numpy(),
                conus[0].cpu().numpy(),
                drn_pred[0].cpu().numpy(),
                samples[0].cpu().numpy(),
            ))

            if args.scan > 0:
                print(f"  Scanned {i+1}/{n_scan}  RMSE={patch_rmse:.4f}")

    # Sort by RMSE ascending, keep best num_samples
    candidates.sort(key=lambda x: x[0])
    best = candidates[:args.num_samples]
    print(f"\n[QuickInference] Best {args.num_samples} patches (lowest RMSE):")
    for rank, (r, *_) in enumerate(best):
        print(f"  rank {rank+1}: RMSE={r:.4f}")

    # Pixel-index axes (real lat/lon not available without conus_lat array)
    pix = np.arange(256)

    for plot_idx, (_, era5_np, conus_np, drn_np, ens_np) in enumerate(best):
        plot_sample(
            era5=era5_np[var_indices],
            target=conus_np[var_indices],
            drn_pred=drn_np[var_indices],
            ensemble=ens_np[:, var_indices],
            var_names=plot_vars,
            stats=stats,
            sample_idx=plot_idx,
            out_dir=out,
            lat=pix,
            lon=pix,
        )

    print(f"\n[QuickInference] {args.num_samples} samples saved to {out}/")


if __name__ == "__main__":
    main()
