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
from src.data.normalization import invert_pretransform

# ERA5 tp is archived in meters (ECMWF convention); CONUS404 PREC_ACC_NC is in mm.
# Convert after inverting the log1p pretransform so ERA5 precip is comparable/plottable
# on the same physical (mm) scale as the target instead of reading as ~0.
ERA5_UNIT_FIX_MM = {"PREC_ACC_NC": 1000.0}

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


def _denorm_channel(arr, mean, std, var_name=None, is_era5=False):
    """Undo z-score, then invert the pretransform (e.g. expm1 for log1p precip) to
    recover true physical units. ERA5 precip additionally gets the m->mm unit fix."""
    phys = arr * std + mean
    if var_name is not None:
        phys = invert_pretransform(phys, var_name)
        if is_era5 and var_name in ERA5_UNIT_FIX_MM:
            phys = phys * ERA5_UNIT_FIX_MM[var_name]
    return phys


def plot_sample(era5, target, drn_pred, ensemble, var_names, stats, sample_idx, out_dir,
                lat=None, lon=None):
    """One figure: rows=vars, cols=Ground Truth|ERA5|DRN|Diffusion|(Ens-GT).
    The first four (value) panels share one horizontal colorbar under the row;
    the last (error) panel keeps its own diverging colorbar. Each var gets its
    own image row (in inches == panel width, so the square 256x256 panels fill
    their cell edge-to-edge with no gutters) plus a thin dedicated colorbar row
    directly under it, so the colorbar never eats into the image's own space."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    n_vars = len(var_names)
    PANEL_IN = 4.3   # inches per square panel (both width and height)
    CBAR_IN = 0.42   # inches for the colorbar strip under each var's row
    GAP_IN = 0.55    # inches between panel row (incl. its Lon xlabel/ticks) and colorbar row
    COL_GAP_IN = 0.06  # inches between adjacent panels in a row

    fig_w = 5 * PANEL_IN + 4 * COL_GAP_IN
    fig_h = n_vars * (PANEL_IN + GAP_IN + CBAR_IN)
    fig = plt.figure(figsize=(fig_w, fig_h))
    # Full-bleed margins: matplotlib's default figure margins (left=0.125,
    # right=0.9, top=0.88, bottom=0.11) otherwise eat asymmetric width vs.
    # height, breaking the PANEL_IN x PANEL_IN square-cell math below and
    # making aspect="equal" silently shrink/reposition axes boxes (which is
    # what was pushing "Lon" down into the colorbar). Overflowing text (Lon,
    # titles, ylabels) is still captured fine since savefig uses
    # bbox_inches="tight", which expands the canvas rather than clipping.
    outer = fig.add_gridspec(n_vars, 1, hspace=0.55 / PANEL_IN,
                              left=0.005, right=0.995, top=0.995, bottom=0.005)
    axes = np.empty((n_vars, 5), dtype=object)
    value_cax = np.empty(n_vars, dtype=object)   # bar under cols 0-3
    error_cax = np.empty(n_vars, dtype=object)   # bar under col 4
    for vi in range(n_vars):
        inner = GridSpecFromSubplotSpec(
            2, 5, subplot_spec=outer[vi],
            height_ratios=[PANEL_IN, CBAR_IN], hspace=GAP_IN / ((PANEL_IN + CBAR_IN) / 2),
            wspace=COL_GAP_IN / PANEL_IN)
        for ci in range(5):
            axes[vi, ci] = fig.add_subplot(inner[0, ci])
        # Two bars, same box size for all 5 panels, small visible gap between
        # them. A single shared linear scale was tried and is scientifically
        # broken on real data: error is a tiny (~+-1-2 unit) difference field,
        # so on the same scale as the absolute-value panels it collapses to a
        # single flat color (verified: solid blank block on real patches).
        # Error needs its own diverging range to show any structure at all.
        cbar_row = GridSpecFromSubplotSpec(1, 5, subplot_spec=inner[1, :],
                                            wspace=COL_GAP_IN / PANEL_IN * 3)
        value_cax[vi] = fig.add_subplot(cbar_row[0, 0:4])
        error_cax[vi] = fig.add_subplot(cbar_row[0, 4])

    col_titles = ["Ground Truth", "ERA5 (interp)", "DRN", "Diffusion", "(Ens−GT)"]

    ens_mean = ensemble.mean(axis=0)   # (C, H, W)

    for vi, v in enumerate(var_names):
        si = CONUS404_VARS.index(v) if v in CONUS404_VARS else vi
        c_mean = float(stats.conus_mean[si])
        c_std  = float(stats.conus_std[si])
        e_mean = float(stats.era5_mean[si]) if si < len(stats.era5_mean) else c_mean
        e_std  = float(stats.era5_std[si])  if si < len(stats.era5_std)  else c_std

        era5_phys  = _denorm_channel(era5[vi],      e_mean, e_std, var_name=v, is_era5=True)
        tgt_phys   = _denorm_channel(target[vi],    c_mean, c_std, var_name=v)
        drn_phys   = _denorm_channel(drn_pred[vi],  c_mean, c_std, var_name=v)
        diff_phys  = _denorm_channel(ens_mean[vi],  c_mean, c_std, var_name=v)
        error_phys = diff_phys - tgt_phys

        unit = VARIABLE_UNITS.get(v, "")
        cmap = CMAPS.get(v, "viridis")

        panels = [tgt_phys, era5_phys, drn_phys, diff_phys, error_phys]

        H, W = tgt_phys.shape
        extent = [0, W, 0, H]
        if lat is not None and lon is not None:
            extent = [lon[0], lon[-1], lat[0], lat[-1]]

        vmin = np.nanpercentile(tgt_phys, 2)
        vmax = np.nanpercentile(tgt_phys, 98)
        err_abs = np.nanpercentile(np.abs(error_phys), 95)
        value_im = None
        error_im = None
        for ci, (ax, data) in enumerate(zip(axes[vi], panels)):
            if ci == 4:  # error: own diverging scale, centered on 0
                evmax = max(err_abs, 1e-6)
                im = ax.imshow(data, origin="lower", cmap="RdBu_r",
                               vmin=-evmax, vmax=evmax, extent=extent, aspect="equal")
                error_im = im
            else:
                im = ax.imshow(data, origin="lower", cmap=cmap,
                               vmin=vmin, vmax=vmax, extent=extent, aspect="equal")
                if ci == 0:
                    value_im = im
            if lat is not None and lon is not None:
                ax.set_xlabel("Lon", fontsize=14)
                # Drop the first/last tick (0 and 250) so they don't overlap
                # the neighboring panel's edge tick in the tight gap between
                # panels, but keep matplotlib's normal 50-step grid instead
                # of letting a pruning locator re-pick a different spacing.
                default_ticks = ax.get_xticks()
                x0, x1 = ax.get_xlim()
                inner_ticks = [t for t in default_ticks if x0 < t < x1]
                ax.set_xticks(inner_ticks)
                if ci == 0:
                    ax.set_ylabel(f"{VARIABLE_NAMES.get(v, v)}\nLat", fontsize=17)
                else:
                    ax.set_yticks([])
                ax.tick_params(labelsize=13)
            else:
                ax.axis("off")
                if ci == 0:
                    ax.set_ylabel(VARIABLE_NAMES.get(v, v), fontsize=18, fontweight="bold")
            if vi == 0:
                ax.set_title(col_titles[ci], fontsize=19, fontweight="bold")

        cbar = fig.colorbar(value_im, cax=value_cax[vi], orientation="horizontal")
        cbar.ax.tick_params(labelsize=13)
        cbar.set_label(unit, fontsize=14)

        err_cbar = fig.colorbar(error_im, cax=error_cax[vi], orientation="horizontal")
        err_cbar.ax.tick_params(labelsize=13)
        err_cbar.set_label(unit, fontsize=14)

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
    parser.add_argument("--sort_by",         default="rmse",
                        choices=["rmse", "diff_contribution"],
                        help="rmse: pick patches model gets right; diff_contribution: pick patches where diffusion adds most")
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
