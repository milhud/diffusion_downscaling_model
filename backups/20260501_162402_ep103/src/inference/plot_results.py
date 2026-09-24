"""Publication-quality result figures from inference output.

Generates:
  - Per-variable comparison maps (target / DRN / DRN+Diff / error)
  - Ensemble spread maps
  - Power spectra per variable
  - Difference maps with colorbars in physical units

Usage:
    python -m src.inference.plot_results \
        --prediction results/prediction_2019_d182.nc \
        --target data/conus404_yearly_2019.nc \
        --day 182 --output results/figures/
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import CONUS404_VARS, VARIABLE_NAMES, VARIABLE_UNITS
from src.evaluation.metrics import power_spectrum_2d


def plot_variable_comparison(pred_nc, target_nc, var, day, output_dir,
                             drn_nc=None):
    """Plot target / DRN / full prediction / error for one variable."""
    with xr.open_dataset(target_nc) as tgt_ds:
        target = tgt_ds[var].isel(time=day).values

    with xr.open_dataset(pred_nc) as pred_ds:
        if f"{var}_ensemble_mean" in pred_ds:
            pred = pred_ds[f"{var}_ensemble_mean"].values
            spread = pred_ds[f"{var}_ensemble_std"].values if f"{var}_ensemble_std" in pred_ds else None
        else:
            pred = pred_ds[var].values

    panels = [target, pred, pred - target]
    titles = [
        f"Target ({VARIABLE_NAMES.get(var, var)})",
        f"Prediction",
        f"Error",
    ]
    cmaps = ["viridis", "viridis", "RdBu_r"]

    n_panels = 4 if spread is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    for i in range(3):
        im = axes[i].imshow(panels[i], origin="lower", aspect="auto", cmap=cmaps[i])
        axes[i].set_title(titles[i])
        plt.colorbar(im, ax=axes[i], shrink=0.7)

    if spread is not None:
        im = axes[3].imshow(spread, origin="lower", aspect="auto", cmap="YlOrRd")
        axes[3].set_title("Ensemble Spread")
        plt.colorbar(im, ax=axes[3], shrink=0.7)

    unit = VARIABLE_UNITS.get(var, "")
    plt.suptitle(f"{VARIABLE_NAMES.get(var, var)} [{unit}] -- Day {day}")
    plt.tight_layout()

    out_path = Path(output_dir) / f"comparison_{var}_d{day}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_spectra_comparison(pred_nc, target_nc, var, day, output_dir):
    """Power spectrum comparison: target vs prediction."""
    with xr.open_dataset(target_nc) as tgt_ds:
        target = tgt_ds[var].isel(time=day).values.astype(np.float32)

    with xr.open_dataset(pred_nc) as pred_ds:
        if f"{var}_ensemble_mean" in pred_ds:
            pred = pred_ds[f"{var}_ensemble_mean"].values.astype(np.float32)
        else:
            pred = pred_ds[var].values.astype(np.float32)

    # Handle NaN by replacing with mean
    target = np.nan_to_num(target, nan=np.nanmean(target))
    pred = np.nan_to_num(pred, nan=np.nanmean(pred))

    wl_tgt, pw_tgt = power_spectrum_2d(target)
    wl_pred, pw_pred = power_spectrum_2d(pred)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(wl_tgt, pw_tgt, "k-", linewidth=2, label="Target")
    ax.loglog(wl_pred, pw_pred, "r-", linewidth=1.5, label="Prediction")
    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel("Power")
    ax.set_title(f"Power Spectrum: {VARIABLE_NAMES.get(var, var)}")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    ax.invert_xaxis()
    plt.tight_layout()

    out_path = Path(output_dir) / f"spectra_{var}_d{day}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, help="Prediction .nc file")
    parser.add_argument("--target", required=True, help="Target CONUS404 .nc file")
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--output", default="results/figures")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    for var in CONUS404_VARS:
        try:
            plot_variable_comparison(args.prediction, args.target, var, args.day, args.output)
            plot_spectra_comparison(args.prediction, args.target, var, args.day, args.output)
        except Exception as e:
            print(f"  SKIP {var}: {e}")


if __name__ == "__main__":
    main()
