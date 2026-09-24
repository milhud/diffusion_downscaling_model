"""Visualization utilities for downscaling pipeline stages."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# Variable names for labeling plots
CONUS_VAR_NAMES = ["T2 (2m Temp)", "TD2 (Dewpoint)", "U10 (U-wind)", "V10 (V-wind)",
                   "PSFC (Sfc Pres)", "Q2 (Humidity)", "PREC_ACC_NC (Precip)"]
ERA5_VAR_NAMES = ["t2m", "d2m", "u10", "v10", "sp", "tp", "z"]


def plot_stage_comparison(
    era5_input: torch.Tensor,
    conus_target: torch.Tensor,
    drn_pred: torch.Tensor,
    residual: torch.Tensor = None,
    noisy_latent: torch.Tensor = None,
    vae_recon: torch.Tensor = None,
    diffusion_sample: torch.Tensor = None,
    final_pred: torch.Tensor = None,
    var_idx: int = 0,
    save_path: str = "stage_comparison.png",
    title_prefix: str = "",
):
    """Plot a multi-panel figure showing each pipeline stage for one variable.

    Args:
        era5_input: (13, H, W) — ERA5 input (first 7 channels are dynamic vars)
        conus_target: (7, H, W) — ground truth CONUS404
        drn_pred: (7, H, W) — DRN prediction
        residual: (7, H, W) — CONUS404 - DRN (optional)
        noisy_latent: (8, h, w) — noisy latent z_t (optional)
        vae_recon: (7, H, W) — VAE reconstruction of residual (optional)
        diffusion_sample: (7, H, W) — diffusion-sampled residual decoded (optional)
        final_pred: (7, H, W) — DRN + diffusion sample (optional)
        var_idx: which output variable to visualize (0-6)
        save_path: where to save the figure
    """
    panels = []
    titles = []

    def to_np(t):
        if t is None:
            return None
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().float().numpy()
        return t

    # 1) ERA5 input (show matching var)
    era5_np = to_np(era5_input)
    panels.append(era5_np[min(var_idx, 6)])
    titles.append(f"ERA5 Input\n({ERA5_VAR_NAMES[min(var_idx, 6)]})")

    # 2) CONUS404 target
    conus_np = to_np(conus_target)
    panels.append(conus_np[var_idx])
    titles.append(f"CONUS404 Target\n({CONUS_VAR_NAMES[var_idx]})")

    # 3) DRN prediction
    drn_np = to_np(drn_pred)
    panels.append(drn_np[var_idx])
    titles.append(f"DRN Prediction\n(Stage 1)")

    # 4) Residual
    if residual is not None:
        res_np = to_np(residual)
        panels.append(res_np[var_idx])
        titles.append("Residual\n(Target - DRN)")

    # 5) Noisy latent (show channel 0)
    if noisy_latent is not None:
        lat_np = to_np(noisy_latent)
        panels.append(lat_np[0])
        titles.append("Noisy Latent\n(z_t, ch0)")

    # 6) VAE reconstruction
    if vae_recon is not None:
        vae_np = to_np(vae_recon)
        panels.append(vae_np[var_idx])
        titles.append("VAE Recon\n(of residual)")

    # 7) Diffusion sample (decoded residual)
    if diffusion_sample is not None:
        diff_np = to_np(diffusion_sample)
        panels.append(diff_np[var_idx])
        titles.append("Diffusion Sample\n(decoded residual)")

    # 8) Final prediction
    if final_pred is not None:
        final_np = to_np(final_pred)
        panels.append(final_np[var_idx])
        titles.append("Final Prediction\n(DRN + Diffusion)")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, panel, title in zip(axes, panels, titles):
        vmin, vmax = np.nanpercentile(panel, [2, 98])
        im = ax.imshow(panel, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14, y=1.02)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {save_path}")


def plot_loss_curves(
    train_losses: list,
    val_losses: list = None,
    title: str = "Training Loss",
    save_path: str = "loss_curve.png",
):
    """Plot training (and optionally validation) loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train", linewidth=2)
    if val_losses:
        ax.plot(val_losses, label="Val", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curve to {save_path}")


def plot_latent_distribution(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    save_path: str = "latent_dist.png",
):
    """Plot histogram of VAE latent mean and std to verify near-Gaussian behavior."""
    mu_np = mu.detach().cpu().float().numpy().flatten()
    std_np = torch.exp(0.5 * logvar).detach().cpu().float().numpy().flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(mu_np, bins=100, density=True, alpha=0.7, color="steelblue")
    ax1.set_title("Latent Mean (μ)")
    ax1.set_xlabel("Value")
    ax1.axvline(0, color="red", linestyle="--", alpha=0.5)

    ax2.hist(std_np, bins=100, density=True, alpha=0.7, color="coral")
    ax2.set_title("Latent Std (σ)")
    ax2.set_xlabel("Value")
    ax2.axvline(1, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved latent distribution to {save_path}")
