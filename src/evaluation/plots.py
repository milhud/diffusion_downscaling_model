"""Periodic evaluation and diagnostic plots during training.

Generates the same diagnostic plots as sanity_check.py but called
periodically from the training loop.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from config import IN_CH

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return t


def _shared_limits(arrays, percentile=(2, 98)):
    """Compute shared vmin/vmax across multiple arrays."""
    all_vals = np.concatenate([_to_np(a).flatten() for a in arrays])
    all_vals = all_vals[np.isfinite(all_vals)]
    return np.nanpercentile(all_vals, percentile)


def plot_loss_curves(losses_dict, save_path, smooth_window=None):
    """Plot multiple loss curves (raw + smoothed) on one figure.

    Args:
        losses_dict: {name: list_of_losses}
    """
    n = len(losses_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4))
    if n == 1:
        axes = [axes]
    colors = ["steelblue", "coral", "forestgreen", "purple"]
    for ax, (name, losses), color in zip(axes, losses_dict.items(), colors):
        sw = smooth_window or max(5, len(losses) // 20)
        ax.plot(losses, linewidth=0.5, alpha=0.25, color=color)
        if len(losses) > sw:
            kernel = np.ones(sw) / sw
            smoothed = np.convolve(losses, kernel, mode="valid")
            offset = sw // 2
            ax.plot(range(offset, offset + len(smoothed)), smoothed,
                    linewidth=2.5, color=color)
        ax.set_title(name); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stage_comparison(panels, titles, save_path, suptitle="",
                          share_groups=None, cmap="RdBu_r"):
    """Plot panels with optional shared colorbar scales.

    Args:
        share_groups: list of group ids, e.g. [0,0,0,1]. Same id = shared scale.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    panels_np = [_to_np(p) for p in panels]

    if share_groups is not None:
        groups = {}
        for i, g in enumerate(share_groups):
            groups.setdefault(g, []).append(i)
        group_ranges = {}
        for g, idxs in groups.items():
            vals = np.concatenate([panels_np[i].flatten() for i in idxs])
            vals = vals[np.isfinite(vals)]
            group_ranges[g] = np.nanpercentile(vals, [2, 98]) if len(vals) > 0 else (0, 1)
    else:
        group_ranges = None

    for i, (ax, p, title) in enumerate(zip(axes, panels_np, titles)):
        if group_ranges is not None:
            vmin, vmax = group_ranges[share_groups[i]]
        else:
            finite = p[np.isfinite(p)] if np.any(np.isfinite(p)) else np.array([0, 1])
            vmin, vmax = np.nanpercentile(finite, [2, 98])
        im = ax.imshow(p, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title, fontsize=10); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def radial_power_spectrum(field_2d):
    """Compute radially averaged power spectrum of a 2D field."""
    f = np.fft.fft2(field_2d)
    f = np.fft.fftshift(f)
    power = np.abs(f) ** 2
    H, W = field_2d.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cy, cx)
    spectrum = np.zeros(max_r)
    for ri in range(max_r):
        mask = r == ri
        if mask.any():
            spectrum[ri] = power[mask].mean()
    return spectrum


def evaluate_drn(drn, era5_batch, conus_batch, plot_dir, epoch, device="cuda"):
    """Generate DRN evaluation plots."""
    drn.eval()
    with torch.no_grad():
        pred = drn(era5_batch.to(device))
        residual = conus_batch.to(device) - pred

    plot_stage_comparison(
        [era5_batch[0, 0], conus_batch[0, 0], pred[0, 0], residual[0, 0]],
        ["ERA5 input", "CONUS404 target", "DRN prediction", "Residual"],
        f"{plot_dir}/drn_epoch{epoch:03d}.png",
        suptitle=f"DRN — Epoch {epoch}",
        share_groups=[0, 0, 0, 1],
    )
    rmse = ((pred.cpu() - conus_batch) ** 2).mean().sqrt().item()
    return rmse


def evaluate_full_pipeline(drn, vae, diff_model, ema, schedule, era5_batch,
                           conus_batch, plot_dir, epoch, latent_ch=4,
                           num_sampling_steps=16, num_ensemble=4,
                           guidance_scale=0.2, device="cuda"):
    """Generate full pipeline evaluation plots (DRN + Diffusion)."""
    from ..models.edm import heun_sampler

    drn.eval(); vae.eval(); diff_model.eval()
    latent_h = latent_w = 64

    era5 = era5_batch[:1].to(device)
    conus = conus_batch[:1].to(device)

    with torch.no_grad():
        drn_pred = drn(era5)
        residual = conus - drn_pred
        # Build conditioning
        era5_down = F.interpolate(era5[:, :IN_CH], (latent_h, latent_w),
                                  mode="bilinear", align_corners=False)
        mu_drn, _ = vae.encode(drn_pred)
        ys = torch.linspace(-1, 1, latent_h, device=device)
        xs = torch.linspace(-1, 1, latent_w, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([yy, xx], dim=0).unsqueeze(0)
        cond = torch.cat([era5_down, mu_drn, pos], dim=1)

    # Ensemble sampling
    ensemble = []
    with ema.apply():
        for _ in range(num_ensemble):
            z = heun_sampler(diff_model, schedule, cond,
                             shape=(1, latent_ch, latent_h, latent_w),
                             num_steps=num_sampling_steps,
                             guidance_scale=guidance_scale)
            with torch.no_grad():
                r = vae.decode(z)
                ensemble.append(drn_pred + r)

    ens_stack = torch.cat(ensemble, dim=0)
    ens_mean = ens_stack.mean(dim=0, keepdim=True)
    final = ensemble[-1]

    # Plot: target / DRN / final / error (shared scale)
    plot_stage_comparison(
        [conus[0, 0], drn_pred[0, 0], final[0, 0],
         (conus[0, 0] - final[0, 0])],
        ["Target", "DRN", f"DRN+Diff", "Error"],
        f"{plot_dir}/pipeline_epoch{epoch:03d}.png",
        suptitle=f"Full Pipeline — Epoch {epoch}",
        share_groups=[0, 0, 0, 1],
    )

    # Power spectra
    target_spec = radial_power_spectrum(_to_np(conus[0, 0]))
    drn_spec = radial_power_spectrum(_to_np(drn_pred[0, 0]))
    final_spec = radial_power_spectrum(_to_np(final[0, 0]))
    ens_mean_spec = radial_power_spectrum(_to_np(ens_mean[0, 0]))

    fig, ax = plt.subplots(figsize=(8, 5))
    k = np.arange(1, len(target_spec))
    ax.loglog(k, target_spec[1:], 'k-', linewidth=2, label='Target')
    ax.loglog(k, drn_spec[1:], 'g-', linewidth=1.5, label='DRN')
    ax.loglog(k, final_spec[1:], 'r-', linewidth=1.5, label='DRN+Diff')
    ax.loglog(k, ens_mean_spec[1:], 'm--', linewidth=1.5, label='Ens Mean')
    ax.set_xlabel("Wavenumber k"); ax.set_ylabel("Power")
    ax.set_title(f"Power Spectrum — Epoch {epoch}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/spectra_epoch{epoch:03d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Error histogram
    drn_err = _to_np((drn_pred - conus)[0, 0]).flatten()
    final_err = _to_np((final - conus)[0, 0]).flatten()
    ens_err = _to_np((ens_mean - conus)[0, 0]).flatten()

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-3, 3, 100)
    ax.hist(drn_err, bins=bins, density=True, alpha=0.5, color="green",
            label=f"DRN (std={drn_err.std():.3f})")
    ax.hist(final_err, bins=bins, density=True, alpha=0.5, color="red",
            label=f"Final (std={final_err.std():.3f})")
    ax.hist(ens_err, bins=bins, density=True, alpha=0.5, color="purple",
            label=f"Ens Mean (std={ens_err.std():.3f})")
    ax.set_xlabel("Error"); ax.set_ylabel("Density")
    ax.set_title(f"Error Distribution — Epoch {epoch}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/error_hist_epoch{epoch:03d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    drn_rmse = np.sqrt((drn_err ** 2).mean())
    final_rmse = np.sqrt((final_err ** 2).mean())
    return drn_rmse, final_rmse
