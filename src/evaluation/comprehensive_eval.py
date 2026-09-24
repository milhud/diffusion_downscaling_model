"""Comprehensive paper-ready evaluation for the latent CorrDiff downscaling model.

Covers metrics from CorrDiff (Mardani et al. 2025) and R2-D2 (Lopez-Gomez et al. 2025):

Baseline:
  - ERA5 bilinear interpolation
  - DRN deterministic prediction

Full model (DRN + VAE + Diffusion, 32 members):
  - CRPS, RMSE, MAE (already in ensemble_eval; reproduced here for baselines)
  - Log-PDF distribution matching (all vars + wind speed)
  - Extreme quantile errors (90th / 95th / 99th percentile)
  - Wind speed distribution (U10^2 + V10^2)^0.5
  - Multivariate joint KDE (pairs of vars)
  - Tail dependence (compound extremes)
  - Spatial bias maps
  - Spatial coherence (structure function)
  - Ensemble spread maps
  - SSIM / PSNR (scikit-image, optional)
  - Calibration: spread-skill ratio curve

Usage:
    python -m src.evaluation.comprehensive_eval --output_dir results/comprehensive
"""

import argparse
import warnings
import torch
import torch.nn.functional as F
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
from matplotlib.colors import LogNorm

# Optional image-quality metrics
try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn("scikit-image not found — SSIM/PSNR skipped")

PALETTE = {
    "ERA5":   "#888888",
    "DRN":    "#F4A261",
    "Diff":   "#2196F3",
    "Target": "#2E7D32",
}


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_pos_embedding(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy, xx], dim=0)


def _era5_interp(era5, target_shape):
    """Bilinear upsample ERA5 to CONUS404 resolution (first OUT_CH channels)."""
    return F.interpolate(
        era5[:, :OUT_CH], target_shape, mode="bilinear", align_corners=False
    )


def _wind_speed(u, v):
    """Element-wise wind speed from U and V."""
    return np.sqrt(u**2 + v**2)


def _log_pdf(values, n_bins=80, eps=1e-6):
    """Return (bin_centers, log10_density)."""
    vmin, vmax = np.nanpercentile(values, 0.5), np.nanpercentile(values, 99.5)
    bins = np.linspace(vmin, vmax, n_bins + 1)
    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, np.log10(np.maximum(counts, eps))


def _structure_function(field, max_lag=32, n_lags=16):
    """Mean squared difference as a function of spatial lag (structure fn)."""
    lags = np.unique(np.logspace(0, np.log10(max_lag), n_lags).astype(int))
    sf = []
    for lag in lags:
        diff_h = field[:, lag:] - field[:, :-lag]
        diff_v = field[lag:, :] - field[:-lag, :]
        sf.append(0.5 * (np.mean(diff_h**2) + np.mean(diff_v**2)))
    return lags, np.array(sf)


def _extreme_quantile_error(pred_mean, target, quantiles=(0.90, 0.95, 0.99)):
    """MAE of predicted vs observed at each empirical quantile threshold."""
    errors = {}
    flat_tgt = target.flatten()
    flat_pred = pred_mean.flatten()
    for q in quantiles:
        thresh = np.nanquantile(flat_tgt, q)
        mask = flat_tgt >= thresh
        if mask.sum() == 0:
            errors[q] = np.nan
        else:
            errors[q] = float(np.mean(np.abs(flat_pred[mask] - flat_tgt[mask])))
    return errors


def _tail_dependence(u, v, threshold=0.90):
    """Empirical coefficient: P(U>q | V>q) where q = threshold quantile."""
    u_thresh = np.nanquantile(u, threshold)
    v_thresh = np.nanquantile(v, threshold)
    both = np.sum((u > u_thresh) & (v > v_thresh))
    v_exc = np.sum(v > v_thresh)
    return float(both / max(v_exc, 1))


# ─── main evaluation ──────────────────────────────────────────────────────────

def run_comprehensive_eval(
    drn, vae, diff_model, ema, schedule,
    test_dl, var_names, output_dir,
    num_members=16, num_steps=32, guidance_scale=0.2,
    device="cuda", max_batches=80,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_vars = len(var_names)
    # U10 index = 2, V10 index = 3 (from CONUS404_VARS order: T2 TD2 U10 V10 PSFC PREC)
    u_idx = var_names.index("U10") if "U10" in var_names else None
    v_idx = var_names.index("V10") if "V10" in var_names else None
    prec_idx = var_names.index("PREC_ACC_NC") if "PREC_ACC_NC" in var_names else None

    # accumulators — flat pixel arrays per variable
    tgt_flat   = {v: [] for v in var_names}
    era5_flat  = {v: [] for v in var_names}
    drn_flat   = {v: [] for v in var_names}
    diff_flat  = {v: [] for v in var_names}   # ensemble mean
    ens_std    = {v: [] for v in var_names}   # ensemble spread

    all_crps   = {v: [] for v in var_names}
    all_rmse_era5 = {v: [] for v in var_names}
    all_rmse_drn  = {v: [] for v in var_names}
    all_rmse_diff = {v: [] for v in var_names}
    all_mae_era5  = {v: [] for v in var_names}
    all_mae_drn   = {v: [] for v in var_names}
    all_mae_diff  = {v: [] for v in var_names}

    # for spread-skill curve
    all_ens_std_vals = {v: [] for v in var_names}
    all_err_vals     = {v: [] for v in var_names}

    # for SSIM/PSNR
    ssim_vals = {v: [] for v in var_names}
    psnr_vals = {v: [] for v in var_names}

    # for spatial bias accumulation (accumulate sum + count)
    bias_sum = {v: None for v in var_names}
    bias_n   = 0

    # for structure function (accumulate all sample fields)
    sf_diff_sf = {v: [] for v in var_names}
    sf_tgt_sf  = {v: [] for v in var_names}

    drn.eval(); vae.eval(); diff_model.eval()

    n_batches = min(len(test_dl), max_batches)
    print(f"[Comprehensive] {n_batches} batches, {num_members} members, {num_steps} steps")

    with torch.no_grad():
        for batch_idx, (era5, conus) in enumerate(test_dl):
            if batch_idx >= max_batches:
                break
            era5 = era5.to(device)
            conus = conus.to(device)

            # ERA5 bilinear baseline
            era5_up = _era5_interp(era5, (PATCH_SIZE, PATCH_SIZE))
            era5_np = era5_up.cpu().numpy()

            # Full pipeline
            with ema.apply():
                drn_pred, samples = run_pipeline(
                    era5, drn, vae, diff_model, schedule,
                    num_steps=num_steps, guidance_scale=guidance_scale,
                    num_samples=num_members, device=device,
                )

            drn_np  = drn_pred.cpu().numpy()       # (B, C, H, W)
            ens_np  = samples.cpu().numpy()         # (B, M, C, H, W)
            tgt_np  = conus.cpu().numpy()           # (B, C, H, W)

            B = tgt_np.shape[0]
            for b in range(B):
                for vi, v in enumerate(var_names):
                    tgt_v  = tgt_np[b, vi]                      # (H, W)
                    era5_v = era5_np[b, vi]
                    drn_v  = drn_np[b, vi]
                    ens_v  = ens_np[b, :, vi, :, :]             # (M, H, W)
                    diff_v = ens_v.mean(axis=0)

                    flat_tgt  = tgt_v.flatten()
                    flat_era5 = era5_v.flatten()
                    flat_drn  = drn_v.flatten()
                    flat_diff = diff_v.flatten()
                    flat_std  = ens_v.std(axis=0).flatten()

                    # store flat arrays (subsample for memory)
                    step = max(1, flat_tgt.shape[0] // 2048)
                    tgt_flat[v].append(flat_tgt[::step])
                    era5_flat[v].append(flat_era5[::step])
                    drn_flat[v].append(flat_drn[::step])
                    diff_flat[v].append(flat_diff[::step])
                    ens_std[v].append(flat_std[::step])

                    # RMSE / MAE
                    all_rmse_era5[v].append(float(np.sqrt(np.mean((flat_era5 - flat_tgt)**2))))
                    all_rmse_drn[v].append(float(np.sqrt(np.mean((flat_drn  - flat_tgt)**2))))
                    all_rmse_diff[v].append(float(np.sqrt(np.mean((flat_diff - flat_tgt)**2))))
                    all_mae_era5[v].append(float(np.mean(np.abs(flat_era5 - flat_tgt))))
                    all_mae_drn[v].append(float(np.mean(np.abs(flat_drn  - flat_tgt))))
                    all_mae_diff[v].append(float(np.mean(np.abs(flat_diff - flat_tgt))))

                    # CRPS (analytical formula, vectorised)
                    M = ens_v.shape[0]
                    ens_r = ens_v.reshape(M, -1)
                    obs_r = flat_tgt
                    mae_ens = np.mean(np.abs(ens_r - obs_r[None, :]), axis=0)
                    sorted_e = np.sort(ens_r, axis=0)
                    weights = 2 * np.arange(1, M+1) - M - 1
                    pw = np.sum(weights[:, None] * sorted_e, axis=0) / (M * M)
                    all_crps[v].append(float(np.mean(mae_ens - pw)))

                    # spread-skill (pixel-level, subsampled)
                    all_ens_std_vals[v].append(flat_std[::step])
                    all_err_vals[v].append(np.abs(flat_diff - flat_tgt)[::step])

                    # spatial bias accumulation
                    if bias_sum[v] is None:
                        bias_sum[v] = (diff_v - tgt_v).copy()
                    else:
                        bias_sum[v] += (diff_v - tgt_v)

                    # structure functions (keep one field per batch to limit memory)
                    sf_diff_sf[v].append(_structure_function(diff_v)[1])
                    sf_tgt_sf[v].append(_structure_function(tgt_v)[1])

                    # SSIM / PSNR
                    if HAS_SKIMAGE:
                        drange = float(tgt_v.max() - tgt_v.min()) + 1e-6
                        ssim_vals[v].append(ssim_fn(tgt_v, diff_v, data_range=drange))
                        psnr_vals[v].append(psnr_fn(tgt_v, diff_v, data_range=drange))

            bias_n += B
            if (batch_idx + 1) % 10 == 0:
                print(f"  [{batch_idx+1}/{n_batches}]")

    # ── concatenate ──────────────────────────────────────────────────────────
    for v in var_names:
        tgt_flat[v]  = np.concatenate(tgt_flat[v])
        era5_flat[v] = np.concatenate(era5_flat[v])
        drn_flat[v]  = np.concatenate(drn_flat[v])
        diff_flat[v] = np.concatenate(diff_flat[v])
        ens_std[v]   = np.concatenate(ens_std[v])
        all_ens_std_vals[v] = np.concatenate(all_ens_std_vals[v])
        all_err_vals[v]     = np.concatenate(all_err_vals[v])
        sf_diff_sf[v] = np.mean(sf_diff_sf[v], axis=0)
        sf_tgt_sf[v]  = np.mean(sf_tgt_sf[v], axis=0)
        bias_sum[v]  = bias_sum[v] / max(bias_n, 1)

    lags, _ = _structure_function(np.zeros((PATCH_SIZE, PATCH_SIZE)))  # just get lag values

    # ── 1. Summary table ─────────────────────────────────────────────────────
    print("\n=== Summary ===")
    with open(out / "summary.txt", "w") as f:
        hdr = f"{'Variable':<18} {'Metric':>6}  {'ERA5':>8}  {'DRN':>8}  {'Diff':>8}  {'CRPS':>8}  {'SSR':>6}"
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")
        print(hdr)
        for v in var_names:
            era5_r = np.mean(all_rmse_era5[v]); drn_r = np.mean(all_rmse_drn[v])
            diff_r = np.mean(all_rmse_diff[v]); crps  = np.mean(all_crps[v])
            spread = ens_std[v].mean()
            skill  = np.sqrt(np.mean((diff_flat[v] - tgt_flat[v])**2))
            ssr    = float(spread / max(skill, 1e-10))
            line = f"{v:<18} {'RMSE':>6}  {era5_r:>8.4f}  {drn_r:>8.4f}  {diff_r:>8.4f}  {crps:>8.4f}  {ssr:>6.3f}"
            f.write(line + "\n"); print(line)
            if HAS_SKIMAGE and ssim_vals[v]:
                s = np.mean(ssim_vals[v]); p = np.mean(psnr_vals[v])
                f.write(f"  SSIM={s:.4f}  PSNR={p:.2f} dB\n")
    print(f"Saved {out/'summary.txt'}")

    # ── 2. Log-PDF distribution matching ─────────────────────────────────────
    ncols = min(n_vars, 3)
    nrows = int(np.ceil(n_vars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for i, (v, ax) in enumerate(zip(var_names, axes)):
        unit = VARIABLE_UNITS.get(v, "")
        for arr, label, color in [
            (tgt_flat[v],  "Target", PALETTE["Target"]),
            (era5_flat[v], "ERA5",   PALETTE["ERA5"]),
            (drn_flat[v],  "DRN",    PALETTE["DRN"]),
            (diff_flat[v], "Diff",   PALETTE["Diff"]),
        ]:
            xc, logp = _log_pdf(arr)
            ax.plot(xc, logp, label=label, color=color, lw=1.5)
        ax.set_title(VARIABLE_NAMES.get(v, v))
        ax.set_xlabel(f"Value ({unit})")
        ax.set_ylabel("log10 density")
        ax.legend(fontsize=7)
    for ax in axes[n_vars:]:
        ax.set_visible(False)
    fig.suptitle("Distribution Matching (Log-PDF)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "log_pdf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved log_pdf.png")

    # ── 3. Wind speed distribution (if U10/V10 available) ───────────────────
    if u_idx is not None and v_idx is not None:
        u_v, v_v = "U10", "V10"
        ws_tgt  = _wind_speed(tgt_flat[u_v],  tgt_flat[v_v])
        ws_era5 = _wind_speed(era5_flat[u_v], era5_flat[v_v])
        ws_drn  = _wind_speed(drn_flat[u_v],  drn_flat[v_v])
        ws_diff = _wind_speed(diff_flat[u_v],  diff_flat[v_v])

        fig, ax = plt.subplots(figsize=(6, 4))
        for arr, label, color in [
            (ws_tgt,  "Target", PALETTE["Target"]),
            (ws_era5, "ERA5",   PALETTE["ERA5"]),
            (ws_drn,  "DRN",    PALETTE["DRN"]),
            (ws_diff, "Diff",   PALETTE["Diff"]),
        ]:
            xc, logp = _log_pdf(arr)
            ax.plot(xc, logp, label=label, color=color, lw=1.8)
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("log10 density")
        ax.set_title("10m Wind Speed Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "wind_speed_pdf.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved wind_speed_pdf.png")

    # ── 4. Extreme quantile errors ────────────────────────────────────────────
    quantiles = (0.90, 0.95, 0.99)
    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 4), squeeze=False)
    axes = axes[0]
    for ax, v in zip(axes, var_names):
        era5_eq = _extreme_quantile_error(era5_flat[v], tgt_flat[v], quantiles)
        drn_eq  = _extreme_quantile_error(drn_flat[v],  tgt_flat[v], quantiles)
        diff_eq = _extreme_quantile_error(diff_flat[v], tgt_flat[v], quantiles)
        x = np.array([q * 100 for q in quantiles])
        ax.plot(x, [era5_eq[q] for q in quantiles], "o-", color=PALETTE["ERA5"],   label="ERA5",   lw=1.5)
        ax.plot(x, [drn_eq[q]  for q in quantiles], "s-", color=PALETTE["DRN"],    label="DRN",    lw=1.5)
        ax.plot(x, [diff_eq[q] for q in quantiles], "^-", color=PALETTE["Diff"],   label="Diff",   lw=1.5)
        ax.set_title(VARIABLE_NAMES.get(v, v))
        ax.set_xlabel("Quantile threshold (%)")
        ax.set_ylabel("MAE at extremes")
        ax.legend(fontsize=7)
    fig.suptitle("Extreme Quantile MAE", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "extreme_quantile_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved extreme_quantile_errors.png")

    # ── 5. Multivariate joint KDE (T2 vs PREC if available, + U10 vs V10) ──
    pairs = []
    if u_idx is not None and v_idx is not None:
        pairs.append(("U10", "V10"))
    if prec_idx is not None and "T2" in var_names:
        pairs.append(("T2", "PREC_ACC_NC"))
    if "T2" in var_names and "TD2" in var_names:
        pairs.append(("T2", "TD2"))

    if pairs:
        fig, axes = plt.subplots(len(pairs), 2, figsize=(10, 4 * len(pairs)))
        if len(pairs) == 1:
            axes = axes[None, :]
        for row, (va, vb) in enumerate(pairs):
            for col, (arr_a, arr_b, title, cmap) in enumerate([
                (tgt_flat[va],  tgt_flat[vb],  "Target", "Greens"),
                (diff_flat[va], diff_flat[vb],  "Diff",   "Blues"),
            ]):
                ax = axes[row, col]
                sub = min(len(arr_a), 30000)
                rng = np.random.default_rng(0)
                idx = rng.choice(len(arr_a), sub, replace=False)
                h, xedges, yedges = np.histogram2d(
                    arr_a[idx], arr_b[idx], bins=60, density=True)
                h = np.maximum(h, 1e-6)
                ax.pcolormesh(xedges, yedges, h.T, norm=LogNorm(), cmap=cmap, shading="auto")
                ax.set_xlabel(f"{VARIABLE_NAMES.get(va, va)} ({VARIABLE_UNITS.get(va, '')})")
                ax.set_ylabel(f"{VARIABLE_NAMES.get(vb, vb)} ({VARIABLE_UNITS.get(vb, '')})")
                ax.set_title(title)
        fig.suptitle("Joint Distribution (2D histogram, log scale)", fontsize=13)
        fig.tight_layout()
        fig.savefig(out / "joint_distributions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved joint_distributions.png")

    # ── 6. Tail dependence ───────────────────────────────────────────────────
    if pairs:
        td_results = {}
        for va, vb in pairs:
            td_results[(va, vb)] = {
                "target": _tail_dependence(tgt_flat[va], tgt_flat[vb]),
                "diff":   _tail_dependence(diff_flat[va], diff_flat[vb]),
                "era5":   _tail_dependence(era5_flat[va], era5_flat[vb]),
                "drn":    _tail_dependence(drn_flat[va], drn_flat[vb]),
            }
        with open(out / "tail_dependence.txt", "w") as f:
            f.write("Tail dependence coefficient P(A>q90 | B>q90)\n")
            f.write(f"{'Pair':<24} {'Target':>8} {'ERA5':>8} {'DRN':>8} {'Diff':>8}\n")
            f.write("-" * 60 + "\n")
            for (va, vb), td in td_results.items():
                line = (f"{va+'/'+vb:<24} {td['target']:>8.4f} "
                        f"{td['era5']:>8.4f} {td['drn']:>8.4f} {td['diff']:>8.4f}")
                f.write(line + "\n")
                print(line)
        print("Saved tail_dependence.txt")

    # ── 7. Spatial bias maps ──────────────────────────────────────────────────
    ncols = min(n_vars, 3)
    nrows = int(np.ceil(n_vars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for ax, v in zip(axes, var_names):
        bias = bias_sum[v]
        vmax = np.nanpercentile(np.abs(bias), 98)
        im = ax.imshow(bias, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(f"{VARIABLE_NAMES.get(v, v)} bias\n(Diff − Target)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=VARIABLE_UNITS.get(v, ""))
    for ax in axes[n_vars:]:
        ax.set_visible(False)
    fig.suptitle("Mean Spatial Bias (Diff ensemble mean − Target)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "spatial_bias.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved spatial_bias.png")

    # ── 8. Spatial coherence (structure function) ─────────────────────────────
    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 4), squeeze=False)
    axes = axes[0]
    dx_km = 4.0
    for ax, v in zip(axes, var_names):
        lags_km = lags * dx_km
        ax.loglog(lags_km, sf_tgt_sf[v],  color=PALETTE["Target"], lw=2, label="Target")
        ax.loglog(lags_km, sf_diff_sf[v], color=PALETTE["Diff"],   lw=2, label="Diff")
        ax.set_xlabel("Lag (km)")
        ax.set_ylabel("Structure function D(r)")
        ax.set_title(VARIABLE_NAMES.get(v, v))
        ax.legend(fontsize=7)
    fig.suptitle("Spatial Coherence: Structure Function D(r) = E[|f(x+r) - f(x)|²]",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "spatial_coherence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved spatial_coherence.png")

    # ── 9. Ensemble spread map (last batch, first sample) ────────────────────
    # Re-run one batch to get spatial spread map
    drn.eval(); vae.eval(); diff_model.eval()
    try:
        era5_b, conus_b = next(iter(test_dl))
        era5_b = era5_b.to(device); conus_b = conus_b.to(device)
        with torch.no_grad(), ema.apply():
            _, samples_b = run_pipeline(
                era5_b, drn, vae, diff_model, schedule,
                num_steps=num_steps, guidance_scale=guidance_scale,
                num_samples=num_members, device=device,
            )
        ens_b = samples_b[0].cpu().numpy()    # (M, C, H, W)
        tgt_b = conus_b[0].cpu().numpy()      # (C, H, W)
        spread_b = ens_b.std(axis=0)          # (C, H, W)
        err_b = np.abs(ens_b.mean(axis=0) - tgt_b)

        ncols = min(n_vars, 3)
        nrows = int(np.ceil(n_vars / ncols))
        fig, axes = plt.subplots(nrows, ncols * 2,
                                 figsize=(5 * ncols * 2, 4 * nrows))
        axes = np.array(axes).reshape(-1)
        ai = 0
        for vi, v in enumerate(var_names):
            for data, title, cmap in [
                (spread_b[vi], "Spread", "YlOrRd"),
                (err_b[vi],    "|Error|", "PuRd"),
            ]:
                ax = axes[ai]; ai += 1
                im = ax.imshow(data, origin="lower", cmap=cmap)
                ax.set_title(f"{VARIABLE_NAMES.get(v, v)}\n{title}")
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label=VARIABLE_UNITS.get(v, ""))
        for ax in axes[ai:]:
            ax.set_visible(False)
        fig.suptitle("Ensemble Spread and Absolute Error (single sample)", fontsize=13)
        fig.tight_layout()
        fig.savefig(out / "ensemble_spread_maps.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved ensemble_spread_maps.png")
    except Exception as e:
        print(f"Spread map skipped: {e}")

    # ── 10. Spread-skill curve ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 4), squeeze=False)
    axes = axes[0]
    for ax, v in zip(axes, var_names):
        std_vals = all_ens_std_vals[v]
        err_vals = all_err_vals[v]
        # bin by spread level, compute mean error per bin
        order = np.argsort(std_vals)
        n_bins = 20
        bin_edges = np.percentile(std_vals, np.linspace(0, 100, n_bins + 1))
        bin_spread, bin_skill = [], []
        for i in range(n_bins):
            mask = (std_vals >= bin_edges[i]) & (std_vals < bin_edges[i + 1])
            if mask.sum() > 10:
                bin_spread.append(std_vals[mask].mean())
                bin_skill.append(err_vals[mask].mean())
        if bin_spread:
            ax.scatter(bin_spread, bin_skill, s=20, color=PALETTE["Diff"])
            lim = max(max(bin_spread), max(bin_skill))
            ax.plot([0, lim], [0, lim], "r--", lw=1, label="Perfect (SSR=1)")
        ax.set_xlabel("Ensemble spread (std)")
        ax.set_ylabel("Mean |error|")
        ax.set_title(VARIABLE_NAMES.get(v, v))
        ax.legend(fontsize=7)
    fig.suptitle("Spread-Skill Diagram", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "spread_skill_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved spread_skill_diagram.png")

    # ── 11. SSIM / PSNR summary ──────────────────────────────────────────────
    if HAS_SKIMAGE:
        with open(out / "image_quality.txt", "w") as f:
            f.write(f"{'Variable':<20} {'SSIM':>8} {'PSNR (dB)':>10}\n")
            f.write("-" * 42 + "\n")
            for v in var_names:
                if ssim_vals[v]:
                    line = f"{v:<20} {np.mean(ssim_vals[v]):>8.4f} {np.mean(psnr_vals[v]):>10.2f}"
                    f.write(line + "\n"); print(line)
        print("Saved image_quality.txt")

    # ── 12. Precipitation-specific: log-log tail plot ────────────────────────
    if prec_idx is not None:
        v = "PREC_ACC_NC"
        fig, ax = plt.subplots(figsize=(6, 4))
        for arr, label, color in [
            (tgt_flat[v],  "Target", PALETTE["Target"]),
            (era5_flat[v], "ERA5",   PALETTE["ERA5"]),
            (drn_flat[v],  "DRN",    PALETTE["DRN"]),
            (diff_flat[v], "Diff",   PALETTE["Diff"]),
        ]:
            pos = arr[arr > 0]
            if pos.size == 0:
                continue
            xc, logp = _log_pdf(pos, n_bins=60)
            ax.plot(xc, logp, label=label, color=color, lw=1.8)
        ax.set_xlabel("Precipitation (mm)")
        ax.set_ylabel("log10 density")
        ax.set_title("Precipitation Distribution (positive values only)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "precip_tail.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved precip_tail.png")

    print(f"\n[Comprehensive] All outputs in {out}/")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_members",    type=int,   default=16)
    parser.add_argument("--num_steps",      type=int,   default=32)
    parser.add_argument("--guidance_scale", type=float, default=0.2)
    parser.add_argument("--output_dir",     default="results/comprehensive")
    parser.add_argument("--max_batches",    type=int,   default=80)
    parser.add_argument("--drn_checkpoint",  default="checkpoints/drn_best.pt")
    parser.add_argument("--vae_checkpoint",  default="checkpoints/vae_best.pt")
    parser.add_argument("--diff_checkpoint", default="checkpoints/diffusion_best.pt")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--data_dir",        default="data")
    parser.add_argument("--cache_dir",       default="cached_data")
    args = parser.parse_args()

    from src.evaluation._eval_setup import build_test_dataloader, load_models

    test_dl, stats, land_mask, valid_origins = build_test_dataloader(
        data_dir=args.data_dir, cache_dir=args.cache_dir,
        batch_size=4, num_workers=2, patches_per_day=1)

    drn, vae, diff_model, ema, schedule = load_models(
        args.drn_checkpoint, args.vae_checkpoint, args.diff_checkpoint, args.device)

    run_comprehensive_eval(
        drn, vae, diff_model, ema, schedule,
        test_dl, CONUS404_VARS, args.output_dir,
        num_members=args.num_members, num_steps=args.num_steps,
        guidance_scale=args.guidance_scale, device=args.device,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
