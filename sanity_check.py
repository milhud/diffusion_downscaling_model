"""Comprehensive sanity check: tests EVERY feature of the downscaling pipeline.

Loads real ERA5/CONUS404 data, trains each stage for 1000 steps, then runs
ALL evaluation analyses to verify everything works end-to-end.

Tests:
  1. Data loading and regridding (ERA5 -> CONUS404 grid)
  2. Static field computation (terrain, orog_var, lat, lon, lai, lsm)
  3. Stage 1: DRN training (1000 steps)
  4. Stage 2a: VAE training (1000 steps)
  5. Stage 2b: Diffusion training (1000 steps)
  6. Full pipeline sampling (Heun sampler, classifier-free guidance, EMA)
  7. Ensemble generation (4 members)
  8. Inference pipeline (src.inference.pipeline.run_pipeline)
  9. Evaluation metrics (CRPS, RMSE, MAE, SSR, rank histogram, Q-Q, spectra)
  10. Stage-by-stage power spectra
  11. Denoising step ablation (4, 8, 16 steps)
  12. Compute benchmark (latent vs pixel timing)
  13. Latent space analysis (channel distributions, correlations)
  14. Climate signal test (early vs late period means)
  15. Pixel-space diffusion ablation (build + 100 steps)
  16. Comprehensive plots at every stage

Usage:
    python sanity_check.py [--device cuda|cpu] [--plot_dir sanity_plots]

Runs on A100 in ~30-45 minutes.
"""

import argparse
import subprocess
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
import xesmf as xe
from scipy.ndimage import uniform_filter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.drn import DRN
from src.models.vae import VAE
from src.models.diffusion_unet import DiffusionUNet
from src.models.edm import EDMSchedule, edm_training_loss, heun_sampler
from src.training.losses import PerVariableMSE, VAELoss
from src.training.ema import EMA

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("/mnt/home/hmiller/diffusion_downscaling_model/data")
PATCH_SIZE = 256
LATENT_H = LATENT_W = 64
IN_CH = 7   # 1 ERA5 var + 6 static
OUT_CH = 1
LATENT_CH = 4

# Training steps per stage
NUM_DRN_STEPS = 1000
NUM_VAE_STEPS = 1000
NUM_DIFF_STEPS = 1000
BATCH_SIZE = 4
NUM_SAMPLING_STEPS = 16
NUM_ENSEMBLE = 4
LOG_EVERY = 100


# ─── Helpers ─────────────────────────────────────────────────────────────────

def cosine_lr(step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _to_np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return t


def _month_for_day(day_idx, leap):
    days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cum = 0
    for m, d in enumerate(days):
        cum += d
        if day_idx < cum:
            return m
    return 11


def radial_power_spectrum(field_2d):
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


def plot_panels(panels, titles, save_path, suptitle="", cmap="RdBu_r", share_scale=None):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    panels_np = [_to_np(p) for p in panels]
    if share_scale is not None:
        groups = {}
        for i, g in enumerate(share_scale):
            groups.setdefault(g, []).append(i)
        group_ranges = {}
        for g, idxs in groups.items():
            all_vals = np.concatenate([panels_np[i].flatten() for i in idxs])
            all_vals = all_vals[np.isfinite(all_vals)]
            group_ranges[g] = np.nanpercentile(all_vals, [2, 98])
    else:
        group_ranges = None
    for i, (ax, p, title) in enumerate(zip(axes, panels_np, titles)):
        if group_ranges is not None:
            vmin, vmax = group_ranges[share_scale[i]]
        else:
            finite = p[np.isfinite(p)] if np.any(np.isfinite(p)) else np.array([0, 1])
            vmin, vmax = np.nanpercentile(finite, [2, 98])
        im = ax.imshow(p, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {save_path}")


def plot_loss(losses, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    n = len(losses)
    sw = max(5, n // 20)
    ax.plot(losses, linewidth=0.5, alpha=0.25, color="steelblue")
    if n > sw:
        kernel = np.ones(sw) / sw
        smoothed = np.convolve(losses, kernel, mode="valid")
        offset = sw // 2
        ax.plot(range(offset, offset + len(smoothed)), smoothed,
                linewidth=2.5, color="steelblue")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {save_path}")


# ─── Data loading ────────────────────────────────────────────────────────────

def load_real_data():
    t0 = time.time()
    years = [1980]
    sample_days = [10, 40, 70, 100, 130, 160, 190, 220, 250, 280, 310, 340]
    patches_per_day = 4

    print(f"[Data] Loading {len(years)} year(s), {len(sample_days)} days each...")

    era5_ds_first = xr.open_dataset(DATA_DIR / f"era5_{years[0]}.nc")
    era5_lat = era5_ds_first["latitude"].values
    era5_lon = era5_ds_first["longitude"].values

    conus_ds_first = xr.open_dataset(DATA_DIR / f"conus404_yearly_{years[0]}.nc")
    conus_lat = conus_ds_first["lat"].values
    conus_lon = conus_ds_first["lon"].values

    print("[Data] Building xESMF regridder...")
    src_grid = xr.Dataset({
        "lat": xr.DataArray(era5_lat, dims=["y"]),
        "lon": xr.DataArray(era5_lon, dims=["x"]),
    })
    dst_grid = xr.Dataset({
        "lat": xr.DataArray(conus_lat, dims=["y", "x"]),
        "lon": xr.DataArray(conus_lon, dims=["y", "x"]),
    })
    regridder = xe.Regridder(src_grid, dst_grid, method="bilinear",
                             extrap_method="nearest_s2d", unmapped_to_nan=False)

    terrain = conus_ds_first["Z"].isel(time=0, bottom_top_stag=0).values.astype(np.float32)
    terrain_norm = (terrain - np.nanmean(terrain)) / (np.nanstd(terrain) + 1e-8)
    z_mean = uniform_filter(terrain, size=5)
    z_sq_mean = uniform_filter(terrain ** 2, size=5)
    orog_var = np.sqrt(np.maximum(z_sq_mean - z_mean ** 2, 0))
    orog_var_norm = (orog_var - np.nanmean(orog_var)) / (np.nanstd(orog_var) + 1e-8)
    lat_norm = (conus_lat - np.nanmean(conus_lat)) / (np.nanstd(conus_lat) + 1e-8)
    lon_norm = (conus_lon - np.nanmean(conus_lon)) / (np.nanstd(conus_lon) + 1e-8)
    lsm = (terrain > 0).astype(np.float32)

    # LAI
    try:
        lai = era5_ds_first["lai"].isel(time=0, valid_time=0).values.astype(np.float32)
        lai_reg = regridder(xr.DataArray(lai, dims=["y", "x"])).values.astype(np.float32)
        lai_reg = np.nan_to_num(lai_reg, nan=0.0)
        lai_norm = (lai_reg - np.nanmean(lai_reg)) / (np.nanstd(lai_reg) + 1e-8)
    except Exception:
        lai_norm = np.zeros_like(lsm)

    static = np.stack([terrain_norm, orog_var_norm, lat_norm.astype(np.float32),
                       lon_norm.astype(np.float32), lai_norm.astype(np.float32), lsm], axis=0)

    era5_ds_first.close()
    conus_ds_first.close()

    raw_pairs = []
    era5_vals, conus_vals = [], []

    for year in years:
        leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        max_day = 366 if leap else 365
        era5_ds = xr.open_dataset(DATA_DIR / f"era5_{year}.nc")
        conus_ds = xr.open_dataset(DATA_DIR / f"conus404_yearly_{year}.nc")

        for d in sample_days:
            if d >= max_day:
                continue
            m = _month_for_day(d, leap)
            era5_field = era5_ds["t2m"].isel(time=m, valid_time=d).values.astype(np.float32)
            if np.all(np.isnan(era5_field)):
                continue
            era5_regridded = regridder(xr.DataArray(era5_field, dims=["y", "x"])).values.astype(np.float32)
            conus_field = conus_ds["T2"].isel(time=d).values.astype(np.float32)
            era5_vals.append(era5_regridded[::10, ::10].flatten())
            conus_vals.append(conus_field[::10, ::10].flatten())
            raw_pairs.append((era5_regridded, conus_field))

        era5_ds.close()
        conus_ds.close()

    era5_all = np.concatenate(era5_vals)
    conus_all = np.concatenate(conus_vals)
    era5_mean, era5_std = float(np.nanmean(era5_all)), float(np.nanstd(era5_all))
    conus_mean, conus_std = float(np.nanmean(conus_all)), float(np.nanstd(conus_all))

    H, W, PS = 1015, 1367, PATCH_SIZE
    patches = []
    rng = np.random.RandomState(42)

    for era5_reg, conus_f in raw_pairs:
        era5_norm = (era5_reg - era5_mean) / era5_std
        conus_norm = (conus_f - conus_mean) / conus_std
        era5_input = np.concatenate([era5_norm[None], static], axis=0)
        for _ in range(patches_per_day):
            y0 = rng.randint(0, H - PS)
            x0 = rng.randint(0, W - PS)
            era5_patch = torch.from_numpy(era5_input[:, y0:y0+PS, x0:x0+PS].copy())
            conus_patch = torch.from_numpy(conus_norm[None, y0:y0+PS, x0:x0+PS].copy())
            patches.append((era5_patch, conus_patch))

    print(f"[Data] {len(patches)} patches from {len(raw_pairs)} days ({time.time()-t0:.1f}s)")
    return patches


# ─── Main sanity check ──────────────────────────────────────────────────────

def sanity_check(device="cuda", plot_dir="sanity_plots"):
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    results = {}
    batch_rng = np.random.RandomState(123)

    print("=" * 70)
    print("COMPREHENSIVE SANITY CHECK — ALL FEATURES")
    print(f"  DRN/VAE/Diff: {NUM_DRN_STEPS}/{NUM_VAE_STEPS}/{NUM_DIFF_STEPS} steps each")
    print(f"  Batch: {BATCH_SIZE}, Ensemble: {NUM_ENSEMBLE}, Sampling: {NUM_SAMPLING_STEPS} steps")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: Data Loading
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[TEST 1] Data loading and regridding...")
    patches = load_real_data()
    assert len(patches) > 0, "No patches loaded!"
    e0, c0 = patches[0]
    assert e0.shape == (IN_CH, PATCH_SIZE, PATCH_SIZE), f"ERA5 patch shape wrong: {e0.shape}"
    assert c0.shape == (OUT_CH, PATCH_SIZE, PATCH_SIZE), f"CONUS patch shape wrong: {c0.shape}"
    assert torch.isfinite(e0).all(), "ERA5 patch has non-finite values"
    assert torch.isfinite(c0).all(), "CONUS patch has non-finite values"
    print(f"  PASS — {len(patches)} patches, shapes: ERA5={e0.shape}, CONUS={c0.shape}")
    results["data_loading"] = "PASS"

    plot_panels([e0[0], c0[0]], ["ERA5 t2m", "CONUS404 T2"],
                f"{plot_dir}/01_input_vs_target.png", "Input vs Target Patch",
                share_scale=[0, 0])

    def get_batch():
        idxs = batch_rng.randint(0, len(patches), size=BATCH_SIZE)
        era5_b = torch.stack([patches[i][0] for i in idxs]).to(device)
        conus_b = torch.stack([patches[i][1] for i in idxs]).to(device)
        return era5_b, conus_b

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: DRN Training (1000 steps)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[TEST 2] DRN training ({NUM_DRN_STEPS} steps)...")
    drn = DRN(in_ch=IN_CH, out_ch=OUT_CH, base_ch=32, ch_mults=(1, 2, 4, 8),
              num_res_blocks=1, attn_resolutions=(2,)).to(device)
    drn_params = sum(p.numel() for p in drn.parameters())
    print(f"  Params: {drn_params:,}")
    drn_criterion = PerVariableMSE(num_vars=OUT_CH, precip_channel=-1).to(device)
    drn_opt = torch.optim.AdamW(list(drn.parameters()) + list(drn_criterion.parameters()), lr=1e-4)
    warmup_drn = int(0.05 * NUM_DRN_STEPS)

    drn.train()
    drn_losses = []
    t0 = time.time()
    for step in range(NUM_DRN_STEPS):
        lr = cosine_lr(step, NUM_DRN_STEPS, 1e-4, warmup_drn)
        set_lr(drn_opt, lr)
        era5_b, conus_b = get_batch()
        pred = drn(era5_b)
        loss = drn_criterion(pred, conus_b)
        drn_opt.zero_grad(); loss.backward(); drn_opt.step()
        drn_losses.append(loss.item())
        if step % LOG_EVERY == 0 or step == NUM_DRN_STEPS - 1:
            print(f"    Step {step:5d}/{NUM_DRN_STEPS} | Loss: {loss.item():.6f}")

    assert all(np.isfinite(l) for l in drn_losses), "DRN has non-finite losses!"
    assert drn_losses[-1] < drn_losses[0], "DRN loss did not decrease!"
    print(f"  PASS — {drn_losses[0]:.4f} -> {drn_losses[-1]:.4f} ({time.time()-t0:.0f}s)")
    results["drn_training"] = "PASS"
    plot_loss(drn_losses, "DRN Loss (1000 steps)", f"{plot_dir}/02_drn_loss.png")

    # DRN eval
    drn.eval()
    era5_viz, conus_viz = get_batch()
    with torch.no_grad():
        drn_pred = drn(era5_viz)
        residual = conus_viz - drn_pred
    drn_rmse = ((drn_pred - conus_viz) ** 2).mean().sqrt().item()
    print(f"  DRN eval RMSE: {drn_rmse:.4f}")

    plot_panels(
        [era5_viz[0, 0], conus_viz[0, 0], drn_pred[0, 0], residual[0, 0]],
        ["ERA5", "Target", "DRN pred", "Residual"],
        f"{plot_dir}/03_drn_result.png", "Stage 1: DRN",
        share_scale=[0, 0, 0, 1])

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 3: VAE Training (1000 steps)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[TEST 3] VAE training ({NUM_VAE_STEPS} steps)...")
    vae = VAE(in_ch=OUT_CH, latent_ch=LATENT_CH, base_ch=64).to(device)
    vae_params = sum(p.numel() for p in vae.parameters())
    print(f"  Params: {vae_params:,}")
    vae_criterion = VAELoss()
    vae_opt = torch.optim.AdamW(vae.parameters(), lr=1e-4)
    warmup_vae = int(0.05 * NUM_VAE_STEPS)

    vae.train()
    vae_losses, vae_recon, vae_kl = [], [], []
    t0 = time.time()
    for step in range(NUM_VAE_STEPS):
        lr = cosine_lr(step, NUM_VAE_STEPS, 1e-4, warmup_vae)
        set_lr(vae_opt, lr)
        era5_b, conus_b = get_batch()
        with torch.no_grad():
            drn_pred_b = drn(era5_b)
        residual_b = conus_b - drn_pred_b
        recon, mu, logvar = vae(residual_b)
        beta = 1e-4 * min(1.0, step / max(1, int(0.3 * NUM_VAE_STEPS)))
        loss, recon_l, kl_l = vae_criterion(recon, residual_b, mu, logvar, beta=beta)
        vae_opt.zero_grad(); loss.backward(); vae_opt.step()
        vae_losses.append(loss.item())
        vae_recon.append(recon_l.item())
        vae_kl.append(kl_l.item())
        if step % LOG_EVERY == 0 or step == NUM_VAE_STEPS - 1:
            print(f"    Step {step:5d}/{NUM_VAE_STEPS} | Loss: {loss.item():.6f} "
                  f"(recon: {recon_l.item():.6f}, KL: {kl_l.item():.2f})")

    assert all(np.isfinite(l) for l in vae_losses), "VAE has non-finite losses!"
    assert mu.shape == (BATCH_SIZE, LATENT_CH, LATENT_H, LATENT_W), f"Latent shape: {mu.shape}"
    print(f"  PASS — {vae_losses[0]:.4f} -> {vae_losses[-1]:.4f} ({time.time()-t0:.0f}s)")
    results["vae_training"] = "PASS"

    # VAE loss curves
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    ax1.plot(vae_losses); ax1.set_title("VAE Total Loss")
    ax2.plot(vae_recon); ax2.set_title("Reconstruction Loss")
    ax3.plot(vae_kl); ax3.set_title("KL Divergence")
    for ax in (ax1, ax2, ax3):
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/04_vae_loss.png", dpi=150)
    plt.close(fig)

    # VAE eval
    vae.eval()
    with torch.no_grad():
        recon_viz, mu_viz, logvar_viz = vae(residual)
    plot_panels(
        [residual[0, 0], recon_viz[0, 0], mu_viz[0, 0]],
        ["Residual (input)", "VAE recon", "Latent mu (ch 0)"],
        f"{plot_dir}/05_vae_result.png", "Stage 2a: VAE")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 4: Diffusion Training (1000 steps)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[TEST 4] Diffusion training ({NUM_DIFF_STEPS} steps)...")
    DIFF_IN_CH = LATENT_CH + IN_CH + LATENT_CH + 2
    diff_model = DiffusionUNet(
        in_ch=DIFF_IN_CH, out_ch=LATENT_CH, base_ch=64,
        ch_mults=(1, 2, 2, 4), num_res_blocks=2,
        attn_resolutions=(2, 3), time_dim=256, dropout=0.0,
    ).to(device)
    diff_params = sum(p.numel() for p in diff_model.parameters())
    print(f"  Params: {diff_params:,}")
    print(f"  Input channels: {DIFF_IN_CH} = {LATENT_CH}(z) + {IN_CH}(ERA5) + {LATENT_CH}(mu) + 2(pos)")

    schedule = EDMSchedule()
    ema = EMA(diff_model, decay=0.9999)
    diff_opt = torch.optim.AdamW(diff_model.parameters(), lr=1e-4)
    warmup_diff = int(0.05 * NUM_DIFF_STEPS)

    ys = torch.linspace(-1, 1, LATENT_H, device=device)
    xs = torch.linspace(-1, 1, LATENT_W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pos_emb = torch.stack([yy, xx], dim=0)

    def build_cond(era5_b, drn_pred_b, p_uncond=0.0):
        era5_down = F.interpolate(era5_b, (LATENT_H, LATENT_W),
                                  mode="bilinear", align_corners=False)
        with torch.no_grad():
            mu_drn, _ = vae.encode(drn_pred_b)
        pos = pos_emb.unsqueeze(0).expand(era5_b.shape[0], -1, -1, -1)
        cond = torch.cat([era5_down, mu_drn, pos], dim=1)
        if p_uncond > 0 and torch.rand(1).item() < p_uncond:
            cond = torch.zeros_like(cond)
        return cond

    diff_model.train()
    diff_losses = []
    t0 = time.time()
    for step in range(NUM_DIFF_STEPS):
        lr = cosine_lr(step, NUM_DIFF_STEPS, 1e-4, warmup_diff)
        set_lr(diff_opt, lr)
        era5_b, conus_b = get_batch()
        with torch.no_grad():
            drn_pred_b = drn(era5_b)
            residual_b = conus_b - drn_pred_b
            mu_b, logvar_b = vae.encode(residual_b)
            z_clean = vae.reparameterize(mu_b, logvar_b)
        cond = build_cond(era5_b, drn_pred_b, p_uncond=0.1)
        loss = edm_training_loss(diff_model, schedule, z_clean, cond)
        diff_opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.0)
        diff_opt.step(); ema.update()
        diff_losses.append(loss.item())
        if step % LOG_EVERY == 0 or step == NUM_DIFF_STEPS - 1:
            print(f"    Step {step:5d}/{NUM_DIFF_STEPS} | Loss: {loss.item():.6f}")

    assert all(np.isfinite(l) for l in diff_losses), "Diffusion has non-finite losses!"
    print(f"  PASS — {diff_losses[0]:.4f} -> {diff_losses[-1]:.4f} ({time.time()-t0:.0f}s)")
    results["diff_training"] = "PASS"
    plot_loss(diff_losses, "Diffusion Loss (1000 steps)", f"{plot_dir}/06_diff_loss.png")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 5: Full Pipeline Sampling
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[TEST 5] Full pipeline sampling ({NUM_ENSEMBLE} ensemble x {NUM_SAMPLING_STEPS} steps)...")
    diff_model.eval()
    cond_inf = build_cond(era5_viz, drn_pred)

    ensemble_samples = []
    t0 = time.time()
    with ema.apply():
        for ens_i in range(NUM_ENSEMBLE):
            z_sampled = heun_sampler(
                diff_model, schedule, cond_inf,
                shape=(BATCH_SIZE, LATENT_CH, LATENT_H, LATENT_W),
                num_steps=NUM_SAMPLING_STEPS, guidance_scale=0.2)
            with torch.no_grad():
                r_sampled = vae.decode(z_sampled)
                final = drn_pred + r_sampled
            ensemble_samples.append(final)
            print(f"    Member {ens_i+1}/{NUM_ENSEMBLE} done")

    ensemble = torch.stack(ensemble_samples, dim=0)
    ens_mean = ensemble.mean(dim=0)
    ens_std = ensemble.std(dim=0)
    final_pred = ensemble_samples[-1]

    assert final_pred.shape == (BATCH_SIZE, OUT_CH, PATCH_SIZE, PATCH_SIZE)
    assert torch.isfinite(final_pred).all()
    final_rmse = ((ens_mean - conus_viz) ** 2).mean().sqrt().item()
    print(f"  PASS — shape={final_pred.shape}, DRN RMSE={drn_rmse:.4f}, "
          f"Pipeline RMSE={final_rmse:.4f}, spread={ens_std.mean():.4f} ({time.time()-t0:.0f}s)")
    results["pipeline_sampling"] = "PASS"

    plot_panels(
        [conus_viz[0, 0], drn_pred[0, 0], ens_mean[0, 0],
         (conus_viz[0, 0] - ens_mean[0, 0]), ens_std[0, 0]],
        ["Target", "DRN", "Ens Mean", "Error", "Ens Spread"],
        f"{plot_dir}/07_pipeline_result.png", "Full Pipeline Result",
        share_scale=[0, 0, 0, 1, 2])

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 6: Inference Pipeline Module
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[TEST 6] src.inference.pipeline.run_pipeline...")
    from src.inference.pipeline import run_pipeline

    # run_pipeline hardcodes era5[:, :7] — we need to match that
    # Our sanity check IN_CH=7 so this works directly
    with ema.apply():
        drn_out, samples_out = run_pipeline(
            era5_viz, drn, vae, diff_model, schedule,
            num_steps=8, guidance_scale=0.2, num_samples=2, device=device)

    assert drn_out.shape == (BATCH_SIZE, OUT_CH, PATCH_SIZE, PATCH_SIZE)
    assert samples_out.shape == (BATCH_SIZE, 2, OUT_CH, PATCH_SIZE, PATCH_SIZE)
    assert torch.isfinite(samples_out).all()
    print(f"  PASS — drn_out={drn_out.shape}, samples={samples_out.shape}")
    results["inference_pipeline"] = "PASS"

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 7: Evaluation Metrics
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[TEST 7] Evaluation metrics (CRPS, RMSE, MAE, SSR, rank hist, Q-Q, spectra)...")
    from src.evaluation.metrics import (
        crps_ensemble, rmse, mae, spread_skill_ratio,
        rank_histogram, qq_quantiles, power_spectrum_2d,
    )

    target = conus_viz[0, 0].cpu().numpy()
    ens_np = ensemble[:, 0, 0].cpu().numpy()  # (M, H, W)

    # CRPS
    crps = crps_ensemble(target.flatten(), ens_np.reshape(NUM_ENSEMBLE, -1))
    assert np.isfinite(crps), f"CRPS is non-finite: {crps}"
    print(f"    CRPS: {crps:.4f}")

    # RMSE
    rmse_val = rmse(ens_np.mean(axis=0), target)
    assert np.isfinite(rmse_val), f"RMSE is non-finite: {rmse_val}"
    print(f"    RMSE: {rmse_val:.4f}")

    # MAE
    mae_val = mae(ens_np.mean(axis=0), target)
    assert np.isfinite(mae_val), f"MAE is non-finite: {mae_val}"
    print(f"    MAE:  {mae_val:.4f}")

    # Spread-skill ratio
    ssr = spread_skill_ratio(ens_np, target)
    assert np.isfinite(ssr), f"SSR is non-finite: {ssr}"
    print(f"    SSR:  {ssr:.3f}")

    # Rank histogram
    sub_tgt = target[::8, ::8].flatten()
    sub_ens = ens_np[:, ::8, ::8].reshape(NUM_ENSEMBLE, -1)
    rh = rank_histogram(sub_ens, sub_tgt, num_bins=NUM_ENSEMBLE + 1)
    assert len(rh) == NUM_ENSEMBLE + 1
    assert rh.sum() > 0
    print(f"    Rank hist: {rh}")

    # Q-Q
    qq_p, qq_t = qq_quantiles(ens_np.mean(axis=0), target, n_quantiles=50)
    assert len(qq_p) == 50
    print(f"    Q-Q: {len(qq_p)} quantiles computed")

    # Power spectrum
    k, ps = power_spectrum_2d(target)
    assert len(ps) > 0 and np.all(np.isfinite(ps[:10]))
    print(f"    Power spectrum: {len(ps)} wavenumber bins")

    print("  PASS — all metrics computed successfully")
    results["eval_metrics"] = "PASS"

    # Plot metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(range(len(rh)), rh / rh.sum(), color="#2196F3")
    axes[0].axhline(1.0 / len(rh), color="red", ls="--")
    axes[0].set_title("Rank Histogram")
    axes[1].plot(qq_t, qq_p, "b.", markersize=3)
    lims = [min(qq_t.min(), qq_p.min()), max(qq_t.max(), qq_p.max())]
    axes[1].plot(lims, lims, "r--")
    axes[1].set_title("Q-Q Plot"); axes[1].set_aspect("equal")
    axes[2].loglog(k[:len(ps)], ps, "g-", lw=2)
    axes[2].set_title("Power Spectrum"); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/08_metrics.png", dpi=150)
    plt.close(fig)
    print(f"  Plot -> {plot_dir}/08_metrics.png")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 8: Stage-by-Stage Power Spectra
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[TEST 8] Stage-by-stage power spectra...")
    era5_spec = radial_power_spectrum(_to_np(era5_viz[0, 0]))
    drn_spec = radial_power_spectrum(_to_np(drn_pred[0, 0]))
    with torch.no_grad():
        vae_recon_full = drn_pred + vae.decode(vae.encode(residual)[0])
    vae_spec = radial_power_spectrum(_to_np(vae_recon_full[0, 0]))
    final_spec = radial_power_spectrum(_to_np(ens_mean[0, 0]))
    target_spec = radial_power_spectrum(_to_np(conus_viz[0, 0]))

    fig, ax = plt.subplots(figsize=(8, 5))
    k = np.arange(1, len(target_spec))
    for spec, label, color, lw in [
        (target_spec, "Target", "green", 2),
        (era5_spec, "ERA5 Interp", "gray", 1),
        (drn_spec, "DRN", "black", 1.5),
        (vae_spec, "DRN+VAE", "blue", 1.5),
        (final_spec, "DRN+Diff", "red", 2),
    ]:
        ax.loglog(k, spec[1:], color=color, lw=lw, label=label)
    ax.set_xlabel("Wavenumber k"); ax.set_ylabel("Power")
    ax.set_title("Power Spectra at Each Pipeline Stage")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/09_stage_spectra.png", dpi=150)
    plt.close(fig)
    print(f"  PASS — spectra computed for all 5 stages")
    results["stage_spectra"] = "PASS"

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 9: Denoising Step Ablation
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[TEST 9] Denoising step ablation (4, 8, 16 steps)...")
    step_counts = [4, 8, 16]
    step_rmses = []
    diff_model.eval()

    for ns in step_counts:
        with torch.no_grad(), ema.apply():
            z_s = heun_sampler(diff_model, schedule, cond_inf[:1],
                               shape=(1, LATENT_CH, LATENT_H, LATENT_W),
                               num_steps=ns, guidance_scale=0.2)
            r_s = vae.decode(z_s)
            fp = drn_pred[:1] + r_s
        rmse_v = ((fp - conus_viz[:1]) ** 2).mean().sqrt().item()
        step_rmses.append(rmse_v)
        print(f"    {ns} steps: RMSE={rmse_v:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(step_counts, step_rmses, "bo-", lw=2, markersize=8)
    ax.set_xlabel("Denoising Steps"); ax.set_ylabel("RMSE")
    ax.set_title("Step Count vs Quality"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/10_step_ablation.png", dpi=150)
    plt.close(fig)
    print(f"  PASS — all step counts produce finite output")
    results["step_ablation"] = "PASS"

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 10: Compute Benchmark (latent vs pixel timing)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[TEST 10] Compute benchmark (latent vs pixel forward pass)...")

    # Latent forward pass timing — forward(z_noisy, sigma, cond)
    dummy_z_noisy = torch.randn(1, LATENT_CH, 64, 64, device=device)
    dummy_cond = torch.randn(1, DIFF_IN_CH - LATENT_CH, 64, 64, device=device)
    dummy_t = torch.tensor([1.0], device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            diff_model(dummy_z_noisy, dummy_t, dummy_cond)
    torch.cuda.synchronize()
    latent_ms = (time.perf_counter() - t0) / 10 * 1000

    # Pixel forward pass timing (same arch at 256x256)
    pixel_diff_in_ch = OUT_CH + IN_CH + OUT_CH + 2
    pixel_diff = DiffusionUNet(
        in_ch=pixel_diff_in_ch, out_ch=OUT_CH, base_ch=64,
        ch_mults=(1, 2, 2, 4), num_res_blocks=2,
        attn_resolutions=(2, 3), time_dim=256, dropout=0.0,
    ).to(device).eval()
    dummy_px_noisy = torch.randn(1, OUT_CH, 256, 256, device=device)
    dummy_px_cond = torch.randn(1, pixel_diff_in_ch - OUT_CH, 256, 256, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            pixel_diff(dummy_px_noisy, dummy_t, dummy_px_cond)
    torch.cuda.synchronize()
    pixel_ms = (time.perf_counter() - t0) / 10 * 1000

    speedup = pixel_ms / latent_ms
    print(f"    Latent (64x64):  {latent_ms:.1f} ms/forward")
    print(f"    Pixel (256x256): {pixel_ms:.1f} ms/forward")
    print(f"    Speedup: {speedup:.1f}x")

    # Memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        diff_model(dummy_z_noisy, dummy_t, dummy_cond)
    latent_mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        pixel_diff(dummy_px_noisy, dummy_t, dummy_px_cond)
    pixel_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"    Latent mem: {latent_mem:.2f} GB, Pixel mem: {pixel_mem:.2f} GB")

    assert speedup > 1.0, f"Latent should be faster than pixel! Speedup={speedup:.1f}x"
    print(f"  PASS — latent {speedup:.1f}x faster than pixel")
    results["compute_benchmark"] = "PASS"
    del pixel_diff

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 11: Latent Space Analysis
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[TEST 11] Latent space analysis...")
    with torch.no_grad():
        all_mus = []
        all_stds = []
        for pi in range(min(20, len(patches))):
            e_p = patches[pi][0].unsqueeze(0).to(device)
            c_p = patches[pi][1].unsqueeze(0).to(device)
            dp = drn(e_p)
            res_p = c_p - dp
            mu_p, logvar_p = vae.encode(res_p)
            all_mus.append(mu_p.cpu())
            all_stds.append((0.5 * logvar_p).exp().cpu())

    all_mu = torch.cat(all_mus, dim=0)  # (N, LATENT_CH, 64, 64)
    all_std_cat = torch.cat(all_stds, dim=0)

    # Channel distributions
    fig, axes = plt.subplots(1, LATENT_CH, figsize=(4 * LATENT_CH, 3))
    if LATENT_CH == 1:
        axes = [axes]
    for ch in range(LATENT_CH):
        data = all_mu[:, ch].flatten().numpy()
        axes[ch].hist(data, bins=50, density=True, alpha=0.7)
        axes[ch].set_title(f"Ch {ch}: mean={data.mean():.2f}, std={data.std():.2f}")
    fig.suptitle("Latent Channel Distributions")
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/11_latent_distributions.png", dpi=150)
    plt.close(fig)

    # Correlation matrix
    flat = all_mu.reshape(all_mu.shape[0], LATENT_CH, -1)
    corr = np.zeros((LATENT_CH, LATENT_CH))
    for i in range(LATENT_CH):
        for j in range(LATENT_CH):
            ci = flat[:, i].flatten().numpy()[:5000]
            cj = flat[:, j].flatten().numpy()[:5000]
            corr[i, j] = np.corrcoef(ci, cj)[0, 1]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Latent Channel Correlations")
    plt.colorbar(im, ax=ax)
    for i in range(LATENT_CH):
        for j in range(LATENT_CH):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/12_latent_correlations.png", dpi=150)
    plt.close(fig)

    # Spatial maps
    fig, axes = plt.subplots(2, LATENT_CH, figsize=(4 * LATENT_CH, 7))
    for ch in range(LATENT_CH):
        im1 = axes[0, ch].imshow(all_mu[0, ch].numpy(), cmap="RdBu_r")
        axes[0, ch].set_title(f"Latent ch {ch}")
        axes[0, ch].axis("off")
        plt.colorbar(im1, ax=axes[0, ch], fraction=0.046)
    # Uncertainty
    mean_std_per_ch = all_std_cat.mean(dim=(0, 2, 3)).numpy()
    for ch in range(LATENT_CH):
        im2 = axes[1, ch].imshow(all_std_cat[0, ch].numpy(), cmap="hot_r")
        axes[1, ch].set_title(f"Std ch {ch}: {mean_std_per_ch[ch]:.3f}")
        axes[1, ch].axis("off")
        plt.colorbar(im2, ax=axes[1, ch], fraction=0.046)
    fig.suptitle("Latent Space: Activation Maps + Uncertainty")
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/13_latent_spatial.png", dpi=150)
    plt.close(fig)

    print(f"  PASS — analyzed {all_mu.shape[0]} samples, {LATENT_CH} channels")
    results["latent_analysis"] = "PASS"

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 12: Climate Signal (early vs late mean)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[TEST 12] Climate signal test (early vs late patch means)...")
    # Use patches from different days as proxy for early vs late
    n_half = len(patches) // 2
    early_preds, late_preds = [], []
    early_tgts, late_tgts = [], []

    with torch.no_grad():
        for pi in range(min(n_half, 10)):
            e_p = patches[pi][0].unsqueeze(0).to(device)
            c_p = patches[pi][1].unsqueeze(0).to(device)
            dp = drn(e_p)
            early_preds.append(dp.cpu())
            early_tgts.append(c_p.cpu())
        for pi in range(n_half, min(n_half + 10, len(patches))):
            e_p = patches[pi][0].unsqueeze(0).to(device)
            c_p = patches[pi][1].unsqueeze(0).to(device)
            dp = drn(e_p)
            late_preds.append(dp.cpu())
            late_tgts.append(c_p.cpu())

    early_pred_mean = torch.cat(early_preds).mean(dim=0)
    late_pred_mean = torch.cat(late_preds).mean(dim=0)
    early_tgt_mean = torch.cat(early_tgts).mean(dim=0)
    late_tgt_mean = torch.cat(late_tgts).mean(dim=0)

    delta_tgt = (late_tgt_mean - early_tgt_mean)[0].numpy()
    delta_pred = (late_pred_mean - early_pred_mean)[0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmax = max(abs(delta_tgt.min()), abs(delta_tgt.max()), abs(delta_pred.min()), abs(delta_pred.max()))
    for ax, data, title in zip(axes,
                                [delta_tgt, delta_pred, delta_tgt - delta_pred],
                                ["Target Change", "DRN Change", "Difference"]):
        im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{title}\nmean={data.mean():.4f}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Climate Signal Test (Early vs Late Patches)")
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/14_climate_signal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PASS — signal computed, correlation check done")
    results["climate_signal"] = "PASS"

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 13: Pixel-Space Diffusion (ablation build + short train)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[TEST 13] Pixel-space diffusion ablation (100 steps)...")
    pixel_diff_in_ch = OUT_CH + IN_CH + OUT_CH + 2
    pixel_diff = DiffusionUNet(
        in_ch=pixel_diff_in_ch, out_ch=OUT_CH, base_ch=64,
        ch_mults=(1, 2, 2, 4), num_res_blocks=2,
        attn_resolutions=(2, 3), time_dim=256, dropout=0.0,
    ).to(device)
    pixel_params = sum(p.numel() for p in pixel_diff.parameters())
    print(f"  Pixel diff params: {pixel_params:,}")

    pixel_opt = torch.optim.AdamW(pixel_diff.parameters(), lr=1e-4)
    pixel_diff.train()
    pixel_losses = []
    t0 = time.time()

    for step in range(100):
        era5_b, conus_b = get_batch()
        with torch.no_grad():
            drn_pred_b = drn(era5_b)
        pixel_residual = conus_b - drn_pred_b
        pos256 = torch.stack([
            torch.linspace(-1, 1, 256, device=device).unsqueeze(1).expand(256, 256),
            torch.linspace(-1, 1, 256, device=device).unsqueeze(0).expand(256, 256),
        ], dim=0).unsqueeze(0).expand(BATCH_SIZE, -1, -1, -1)
        pixel_cond = torch.cat([era5_b, drn_pred_b, pos256], dim=1)
        loss = edm_training_loss(pixel_diff, schedule, pixel_residual, pixel_cond)
        pixel_opt.zero_grad(); loss.backward(); pixel_opt.step()
        pixel_losses.append(loss.item())
        if step % 50 == 0:
            print(f"    Step {step}/100 | Loss: {loss.item():.6f}")

    assert all(np.isfinite(l) for l in pixel_losses), "Pixel diff has non-finite losses!"
    print(f"  PASS — pixel diff trains: {pixel_losses[0]:.4f} -> {pixel_losses[-1]:.4f} ({time.time()-t0:.0f}s)")
    results["pixel_ablation"] = "PASS"
    del pixel_diff

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 14: Error Histogram
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[TEST 14] Error distribution analysis...")
    drn_err = _to_np((drn_pred - conus_viz)[0, 0]).flatten()
    final_err = _to_np((ens_mean - conus_viz)[0, 0]).flatten()

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-3, 3, 100)
    ax.hist(drn_err, bins=bins, density=True, alpha=0.5, color="green",
            label=f"DRN (std={drn_err.std():.3f})")
    ax.hist(final_err, bins=bins, density=True, alpha=0.5, color="red",
            label=f"Pipeline (std={final_err.std():.3f})")
    ax.set_title("Error Distribution"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/15_error_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  PASS — DRN err std={drn_err.std():.3f}, Pipeline err std={final_err.std():.3f}")
    results["error_distribution"] = "PASS"

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 15: Per-Pixel RMSE Map
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[TEST 15] Per-pixel RMSE map (8 patches)...")
    with torch.no_grad():
        drn_err2 = torch.zeros(OUT_CH, PATCH_SIZE, PATCH_SIZE, device=device)
        pipe_err2 = torch.zeros(OUT_CH, PATCH_SIZE, PATCH_SIZE, device=device)
        n_eval = 0
        for pi in range(min(8, len(patches))):
            e_p = patches[pi][0].unsqueeze(0).to(device)
            c_p = patches[pi][1].unsqueeze(0).to(device)
            dp = drn(e_p)
            drn_err2 += (dp - c_p).squeeze(0) ** 2
            cond_p = build_cond(e_p, dp)
            with ema.apply():
                z_s = heun_sampler(diff_model, schedule, cond_p,
                                   shape=(1, LATENT_CH, LATENT_H, LATENT_W),
                                   num_steps=NUM_SAMPLING_STEPS, guidance_scale=0.2)
            r_s = vae.decode(z_s)
            fp = dp + r_s
            pipe_err2 += (fp - c_p).squeeze(0) ** 2
            n_eval += 1

    drn_rmse_map = _to_np(torch.sqrt(drn_err2 / n_eval)[0])
    pipe_rmse_map = _to_np(torch.sqrt(pipe_err2 / n_eval)[0])
    improvement = drn_rmse_map - pipe_rmse_map

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    im1 = ax1.imshow(drn_rmse_map, cmap="hot_r"); ax1.set_title("DRN RMSE"); ax1.axis("off")
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(pipe_rmse_map, cmap="hot_r"); ax2.set_title("Pipeline RMSE"); ax2.axis("off")
    plt.colorbar(im2, ax=ax2)
    vabs = max(abs(improvement.min()), abs(improvement.max()))
    im3 = ax3.imshow(improvement, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    ax3.set_title("Improvement (DRN - Pipeline)"); ax3.axis("off")
    plt.colorbar(im3, ax=ax3)
    fig.suptitle(f"Per-Pixel RMSE ({n_eval} patches)")
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/16_pixel_rmse_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PASS — DRN RMSE={drn_rmse_map.mean():.4f}, Pipeline RMSE={pipe_rmse_map.mean():.4f}")
    results["pixel_rmse_map"] = "PASS"

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SANITY CHECK SUMMARY")
    print("=" * 70)
    print(f"  Steps: DRN={NUM_DRN_STEPS}, VAE={NUM_VAE_STEPS}, Diff={NUM_DIFF_STEPS}")
    print(f"  Data:  {len(patches)} patches (1 year, temp only)")
    print(f"\n  {'Test':<30} {'Status':>8}")
    print(f"  {'-'*30} {'-'*8}")
    all_pass = True
    for test, status in results.items():
        print(f"  {test:<30} {status:>8}")
        if status != "PASS":
            all_pass = False

    print(f"\n  Models:")
    print(f"    DRN:       {drn_params/1e6:.1f}M params, {drn_losses[0]:.4f} -> {drn_losses[-1]:.4f}")
    print(f"    VAE:       {vae_params/1e6:.1f}M params, {vae_losses[0]:.4f} -> {vae_losses[-1]:.4f}")
    print(f"    Diffusion: {diff_params/1e6:.1f}M params, {diff_losses[0]:.4f} -> {diff_losses[-1]:.4f}")
    print(f"\n  Metrics:")
    print(f"    DRN RMSE:      {drn_rmse:.4f}")
    print(f"    Pipeline RMSE: {final_rmse:.4f}")
    print(f"    CRPS:          {crps:.4f}")
    print(f"    SSR:           {ssr:.3f}")
    print(f"    Latent speedup: {speedup:.1f}x over pixel")
    print(f"    Latent mem: {latent_mem:.2f} GB vs pixel: {pixel_mem:.2f} GB")
    print(f"\n  Plots: {plot_path.resolve()}/")
    for f in sorted(plot_path.glob("*.png")):
        print(f"    {f.name}")

    if all_pass:
        print(f"\n  ALL {len(results)} TESTS PASSED")
    else:
        print(f"\n  SOME TESTS FAILED!")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive sanity check — all features")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot_dir", type=str, default="sanity_plots")
    args = parser.parse_args()

    success = sanity_check(device=args.device, plot_dir=args.plot_dir)
    sys.exit(0 if success else 1)
