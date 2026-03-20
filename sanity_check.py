"""End-to-end sanity check using REAL ERA5/CONUS404 data (temperature only).

Loads a few days from disk, regrids ERA5→CONUS404 in-memory, extracts random
256×256 patches, and trains 1 epoch of each stage (DRN → VAE → Diffusion).
Generates diagnostic plots at every stage including power spectra, ensemble
spread, error histograms, and multi-patch comparisons.

No data is written to disk except plots and the final summary.

Usage:
  python sanity_check.py [--device cuda|cpu] [--plot_dir sanity_plots]
  python sanity_check.py --extensive    # 500 steps/stage, bigger models, stricter checks
"""

import argparse
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
DATA_DIR = Path("/gpfsm/dnb33/hpmille1/diffusion_downscaling_model/data")
PATCH_SIZE = 256
LATENT_H = LATENT_W = 64


def get_config(extensive: bool):
    """Return config dict based on mode."""
    if extensive:
        return dict(
            num_train_steps=500,
            batch_size=4,
            years=[1980, 1981],
            sample_days=[10, 30, 50, 70, 90, 110, 130, 150, 170, 190,
                         210, 230, 250, 270, 290, 310, 330, 350],
            patches_per_day=8,
            drn_base_ch=64,
            drn_num_res=2,
            vae_base_ch=128,
            diff_base_ch=96,
            diff_num_res=3,
            num_sampling_steps=24,
            num_ensemble=8,
            num_eval_patches=48,
            lr=1e-4,
            log_every=25,
        )
    else:
        return dict(
            num_train_steps=50,
            batch_size=4,
            years=[1980],
            sample_days=[10, 40, 70, 100, 130, 160, 190, 220, 250, 280, 310, 340],
            patches_per_day=4,
            drn_base_ch=32,
            drn_num_res=1,
            vae_base_ch=64,
            diff_base_ch=64,
            diff_num_res=2,
            num_sampling_steps=12,
            num_ensemble=4,
            num_eval_patches=24,
            lr=3e-4,
            log_every=10,
        )


# ─── Data loading (all in-memory) ────────────────────────────────────────────

def _month_for_day(day_idx: int, leap: bool) -> int:
    days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cum = 0
    for m, d in enumerate(days):
        cum += d
        if day_idx < cum:
            return m
    return 11


def load_real_data(cfg: dict):
    """Load ERA5 t2m and CONUS404 T2 for a handful of days. All in-memory."""
    t0 = time.time()
    years = cfg["years"]
    sample_days = cfg["sample_days"]
    patches_per_day = cfg["patches_per_day"]

    print(f"[Data] Loading {len(years)} year(s), {len(sample_days)} days each...")

    # Coordinates from first year
    era5_ds_first = xr.open_dataset(DATA_DIR / f"era5_{years[0]}.nc")
    era5_lat = era5_ds_first["latitude"].values
    era5_lon = era5_ds_first["longitude"].values

    conus_ds_first = xr.open_dataset(DATA_DIR / f"conus404_yearly_{years[0]}.nc")
    conus_lat = conus_ds_first["lat"].values
    conus_lon = conus_ds_first["lon"].values

    print("[Data] Building xESMF regridder (bilinear + nearest extrap)...")
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
    print(f"[Data] Regridder built in {time.time()-t0:.1f}s")

    print("[Data] Loading static fields from CONUS404...")
    terrain = conus_ds_first["Z"].isel(time=0, bottom_top_stag=0).values.astype(np.float32)
    terrain_norm = (terrain - np.nanmean(terrain)) / (np.nanstd(terrain) + 1e-8)

    z_mean = uniform_filter(terrain, size=5)
    z_sq_mean = uniform_filter(terrain ** 2, size=5)
    orog_var = np.sqrt(np.maximum(z_sq_mean - z_mean ** 2, 0))
    orog_var_norm = (orog_var - np.nanmean(orog_var)) / (np.nanstd(orog_var) + 1e-8)

    lat_norm = (conus_lat - np.nanmean(conus_lat)) / (np.nanstd(conus_lat) + 1e-8)
    lon_norm = (conus_lon - np.nanmean(conus_lon)) / (np.nanstd(conus_lon) + 1e-8)
    lsm = (terrain > 0).astype(np.float32)

    static = np.stack([terrain_norm, orog_var_norm, lat_norm.astype(np.float32),
                       lon_norm.astype(np.float32), lsm], axis=0)

    era5_ds_first.close()
    conus_ds_first.close()

    era5_t2m_vals = []
    conus_t2_vals = []
    raw_pairs = []
    day_labels = []

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
            era5_t2m_vals.append(era5_regridded[::10, ::10].flatten())
            conus_t2_vals.append(conus_field[::10, ::10].flatten())
            raw_pairs.append((era5_regridded, conus_field))
            day_labels.append(f"Y{year}D{d}")

        era5_ds.close()
        conus_ds.close()

    era5_all = np.concatenate(era5_t2m_vals)
    conus_all = np.concatenate(conus_t2_vals)
    era5_mean, era5_std = float(np.nanmean(era5_all)), float(np.nanstd(era5_all))
    conus_mean, conus_std = float(np.nanmean(conus_all)), float(np.nanstd(conus_all))
    static_means = static.mean(axis=(1, 2), keepdims=True)
    static_stds = np.maximum(static.std(axis=(1, 2), keepdims=True), 1e-8)
    print(f"[Data] ERA5 t2m:  mean={era5_mean:.2f} K, std={era5_std:.2f} K")
    print(f"[Data] CONUS T2:  mean={conus_mean:.2f} K, std={conus_std:.2f} K")

    H, W = 1015, 1367
    PS = PATCH_SIZE
    patches = []
    patch_day_idx = []
    rng = np.random.RandomState(42)

    for day_i, (era5_reg, conus_f) in enumerate(raw_pairs):
        era5_norm = (era5_reg - era5_mean) / era5_std
        conus_norm = (conus_f - conus_mean) / conus_std
        static_norm = (static - static_means) / static_stds
        era5_input = np.concatenate([era5_norm[None], static_norm], axis=0)

        for _ in range(patches_per_day):
            y0 = rng.randint(0, H - PS)
            x0 = rng.randint(0, W - PS)
            era5_patch = torch.from_numpy(era5_input[:, y0:y0+PS, x0:x0+PS].copy())
            conus_patch = torch.from_numpy(conus_norm[None, y0:y0+PS, x0:x0+PS].copy())
            patches.append((era5_patch, conus_patch))
            patch_day_idx.append(day_i)

    print(f"[Data] Extracted {len(patches)} patches of size {PS}x{PS} "
          f"from {len(raw_pairs)} days across {len(years)} year(s)")
    print(f"[Data] Total load time: {time.time()-t0:.1f}s")

    era5_full = (raw_pairs[0][0] - era5_mean) / era5_std
    conus_full = (raw_pairs[0][1] - conus_mean) / conus_std

    return patches, era5_full, conus_full, day_labels, patch_day_idx


# ─── Plotting helpers ────────────────────────────────────────────────────────

def _to_np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return t


def plot_stage_panels(panels, titles, save_path, suptitle="", cmap="RdBu_r"):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, panel, title in zip(axes, panels, titles):
        p = _to_np(panel)
        vmin, vmax = np.nanpercentile(p, [2, 98])
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
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, linewidth=2)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {save_path}")


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


# ─── Main sanity check ──────────────────────────────────────────────────────

def sanity_check(device: str = "cuda", plot_dir: str = "sanity_plots", extensive: bool = False):
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    cfg = get_config(extensive)
    mode_str = "EXTENSIVE" if extensive else "QUICK"
    NUM_TRAIN_STEPS = cfg["num_train_steps"]
    BATCH_SIZE = cfg["batch_size"]
    NUM_SAMPLING_STEPS = cfg["num_sampling_steps"]
    NUM_ENSEMBLE = cfg["num_ensemble"]
    LOG_EVERY = cfg["log_every"]
    LR = cfg["lr"]

    IN_CH = 6
    OUT_CH = 1
    LATENT_CH = 4

    print("=" * 70)
    print(f"SANITY CHECK [{mode_str}] — Real data, temperature only")
    print(f"  Steps/stage: {NUM_TRAIN_STEPS}, Ensemble: {NUM_ENSEMBLE}, "
          f"Sampling: {NUM_SAMPLING_STEPS} Heun steps")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # LOAD REAL DATA
    # ═══════════════════════════════════════════════════════════════════════
    patches, era5_full, conus_full, day_labels, patch_day_idx = load_real_data(cfg)

    plot_stage_panels(
        [era5_full, conus_full],
        ["ERA5 t2m (regridded)", "CONUS404 T2 (target)"],
        f"{plot_dir}/00_input_vs_target_full.png",
        suptitle=f"Raw Input vs Target — Full Domain (Day {day_labels[0]})",
    )

    e0, c0 = patches[0]
    plot_stage_panels(
        [e0[0], c0[0]],
        ["ERA5 t2m patch", "CONUS404 T2 patch"],
        f"{plot_dir}/01_patch_example.png",
        suptitle="Example 256x256 Patch",
    )

    def get_batch(step):
        idxs = [(step * BATCH_SIZE + i) % len(patches) for i in range(BATCH_SIZE)]
        era5_b = torch.stack([patches[i][0] for i in idxs]).to(device)
        conus_b = torch.stack([patches[i][1] for i in idxs]).to(device)
        return era5_b, conus_b

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1: DRN
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STAGE 1: Deterministic Regression Network (DRN)")
    print("=" * 70)

    drn = DRN(in_ch=IN_CH, out_ch=OUT_CH, base_ch=cfg["drn_base_ch"],
              ch_mults=(1, 2, 4, 8), num_res_blocks=cfg["drn_num_res"],
              attn_resolutions=(2,)).to(device)
    print(f"[DRN] Params: {sum(p.numel() for p in drn.parameters()):,}")

    drn_criterion = PerVariableMSE(num_vars=OUT_CH, precip_channel=-1).to(device)
    drn_opt = torch.optim.AdamW(
        list(drn.parameters()) + list(drn_criterion.parameters()), lr=LR)

    drn.train()
    drn_losses = []
    for step in range(NUM_TRAIN_STEPS):
        era5_b, conus_b = get_batch(step)
        pred = drn(era5_b)
        loss = drn_criterion(pred, conus_b)
        drn_opt.zero_grad(); loss.backward(); drn_opt.step()
        drn_losses.append(loss.item())
        if step % LOG_EVERY == 0:
            print(f"  Step {step:3d}/{NUM_TRAIN_STEPS} | Loss: {loss.item():.6f}")

    assert all(np.isfinite(l) for l in drn_losses), "DRN has non-finite losses!"
    if extensive:
        assert drn_losses[-1] < drn_losses[0], \
            f"DRN loss did not decrease: {drn_losses[0]:.4f} -> {drn_losses[-1]:.4f}"
        # Check last 20% of steps are lower than first 20%
        early = np.mean(drn_losses[:NUM_TRAIN_STEPS // 5])
        late = np.mean(drn_losses[-NUM_TRAIN_STEPS // 5:])
        assert late < early, f"DRN not converging: early avg {early:.4f} vs late avg {late:.4f}"
    print(f"[DRN] PASS — loss {drn_losses[0]:.4f} -> {drn_losses[-1]:.4f}")
    plot_loss(drn_losses, f"DRN Training Loss [{mode_str}]", f"{plot_dir}/02_drn_loss.png")

    # DRN eval on first batch
    drn.eval()
    era5_viz, conus_viz = get_batch(0)
    with torch.no_grad():
        drn_pred = drn(era5_viz)
        residual = conus_viz - drn_pred

    plot_stage_panels(
        [era5_viz[0, 0], conus_viz[0, 0], drn_pred[0, 0], residual[0, 0]],
        ["ERA5 t2m", "CONUS404 T2\n(target)", "DRN prediction\n(Stage 1)", "Residual\n(target - DRN)"],
        f"{plot_dir}/03_stage1_result.png",
        suptitle="Stage 1: DRN Downscaling",
    )

    # ── DRN: multi-patch comparison across different days ──
    print("[DRN] Evaluating across multiple patches from different days...")
    drn_rmses = []
    fig_mp, axes_mp = plt.subplots(3, 4, figsize=(18, 13))
    for pi in range(min(12, len(patches))):
        e_p, c_p = patches[pi][0].unsqueeze(0).to(device), patches[pi][1].unsqueeze(0).to(device)
        with torch.no_grad():
            p_p = drn(e_p)
        rmse = ((p_p - c_p) ** 2).mean().sqrt().item()
        drn_rmses.append(rmse)
        ax = axes_mp[pi // 4, pi % 4]
        err = _to_np((c_p - p_p)[0, 0])
        vabs = max(abs(np.nanpercentile(err, 2)), abs(np.nanpercentile(err, 98)))
        ax.imshow(err, cmap="RdBu_r", vmin=-vabs, vmax=vabs, origin="lower")
        ax.set_title(f"Day {day_labels[patch_day_idx[pi]]}\nRMSE={rmse:.3f}", fontsize=9)
        ax.axis("off")
    fig_mp.suptitle("DRN Error Maps Across Days/Patches", fontsize=14, y=1.01)
    plt.tight_layout()
    fig_mp.savefig(f"{plot_dir}/03b_drn_multi_patch_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig_mp)
    print(f"  Plot -> {plot_dir}/03b_drn_multi_patch_errors.png")
    print(f"  DRN RMSE across patches: mean={np.mean(drn_rmses):.4f}, "
          f"std={np.std(drn_rmses):.4f}, range=[{np.min(drn_rmses):.4f}, {np.max(drn_rmses):.4f}]")

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2a: VAE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STAGE 2a: Variational Autoencoder (VAE)")
    print("=" * 70)

    vae = VAE(in_ch=OUT_CH, latent_ch=LATENT_CH, base_ch=cfg["vae_base_ch"]).to(device)
    print(f"[VAE] Params: {sum(p.numel() for p in vae.parameters()):,}")

    vae_criterion = VAELoss()
    vae_opt = torch.optim.AdamW(vae.parameters(), lr=LR)

    vae.train()
    vae_losses, vae_recon_losses, vae_kl_losses = [], [], []
    for step in range(NUM_TRAIN_STEPS):
        era5_b, conus_b = get_batch(step)
        with torch.no_grad():
            drn_pred_b = drn(era5_b)
        residual_b = conus_b - drn_pred_b
        recon, mu, logvar = vae(residual_b)
        beta = min(1e-4 * step / 15, 1e-4)
        loss, recon_l, kl_l = vae_criterion(recon, residual_b, mu, logvar, beta=beta)
        vae_opt.zero_grad(); loss.backward(); vae_opt.step()
        vae_losses.append(loss.item())
        vae_recon_losses.append(recon_l.item())
        vae_kl_losses.append(kl_l.item())
        if step % LOG_EVERY == 0:
            print(f"  Step {step:3d}/{NUM_TRAIN_STEPS} | Loss: {loss.item():.6f} "
                  f"(recon: {recon_l.item():.6f}, KL: {kl_l.item():.2f}, beta: {beta:.1e})")

    assert all(np.isfinite(l) for l in vae_losses), "VAE has non-finite losses!"
    assert mu.shape == (BATCH_SIZE, LATENT_CH, LATENT_H, LATENT_W), f"Latent shape wrong: {mu.shape}"
    if extensive:
        assert vae_recon_losses[-1] < vae_recon_losses[0], \
            f"VAE recon loss did not decrease: {vae_recon_losses[0]:.4f} -> {vae_recon_losses[-1]:.4f}"
        # KL should be bounded and positive (not collapsed)
        assert vae_kl_losses[-1] > 0.1, f"VAE KL collapsed to {vae_kl_losses[-1]:.4f}"
        assert vae_kl_losses[-1] < 100, f"VAE KL exploded to {vae_kl_losses[-1]:.4f}"
    print(f"[VAE] PASS — loss {vae_losses[0]:.4f} -> {vae_losses[-1]:.4f}, latent: {mu.shape}")

    # ── VAE loss curves: combined + separate recon/KL ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    ax1.plot(vae_losses, linewidth=2); ax1.set_title("VAE Total Loss"); ax1.grid(True, alpha=0.3)
    ax2.plot(vae_recon_losses, linewidth=2, color="steelblue"); ax2.set_title("Reconstruction Loss"); ax2.grid(True, alpha=0.3)
    ax3.plot(vae_kl_losses, linewidth=2, color="coral"); ax3.set_title("KL Divergence"); ax3.grid(True, alpha=0.3)
    for ax in [ax1, ax2, ax3]: ax.set_xlabel("Step")
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/04_vae_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {plot_dir}/04_vae_loss.png")

    # VAE visualization
    vae.eval()
    with torch.no_grad():
        recon_viz, mu_viz, logvar_viz = vae(residual)
        z_clean = vae.reparameterize(mu_viz, logvar_viz)

    schedule = EDMSchedule()
    noise = torch.randn_like(z_clean)
    z_noisy = z_clean + noise * 10.0

    plot_stage_panels(
        [residual[0, 0], recon_viz[0, 0], z_clean[0, 0], z_noisy[0, 0]],
        ["Residual\n(VAE input)", "VAE reconstruction", "Clean latent z\n(ch 0, 64x64)",
         "Noisy latent z_t\n(sigma=10)"],
        f"{plot_dir}/05_stage2a_vae.png",
        suptitle="Stage 2a: VAE Encoding/Decoding of Residual",
    )

    # ── Latent distribution with Gaussian overlay ──
    mu_np = mu_viz.cpu().numpy().flatten()
    std_np = torch.exp(0.5 * logvar_viz).cpu().numpy().flatten()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(mu_np, bins=80, density=True, alpha=0.7, color="steelblue", label="Actual")
    xg = np.linspace(mu_np.min(), mu_np.max(), 200)
    ax1.plot(xg, np.exp(-xg**2 / 2) / np.sqrt(2*np.pi), 'r--', linewidth=2, label="N(0,1)")
    ax1.set_title("Latent mu"); ax1.legend()
    ax2.hist(std_np, bins=80, density=True, alpha=0.7, color="coral")
    ax2.axvline(1, color="red", ls="--", alpha=0.5, label="Target sigma=1")
    ax2.set_title("Latent sigma"); ax2.legend()
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/06_vae_latent_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {plot_dir}/06_vae_latent_dist.png")

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2b: Conditional Diffusion
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STAGE 2b: Conditional EDM Diffusion UNet")
    print("=" * 70)

    DIFF_IN_CH = LATENT_CH + 1 + LATENT_CH + 2  # 11

    diff_model = DiffusionUNet(
        in_ch=DIFF_IN_CH, out_ch=LATENT_CH, base_ch=cfg["diff_base_ch"],
        ch_mults=(1, 2, 2, 4), num_res_blocks=cfg["diff_num_res"],
        attn_resolutions=(2, 3), time_dim=256, dropout=0.0,
    ).to(device)
    print(f"[Diff] Params: {sum(p.numel() for p in diff_model.parameters()):,}")

    ema = EMA(diff_model, decay=0.9999)
    diff_opt = torch.optim.AdamW(diff_model.parameters(), lr=LR)

    ys = torch.linspace(-1, 1, LATENT_H, device=device)
    xs = torch.linspace(-1, 1, LATENT_W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pos_emb = torch.stack([yy, xx], dim=0)

    def build_cond(era5_b, drn_pred_b):
        era5_down = F.interpolate(era5_b[:, :1], (LATENT_H, LATENT_W),
                                  mode="bilinear", align_corners=False)
        with torch.no_grad():
            mu_drn, _ = vae.encode(drn_pred_b)
        pos = pos_emb.unsqueeze(0).expand(era5_b.shape[0], -1, -1, -1)
        return torch.cat([era5_down, mu_drn, pos], dim=1)

    diff_model.train()
    diff_losses = []
    for step in range(NUM_TRAIN_STEPS):
        era5_b, conus_b = get_batch(step)
        with torch.no_grad():
            drn_pred_b = drn(era5_b)
            residual_b = conus_b - drn_pred_b
            mu_b, logvar_b = vae.encode(residual_b)
            z_clean_b = vae.reparameterize(mu_b, logvar_b)

        cond = build_cond(era5_b, drn_pred_b)
        if torch.rand(1).item() < 0.1:
            cond = torch.zeros_like(cond)

        loss = edm_training_loss(diff_model, schedule, z_clean_b, cond)
        diff_opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.0)
        diff_opt.step(); ema.update()
        diff_losses.append(loss.item())
        if step % LOG_EVERY == 0:
            print(f"  Step {step:3d}/{NUM_TRAIN_STEPS} | Loss: {loss.item():.6f}")

    assert all(np.isfinite(l) for l in diff_losses), "Diffusion has non-finite losses!"
    if extensive:
        early = np.mean(diff_losses[:NUM_TRAIN_STEPS // 5])
        late = np.mean(diff_losses[-NUM_TRAIN_STEPS // 5:])
        assert late < early, f"Diffusion not converging: early avg {early:.4f} vs late avg {late:.4f}"
    print(f"[Diff] PASS — loss {diff_losses[0]:.4f} -> {diff_losses[-1]:.4f}")
    plot_loss(diff_losses, f"Diffusion Training Loss [{mode_str}]", f"{plot_dir}/07_diffusion_loss.png")

    # ═══════════════════════════════════════════════════════════════════════
    # FULL PIPELINE SAMPLING
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FULL PIPELINE: Heun sampling + ensemble + diagnostics")
    print("=" * 70)

    diff_model.eval()
    with torch.no_grad():
        cond_inf = build_cond(era5_viz, drn_pred)

    print(f"Sampling {NUM_ENSEMBLE} ensemble members with {NUM_SAMPLING_STEPS} Heun steps each...")

    ensemble_samples = []
    with ema.apply():
        for ens_i in range(NUM_ENSEMBLE):
            z_sampled = heun_sampler(
                diff_model, schedule, cond_inf,
                shape=(BATCH_SIZE, LATENT_CH, LATENT_H, LATENT_W),
                num_steps=NUM_SAMPLING_STEPS,
                guidance_scale=0.2,
            )
            with torch.no_grad():
                r_sampled = vae.decode(z_sampled)
                final = drn_pred + r_sampled
            ensemble_samples.append(final)
            print(f"  Ensemble member {ens_i+1}/{NUM_ENSEMBLE} done")

    ensemble = torch.stack(ensemble_samples, dim=0)  # (NUM_ENSEMBLE, B, 1, H, W)
    ens_mean = ensemble.mean(dim=0)
    ens_std = ensemble.std(dim=0)

    # Use last sample as representative
    final_pred = ensemble_samples[-1]
    r_sampled_last = final_pred - drn_pred

    assert final_pred.shape == (BATCH_SIZE, OUT_CH, PATCH_SIZE, PATCH_SIZE), \
        f"Final output shape wrong: {final_pred.shape}"
    assert torch.isfinite(final_pred).all(), "Final prediction has non-finite values!"
    print(f"[Pipeline] PASS — output: {final_pred.shape}, all finite")

    # ── 08: Full pipeline 8-panel ──
    sigma_mid = schedule.get_sigmas(NUM_SAMPLING_STEPS, device)[NUM_SAMPLING_STEPS // 2]
    z_noisy_plot = z_sampled + torch.randn_like(z_sampled) * sigma_mid

    plot_stage_panels(
        [era5_viz[0, 0], conus_viz[0, 0], drn_pred[0, 0], residual[0, 0],
         z_noisy_plot[0, 0], z_sampled[0, 0], r_sampled_last[0, 0], final_pred[0, 0]],
        ["ERA5 t2m\n(input)", "CONUS404 T2\n(target)", "DRN pred\n(Stage 1)",
         "Residual\n(target-DRN)", "Noisy latent\n(mid-sigma)", "Denoised latent\n(sampled)",
         "Decoded residual\n(diffusion)", "Final prediction\n(DRN + diff)"],
        f"{plot_dir}/08_full_pipeline.png",
        suptitle="Full Pipeline: ERA5 -> DRN -> Diffusion -> Final",
    )

    # ── 09: Target vs final vs error ──
    plot_stage_panels(
        [conus_viz[0, 0], final_pred[0, 0], (conus_viz[0, 0] - final_pred[0, 0])],
        ["CONUS404 T2 (target)", "Final prediction", "Error (target - pred)"],
        f"{plot_dir}/09_target_vs_prediction.png",
        suptitle="Target vs Final Prediction (temperature)",
    )

    # ── 10: Ensemble spread visualization ──
    fig, axes = plt.subplots(1, NUM_ENSEMBLE + 2, figsize=(4.5 * (NUM_ENSEMBLE + 2), 4))
    for i in range(NUM_ENSEMBLE):
        p = _to_np(ensemble_samples[i][0, 0])
        vmin, vmax = np.nanpercentile(p, [2, 98])
        im = axes[i].imshow(p, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        axes[i].set_title(f"Sample {i+1}", fontsize=10); axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    # Ensemble mean
    p = _to_np(ens_mean[0, 0])
    vmin, vmax = np.nanpercentile(p, [2, 98])
    im = axes[NUM_ENSEMBLE].imshow(p, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
    axes[NUM_ENSEMBLE].set_title("Ensemble Mean", fontsize=10); axes[NUM_ENSEMBLE].axis("off")
    plt.colorbar(im, ax=axes[NUM_ENSEMBLE], fraction=0.046, pad=0.04)
    # Ensemble std
    p = _to_np(ens_std[0, 0])
    im = axes[NUM_ENSEMBLE+1].imshow(p, cmap="hot_r", origin="lower")
    axes[NUM_ENSEMBLE+1].set_title("Ensemble Spread\n(std dev)", fontsize=10)
    axes[NUM_ENSEMBLE+1].axis("off")
    plt.colorbar(im, ax=axes[NUM_ENSEMBLE+1], fraction=0.046, pad=0.04)
    fig.suptitle(f"Diffusion Ensemble ({NUM_ENSEMBLE} members)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/10_ensemble_spread.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {plot_dir}/10_ensemble_spread.png")

    # ── 11: Power spectra comparison ──
    print("[Diagnostics] Computing power spectra...")
    target_spec = radial_power_spectrum(_to_np(conus_viz[0, 0]))
    drn_spec = radial_power_spectrum(_to_np(drn_pred[0, 0]))
    final_spec = radial_power_spectrum(_to_np(final_pred[0, 0]))
    era5_spec = radial_power_spectrum(_to_np(era5_viz[0, 0]))
    ens_mean_spec = radial_power_spectrum(_to_np(ens_mean[0, 0]))

    fig, ax = plt.subplots(figsize=(8, 5))
    k = np.arange(1, len(target_spec))
    ax.loglog(k, target_spec[1:], 'k-', linewidth=2, label='CONUS404 Target')
    ax.loglog(k, era5_spec[1:], 'b--', linewidth=1.5, label='ERA5 (regridded)')
    ax.loglog(k, drn_spec[1:], 'g-', linewidth=1.5, label='DRN (Stage 1)')
    ax.loglog(k, final_spec[1:], 'r-', linewidth=1.5, label='Final (DRN+Diff)')
    ax.loglog(k, ens_mean_spec[1:], 'm--', linewidth=1.5, label='Ensemble Mean')
    ax.set_xlabel("Wavenumber k"); ax.set_ylabel("Power")
    ax.set_title("Radially Averaged Power Spectrum — Temperature")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/11_power_spectra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {plot_dir}/11_power_spectra.png")

    # ── 12: Error histogram ──
    drn_err = _to_np((drn_pred - conus_viz)[0, 0]).flatten()
    final_err = _to_np((final_pred - conus_viz)[0, 0]).flatten()
    ens_err = _to_np((ens_mean - conus_viz)[0, 0]).flatten()

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-3, 3, 100)
    ax.hist(drn_err, bins=bins, density=True, alpha=0.5, color="green", label=f"DRN (std={drn_err.std():.3f})")
    ax.hist(final_err, bins=bins, density=True, alpha=0.5, color="red", label=f"Final (std={final_err.std():.3f})")
    ax.hist(ens_err, bins=bins, density=True, alpha=0.5, color="purple", label=f"Ens Mean (std={ens_err.std():.3f})")
    ax.set_xlabel("Error (normalized)"); ax.set_ylabel("Density")
    ax.set_title("Error Distribution: DRN vs Final vs Ensemble Mean")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/12_error_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {plot_dir}/12_error_histogram.png")

    # ── 13: Per-pixel RMSE map ──
    with torch.no_grad():
        all_drn_err2 = torch.zeros(OUT_CH, PATCH_SIZE, PATCH_SIZE, device=device)
        all_final_err2 = torch.zeros(OUT_CH, PATCH_SIZE, PATCH_SIZE, device=device)
        n_eval = 0
        for pi in range(min(cfg["num_eval_patches"], len(patches))):
            e_p = patches[pi][0].unsqueeze(0).to(device)
            c_p = patches[pi][1].unsqueeze(0).to(device)
            dp = drn(e_p)
            all_drn_err2 += (dp - c_p).squeeze(0) ** 2
            res_p = c_p - dp
            mu_p, lv_p = vae.encode(res_p)
            z_p = vae.reparameterize(mu_p, lv_p)
            cond_p = build_cond(e_p, dp)
            with ema.apply():
                z_s = heun_sampler(diff_model, schedule, cond_p,
                                   shape=(1, LATENT_CH, LATENT_H, LATENT_W),
                                   num_steps=NUM_SAMPLING_STEPS, guidance_scale=0.2)
            r_s = vae.decode(z_s)
            fp = dp + r_s
            all_final_err2 += (fp - c_p).squeeze(0) ** 2
            n_eval += 1

    drn_rmse_map = _to_np(torch.sqrt(all_drn_err2 / n_eval)[0])
    final_rmse_map = _to_np(torch.sqrt(all_final_err2 / n_eval)[0])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    im1 = ax1.imshow(drn_rmse_map, cmap="hot_r", origin="lower"); ax1.set_title("DRN RMSE"); ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    im2 = ax2.imshow(final_rmse_map, cmap="hot_r", origin="lower"); ax2.set_title("Full Pipeline RMSE"); ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    improvement = drn_rmse_map - final_rmse_map
    vabs = max(abs(np.nanpercentile(improvement, 5)), abs(np.nanpercentile(improvement, 95)))
    im3 = ax3.imshow(improvement, cmap="RdBu_r", vmin=-vabs, vmax=vabs, origin="lower")
    ax3.set_title("Improvement\n(DRN RMSE - Full RMSE)"); ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    fig.suptitle(f"Per-Pixel RMSE (averaged over {n_eval} patches)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/13_pixel_rmse_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {plot_dir}/13_pixel_rmse_map.png")
    print(f"  DRN mean pixel RMSE:  {drn_rmse_map.mean():.4f}")
    print(f"  Full mean pixel RMSE: {final_rmse_map.mean():.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SANITY CHECK SUMMARY")
    print("=" * 70)
    drn_params = sum(p.numel() for p in drn.parameters())
    vae_params = sum(p.numel() for p in vae.parameters())
    diff_params = sum(p.numel() for p in diff_model.parameters())
    print(f"  Mode:      {mode_str} ({NUM_TRAIN_STEPS} steps/stage)")
    print(f"  Data:      {len(patches)} real patches loaded (temp only, in-memory)")
    print(f"  DRN:       {drn_losses[0]:.4f} -> {drn_losses[-1]:.4f}  ({drn_params/1e6:.1f}M params)")
    print(f"  VAE:       {vae_losses[0]:.4f} -> {vae_losses[-1]:.4f}  ({vae_params/1e6:.1f}M params)")
    print(f"  Diffusion: {diff_losses[0]:.4f} -> {diff_losses[-1]:.4f}  ({diff_params/1e6:.1f}M params)")
    print(f"  Pipeline:  shape={final_pred.shape}, all finite")
    print(f"  Ensemble:  {NUM_ENSEMBLE} members, spread std={_to_np(ens_std).mean():.4f}")
    print(f"  RMSE:      DRN={drn_rmse_map.mean():.4f}, Full={final_rmse_map.mean():.4f}")
    print(f"\n  Plots saved to: {plot_path.resolve()}/")
    files = sorted(plot_path.glob("[0-9]*.png"))
    for f in files:
        print(f"    {f.name}")
    print(f"\n  ALL CHECKS PASSED")
    print("=" * 70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check — real data, temperature only")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot_dir", type=str, default="sanity_plots")
    parser.add_argument("--extensive", action="store_true",
                        help="Run 500 steps/stage with bigger models, more data, stricter checks")
    args = parser.parse_args()

    success = sanity_check(device=args.device, plot_dir=args.plot_dir, extensive=args.extensive)
    sys.exit(0 if success else 1)
