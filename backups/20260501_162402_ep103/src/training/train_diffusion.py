"""Stage 2b: Conditional EDM diffusion training loop."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from ..models.drn import DRN
from ..models.vae import VAE
from ..models.diffusion_unet import DiffusionUNet
from ..models.edm import EDMSchedule, edm_training_loss, heun_sampler
from ..training.ema import EMA
from src.evaluation.plots import plot_loss_curves, plot_stage_comparison, radial_power_spectrum

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _make_pos_embedding(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Create 2-channel positional embedding (normalized y, x coords)."""
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy, xx], dim=0)  # (2, H, W)


def _build_diffusion_cond(era5, drn_pred, vae, pos_emb, p_uncond=0.1):
    """Build conditioning tensor for diffusion UNet.

    Concatenates:
        - ERA5 (all input channels) downsampled to latent resolution
        - VAE-encoded DRN mean (LATENT_CH channels)
        - positional embedding (2ch)

    With probability p_uncond, zeros out conditioning for classifier-free guidance.
    """
    B = era5.shape[0]
    latent_h, latent_w = 64, 64

    # Downsample all ERA5 input channels (dynamic vars + static fields) to latent resolution
    era5_down = F.interpolate(era5, size=(latent_h, latent_w), mode="bilinear", align_corners=False)

    # Encode DRN prediction with VAE encoder (mean only, no sampling)
    with torch.no_grad():
        mu_drn, _ = vae.encode(drn_pred)  # (B, LATENT_CH, 64, 64)

    # Positional embedding
    pos = pos_emb.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, 64, 64)

    cond = torch.cat([era5_down, mu_drn, pos], dim=1)

    # Classifier-free guidance: randomly zero out conditioning
    if p_uncond > 0 and torch.rand(1).item() < p_uncond:
        cond = torch.zeros_like(cond)

    return cond


def train_diffusion(
    diff_model: DiffusionUNet,
    drn: DRN,
    vae: VAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 2e-4,
    warmup_epochs: int = 5,
    ema_decay: float = 0.9999,
    p_uncond: float = 0.1,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
    plot_dir: str = "train_plots",
    log_interval: int = 50,
    eval_every: int = 3,
    latent_ch: int = 4,
    resume: bool = False,
    grad_accum: int = 1,
    cosine_restart_period: int = 0,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
    train_sampler=None,
):
    """Train the conditional diffusion UNet in VAE latent space.

    Args:
        resume: If True, load from diffusion_latest.pt and continue training.
        grad_accum: Gradient accumulation steps (effective batch = batch_size * grad_accum).
        cosine_restart_period: If >0, use CosineAnnealingWarmRestarts with this
            period (in epochs) after warmup. If 0, use single cosine decay.
        p_mean: EDM noise schedule log-normal mean. Lower values bias toward
            higher noise levels; higher (e.g. -0.8) trains more on fine details.
        p_std: EDM noise schedule log-normal std. Smaller values concentrate
            training around the mean noise level.
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    is_main = rank == 0

    drn = drn.eval()
    vae = vae.eval()
    for p in drn.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)

    if world_size > 1:
        diff_model = DDP(diff_model, device_ids=[local_rank], output_device=local_rank)
    raw_model = diff_model.module if world_size > 1 else diff_model

    schedule = EDMSchedule(p_mean=p_mean, p_std=p_std)
    ema = EMA(raw_model, decay=ema_decay)
    optimizer = torch.optim.AdamW(diff_model.parameters(), lr=lr, weight_decay=1e-5)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    if cosine_restart_period > 0:
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cosine_restart_period, T_mult=1)
    else:
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    if is_main:
        print(f"  [Diff] Noise schedule: p_mean={p_mean}, p_std={p_std}")
        print(f"  [Diff] Grad accumulation: {grad_accum} (effective batch={train_loader.batch_size * grad_accum})")
        if cosine_restart_period > 0:
            print(f"  [Diff] Cosine warm restarts: T_0={cosine_restart_period} epochs")

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    pos_emb = _make_pos_embedding(64, 64, device)
    best_val_loss = float("inf")
    all_train_losses = []
    all_val_losses = []
    start_epoch = 0

    # Resume from checkpoint
    if resume:
        ckpt_path = ckpt_dir / "diffusion_latest.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            raw_model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
            # Do NOT restore scheduler state — old checkpoint carries a stale
            # T_max from when diff_epochs was lower. Fast-forward the fresh
            # scheduler (built with the current total_epochs) to the resume
            # epoch so LR lands at the correct point on the new cosine.
            for _ in range(start_epoch):
                scheduler.step()
            if "train_losses" in ckpt:
                all_train_losses = ckpt["train_losses"]
            if "val_losses" in ckpt:
                all_val_losses = ckpt["val_losses"]
            if is_main:
                print(f"[Diff] Resumed from epoch {start_epoch} (best_val={best_val_loss:.6f})")
        else:
            if is_main:
                print(f"[Diff] No checkpoint at {ckpt_path}, starting from scratch")

    # Grab fixed eval batch
    eval_era5, eval_conus = next(iter(val_loader))

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        diff_model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (era5, conus) in enumerate(train_loader):
            era5 = era5.to(device)
            conus = conus.to(device)

            # Compute residual and encode to latent
            with torch.no_grad():
                drn_pred = drn(era5)
                residual = conus - drn_pred
                mu, _ = vae.encode(residual)
                z_clean = mu

            # Build conditioning
            cond = _build_diffusion_cond(era5, drn_pred, vae, pos_emb, p_uncond=p_uncond)

            # EDM training loss (scaled for gradient accumulation)
            loss = edm_training_loss(diff_model, schedule, z_clean, cond)
            (loss / grad_accum).backward()

            epoch_loss += loss.item()
            all_train_losses.append(loss.item())

            # Step optimizer every grad_accum mini-batches
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.0)
                optimizer.step()
                ema.update()
                optimizer.zero_grad()

            if is_main and step % log_interval == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"  [Diff] Epoch {epoch+1}, Step {step}, Loss: {loss.item():.6f}, LR: {cur_lr:.2e}")

        # Handle leftover gradients at end of epoch
        if (step + 1) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.0)
            optimizer.step()
            ema.update()
            optimizer.zero_grad()

        scheduler.step()
        avg_train = epoch_loss / max(len(train_loader), 1)

        # Validation — aggregate across all ranks (use EMA weights for the
        # measurement that actually matches inference)
        diff_model.eval()
        val_loss = torch.tensor(0.0, device=device)
        val_count = torch.tensor(0, device=device)
        with torch.no_grad(), ema.apply():
            for era5, conus in val_loader:
                era5 = era5.to(device)
                conus = conus.to(device)
                drn_pred = drn(era5)
                residual = conus - drn_pred
                mu, _ = vae.encode(residual)
                z_clean = mu
                cond = _build_diffusion_cond(era5, drn_pred, vae, pos_emb, p_uncond=0)
                loss = edm_training_loss(diff_model, schedule, z_clean, cond)
                val_loss += loss
                val_count += 1
        if world_size > 1:
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
        avg_val = (val_loss / val_count.clamp(min=1)).item()
        all_val_losses.append(avg_val)

        if is_main:
            print(f"[Diff] Epoch {epoch+1}/{epochs} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

            # Save best checkpoint
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": avg_val,
                    "best_val_loss": best_val_loss,
                    "train_losses": all_train_losses,
                    "val_losses": all_val_losses,
                }, ckpt_dir / "diffusion_best.pt")

            # Always save latest (for resuming)
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": avg_val,
                "best_val_loss": best_val_loss,
                "train_losses": all_train_losses,
                "val_losses": all_val_losses,
            }, ckpt_dir / "diffusion_latest.pt")

            # Periodic eval plots
            if (epoch + 1) % eval_every == 0 or epoch == 0:
                _eval_diffusion(
                    raw_model, drn, vae, ema, schedule, pos_emb,
                    eval_era5, eval_conus, plot_dir, epoch + 1,
                    latent_ch=latent_ch, device=device,
                )
                plot_loss_curves(
                    {"Diff Train Loss (per step)": all_train_losses,
                     "Diff Val Loss (per epoch)": all_val_losses},
                    f"{plot_dir}/diff_loss_curves.png",
                )

    return raw_model, ema


def _eval_diffusion(diff_model, drn, vae, ema, schedule, pos_emb,
                    era5_batch, conus_batch, plot_dir, epoch,
                    latent_ch=4, num_steps=32, num_ensemble=16, device="cuda"):
    """Generate diffusion evaluation plots: target / DRN / DRN+Diff / error + spectra."""
    diff_model.eval()
    era5 = era5_batch[:1].to(device)
    conus = conus_batch[:1].to(device)

    with torch.no_grad():
        drn_pred = drn(era5)
        cond = _build_diffusion_cond(era5, drn_pred, vae, pos_emb, p_uncond=0)

    # Ensemble sampling with EMA weights
    ensemble = []
    with ema.apply():
        for _ in range(num_ensemble):
            z = heun_sampler(diff_model, schedule, cond,
                             shape=(1, latent_ch, 64, 64),
                             num_steps=num_steps, guidance_scale=0.2)
            with torch.no_grad():
                resid_recon = vae.decode(z)
                ensemble.append(drn_pred + resid_recon)

    ens_stack = torch.cat(ensemble, dim=0)   # (N, C, H, W)
    ens_mean = ens_stack.mean(dim=0, keepdim=True)
    full_recon = ensemble[-1]

    # Plot comparison using ensemble mean (normalized space)
    target_np = conus[0, 0].cpu().numpy()
    drn_np = drn_pred[0, 0].cpu().numpy()
    full_np = ens_mean[0, 0].cpu().numpy()
    err_np = full_np - target_np

    plot_stage_comparison(
        [target_np, drn_np, full_np, err_np],
        ["Target", "DRN", "DRN+Diff (ens mean)", "Error"],
        f"{plot_dir}/diff_epoch{epoch:03d}.png",
        suptitle=f"Diffusion — Epoch {epoch} (N={num_ensemble})",
        share_groups=[0, 0, 0, 1],
    )

    # Power spectra
    single_np = full_recon[0, 0].cpu().numpy()
    target_spec = radial_power_spectrum(target_np)
    drn_spec = radial_power_spectrum(drn_np)
    full_spec = radial_power_spectrum(single_np)
    ens_spec = radial_power_spectrum(full_np)

    fig, ax = plt.subplots(figsize=(8, 5))
    k = np.arange(1, len(target_spec))
    ax.loglog(k, target_spec[1:], 'k-', linewidth=2, label='Target')
    ax.loglog(k, drn_spec[1:], 'g-', linewidth=1.5, label='DRN')
    ax.loglog(k, full_spec[1:], 'r-', linewidth=1.5, label='DRN+Diff (single)')
    ax.loglog(k, ens_spec[1:], 'm--', linewidth=1.5, label='DRN+Diff (ens mean)')
    ax.set_xlabel("Wavenumber k"); ax.set_ylabel("Power")
    ax.set_title(f"Power Spectrum — Epoch {epoch}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/diff_spectra_epoch{epoch:03d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Metrics: RMSE and CRPS
    drn_rmse = np.sqrt(((drn_np - target_np) ** 2).mean())
    full_rmse = np.sqrt(((full_np - target_np) ** 2).mean())

    # CRPS (energy score): E|x-y| - 0.5*E|x-x'|
    mae_term = (ens_stack - conus).abs().mean(dim=0)
    spread_term = (ens_stack.unsqueeze(0) - ens_stack.unsqueeze(1)).abs().mean(dim=(0, 1))
    crps = (mae_term - 0.5 * spread_term).mean().item()

    print(f"  [Diff] Epoch {epoch} eval — DRN RMSE: {drn_rmse:.4f}, "
          f"DRN+Diff RMSE: {full_rmse:.4f}, CRPS: {crps:.4f}")
