"""Training loop for the R2-D2-style baseline (PyTorch reproduction — see docs/R2D2_BASELINE.md).

Single-stage residual diffusion: no DRN mean predictor, no VAE latent space. The
network denoises the residual between the (already-regridded, on our CONUS404 grid)
ERA5 conditioning field and the true CONUS404 target, directly in pixel space.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from ..models.r2d2_unet import R2D2UNet
from ..models.edm import EDMSchedule, edm_training_loss, heun_sampler
from ..training.ema import EMA
from src.evaluation.plots import plot_loss_curves, plot_stage_comparison, radial_power_spectrum

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _residual_and_cond(era5: torch.Tensor, conus: torch.Tensor, out_ch: int, p_uncond: float = 0.0):
    """Baseline = matching ERA5 channels (already regridded onto the CONUS404 grid,
    config.VARIABLE_PAIRS keeps channel order aligned 1:1 with the CONUS404 output
    channels). residual = target - baseline. cond = full ERA5+static stack.
    """
    baseline = era5[:, :out_ch]
    residual = conus - baseline
    cond = era5
    if p_uncond > 0 and torch.rand(1).item() < p_uncond:
        cond = torch.zeros_like(cond)
    return residual, cond, baseline


def train_r2d2(
    model: R2D2UNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    out_ch: int,
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
    resume: bool = False,
    grad_accum: int = 1,
    p_mean: float = 0.0,
    p_std: float = 1.2,
    sigma_data: float = None,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
    train_sampler=None,
    max_steps: int = None,
):
    """Train the R2-D2-style single-stage residual diffusion model.

    Args:
        max_steps: if set, stop after this many optimizer steps total (smoke testing).
        sigma_data: EDM sigma_data for the preconditioning; if None, estimated
            empirically from one training batch's residual std before training starts.
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    is_main = rank == 0

    # Estimate sigma_data from data if not given — residual scale differs from the
    # VAE-latent model's, since this operates directly on z-scored physical fields.
    if sigma_data is None:
        probe_era5, probe_conus = next(iter(train_loader))
        probe_residual, _, _ = _residual_and_cond(probe_era5, probe_conus, out_ch)
        sigma_data = float(probe_residual.std().clamp(min=1e-3))
        if is_main:
            print(f"  [R2D2] Estimated sigma_data={sigma_data:.4f} from training batch residual std")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module if world_size > 1 else model

    schedule = EDMSchedule(p_mean=p_mean, p_std=p_std, sigma_data=sigma_data)
    ema = EMA(raw_model, decay=ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    if is_main:
        print(f"  [R2D2] sigma_data={sigma_data:.4f}, p_mean={p_mean}, p_std={p_std}")
        print(f"  [R2D2] Grad accumulation: {grad_accum} (effective batch={train_loader.batch_size * grad_accum})")

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    all_train_losses = []
    all_val_losses = []
    start_epoch = 0
    global_step = 0

    if resume:
        ckpt_path = ckpt_dir / "r2d2_latest.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            raw_model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
            sigma_data = ckpt.get("sigma_data", sigma_data)
            schedule = EDMSchedule(p_mean=p_mean, p_std=p_std, sigma_data=sigma_data)
            for _ in range(start_epoch):
                scheduler.step()
            if "train_losses" in ckpt:
                all_train_losses = ckpt["train_losses"]
            if "val_losses" in ckpt:
                all_val_losses = ckpt["val_losses"]
            if is_main:
                print(f"[R2D2] Resumed from epoch {start_epoch} (best_val={best_val_loss:.6f})")
        else:
            if is_main:
                print(f"[R2D2] No checkpoint at {ckpt_path}, starting from scratch")

    eval_era5, eval_conus = next(iter(val_loader))

    stop_training = False
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        step = 0

        for step, (era5, conus) in enumerate(train_loader):
            era5 = era5.to(device)
            conus = conus.to(device)

            residual, cond, _ = _residual_and_cond(era5, conus, out_ch, p_uncond=p_uncond)

            loss = edm_training_loss(model, schedule, residual, cond)
            (loss / grad_accum).backward()

            epoch_loss += loss.item()
            all_train_losses.append(loss.item())

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ema.update()
                optimizer.zero_grad()
                global_step += 1

            if is_main and step % log_interval == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"  [R2D2] Epoch {epoch+1}, Step {step}, Loss: {loss.item():.6f}, LR: {cur_lr:.2e}")

            if max_steps is not None and global_step >= max_steps:
                stop_training = True
                break

        if (step + 1) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update()
            optimizer.zero_grad()

        scheduler.step()
        avg_train = epoch_loss / max(step + 1, 1)

        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        val_count = torch.tensor(0, device=device)
        with torch.no_grad(), ema.apply():
            for era5, conus in val_loader:
                era5 = era5.to(device)
                conus = conus.to(device)
                residual, cond, _ = _residual_and_cond(era5, conus, out_ch, p_uncond=0)
                loss = edm_training_loss(model, schedule, residual, cond)
                val_loss += loss
                val_count += 1
                if max_steps is not None:
                    break  # smoke test: one val batch is enough
        if world_size > 1:
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
        avg_val = (val_loss / val_count.clamp(min=1)).item()
        all_val_losses.append(avg_val)

        if is_main:
            print(f"[R2D2] Epoch {epoch+1}/{epochs} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

            save_common = dict(
                epoch=epoch,
                model_state_dict=raw_model.state_dict(),
                ema_state_dict=ema.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                val_loss=avg_val,
                best_val_loss=best_val_loss,
                train_losses=all_train_losses,
                val_losses=all_val_losses,
                sigma_data=sigma_data,
            )
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_common["best_val_loss"] = best_val_loss
                torch.save(save_common, ckpt_dir / "r2d2_best.pt")
            torch.save(save_common, ckpt_dir / "r2d2_latest.pt")

            if (epoch + 1) % eval_every == 0 or epoch == 0 or stop_training:
                _eval_r2d2(raw_model, ema, schedule, eval_era5, eval_conus,
                           plot_dir, epoch + 1, out_ch=out_ch, device=device,
                           num_steps=8 if stop_training else 32,
                           num_ensemble=2 if stop_training else 16)
                plot_loss_curves(
                    {"R2D2 Train Loss (per step)": all_train_losses,
                     "R2D2 Val Loss (per epoch)": all_val_losses},
                    f"{plot_dir}/r2d2_loss_curves.png",
                )

        if stop_training:
            break

    return raw_model, ema


def _eval_r2d2(model, ema, schedule, era5_batch, conus_batch, plot_dir, epoch,
               out_ch, num_steps=32, num_ensemble=16, device="cuda"):
    """Generate diffusion evaluation plots: target / interp baseline / R2D2 pred / error."""
    model.eval()
    era5 = era5_batch[:1].to(device)
    conus = conus_batch[:1].to(device)

    with torch.no_grad():
        _, cond, baseline = _residual_and_cond(era5, conus, out_ch)

    ensemble = []
    with ema.apply():
        for _ in range(num_ensemble):
            resid = heun_sampler(model, schedule, cond,
                                  shape=(1, out_ch, era5.shape[2], era5.shape[3]),
                                  num_steps=num_steps, guidance_scale=0.2)
            ensemble.append(baseline + resid)

    ens_stack = torch.cat(ensemble, dim=0)
    ens_mean = ens_stack.mean(dim=0, keepdim=True)

    target_np = conus[0, 0].cpu().numpy()
    baseline_np = baseline[0, 0].cpu().numpy()
    full_np = ens_mean[0, 0].cpu().numpy()
    err_np = full_np - target_np

    plot_stage_comparison(
        [target_np, baseline_np, full_np, err_np],
        ["Target", "ERA5 baseline (interp)", "R2D2-style (ens mean)", "Error"],
        f"{plot_dir}/r2d2_epoch{epoch:03d}.png",
        suptitle=f"R2D2-style (PyTorch repro) — Epoch {epoch} (N={num_ensemble})",
        share_groups=[0, 0, 0, 1],
    )

    rmse_base = np.sqrt(((baseline_np - target_np) ** 2).mean())
    rmse_full = np.sqrt(((full_np - target_np) ** 2).mean())
    mae_term = (ens_stack - conus).abs().mean(dim=0)
    spread_term = (ens_stack.unsqueeze(0) - ens_stack.unsqueeze(1)).abs().mean(dim=(0, 1))
    crps = (mae_term - 0.5 * spread_term).mean().item()

    print(f"  [R2D2] Epoch {epoch} eval — Baseline RMSE: {rmse_base:.4f}, "
          f"R2D2-style RMSE: {rmse_full:.4f}, CRPS: {crps:.4f}")
