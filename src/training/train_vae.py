"""Stage 2a: VAE training loop with beta annealing."""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from pathlib import Path

from ..models.drn import DRN
from ..models.vae import VAE
from ..training.losses import VAELoss


def train_vae(
    vae: VAE,
    drn: DRN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-4,
    beta_max: float = 1e-3,
    beta_anneal_frac: float = 0.3,
    warmup_epochs: int = 3,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
    log_interval: int = 50,
    resume: bool = False,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
    train_sampler=None,
):
    """Train VAE on residuals (CONUS404 - DRN prediction).

    DRN runs in eval mode to generate residuals on the fly.
    Beta annealing ramps from 0 to beta_max over beta_anneal_frac of total steps.

    Args:
        resume: If True, load from vae_latest.pt and continue training.
    """
    is_main = rank == 0

    drn = drn.eval()
    for p in drn.parameters():
        p.requires_grad_(False)

    if world_size > 1:
        vae = DDP(vae, device_ids=[local_rank], output_device=local_rank)
    raw_vae = vae.module if world_size > 1 else vae

    criterion = VAELoss()
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-5)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    total_steps = epochs * max(len(train_loader), 1)
    beta_ramp_steps = int(beta_anneal_frac * total_steps)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    global_step = 0
    start_epoch = 0

    # Resume from checkpoint
    if resume:
        ckpt_path = ckpt_dir / "vae_latest.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            raw_vae.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
            global_step = ckpt.get("global_step", start_epoch * max(len(train_loader), 1))
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if is_main:
                print(f"[VAE] Resumed from epoch {start_epoch} (best_val={best_val_loss:.6f})")
        else:
            if is_main:
                print(f"[VAE] No checkpoint at {ckpt_path}, starting from scratch")

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        vae.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        for step, (era5, conus) in enumerate(train_loader):
            era5 = era5.to(device)
            conus = conus.to(device)

            # Per-step beta annealing: 0 → beta_max over beta_ramp_steps
            beta = beta_max * min(1.0, global_step / max(1, beta_ramp_steps))

            # Compute residual: CONUS404 - DRN(ERA5)
            with torch.no_grad():
                drn_pred = drn(era5)
            residual = conus - drn_pred

            # VAE forward
            recon, mu, logvar = vae(residual)
            loss, recon_l, kl_l = criterion(recon, residual, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_recon += recon_l.item()
            epoch_kl += kl_l.item()

            if is_main and step % log_interval == 0:
                print(f"  [VAE] Epoch {epoch+1}, Step {step}, Loss: {loss.item():.6f} "
                      f"(recon: {recon_l.item():.6f}, KL: {kl_l.item():.6f}, beta: {beta:.2e})")

        scheduler.step()
        n = max(len(train_loader), 1)
        avg_train = epoch_loss / n

        # Validation — aggregate across all ranks
        vae.eval()
        val_loss = torch.tensor(0.0, device=device)
        val_count = torch.tensor(0, device=device)
        with torch.no_grad():
            for era5, conus in val_loader:
                era5 = era5.to(device)
                conus = conus.to(device)
                drn_pred = drn(era5)
                residual = conus - drn_pred
                recon, mu, logvar = vae(residual)
                loss, _, _ = criterion(recon, residual, mu, logvar, beta=beta)
                val_loss += loss
                val_count += 1
        if world_size > 1:
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
        avg_val = (val_loss / val_count.clamp(min=1)).item()

        if is_main:
            print(f"[VAE] Epoch {epoch+1}/{epochs} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | beta: {beta:.2e}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": avg_val,
                    "best_val_loss": best_val_loss,
                    "global_step": global_step,
                }, ckpt_dir / "vae_best.pt")

            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": avg_val,
                "best_val_loss": best_val_loss,
                "global_step": global_step,
            }, ckpt_dir / "vae_latest.pt")

    return raw_vae
