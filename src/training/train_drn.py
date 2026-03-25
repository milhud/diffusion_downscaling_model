"""Stage 1: DRN training loop."""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from pathlib import Path

from ..models.drn import DRN
from ..training.losses import PerVariableMSE
from src.evaluation.plots import evaluate_drn, plot_loss_curves


def train_drn(
    model: DRN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    warmup_epochs: int = 5,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
    plot_dir: str = "train_plots",
    log_interval: int = 50,
    eval_every: int = 3,
    num_output_vars: int = 1,
    precip_channel: int = -1,
    resume: bool = False,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
    train_sampler=None,
):
    """Train the DRN (Stage 1) with periodic eval plots.

    Args:
        resume: If True, load from drn_latest.pt and continue training.
    """
    is_main = rank == 0

    # Wrap model in DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module if world_size > 1 else model

    if num_output_vars > 1:
        criterion = PerVariableMSE(num_vars=num_output_vars, precip_channel=precip_channel).to(device)
        # Scale LR linearly with world_size (linear scaling rule)
        opt_params = list(model.parameters()) + list(criterion.parameters())
    else:
        criterion = nn.MSELoss()
        opt_params = model.parameters()
    optimizer = torch.optim.AdamW(opt_params, lr=lr * world_size, weight_decay=weight_decay)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    # Grab a fixed eval batch from validation set
    eval_era5, eval_conus = next(iter(val_loader))

    best_val_loss = float("inf")
    all_train_losses = []
    all_val_losses = []
    start_epoch = 0

    # Resume from checkpoint
    if resume:
        ckpt_path = ckpt_dir / "drn_latest.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            raw_model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "criterion_state_dict" in ckpt and hasattr(criterion, "load_state_dict"):
                criterion.load_state_dict(ckpt["criterion_state_dict"])
            if "train_losses" in ckpt:
                all_train_losses = ckpt["train_losses"]
            if "val_losses" in ckpt:
                all_val_losses = ckpt["val_losses"]
            if is_main:
                print(f"[DRN] Resumed from epoch {start_epoch} (best_val={best_val_loss:.6f})")
        else:
            if is_main:
                print(f"[DRN] No checkpoint at {ckpt_path}, starting from scratch")

    for epoch in range(start_epoch, epochs):
        # Sync DistributedSampler shuffling seed and criterion weights
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if world_size > 1 and hasattr(criterion, "log_var"):
            dist.broadcast(criterion.log_var.data, src=0)

        model.train()
        epoch_loss = 0.0

        for step, (era5, conus) in enumerate(train_loader):
            era5 = era5.to(device)
            conus = conus.to(device)

            pred = model(era5)
            loss = criterion(pred, conus)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            all_train_losses.append(loss.item())
            if is_main and step % log_interval == 0:
                print(f"  [DRN] Epoch {epoch+1}/{epochs}, Step {step}, Loss: {loss.item():.6f}")

        scheduler.step()
        avg_train = epoch_loss / max(len(train_loader), 1)

        # Validation — aggregate across all ranks
        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        val_count = torch.tensor(0, device=device)
        with torch.no_grad():
            for era5, conus in val_loader:
                era5 = era5.to(device)
                conus = conus.to(device)
                pred = model(era5)
                val_loss += criterion(pred, conus)
                val_count += 1
        if world_size > 1:
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
        avg_val = (val_loss / val_count.clamp(min=1)).item()
        all_val_losses.append(avg_val)

        if is_main:
            print(f"[DRN] Epoch {epoch+1}/{epochs} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

        criterion_state = criterion.state_dict() if hasattr(criterion, "state_dict") else None

        if is_main:
            # Save best checkpoint
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                ckpt_dict = {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": avg_val,
                    "best_val_loss": best_val_loss,
                    "train_losses": all_train_losses,
                    "val_losses": all_val_losses,
                }
                if criterion_state is not None:
                    ckpt_dict["criterion_state_dict"] = criterion_state
                torch.save(ckpt_dict, ckpt_dir / "drn_best.pt")

            # Always save latest (for resuming)
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": avg_val,
                "best_val_loss": best_val_loss,
                "train_losses": all_train_losses,
                "val_losses": all_val_losses,
            }
            if criterion_state is not None:
                ckpt_dict["criterion_state_dict"] = criterion_state
            torch.save(ckpt_dict, ckpt_dir / "drn_latest.pt")

            # Periodic eval plots
            if (epoch + 1) % eval_every == 0 or epoch == 0:
                rmse = evaluate_drn(raw_model, eval_era5, eval_conus, plot_dir,
                                    epoch=epoch + 1, device=device)
                print(f"  [DRN] Epoch {epoch+1} eval RMSE: {rmse:.4f}")
                plot_loss_curves(
                    {"Train Loss (per step)": all_train_losses,
                     "Val Loss (per epoch)": all_val_losses},
                    f"{plot_dir}/drn_loss_curves.png",
                )

    return raw_model
