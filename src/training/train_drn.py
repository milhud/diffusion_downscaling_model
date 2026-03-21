"""Stage 1: DRN training loop."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from ..models.drn import DRN
from ..training.losses import PerVariableMSE
from ..training.evaluation import evaluate_drn, plot_loss_curves


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
):
    """Train the DRN (Stage 1) with periodic eval plots."""
    model = model.to(device)
    if num_output_vars > 1:
        criterion = PerVariableMSE(num_vars=num_output_vars, precip_channel=precip_channel).to(device)
        opt_params = list(model.parameters()) + list(criterion.parameters())
    else:
        criterion = nn.MSELoss()
        opt_params = model.parameters()
    optimizer = torch.optim.AdamW(opt_params, lr=lr, weight_decay=weight_decay)
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

    for epoch in range(epochs):
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
            if step % log_interval == 0:
                print(f"  [DRN] Epoch {epoch+1}/{epochs}, Step {step}, Loss: {loss.item():.6f}")

        scheduler.step()
        avg_train = epoch_loss / max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for era5, conus in val_loader:
                era5 = era5.to(device)
                conus = conus.to(device)
                pred = model(era5)
                val_loss += criterion(pred, conus).item()
        avg_val = val_loss / max(len(val_loader), 1)
        all_val_losses.append(avg_val)

        print(f"[DRN] Epoch {epoch+1}/{epochs} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

        # Save best checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
            }, ckpt_dir / "drn_best.pt")

        # Always save latest (for resuming)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val,
        }, ckpt_dir / "drn_latest.pt")

        # Periodic eval plots
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            rmse = evaluate_drn(model, eval_era5, eval_conus, plot_dir,
                                epoch=epoch + 1, device=device)
            print(f"  [DRN] Epoch {epoch+1} eval RMSE: {rmse:.4f}")

            # Loss curves
            plot_loss_curves(
                {"Train Loss (per step)": all_train_losses,
                 "Val Loss (per epoch)": all_val_losses},
                f"{plot_dir}/drn_loss_curves.png",
            )

    return model
