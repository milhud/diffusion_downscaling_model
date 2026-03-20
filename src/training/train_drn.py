"""Stage 1: DRN training loop."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from ..models.drn import DRN
from ..training.losses import PerVariableMSE


def train_drn(
    model: DRN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
    log_interval: int = 50,
):
    """Train the DRN (Stage 1)."""
    model = model.to(device)
    criterion = PerVariableMSE(num_vars=7, precip_channel=6).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

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

        print(f"[DRN] Epoch {epoch+1}/{epochs} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
            }, ckpt_dir / "drn_best.pt")

    return model
