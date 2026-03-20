"""Stage 2b: Conditional EDM diffusion training loop."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from ..models.drn import DRN
from ..models.vae import VAE
from ..models.diffusion_unet import DiffusionUNet
from ..models.edm import EDMSchedule, edm_training_loss
from ..training.ema import EMA


def _make_pos_embedding(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Create 2-channel positional embedding (normalized y, x coords)."""
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy, xx], dim=0)  # (2, H, W)


def _build_diffusion_cond(era5, drn_pred, vae, pos_emb, p_uncond=0.1):
    """Build conditioning tensor for diffusion UNet.

    Concatenates:
        - ERA5 downsampled to latent resolution (7ch)
        - VAE-encoded DRN mean (8ch)
        - positional embedding (2ch)
    Total: 17 channels at latent resolution (64×64)

    With probability p_uncond, zeros out conditioning for classifier-free guidance.
    """
    B = era5.shape[0]
    latent_h, latent_w = 64, 64  # latent resolution

    # Downsample ERA5 (first 7 channels only — dynamic vars) to latent resolution
    era5_dynamic = era5[:, :7]  # (B, 7, 256, 256) — only dynamic ERA5 vars
    era5_down = F.interpolate(era5_dynamic, size=(latent_h, latent_w), mode="bilinear", align_corners=False)

    # Encode DRN prediction with VAE encoder (mean only, no sampling)
    with torch.no_grad():
        mu_drn, _ = vae.encode(drn_pred)  # (B, 8, 64, 64)

    # Positional embedding
    pos = pos_emb.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, 64, 64)

    cond = torch.cat([era5_down, mu_drn, pos], dim=1)  # (B, 17, 64, 64)

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
    lr: float = 1e-4,
    ema_decay: float = 0.9999,
    p_uncond: float = 0.1,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
    log_interval: int = 50,
):
    """Train the conditional diffusion UNet in VAE latent space."""
    diff_model = diff_model.to(device)
    drn = drn.to(device).eval()
    vae = vae.to(device).eval()
    for p in drn.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)

    schedule = EDMSchedule()
    ema = EMA(diff_model, decay=ema_decay)
    optimizer = torch.optim.AdamW(diff_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pos_emb = _make_pos_embedding(64, 64, device)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        diff_model.train()
        epoch_loss = 0.0

        for step, (era5, conus) in enumerate(train_loader):
            era5 = era5.to(device)
            conus = conus.to(device)

            # Compute residual and encode to latent
            with torch.no_grad():
                drn_pred = drn(era5)
                residual = conus - drn_pred
                mu, logvar = vae.encode(residual)
                z_clean = vae.reparameterize(mu, logvar)  # (B, 8, 64, 64)

            # Build conditioning
            cond = _build_diffusion_cond(era5, drn_pred, vae, pos_emb, p_uncond=p_uncond)

            # EDM training loss
            loss = edm_training_loss(diff_model, schedule, z_clean, cond)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.0)
            optimizer.step()
            ema.update()

            epoch_loss += loss.item()
            if step % log_interval == 0:
                print(f"  [Diff] Epoch {epoch+1}, Step {step}, Loss: {loss.item():.6f}")

        scheduler.step()
        avg_train = epoch_loss / max(len(train_loader), 1)

        # Validation
        diff_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for era5, conus in val_loader:
                era5 = era5.to(device)
                conus = conus.to(device)
                drn_pred = drn(era5)
                residual = conus - drn_pred
                mu, logvar = vae.encode(residual)
                z_clean = vae.reparameterize(mu, logvar)
                cond = _build_diffusion_cond(era5, drn_pred, vae, pos_emb, p_uncond=0)
                loss = edm_training_loss(diff_model, schedule, z_clean, cond)
                val_loss += loss.item()
        avg_val = val_loss / max(len(val_loader), 1)

        print(f"[Diff] Epoch {epoch+1}/{epochs} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": diff_model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
            }, ckpt_dir / "diffusion_best.pt")

    return diff_model, ema
