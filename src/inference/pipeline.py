"""Full downscaling inference pipeline: ERA5 → DRN → VAE → Diffusion → final prediction."""

import torch
import torch.nn.functional as F
from ..models.drn import DRN
from ..models.vae import VAE
from ..models.diffusion_unet import DiffusionUNet
from ..models.edm import EDMSchedule, heun_sampler


def _make_pos_embedding(H: int, W: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy, xx], dim=0)


@torch.no_grad()
def run_pipeline(
    era5_input: torch.Tensor,
    drn: DRN,
    vae: VAE,
    diff_model: DiffusionUNet,
    schedule: EDMSchedule = None,
    num_steps: int = 32,
    guidance_scale: float = 0.2,
    num_samples: int = 1,
    device: str = "cuda",
):
    """Run the full two-stage downscaling pipeline.

    Args:
        era5_input: (B, 13, H, W) — normalized ERA5 + static fields
        drn: trained DRN model
        vae: trained VAE model
        diff_model: trained diffusion UNet
        schedule: EDM schedule (uses default if None)
        num_steps: denoising steps
        guidance_scale: classifier-free guidance weight
        num_samples: number of diffusion ensemble members per input

    Returns:
        drn_pred: (B, 7, H, W) — DRN deterministic prediction
        samples: (B, num_samples, 7, H, W) — diffusion ensemble predictions
    """
    if schedule is None:
        schedule = EDMSchedule()

    B = era5_input.shape[0]
    era5_input = era5_input.to(device)

    # Stage 1: DRN prediction
    drn_pred = drn(era5_input)  # (B, 7, H, W)

    # Build diffusion conditioning
    latent_h, latent_w = 64, 64
    era5_down = F.interpolate(era5_input[:, :7], (latent_h, latent_w), mode="bilinear", align_corners=False)
    mu_drn, _ = vae.encode(drn_pred)
    pos = _make_pos_embedding(latent_h, latent_w, device).unsqueeze(0).expand(B, -1, -1, -1)
    cond = torch.cat([era5_down, mu_drn, pos], dim=1)  # (B, 17, 64, 64)

    all_samples = []
    for _ in range(num_samples):
        # Stage 2: Sample residual in latent space
        z_sample = heun_sampler(
            diff_model, schedule, cond,
            shape=(B, 8, latent_h, latent_w),
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        # Decode latent to residual
        r_sample = vae.decode(z_sample)  # (B, 7, 256, 256)
        # Final prediction = DRN mean + sampled residual
        final = drn_pred + r_sample
        all_samples.append(final)

    samples = torch.stack(all_samples, dim=1)  # (B, num_samples, 7, H, W)
    return drn_pred, samples
