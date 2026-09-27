"""EDM (Elucidated Diffusion Model) noise schedule, preconditioning, and Heun sampler."""

import torch
import torch.nn as nn
import numpy as np


class EDMSchedule:
    """Variance-exploding EDM schedule (Karras et al. 2022)."""

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_data: float = 1.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data

    def sample_sigma(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample noise levels from log-normal for training."""
        log_sigma = torch.randn(n, device=device) * self.p_std + self.p_mean
        return log_sigma.exp()

    def get_sigmas(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Deterministic sigma schedule for sampling."""
        step_indices = torch.arange(num_steps, device=device, dtype=torch.float64)
        t_steps = (
            self.sigma_max ** (1 / self.rho)
            + step_indices / (num_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros(1, device=device, dtype=torch.float64)])
        return t_steps.float()

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1 / (sigma**2 + self.sigma_data**2).sqrt()

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return 0.25 * sigma.log()

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def preconditioned_forward(self, model, z_noisy, sigma, cond):
        """Apply EDM preconditioning: D(x; σ) = c_skip·x + c_out·F(c_in·x; c_noise(σ), cond)."""
        sigma_v = sigma.view(-1, 1, 1, 1)
        c_in = self.c_in(sigma_v)
        c_noise = self.c_noise(sigma)
        c_skip = self.c_skip(sigma_v)
        c_out = self.c_out(sigma_v)
        F_out = model(c_in * z_noisy, c_noise, cond)
        return c_skip * z_noisy + c_out * F_out


def edm_training_loss(model, schedule: EDMSchedule, z_clean, cond):
    """Compute EDM denoising loss with preconditioning."""
    B = z_clean.shape[0]
    sigma = schedule.sample_sigma(B, z_clean.device)
    noise = torch.randn_like(z_clean)
    z_noisy = z_clean + noise * sigma.view(-1, 1, 1, 1)
    D_out = schedule.preconditioned_forward(model, z_noisy, sigma, cond)
    weight = schedule.loss_weight(sigma).view(-1, 1, 1, 1)
    loss = weight * (D_out - z_clean) ** 2
    return loss.mean()


@torch.no_grad()
def heun_sampler(
    model,
    schedule: EDMSchedule,
    cond: torch.Tensor,
    shape: tuple,
    num_steps: int = 32,
    guidance_scale: float = 0.2,
    cond_drop: bool = False,
):
    """Heun's 2nd-order sampler for EDM.

    Args:
        model: diffusion UNet
        schedule: EDM schedule
        cond: conditioning tensor
        shape: (B, C, H, W) of latent
        num_steps: number of denoising steps
        guidance_scale: classifier-free guidance weight (0 = no guidance)
    """
    device = cond.device
    sigmas = schedule.get_sigmas(num_steps, device)
    x = torch.randn(shape, device=device) * sigmas[0]

    for i in range(num_steps):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_t = torch.full((shape[0],), sigma_cur, device=device)

        # Denoised prediction
        d_cur = schedule.preconditioned_forward(model, x, sigma_t, cond)

        # Classifier-free guidance
        if guidance_scale > 0:
            d_uncond = schedule.preconditioned_forward(model, x, sigma_t, torch.zeros_like(cond))
            d_cur = d_cur + guidance_scale * (d_cur - d_uncond)

        # Euler step
        d_prime = (x - d_cur) / sigma_cur
        x_next = x + d_prime * (sigma_next - sigma_cur)

        # Heun correction (skip at last step)
        if sigma_next > 0 and i < num_steps - 1:
            sigma_t_next = torch.full((shape[0],), sigma_next, device=device)
            d_next = schedule.preconditioned_forward(model, x_next, sigma_t_next, cond)
            if guidance_scale > 0:
                d_uncond_next = schedule.preconditioned_forward(
                    model, x_next, sigma_t_next, torch.zeros_like(cond)
                )
                d_next = d_next + guidance_scale * (d_next - d_uncond_next)
            d_prime_next = (x_next - d_next) / sigma_next
            x_next = x + 0.5 * (d_prime + d_prime_next) * (sigma_next - sigma_cur)

        x = x_next

    return x
