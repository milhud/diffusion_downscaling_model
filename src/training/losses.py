"""Loss functions for all training stages."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """Wavenumber-weighted FFT power-spectrum penalty.

    Penalizes mismatch in radially-averaged 2D power spectrum, with a linear
    wavenumber weighting (1 + k) so high-k / fine-scale structure contributes
    more. Encourages reconstruction of fine-scale spatial variation that plain
    MSE tends to smooth out.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        pred_power = pred_fft.abs() ** 2
        target_power = target_fft.abs() ** 2
        H, W = pred.shape[-2:]
        ky = torch.fft.fftfreq(H, device=pred.device).abs()
        kx = torch.fft.rfftfreq(W, device=pred.device).abs()
        k_mag = (ky[:, None] ** 2 + kx[None, :] ** 2).sqrt()
        weight = 1.0 + k_mag
        return (weight * (pred_power - target_power).abs()).mean()


class GradientLoss(nn.Module):
    """Sobel-gradient MAE: penalizes blurry spatial gradients (fronts, terrain edges)."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.t().contiguous()
        self.register_buffer("sx", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sy", sobel_y.view(1, 1, 3, 3))

    def _grad(self, x: torch.Tensor) -> tuple:
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)
        gx = F.conv2d(x_flat, self.sx, padding=1).reshape(B, C, H, W)
        gy = F.conv2d(x_flat, self.sy, padding=1).reshape(B, C, H, W)
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pgx, pgy = self._grad(pred)
        tgx, tgy = self._grad(target)
        return (pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()


class PerVariableMSE(nn.Module):
    """Inverse-variance weighted MSE with optional L1 on precipitation and
    optional spectral/gradient auxiliary losses.

    Args:
        num_vars: number of output variables
        precip_channel: index of precipitation channel for L1 loss (or -1 to disable)
        precip_l1_weight: weight for the L1 precipitation term
        spectral_weight: weight for wavenumber-weighted FFT power penalty (0 disables)
        gradient_weight: weight for Sobel-gradient MAE penalty (0 disables)
    """

    def __init__(
        self,
        num_vars: int = 7,
        precip_channel: int = 6,
        precip_l1_weight: float = 0.1,
        spectral_weight: float = 0.0,
        gradient_weight: float = 0.0,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.precip_channel = precip_channel
        self.precip_l1_weight = precip_l1_weight
        self.spectral_weight = spectral_weight
        self.gradient_weight = gradient_weight
        self.log_var = nn.Parameter(torch.zeros(num_vars))
        self.spectral = SpectralLoss() if spectral_weight > 0 else None
        self.gradient = GradientLoss() if gradient_weight > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        precision = torch.exp(-self.log_var)
        mse_per_var = ((pred - target) ** 2).mean(dim=(0, 2, 3))
        loss = (precision * mse_per_var + self.log_var).mean()

        if 0 <= self.precip_channel < self.num_vars:
            l1 = F.l1_loss(pred[:, self.precip_channel], target[:, self.precip_channel])
            loss = loss + self.precip_l1_weight * l1

        if self.spectral is not None:
            loss = loss + self.spectral_weight * self.spectral(pred, target)
        if self.gradient is not None:
            loss = loss + self.gradient_weight * self.gradient(pred, target)

        return loss


class KLDivLoss(nn.Module):
    """KL divergence loss for VAE: KL(q(z|x) || N(0,I))."""

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class VAELoss(nn.Module):
    """Combined VAE loss: reconstruction + β * KL + λ_spec * spectral."""

    def __init__(self, spectral_weight: float = 0.0):
        super().__init__()
        self.recon_loss = nn.MSELoss()
        self.kl_loss = KLDivLoss()
        self.spectral_weight = spectral_weight
        self.spectral = SpectralLoss() if spectral_weight > 0 else None

    def forward(self, recon, target, mu, logvar, beta: float = 1e-4):
        recon_l = self.recon_loss(recon, target)
        kl_l = self.kl_loss(mu, logvar)
        total = recon_l + beta * kl_l
        if self.spectral is not None:
            total = total + self.spectral_weight * self.spectral(recon, target)
        return total, recon_l, kl_l
