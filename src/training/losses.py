"""Loss functions for all training stages."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerVariableMSE(nn.Module):
    """Inverse-variance weighted MSE with optional L1 on specific channels (e.g., precipitation).

    Args:
        num_vars: number of output variables
        precip_channel: index of precipitation channel for L1 loss (or -1 to disable)
        precip_l1_weight: weight for the L1 precipitation term
    """

    def __init__(self, num_vars: int = 7, precip_channel: int = 6, precip_l1_weight: float = 0.1):
        super().__init__()
        self.num_vars = num_vars
        self.precip_channel = precip_channel
        self.precip_l1_weight = precip_l1_weight
        # Learnable inverse-variance weights (log scale for stability)
        self.log_var = nn.Parameter(torch.zeros(num_vars))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, num_vars, H, W)
            target: (B, num_vars, H, W)
        """
        precision = torch.exp(-self.log_var)  # 1/variance
        mse_per_var = ((pred - target) ** 2).mean(dim=(0, 2, 3))  # (num_vars,)
        # Weighted MSE + regularization term to prevent variance from growing
        loss = (precision * mse_per_var + self.log_var).mean()

        # L1 on precipitation channel
        if 0 <= self.precip_channel < self.num_vars:
            l1 = F.l1_loss(pred[:, self.precip_channel], target[:, self.precip_channel])
            loss = loss + self.precip_l1_weight * l1

        return loss


class KLDivLoss(nn.Module):
    """KL divergence loss for VAE: KL(q(z|x) || N(0,I))."""

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class VAELoss(nn.Module):
    """Combined VAE loss: reconstruction + beta * KL."""

    def __init__(self):
        super().__init__()
        self.recon_loss = nn.MSELoss()
        self.kl_loss = KLDivLoss()

    def forward(self, recon, target, mu, logvar, beta: float = 1e-4):
        recon_l = self.recon_loss(recon, target)
        kl_l = self.kl_loss(mu, logvar)
        return recon_l + beta * kl_l, recon_l, kl_l
