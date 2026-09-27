"""Stage 2a: Variational Autoencoder for residual compression."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import ResBlock, AttentionBlock, Downsample, Upsample


class VAEEncoder(nn.Module):
    """Encodes residual (7ch, 256×256) → latent params (16ch, 64×64)."""

    def __init__(self, in_ch: int = 7, latent_ch: int = 8, base_ch: int = 128, ch_mults: tuple = (1, 2, 4)):
        super().__init__()
        channels = [base_ch * m for m in ch_mults]
        layers = [nn.Conv2d(in_ch, channels[0], 3, padding=1)]

        prev_ch = channels[0]
        for i, ch in enumerate(channels):
            layers.append(ResBlock(prev_ch, ch))
            layers.append(ResBlock(ch, ch))
            if i < len(channels) - 1:
                layers.append(Downsample(ch))
            prev_ch = ch

        layers.append(AttentionBlock(channels[-1]))
        layers.append(ResBlock(channels[-1], channels[-1]))
        layers.append(nn.GroupNorm(32, channels[-1]))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(channels[-1], latent_ch * 2, 1))  # mu + logvar
        self.net = nn.Sequential(*layers)
        self.latent_ch = latent_ch

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decodes latent (8ch, 64×64) → reconstruction (7ch, 256×256)."""

    def __init__(self, out_ch: int = 7, latent_ch: int = 8, base_ch: int = 128, ch_mults: tuple = (4, 2, 1)):
        super().__init__()
        channels = [base_ch * m for m in ch_mults]
        layers = [
            nn.Conv2d(latent_ch, channels[0], 3, padding=1),
            ResBlock(channels[0], channels[0]),
            AttentionBlock(channels[0]),
        ]

        prev_ch = channels[0]
        for i, ch in enumerate(channels):
            layers.append(ResBlock(prev_ch, ch))
            layers.append(ResBlock(ch, ch))
            if i < len(channels) - 1:
                layers.append(Upsample(ch))
            prev_ch = ch

        layers.append(nn.GroupNorm(32, channels[-1]))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(channels[-1], out_ch, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    """Full VAE: encode, sample, decode."""

    def __init__(self, in_ch: int = 7, latent_ch: int = 8, base_ch: int = 128):
        super().__init__()
        self.encoder = VAEEncoder(in_ch=in_ch, latent_ch=latent_ch, base_ch=base_ch)
        self.decoder = VAEDecoder(out_ch=in_ch, latent_ch=latent_ch, base_ch=base_ch)
        self.latent_ch = latent_ch

    def encode(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
