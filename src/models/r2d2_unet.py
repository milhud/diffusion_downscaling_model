"""R2-D2-style residual diffusion UNet — PyTorch reproduction, NOT the original checkpoint.

Reproduces the published architecture of R2-D2 (Lopez-Gomez et al., "Dynamical-generative
downscaling of climate model ensembles," PNAS 2025 / arXiv:2410.01776; original code:
google-research/swirl-dynamics, JAX/Flax/Orbax). The original is a single-stage residual
diffusion model — no VAE, no separate regression network — trained to denoise the residual
between a naive interpolated coarse field and the true high-res field. This module ports
that architecture spec into our PyTorch stack so it can train on our own ERA5->CONUS404
cache and be benchmarked against our in-house Latent CorrDiff model and NVIDIA CorrDiff.

See docs/R2D2_BASELINE.md for exactly what is matched vs. approximated from the paper/code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import ResBlock, Downsample, Upsample, TimeEmbedding


class R2D2UNet(nn.Module):
    """3-level UNet, no self-attention, FiLM noise-embedding conditioning.

    Matches the published `PreconditionedDenoiserUNet` spec: num_channels=(256,512,768),
    num_blocks=4 per level, noise_embed_dim=192, dropout_rate=0.1, use_attention=False,
    use_position_encoding=False. Conditioning (upsampled/regridded LR field + static
    fields) is channel-concatenated with the noisy input at the first conv, approximating
    the original `InterpConvMerge` (see docs/R2D2_BASELINE.md for the deviation note).

    Downsampling: this repo's other UNets (DRN, DiffusionUNet) downsample between levels
    but not after the last one; we follow that same convention here (2 downsamples across
    3 levels, 4x total reduction) rather than the paper's 3 downsamples (8x), to stay
    consistent with the rest of the codebase's style — documented in R2D2_BASELINE.md.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_ch: int,
        channels: tuple = (256, 512, 768),
        num_blocks: int = 4,
        noise_embed_dim: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        num_levels = len(channels)
        self.time_embed = TimeEmbedding(noise_embed_dim)
        time_emb_dim = noise_embed_dim * 4

        self.input_conv = nn.Conv2d(in_ch + cond_ch, channels[0], 3, padding=1)

        self.enc_res = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = []
        prev_ch = channels[0]
        for i, ch in enumerate(channels):
            level_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                level_blocks.append(ResBlock(prev_ch, ch, time_dim=time_emb_dim, dropout=dropout))
                prev_ch = ch
                skip_channels.append(ch)
            self.enc_res.append(level_blocks)
            if i < num_levels - 1:
                self.downsamples.append(Downsample(ch))

        self.mid_block1 = ResBlock(channels[-1], channels[-1], time_dim=time_emb_dim, dropout=dropout)
        self.mid_block2 = ResBlock(channels[-1], channels[-1], time_dim=time_emb_dim, dropout=dropout)

        self.dec_res = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        prev_ch = channels[-1]
        for i in reversed(range(num_levels)):
            ch = channels[i]
            level_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                sc = skip_channels.pop()
                level_blocks.append(ResBlock(prev_ch + sc, ch, time_dim=time_emb_dim, dropout=dropout))
                prev_ch = ch
            self.dec_res.append(level_blocks)
            if i > 0:
                self.upsamples.append(Upsample(ch))

        self.out_norm = nn.GroupNorm(min(32, channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_ch, 3, padding=1)

    def forward(self, x_noisy: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(sigma)
        h = self.input_conv(torch.cat([x_noisy, cond], dim=1))

        skips = []
        for i, res_blocks in enumerate(self.enc_res):
            for block in res_blocks:
                h = block(h, t_emb)
                skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        for i, res_blocks in enumerate(self.dec_res):
            for block in res_blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        return self.out_conv(F.silu(self.out_norm(h)))
