"""Stage 2b: Conditional Diffusion UNet operating in VAE latent space."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import ResBlock, AttentionBlock, Downsample, Upsample, TimeEmbedding


class DiffusionUNet(nn.Module):
    """4-level UNet with FiLM time conditioning for EDM diffusion.

    Clean skip-connection pattern:
      Encoder saves h after each ResBlock → skips list
      Decoder pops from skips in reverse, concatenates, then processes
    """

    def __init__(
        self,
        in_ch: int = 25,
        out_ch: int = 8,
        base_ch: int = 128,
        ch_mults: tuple = (1, 2, 2, 4),
        num_res_blocks: int = 4,
        attn_resolutions: tuple = (1, 2, 3),
        time_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        channels = [base_ch * m for m in ch_mults]
        num_levels = len(channels)
        self.time_embed = TimeEmbedding(time_dim)
        time_emb_dim = time_dim * 4

        self.input_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        # ── Encoder ──
        self.enc_res = nn.ModuleList()   # [level][block]
        self.enc_attn = nn.ModuleList()  # [level] — Identity if no attn
        self.downsamples = nn.ModuleList()

        skip_channels = []  # track what each skip's channel count is
        prev_ch = channels[0]
        for i, ch in enumerate(channels):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(prev_ch, ch, time_dim=time_emb_dim, dropout=dropout))
                prev_ch = ch
                skip_channels.append(ch)
            self.enc_res.append(level_blocks)
            self.enc_attn.append(AttentionBlock(ch) if i in attn_resolutions else nn.Identity())
            if i < num_levels - 1:
                self.downsamples.append(Downsample(ch))

        # ── Bottleneck ──
        self.mid_block1 = ResBlock(channels[-1], channels[-1], time_dim=time_emb_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(channels[-1])
        self.mid_block2 = ResBlock(channels[-1], channels[-1], time_dim=time_emb_dim, dropout=dropout)

        # ── Decoder ──
        self.dec_res = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        prev_ch = channels[-1]
        for i in reversed(range(num_levels)):
            ch = channels[i]
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                sc = skip_channels.pop()
                level_blocks.append(ResBlock(prev_ch + sc, ch, time_dim=time_emb_dim, dropout=dropout))
                prev_ch = ch
            self.dec_res.append(level_blocks)
            self.dec_attn.append(AttentionBlock(ch) if i in attn_resolutions else nn.Identity())
            if i > 0:
                self.upsamples.append(Upsample(ch))

        self.out_norm = nn.GroupNorm(min(32, channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_ch, 3, padding=1)

    def forward(self, z_noisy: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(sigma)
        x = torch.cat([z_noisy, cond], dim=1)
        h = self.input_conv(x)

        # Encoder — save skip after every ResBlock
        skips = []
        for i, (res_blocks, attn) in enumerate(zip(self.enc_res, self.enc_attn)):
            for block in res_blocks:
                h = block(h, t_emb)
                skips.append(h)
            h = attn(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder — pop skips in reverse
        for i, (res_blocks, attn) in enumerate(zip(self.dec_res, self.dec_attn)):
            for block in res_blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
            h = attn(h)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        return self.out_conv(F.silu(self.out_norm(h)))
