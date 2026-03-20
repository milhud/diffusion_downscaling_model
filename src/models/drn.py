"""Stage 1: Deterministic Regression Network (DRN) — predicts E[x|y]."""

import torch
import torch.nn as nn
from .components import ResBlock, AttentionBlock, Downsample, Upsample


class DRN(nn.Module):
    """4-level UNet for deterministic downscaling regression.

    Input:  (B, in_ch, H, W)  — ERA5 regridded + static fields
    Output: (B, out_ch, H, W) — predicted CONUS404 conditional mean
    """

    def __init__(
        self,
        in_ch: int = 13,
        out_ch: int = 7,
        base_ch: int = 64,
        ch_mults: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (2,),  # indices into ch_mults where attention is applied
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = [base_ch * m for m in ch_mults]
        self.input_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_ch = channels[0]
        for i, ch in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(prev_ch, ch, dropout=dropout))
                prev_ch = ch
            if i in attn_resolutions:
                blocks.append(AttentionBlock(ch))
            self.enc_blocks.append(blocks)
            if i < len(channels) - 1:
                self.downsamples.append(Downsample(ch))

        # Bottleneck
        self.mid_block1 = ResBlock(channels[-1], channels[-1], dropout=dropout)
        self.mid_attn = AttentionBlock(channels[-1])
        self.mid_block2 = ResBlock(channels[-1], channels[-1], dropout=dropout)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(len(channels))):
            ch = channels[i]
            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                in_c = prev_ch + ch if j == 0 else ch
                blocks.append(ResBlock(in_c, ch, dropout=dropout))
                prev_ch = ch
            if i in attn_resolutions:
                blocks.append(AttentionBlock(ch))
            self.dec_blocks.append(blocks)
            if i > 0:
                self.upsamples.append(Upsample(ch))

        self.out_norm = nn.GroupNorm(min(32, channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(x)
        skips = []

        # Encoder
        for i, blocks in enumerate(self.enc_blocks):
            for block in blocks:
                h = block(h)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        # Bottleneck
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        # Decoder
        for i, blocks in enumerate(self.dec_blocks):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for j, block in enumerate(blocks):
                if j == 0:
                    h = block(h)
                else:
                    h = block(h)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        return self.out_conv(torch.nn.functional.silu(self.out_norm(h)))
