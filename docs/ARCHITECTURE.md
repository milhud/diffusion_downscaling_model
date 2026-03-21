# Model Architecture

## Overview: Three-Stage Latent CorrDiff Pipeline

```
ERA5 (0.25deg, ~27km)
    |
    v  [bilinear regrid to CONUS404 grid]
ERA5* (1015x1367, 4km) + 6 static fields
    |
    v  Stage 1: DRN (Deterministic Regression Network)
mu = E[CONUS404 | ERA5]                    # conditional mean
    |
    v  residual = CONUS404 - mu
    |
    v  Stage 2a: VAE Encoder
z = encode(residual)                        # (B, 8, 64, 64) latent
    |
    v  Stage 2b: Latent Diffusion (EDM)
z_sample ~ p(z | ERA5, mu)                 # denoised latent sample
    |
    v  Stage 2a: VAE Decoder
r_sample = decode(z_sample)                 # reconstructed residual
    |
    v  mu + r_sample
CONUS404 prediction (B, 7, 256, 256)        # final output
```

## Innovation vs Prior Work

| | CorrDiff (Mardani) | R2-D2 (Lopez-Gomez) | **Ours** |
|---|---|---|---|
| Regression step | UNet (pixel space) | WRF (physics model) | UNet DRN |
| Diffusion space | Pixel (448x448) | Pixel (340x270) | **Latent (64x64)** |
| Compression | None | None | **VAE 4x spatial** |
| Variables | 4 (T, u, v, radar) | 6 | 7 (incl. Q2 synthesis) |
| Domain | Taiwan (2km) | Western US (9km) | CONUS (4km) |

Key advantage: Diffusion operates on 64x64 latent space instead of 256x256 pixel space, reducing compute by ~16x per denoising step while preserving residual variance reduction.

## Stage 1: DRN (Deterministic Regression Network)

4-level UNet predicting the conditional mean of CONUS404 given ERA5 input.

**Architecture:**
```
Input: (B, IN_CH, 256, 256)     # ERA5 vars + 6 static fields
  Conv2d(IN_CH -> 96)

  Encoder:
    Level 0: 2x ResBlock(96, 96) -> Downsample
    Level 1: 2x ResBlock(96, 192) -> Downsample
    Level 2: 2x ResBlock(192, 384) + Attention -> Downsample
    Level 3: 2x ResBlock(384, 768)

  Bottleneck:
    ResBlock(768, 768) + Attention + ResBlock(768, 768)

  Decoder (with skip connections):
    Level 3: 2x ResBlock(768+768, 768) -> Upsample
    Level 2: 2x ResBlock(768+384, 384) + Attention -> Upsample
    Level 1: 2x ResBlock(384+192, 192) -> Upsample
    Level 0: 2x ResBlock(192+96, 96)

  GroupNorm + SiLU + Conv2d(96 -> OUT_CH)

Output: (B, OUT_CH, 256, 256)   # 7 CONUS404 variables
```

**Parameters:** ~7M (with base_ch=96)
**Loss:** PerVariableMSE with learnable inverse-variance weights + L1 on precipitation

## Stage 2a: VAE (Variational Autoencoder)

Compresses DRN residuals from 256x256 to 64x64 latent space.

**Encoder:**
```
Input: (B, OUT_CH, 256, 256)    # residual = CONUS - DRN pred
  Conv2d(OUT_CH -> 128)
  2x ResBlock(128, 128) -> Downsample(128)     # 128x128
  2x ResBlock(128, 256) -> Downsample(256)     # 64x64
  2x ResBlock(256, 512)
  Attention(512) + ResBlock(512)
  GroupNorm + SiLU + Conv2d(512 -> 2*LATENT_CH)
  Split -> mu (B, 8, 64, 64), logvar (B, 8, 64, 64)
```

**Decoder:** Mirror of encoder with Upsample replacing Downsample.

**Parameters:** ~12M
**Loss:** MSE reconstruction + KL divergence (beta annealed 0 -> 1e-3)
**Compression ratio:** 256x256 -> 64x64 (4x spatial, LDM-4 style)

## Stage 2b: Diffusion UNet (Conditional EDM)

Denoises latent codes conditioned on ERA5 input and DRN prediction.

**Conditioning (concatenated at input):**
```
z_noisy:    (B, 8, 64, 64)     # noisy latent
ERA5_down:  (B, IN_CH, 64, 64) # ERA5 bilinear-downsampled to 64x64
mu_encoded: (B, 8, 64, 64)     # VAE.encode(DRN_prediction).mu
pos_embed:  (B, 2, 64, 64)     # normalized (y, x) coordinates
Total:      (B, 8+IN_CH+8+2, 64, 64)
```

**Architecture:**
```
Input conv: total_cond_ch -> 128

Encoder:
  Level 0: 4x ResBlock(128, 128) + FiLM(time) -> Downsample    # 32x32
  Level 1: 4x ResBlock(128, 256) + FiLM(time) + Attn -> Downsample  # 16x16
  Level 2: 4x ResBlock(256, 256) + FiLM(time) + Attn -> Downsample  # 8x8
  Level 3: 4x ResBlock(256, 512) + FiLM(time) + Attn

Bottleneck:
  ResBlock(512) + Attention + ResBlock(512)

Decoder (skip connections):
  Reverse of encoder

Output conv: 128 -> LATENT_CH (8)
```

**Parameters:** ~142M
**Time conditioning:** Sinusoidal embedding (dim=512) -> MLP -> FiLM modulation in ResBlocks

## EDM Schedule (Karras et al. 2022)

- **Type:** Variance-exploding
- **sigma_min:** 0.002
- **sigma_max:** 80.0
- **Training noise:** LogNormal(mean=-1.2, std=1.2)
- **Preconditioning:** c_skip, c_out, c_in, c_noise applied per Karras formulation
- **Loss weighting:** (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2

## Sampling

- **Method:** Heun's 2nd-order (deterministic ODE solver)
- **Steps:** 32 denoising steps
- **Classifier-free guidance:** g=0.2 (10% unconditional dropout during training)
- **EMA:** decay=0.9999, applied during inference

## Key Components (src/models/components.py)

- **ResBlock:** GroupNorm(32) -> SiLU -> Conv2d -> GroupNorm -> SiLU -> Dropout -> Conv2d + skip
- **AttentionBlock:** Multi-head self-attention (heads inferred from channels)
- **TimeEmbedding:** Sinusoidal -> Linear -> SiLU -> Linear (FiLM modulation)
- **Downsample:** Conv2d(stride=2)
- **Upsample:** Nearest-neighbor + Conv2d
