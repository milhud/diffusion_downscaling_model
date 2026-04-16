# Plan: Two-Stage Latent Diffusion Model for ERA5→CONUS404 Downscaling

## Context

Building a greenfield atmospheric downscaling system. ERA5 (~27km coarse reanalysis) is the input; CONUS404 (~4km WRF-downscaled) is the high-resolution truth. The goal is a two-stage pipeline:
1. **Stage 1 (DRN)**: Deterministic regression network — predicts the conditional mean of CONUS404 given ERA5. No disk writes needed; just build the model.
2. **Stage 2 (LDM)**: Latent diffusion model — generates fine-scale stochastic variations (residual) in compressed latent space.

This is inspired by CorrDiff (Mardani et al.), LDM (Rombach et al.), and R2-D2 (Lopez-Gomez et al.).

---

## Phase 0: Data Safety — Make .nc Files Read-Only

Before any code runs, protect the source data:

```bash
chmod -R 444 /gpfsm/dnb33/hpmille1/diffusion_downscaling_model/data/
# Verify:
ls -l /gpfsm/dnb33/hpmille1/diffusion_downscaling_model/data/ | head -5
```

This applies to both the symlinks and (if permissions allow) the underlying files at:
- `/discover/nobackup/sduan/pipeline/data/processed/era5_YYYY.nc`
- `/discover/nobackup/hpmille1/final_data/conus404_yearly_YYYY.nc`

All normalization, regridding, and preprocessing happens **in-memory only**. No intermediate files are written to disk at any stage.

---

## Data Facts

| | ERA5 | CONUS404 |
|---|---|---|
| Grid | 111×235 (regular lat/lon) | 1015×1367 (WRF Lambert conformal) |
| Resolution | ~27 km | ~4 km |
| Shape | (12, 366, 111, 235) per year | (366, 1015, 1367) per year |
| Key vars | t2m, d2m, u10, v10, sp, tp, z | T2, TD2, U10, V10, PSFC, Q2, PREC_ACC_NC |
| Years | 1980–2020 | 1980–2020 |

**ERA5 time structure**: The (12, 366) leading dims are (month, day-of-year). Each month only populates its own days; others are NaN. Requires a `(day → month_index)` lookup table computed in memory at dataset init.

**Grid mismatch**: ERA5 is regular lat/lon; CONUS404 is WRF Lambert conformal. ~32% of CONUS404 pixels fall outside ERA5 lat/lon bounds. Regridding via xESMF bilinear + nearest-neighbor extrapolation — weight matrix computed once and held in memory, never written to disk.

**Normalization**: All transforms (z-score, log1p, sqrt) computed from training set statistics held in memory. Stats can be recomputed at each run or saved to a small `norm_stats.npz` (not raw data).

**Split**: Train 1980–2014 | Val 2015–2017 | Test 2018–2020

---

## Architecture

### Data Flow
```
ERA5 (111×235) → bilinear regrid [in-memory] → ERA5* (1015×1367)
                     ↓
             Stage 1 DRN → μ (1015×1367, 7 vars)  [in-memory]
             CONUS404 - μ = residual r              [in-memory]
                     ↓
             VAE Encoder → z (64×64, 8 ch)          [4× spatial compression]
                     ↓
             Diffusion UNet (conditioned on ERA5* + μ) → z_sample
                     ↓
             VAE Decoder → r_sample (256×256, 7 vars)
                     ↓
             μ + r_sample → final CONUS404 prediction
```

### Stage 1: Deterministic Regression Network (DRN)
- **Purpose**: Predict E[x|y] — reduces residual variance for Stage 2
- **Input**: ERA5 regridded to CONUS404 grid (in-memory) + static fields = 13 channels
- **Output**: 7 CONUS404 variables (T2, TD2, U10, V10, PSFC, Q2, PREC_ACC_NC)
- **Architecture**: 4-level UNet with ResBlocks + attention (~34M params)
  - Encoder: 64→128→256→512 channels with stride-2 downsampling
  - Bottleneck: 512 channels + attention
  - Decoder: skip connections + up-convolutions
  - 256×256 patches (patch-based training)
- **Loss**: Per-variable inverse-variance weighted MSE + L1 on precipitation
- **Training**: AdamW lr=1e-4, 100 epochs, batch=16, 4×A100

### Stage 2a: Variational Autoencoder (VAE)
- **Input**: Residual r = CONUS404_normalized - DRN_prediction (7 channels, 256×256 patches)
- **Latent space**: (8, 64, 64) — 4× spatial compression
- **Encoder**: Conv(7→128) → 2 down-blocks (128→256→512) → attention → Conv(512→16) → split μ/logvar
- **Decoder**: Conv(8→512) → attention → 2 up-blocks (512→256→128) → Conv(128→7)
- **Loss**: MSE reconstruction + KL (beta annealed 0→1e-4 over 15 epochs)
- DRN runs in eval() mode to generate residuals; no intermediate data written

### Stage 2b: Conditional EDM Diffusion UNet
- **Noise schedule**: Variance-exploding EDM (Karras 2022), σ_min=0.002, σ_max=80.0, LogNormal(−1.2, 1.2)
- **Input**: Noisy latent z_t (8ch) + ERA5 at 64×64 (7ch) + VAE-encoded DRN mean (8ch) + pos emb (2ch) = 25ch
- **Architecture**: 4-level UNet with FiLM time conditioning (~180M params)
- **Classifier-free guidance**: p_uncond=0.1 during training, g=0.2 at inference
- **Sampling**: Heun's 2nd-order method, 32 denoising steps
- **EMA**: decay=0.9999

---

## Normalization (all in-memory)
| Variable | Transform |
|---|---|
| t2m, d2m, T2, TD2 | z-score |
| u10, v10, U10, V10, sp, PSFC | z-score |
| tp, PREC_ACC_NC | log1p then z-score |
| Q2 | sqrt then z-score |
| z/geopotential | divide by g=9.81 then z-score |

Stats computed from training years (1980–2014) in memory at startup. Optionally cached to `norm_stats.npz` (not raw data).

---

## File Structure

```
src/
  data/
    dataset.py         # DownscalingDataset, patch extraction, DataModule
    regrid.py          # ERA5Regridder (xESMF, weights held in memory)
    normalization.py   # NormalizationStats, normalize/denormalize (in-memory)
    static_fields.py   # Load CONUS404 terrain, LAI, lat/lon (read-only)

  models/
    components.py      # ResBlock, AttentionBlock, DownConv, UpConv, FiLM, TimeEmbedding
    drn.py             # DeterministicRegressionNetwork (Stage 1)
    vae.py             # VAEEncoder, VAEDecoder, VAE wrapper
    diffusion_unet.py  # DiffusionUNet with EDM preconditioning
    edm.py             # EDMSchedule, Heun sampler, loss weighting

  training/
    train_drn.py       # Stage 1 training loop
    train_vae.py       # Stage 2a training loop (beta annealing)
    train_diffusion.py # Stage 2b training loop (EMA, CFG)
    losses.py          # PerVariableMSE, KLDivLoss, EDMWeightedLoss
    ema.py             # ExponentialMovingAverage

  inference/
    pipeline.py        # DownscalingPipeline (tiled + Hann window stitching)
    sample.py          # CLI: load models, run inference, save .nc output

  utils/
    metrics.py         # RMSE, CRPS, power_spectrum_2d, ETS
    visualization.py   # plot_comparison, plot_spectra
    checkpoint.py      # save/load checkpoint

  configs/
    data_config.yaml
    drn_config.yaml
    vae_config.yaml
    diffusion_config.yaml

scripts/
  evaluate_test_set.py  # Evaluation on 2018–2020

train.py                # Entry point: --stage [drn|vae|diffusion]
ARCHITECTURE.md         # Detailed architecture documentation (to be written first)
```

---

## Training Sequence
1. **chmod**: `chmod -R 444 data/` to protect source .nc files
2. **Phase 1** (DRN, 2–4 days, 4×A100): `torchrun train.py --stage drn`
3. **Phase 2** (VAE, 1–2 days, 4×A100): `torchrun train.py --stage vae --drn_checkpoint ...`
4. **Phase 3** (Diffusion, 3–5 days, 4×A100): `torchrun train.py --stage diffusion ...`

---

## Verification
- After DRN: val MSE below bicubic interpolation baseline for T2
- After VAE: reconstruction loss <5% variance unexplained; latent histograms near-Gaussian
- After Diffusion: power spectra of generated CONUS404 vs truth; CRPS < DRN-only
- Full pipeline: ensemble mean vs CONUS404 truth for test day

---

## Key Implementation Notes
- **.nc files are read-only** — `chmod 444` first; never write to them
- **All normalization in-memory** — no intermediate .nc or binary data files written
- **xESMF regrid weights** held in memory (computed at dataset init from lat/lon arrays read from files)
- **CONUS404 Z variable**: Only load level 0 for terrain height (3D Z is ~104 GB/year — skip other levels)
- **Precipitation**: Clip to ≥0 after denormalization
- **Patch size**: 256×256 throughout (fits 4×A100 with batch=16)
- **Hann window stitching** at inference to avoid boundary artifacts
