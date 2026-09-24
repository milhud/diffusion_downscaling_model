# Publishable Points: Latent CorrDiff for Atmospheric Downscaling

## Core Innovation

This work introduces **Latent CorrDiff** — the first model to combine CorrDiff's residual corrective decomposition (Mardani et al. 2025) with latent diffusion (Rombach et al. 2022) for atmospheric downscaling. The model performs ERA5 (0.25deg, ~27km) to CONUS404 (4km) downscaling over the contiguous United States using a fully learned three-stage pipeline with no physics-based model in the loop.

The key idea: rather than running an expensive diffusion process in full pixel space (256x256 or larger), we compress the DRN residual field into a 64x64 latent space via a VAE, then run the diffusion model in that compressed space. This retains the variance reduction benefit of residual learning (CorrDiff) while adding the compute efficiency of latent-space generation (LDM).

---

## Detailed Comparison with Prior Work

### vs. CorrDiff (Mardani et al., Communications Earth & Environment, 2025)

CorrDiff introduced the two-step corrective decomposition x = E[x|y] + r, where a UNet regression model predicts the conditional mean and a diffusion model generates the stochastic residual. Our model preserves this decomposition but makes three key changes:

1. **Latent-space diffusion instead of pixel-space diffusion.** CorrDiff runs its EDM diffusion model on 448x448 pixel-space residuals. We insert a VAE between the regression and diffusion stages, compressing residuals from 256x256 to 64x64 latent codes. The diffusion model then operates entirely in this latent space. This reduces the spatial dimensionality by 16x per denoising step (64x64 = 4,096 vs 448x448 = 200,704 spatial positions). The variance reduction property of the residual decomposition (var(r) <= var(x), proven by Mardani et al.) is preserved — the VAE simply provides a more compact representation of r.

2. **Domain scale.** CorrDiff covers Taiwan (~36x448 input to 448x448 output, 2km resolution). We cover the contiguous United States (1015x1367 full grid, 4km resolution), a domain roughly 50x larger in area. Our patch-based training (256x256 patches from 87 valid land origins) enables scaling to this domain while maintaining training efficiency.

3. **Data regime.** CorrDiff trains on 3 years (2018-2020) of CWA Taiwan WRF data. We train on 35 years (1980-2014) of CONUS404 data (WRF dynamical downscaling of ERA5 by NCAR), with 3 years validation (2015-2017) and 3 years test (2018-2020). The longer training period provides more diverse weather conditions and seasonal cycles.

**What we keep from CorrDiff:**
- EDM (Elucidated Diffusion Model) noise schedule with variance-exploding formulation
- Karras et al. (2022) preconditioning (c_skip, c_out, c_in, c_noise)
- Heun's 2nd-order ODE sampler for inference
- Classifier-free guidance (p_uncond=0.1 training, g=0.2 inference)
- UNet backbone with FiLM time conditioning
- Concatenation-based spatial conditioning

### vs. R2-D2 (Lopez-Gomez et al., PNAS, 2025)

R2-D2 uses a two-stage dynamical-generative framework where WRF first downscales ESM output from ~100km to 45km, then a diffusion model refines from 45km to 9km. Our approach differs in three fundamental ways:

1. **Fully learned pipeline — no physics model required.** R2-D2's first stage is a full WRF simulation at 45km resolution, which still requires substantial compute and physics-based infrastructure. We replace this entirely with a learned DRN (Deterministic Regression Network), making the full pipeline trainable end-to-end with standard deep learning tools. This eliminates the need for WRF expertise, configuration, and compute for the coarse-resolution stage.

2. **Latent-space diffusion.** R2-D2 runs its diffusion model in pixel space (340x270 spatial grid, ~180M parameters). We compress to 64x64 latent space before diffusion. While R2-D2's pixel grid is smaller than CorrDiff's, the latent approach still provides significant compute savings per denoising step.

3. **Single-source training with higher target resolution.** R2-D2 trains on WRF downscaling of a single CMIP6 model (CanESM5) at 9km resolution and demonstrates generalization to 7 other ESMs. We train on CONUS404 (WRF downscaling of ERA5) at 4km resolution. While we do not yet demonstrate multi-ESM generalization, our higher target resolution captures finer-scale features.

**What R2-D2 does that we don't (yet):**
- Multi-ESM generalization testing
- End-of-century climate projection evaluation
- 32-member ensemble CRPS evaluation across multiple ESMs
- Compound extreme event case studies (Santa Ana winds)

### vs. Latent Diffusion Models (Rombach et al., CVPR 2022)

LDM introduced the idea of training diffusion models in the latent space of a pretrained autoencoder, achieving near-optimal balance between complexity reduction and detail preservation for image synthesis. We adapt this framework to atmospheric fields with several domain-specific modifications:

1. **Residual compression instead of direct compression.** LDM compresses raw images via a VQGAN/KL-regularized autoencoder. We compress DRN *residuals* (CONUS404 - DRN prediction), not raw atmospheric fields. This is critical because:
   - Residuals have lower variance than raw fields (by CorrDiff's variance reduction theorem)
   - The large-scale spatial structure is already captured by the DRN mean prediction
   - The VAE only needs to represent the stochastic fine-scale variability
   - This makes the latent space more compact and the diffusion task easier

2. **Concatenation conditioning instead of cross-attention.** LDM uses cross-attention for flexible multi-modal conditioning (text, semantic maps). For atmospheric downscaling, all conditions are spatially aligned (ERA5 on the same grid), so we use direct concatenation at the UNet input — simpler and more parameter-efficient for this use case.

3. **Compression ratio.** We use LDM-4 style compression (4x spatial downsampling: 256x256 -> 64x64), which Rombach et al. identified as near-optimal for balancing efficiency and fidelity. More aggressive compression (LDM-8, LDM-16) risks losing fine-scale atmospheric detail.

4. **KL-regularized VAE with beta annealing.** We use KL divergence regularization with beta annealed from 0 to 1e-3 over 30% of training, following Rombach et al.'s finding that very light regularization preserves reconstruction quality while preventing arbitrarily high-variance latent spaces.

### vs. AI-based LAM (Sha et al., arXiv 2026)

Sha et al. develop a Swin-Transformer-based Limited Area Model for autoregressive dynamical downscaling — a fundamentally different approach:

1. **Single-shot vs. autoregressive.** Sha et al.'s LAM iterates hourly, taking its own previous output as input (like a weather model). We perform single-shot downscaling — one forward pass produces the downscaled field for a given time step. This means we don't accumulate errors over time but also don't capture temporal dynamics.

2. **Deterministic vs. probabilistic.** The LAM is deterministic — ensemble diversity comes only from different boundary forcings. Our diffusion model is inherently probabilistic — each forward pass through the diffusion sampler produces a different realization, enabling ensemble generation from a single input.

3. **Architecture.** They use Swin-Transformer (windowed self-attention) with 36 stacks and 1280-dimensional tensors. We use a standard UNet backbone (~142M parameters for diffusion, ~7M for DRN, ~12M for VAE).

4. **Post-processing.** Sha et al. use a separate U-Net to derive diagnostic variables (precipitation, soil moisture, OLR) from prognostic LAM output. We include all target variables in the same pipeline.

**What Sha et al. do that we don't:**
- Multi-year continuous integration stability testing
- Boundary forcing generalization (ERA5, GDAS/FNL, CESM)
- Future climate downscaling (SSP370)
- Ensemble spread preservation analysis
- Hurricane/extratropical cyclone case studies

---

## Architecture Summary

**Variables (6 ERA5 → 6 CONUS404):**
| ERA5 Input | CONUS404 Target |
|---|---|
| t2m (2m temperature) | T2 |
| d2m (2m dewpoint) | TD2 |
| u10 (10m U-wind) | U10 |
| v10 (10m V-wind) | V10 |
| sp (surface pressure) | PSFC |
| tp (total precipitation) | PREC_ACC_NC |

Plus 6 static fields (terrain height, orographic variance, lat, lon, LAI, land-sea mask).

```
ERA5 (0.25deg, 6 vars) --[bilinear regrid]--> ERA5* (4km, 1015x1367)
                                         |
                              + 6 static fields
                                         |
                                         v
                    Stage 1: DRN  IN_CH=12, OUT_CH=6
                    mu = E[CONUS404 | ERA5*]         # (B, 6, 256, 256)
                                         |
                    residual = CONUS404 - mu
                                         |
                                         v
                    Stage 2a: VAE Encoder  LATENT_CH=8
                    z = encode(residual)              # (B, 8, 64, 64)
                                         |
                                         v
                    Stage 2b: Latent Diffusion
                    z_sample ~ p(z | ERA5*, mu)       # conditioned denoising
                                         |
                                         v
                    Stage 2a: VAE Decoder
                    r_hat = decode(z_sample)           # (B, 6, 256, 256)
                                         |
                                         v
                    output = mu + r_hat                # final prediction
```

**Diffusion conditioning (concatenated at 64x64):**
- z_noisy: (B, 8, 64, 64) — noisy latent code
- ERA5_down: (B, 12, 64, 64) — ERA5 + static fields downsampled to latent resolution
- mu_encoded: (B, 8, 64, 64) — VAE.encode(DRN prediction).mu
- pos_embed: (B, 2, 64, 64) — normalized (y, x) coordinate grids
- Total input: (B, 30, 64, 64)

---

## Results

### Single-Variable Baseline (2m Temperature only — completed)

Checkpoints saved to HuggingFace (`mudhil/diffusion-downscaling-model`, `checkpoints_single_var/`).

#### Training Convergence
| Stage | Epochs | Best Val Loss | Notes |
|-------|--------|--------------|-------|
| DRN | 50 | 0.0157 (MSE) | Converged by epoch ~4 |
| VAE | 25 | 0.000862 (recon) | Beta annealing critical |
| Diffusion | 37/50 | 0.916 | Hit 12h walltime; loss plateau observed |

#### Downscaling Skill Progression (Normalized RMSE, T2 only)
| Configuration | RMSE | vs. DRN Alone |
|--------------|------|---------------|
| DRN only | 0.118 | baseline |
| DRN + Diffusion (epoch 1) | 0.246 | worse (noise-adding phase) |
| DRN + Diffusion (epoch 9) | 0.085 | +28% better |
| DRN + Diffusion (epoch 18) | 0.070 | **+41% better** |

The diffusion model initially degrades performance (adds noise faster than it corrects), then crosses the DRN baseline around epoch 6-9 and steadily improves — consistent with CorrDiff's reported behavior.

---

### Multi-Variable Run (6 variables — in progress)

**Variables:** T2, TD2, U10, V10, PSFC, PREC_ACC_NC
**IN_CH=12, OUT_CH=6, LATENT_CH=8**
**Training data:** CONUS404 cached data, 35 train years (1980-2014)

#### DRN Early Training (epochs 1-5)
| Epoch | Train Loss (NLL) | Val Loss (NLL) | RMSE |
|-------|-----------------|----------------|------|
| 1 | 0.557 | 0.387 | 0.924 |
| 3 | -0.150 | -0.263 | 0.299 |
| 5+ | continuing... | — | — |

Note: The PerVariableMSE loss uses learnable inverse-variance weighting (Gaussian NLL), so negative values are expected and indicate the model is learning to assign high precision to each variable. RMSE is the meaningful quality metric.

Physical-unit RMSE at epoch 3 (~0.30 normalized) corresponds roughly to 1-2 K for temperature, ~1-2 m/s for winds — already in a physically reasonable range after 3 epochs.

VAE and Diffusion results pending completion of DRN training.

---

## Publishability Assessment

### Strong Points (Publication-Ready)
1. **Novel combination** — No prior work combines CorrDiff residual decomposition with latent diffusion for atmospheric downscaling. This is a clear methodological contribution.
2. **Compute efficiency argument** — 16x spatial reduction in diffusion operations is a concrete, quantifiable advantage over pixel-space approaches.
3. **Working three-stage pipeline** — All stages train successfully and the diffusion model demonstrably improves upon the deterministic baseline.
4. **Large domain** — CONUS-scale 4km downscaling is more ambitious than Taiwan-scale (CorrDiff) or Western-US 9km (R2-D2).
5. **40-year training dataset** — CONUS404 provides a rich, validated training source.

### Planned Additions for Full Publication
1. **Multi-variable results** — Full training run (DRN → VAE → Diffusion) underway with 6 variables. Final CRPS, RMSE, and power spectra results pending.
2. **32-member ensemble evaluation** — CRPS, rank histograms, spread-skill ratio, Q-Q plots across all 6 variables.
3. **Power spectra analysis** — Radially averaged FFT with Hann windowing: ERA5 interpolation vs. DRN vs. DRN+Diffusion vs. CONUS404 at each stage, per variable.
4. **Ablation study** — Pixel-space CorrDiff variant (remove VAE, run diffusion on 256x256) to isolate the latent-space advantage in convergence speed and sample quality.
5. **Precipitation synthesis** — PREC_ACC_NC predicted from log1p-pretransformed ERA5 tp, testing whether the model adds fine-scale convective structure beyond the coarse ERA5 input.
6. **Case studies** — Selected extreme events from the 2018-2020 test period (e.g., heat waves, cold fronts, precipitation extremes).
7. **Full-CONUS inference** — Tiled prediction over the complete 1015x1367 domain with overlap blending.

---

## How to Run

### Prerequisites
- Data: ERA5 and CONUS404 .nc files symlinked in `data/` (1980-2020)
- Hardware: NVIDIA A100 GPU (SLURM gpu_a100 partition, alla100 QOS, 12h max)
- Environment: PyTorch, xarray, xESMF, numpy, matplotlib

### Build Cache (One-Time)
```bash
python -m src.preprocessing.cache_builder --all
# Or parallel via SLURM: sbatch scripts/run_preprocess.sh
```
Produces: `cached_data/era5_{year}.npy`, `conus_{year}.npy`, `static_fields.npy`, `nan_days.json`

### Verify Preprocessing
```bash
python scripts/verify_preprocessing.py --plot     # cache vs raw .nc comparison plots
python scripts/verify_normalization.py             # normalization stats + histograms
python -m src.preprocessing.verify_cache           # integrity, spot-checks, correlations
```

### Train
```bash
# All three stages sequentially (~12h on A100):
python train.py --stage all --data_dir data --cache_dir cached_data

# Or individually:
python train.py --stage drn        # Stage 1: ~1h
python train.py --stage vae        # Stage 2a: ~2h
python train.py --stage diffusion  # Stage 2b: ~8h+

# Resume from latest checkpoint (e.g., after SLURM walltime):
python train.py --stage diffusion --resume

# Via SLURM:
sbatch scripts/run_training.sh --stage diffusion --resume
```

The `--resume` flag loads model weights, optimizer state, LR scheduler, EMA
weights (diffusion), and loss history from `*_latest.pt` checkpoints, then
continues training from the next epoch.

#### Auto-resubmit (handles 12h SLURM walltime limit)

```bash
# Preferred — single command, handles resume automatically:
./scripts/resume drn        # restart DRN from last checkpoint
./scripts/resume vae        # restart VAE from last checkpoint
./scripts/resume diffusion  # restart diffusion from last checkpoint

# Monitor:
tail -f train_loop.log

# Stop daemon:
kill $(cat .train_daemon.pid)
```

The `resume` script kills any existing daemon, starts a new one that polls the job every 2 minutes, and resubmits with `--resume` up to 20 times (240h total budget). Checkpoints saved at end of every epoch so at most one epoch is lost on walltime kill.

### Inference
```bash
python -m src.inference.sample_nc --input era5_input.nc --output downscaled.nc
```

### Evaluation
```bash
sbatch scripts/run_evaluation.sh
```

### Checkpoints
HuggingFace: [mudhil/diffusion-downscaling-model](https://huggingface.co/mudhil/diffusion-downscaling-model)

**Single-variable (T2 only) — completed, archived:**
- `checkpoints_single_var/drn_best.pt` (568 MB)
- `checkpoints_single_var/vae_best.pt` (454 MB)
- `checkpoints_single_var/diffusion_best.pt` (2.2 GB)
- `checkpoints_single_var/diffusion_latest.pt` (2.2 GB)

**Multi-variable (6 vars) — in progress:**
- `checkpoints/drn_best.pt` — updated as training proceeds
- `checkpoints/vae_best.pt` — after DRN completes
- `checkpoints/diffusion_best.pt` — after VAE completes

---

## Key Citations

1. Mardani, M. et al. (2025). Residual Corrective Diffusion Modeling for km-Scale Atmospheric Downscaling. *Communications Earth & Environment*, 6:124.
2. Lopez-Gomez, I. et al. (2025). Dynamical-Generative Downscaling of Climate Model Ensembles. *PNAS*, 122(17).
3. Rombach, R. et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*.
4. Sha, Y. et al. (2026). AI-Based Regional Emulation for Kilometer-Scale Dynamical Downscaling. *arXiv:2602.18646v2*.
5. Karras, T. et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. *NeurIPS 2022*.
