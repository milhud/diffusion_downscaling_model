
# Loss Function Improvements for Fine-Scale Temperature Variation

---

## Context

The current pipeline achieves DRN RMSE ~0.13 normalized (~1.7 K physical), with the diffusion model not yet consistently beating the DRN baseline at only ~50 epochs. The user wants to capture **local and small-scale temperature variations** better — ±0.5 normalized ≈ ±6.5 K physical (T2 std ≈ 13 K from `norm_stats.npz`).

The root problem: **MSE minimizes the mean of squared errors**, which encourages spatially smooth predictions that miss fine structure (terrain-induced gradients, fronts, convective boundaries). The diffusion EDM loss trains on all noise levels equally, not biasing toward the low-σ fine-detail regime.

---

## Proposed Loss Changes

### 1. DRN — Add Spectral + Gradient Auxiliary Losses (requires DRN retrain)

**File to modify:** `src/training/losses.py`, `src/training/train_drn.py`

**New combined loss:**
```
L_drn = L_NLL (current) + λ_spec * L_spectral + λ_grad * L_gradient
```

**Spectral loss** — penalize mismatch in radially-averaged power spectrum at high wavenumbers:
```python
class SpectralLoss(nn.Module):
    def forward(self, pred, target):
        # FFT per-channel, average over batch
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        pred_power = pred_fft.abs() ** 2
        target_power = target_fft.abs() ** 2
        # Weight by wavenumber (emphasize high-k)
        H, W = pred.shape[-2:]
        ky = torch.fft.fftfreq(H, device=pred.device).abs()
        kx = torch.fft.rfftfreq(W, device=pred.device).abs()
        k_mag = (ky[:, None] ** 2 + kx[None, :] ** 2).sqrt()
        weight = 1 + k_mag  # linear wavenumber weighting
        return (weight * (pred_power - target_power).abs()).mean()
```

**Gradient (Sobel) loss** — penalize blurry spatial gradients (critical for temperature fronts):
```python
class GradientLoss(nn.Module):
    # Sobel kernels applied per-channel
    # L = MAE(∂pred/∂x - ∂target/∂x) + MAE(∂pred/∂y - ∂target/∂y)
```

**Recommended weights:** `λ_spec=0.1`, `λ_grad=0.1` — auxiliary to NLL, not dominant.

**Trade-off:** Requires DRN retrain from scratch (~12h SLURM). But DRN RMSE improvement propagates through all downstream stages — the biggest leverage point.

---

### 2. Diffusion — Shift p_mean Toward Finer Scales (can apply now, mid-training)

**File to modify:** `config.py`

**Current:** `diff_p_mean=-0.8` → noise centered around σ ≈ 0.45  
**Proposed:** `diff_p_mean=-1.2` → noise centered around σ ≈ 0.30 (more training on fine-detail, low-σ regime)

This biases the EDM training loss toward denoising at lower noise levels, where fine spatial structure is resolved. The change takes effect immediately on resume since `sample_sigma()` reads `p_mean` each forward pass.

**Effect:** The model spends proportionally more gradient steps learning to reconstruct fine-scale structure (σ ≈ 0.1–0.5) rather than large-scale structure (σ ≈ 1–5). This directly targets the small-scale variation problem.

---

### 3. VAE — Add Spectral Reconstruction Loss (requires VAE retrain)

**File to modify:** `src/training/losses.py` (VAELoss class)

**Current:** `L_VAE = MSE_recon + β * KL`  
**Proposed:** `L_VAE = MSE_recon + λ_spec * L_spectral + β * KL`

Since the VAE compresses 256×256 residuals → 64×64 latent (4× spatial downsampling), fine-scale structure is lost at the bottleneck. A spectral reconstruction penalty ensures the decoder is forced to reconstruct high-wavenumber components of the residual, not just the smooth mean.

**Weight:** `λ_spec=0.05` — keep spectral loss weaker than reconstruction to maintain stability.

---

## Implementation Order (by impact vs. cost)

| Change | Stage | Impact | Cost | When |
|--------|-------|--------|------|------|
| Shift `p_mean` -0.8 → -1.2 | Diffusion | Medium | Zero (just config) | Now |
| Spectral + gradient loss | DRN | High | DRN retrain (~12h) | Next run |
| Spectral reconstruction | VAE | Medium | VAE retrain (~6h) | After DRN |

---

## Files to Modify

| File | Change |
|------|--------|
| `config.py` | `diff_p_mean=-0.8` → `-1.2` |
| `src/training/losses.py` | Add `SpectralLoss`, `GradientLoss`; extend `PerVariableMSE` and `VAELoss` |
| `src/training/train_drn.py` | Use new combined loss; add `λ_spec`, `λ_grad` hyperparams |
| `config.py` | Add `drn_spectral_weight=0.1`, `drn_gradient_weight=0.1`, `vae_spectral_weight=0.05` |

---

## Verification

- After `p_mean` change: check that `diff_spectra_epoch*.png` plots show DRN+Diff spectrum closer to target at high wavenumbers within 5–10 epochs
- After DRN retrain: compare `drn_epoch*.png` spatial sharpness vs. old — fronts and terrain edges should be crisper
- Metric to watch: per-variable RMSE on T2 should drop below 0.10 normalized; DRN+Diff CRPS should show clear improvement over DRN-only

---

# Plan: Paper-Publishable Latent CorrDiff for Atmospheric Downscaling

---

## New Question: Can All Three Stages Train Simultaneously?

**Short answer: No, not with the current architecture.** The stages have hard data dependencies:

| Stage | Requires |
|-------|---------|
| DRN | Nothing — trains on raw ERA5 + CONUS404 |
| VAE | **DRN checkpoint** — trains on `residual = CONUS404 - DRN(ERA5)` |
| Diffusion | **DRN + VAE checkpoints** — conditions on `vae.encode(drn_pred)` and trains on `vae.encode(residual)` |

**Why it can't parallelize:**
- VAE computes `residual = conus - drn(era5)` for every batch — needs a trained DRN
- Diffusion builds its conditioning as `[era5_down | vae.encode(drn_pred) | pos_embed]` — needs both

**Option: Decouple VAE from DRN (architectural change)**

Train the VAE on **raw CONUS404** instead of residuals. This lets DRN and VAE train simultaneously on two separate GPU nodes. Tradeoffs:
- ✓ DRN + VAE run in parallel → saves ~12-24h wall time
- ✗ VAE latent space optimized for full CONUS404, not just residuals → slightly higher variance → harder for diffusion
- ✗ The conditioning term `vae.encode(drn_pred)` is less meaningful (encodes an imperfect prediction in a general CONUS latent, not a residual latent)
- ✗ Residuals have ~50-70% lower variance than raw CONUS → VAE compression is more lossy on a general target

**Recommendation:** Don't change the architecture. The residual design is the core contribution (following CorrDiff's variance reduction principle). Parallelism isn't worth the quality tradeoff. Sequential training with auto-resubmit is already in place.

---

## Context
We have a working single-variable (temperature) three-stage pipeline: DRN → VAE → Latent Diffusion. DRN+Diff RMSE 0.070 vs DRN 0.118 (41% improvement). To make this publishable, we need: multi-variable support, project reorganization, comprehensive evaluation, verification scripts, end-to-end inference from .nc → .nc, and proper documentation.

**Core novelty:** First latent-space CorrDiff for atmospheric downscaling. CorrDiff and R2-D2 run diffusion in pixel space. We compress residuals via VAE (256×256 → 64×64) before diffusion — faster, more memory-efficient, same or better quality.

**Q2 synthesis (CONFIRMED):** Predict specific humidity with no direct ERA5 input — like CorrDiff's radar reflectivity synthesis.

**Checkpoint resume (CONFIRMED):** Keep 142M param diffusion model, add resume logic for 12h SLURM restarts.

---

## Phase 0: Project Reorganization

### Current structure (flat, messy)
```
src/data/          — dataset, normalization, regrid, land_mask
src/models/        — drn, vae, diffusion_unet, edm, components
src/training/      — train loops, evaluation, losses, ema
src/inference/     — pipeline (incomplete)
src/utils/         — visualization (barely used)
root/              — train.py, config.py, sample_checkpoint.py, sanity_check.py, preprocess_cache.py
```

### New structure (organized by theme)
```
src/
  preprocessing/
    __init__.py
    cache_builder.py       ← preprocess_cache.py (moved + cleaned)
    regrid.py              ← src/data/regrid.py
    normalization.py       ← src/data/normalization.py
    land_mask.py           ← src/data/land_mask.py
    verify_cache.py        ← NEW: sanity checks on cached data
  data/
    __init__.py
    dataset.py             ← src/data/dataset.py (keep here, it's data loading)
  models/
    __init__.py
    components.py          ← unchanged
    drn.py                 ← unchanged
    vae.py                 ← unchanged
    diffusion_unet.py      ← unchanged
    edm.py                 ← unchanged
  training/
    __init__.py
    train_drn.py           ← unchanged
    train_vae.py           ← unchanged
    train_diffusion.py     ← add resume support
    losses.py              ← unchanged
    ema.py                 ← unchanged
  evaluation/
    __init__.py
    metrics.py             ← NEW: CRPS, rank histograms, spread-skill, etc.
    test_evaluation.py     ← NEW: full test set eval script
    plots.py               ← src/training/evaluation.py (moved + expanded)
  inference/
    __init__.py
    pipeline.py            ← expanded: Hann tiling, full-CONUS
    sample_nc.py           ← NEW: .nc input → .nc output end-to-end
    plot_results.py        ← NEW: publication-quality result figures

scripts/
  run_training.sh          ← SLURM training
  run_preprocess.sh        ← SLURM preprocessing
  run_evaluation.sh        ← NEW: SLURM test set evaluation
  verify_preprocessing.py  ← NEW: verify cached data correctness
  verify_normalization.py  ← NEW: verify norm stats are correct
  compare_baselines.py     ← NEW: bicubic + DRN-only baselines

docs/
  ARCHITECTURE.md          ← NEW: model architecture documentation
  PREPROCESSING.md         ← NEW: data pipeline documentation
  DATA_SPECIFICATION.md    ← NEW: variable specs, grid info, splits
  RESULTS.md               ← NEW: results, plots, interpretation

config.py                  ← stays in root (central config)
train.py                   ← stays in root (entry point)
```

### Files to move/rename
| From | To |
|------|-----|
| `preprocess_cache.py` | `src/preprocessing/cache_builder.py` |
| `src/data/regrid.py` | `src/preprocessing/regrid.py` |
| `src/data/normalization.py` | `src/preprocessing/normalization.py` |
| `src/data/land_mask.py` | `src/preprocessing/land_mask.py` |
| `src/training/evaluation.py` | `src/evaluation/plots.py` |
| `src/utils/visualization.py` | merge into `src/evaluation/plots.py` |
| `sample_checkpoint.py` | `src/inference/sample_nc.py` |
| `sanity_check.py` | `scripts/verify_preprocessing.py` |
| `gridding_test.py` | delete (one-off test) |

Update all imports in every file that references moved modules.

---

## Phase 1: Multi-Variable Data Pipeline

### 1a. Fix variable mappings in `config.py`
```python
VARIABLE_PAIRS = {
    "t2m": "T2",           # 2m temperature
    "d2m": "TD2",          # 2m dewpoint
    "u10": "U10",          # 10m u-wind
    "v10": "V10",          # 10m v-wind
    "sp": "PSFC",          # surface pressure
    "tp": "PREC_ACC_NC",   # precipitation
}
SYNTHESIS_VARS = ["Q2"]    # output-only, no ERA5 counterpart
# IN_CH = 6 ERA5 + 6 static = 12
# OUT_CH = 6 paired + 1 synthesis = 7
# LATENT_CH = 8
```

### 1b. Regenerate cached data
- Delete old 1-var cache (`cached_data/`)
- Update `cache_builder.py` for new vars + Q2 synthesis output
- Re-run SLURM array job (41 tasks)
- Use float16 if storage is tight (~500GB vs ~1TB at float32)
- Regenerate `norm_stats.npz`

### 1c. Model capacity
- Increase `drn_base_ch`: 64 → 96 (~7M params) for 7-var output
- Keep VAE `base_ch=128`, diffusion UNet 142M params
- `precip_channel=5` in PerVariableMSE

---

## Phase 2: Verification Scripts

### 2a. `scripts/verify_preprocessing.py`
- Load cached .npy files, compare against raw .nc for random samples
- Check value ranges per variable (temperature ~220-330K, wind ±30 m/s, etc.)
- Verify no unexpected NaN beyond nan_days.json
- Plot side-by-side: raw .nc vs cached .npy for 3-5 random days
- Check static fields match expected terrain/lat/lon

### 2b. `scripts/verify_normalization.py`
- Load norm_stats.npz, print per-variable mean/std
- Verify normalized data has mean≈0, std≈1 on training set
- Check pretransforms (log1p for precip, sqrt for Q2) produce reasonable ranges
- Plot histograms of raw vs normalized data per variable

### 2c. `src/preprocessing/verify_cache.py`
- Quick integrity check: file sizes, shapes, dtype
- Random spot-check: load day 100 from year 2000, verify not all zeros/NaN
- Cross-variable correlation check: T2 and TD2 should be correlated, etc.

---

## Phase 3: Retrain All Three Stages

### 3a. Add checkpoint resume to all training loops
In `train_drn.py`, `train_vae.py`, `train_diffusion.py`:
```python
# At start of training function:
if (ckpt_dir / "XXX_latest.pt").exists():
    ckpt = torch.load(ckpt_dir / "XXX_latest.pt")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    # Also restore scheduler, EMA for diffusion
```

### 3b. Training sequence
1. DRN: 50 epochs with PerVariableMSE (7 vars, precip L1)
2. VAE: 25 epochs on DRN residuals (7ch → 8ch latent)
3. Diffusion: 50+ epochs with resume support (resubmit SLURM as needed)

### 3c. Generate training plots during training
- Already have per-epoch eval plots for DRN and diffusion
- Add: per-variable loss curves (not just aggregate)
- Add: per-variable RMSE tracking during validation
- Add: latent space visualization for VAE (t-SNE or PCA of latent codes)

---

## Phase 4: End-to-End Inference Pipeline

### 4a. `src/inference/sample_nc.py` — .nc input → .nc output
```python
# Usage: python -m src.inference.sample_nc \
#   --era5 data/era5_2019.nc \
#   --output results/prediction_2019.nc \
#   --day 182 --num_ensemble 32
```
- Load ERA5 .nc file for specified day
- Regrid ERA5 to CONUS404 grid
- Run full pipeline: DRN → VAE → Diffusion (N ensemble members)
- Denormalize all variables back to physical units
- Save output as .nc with proper coordinates, attributes, variable metadata
- Include ensemble mean, ensemble spread, and individual members

### 4b. `src/inference/pipeline.py` — Hann-window tiled inference
- Tile full CONUS404 domain (1015×1367) with overlapping 256×256 patches
- 2D Hann window weighting for smooth stitching
- Support both single-patch and full-domain modes

### 4c. `src/inference/plot_results.py` — Publication figures from .nc output
- Load prediction .nc and target .nc
- Generate per-variable comparison maps (target / DRN / DRN+Diff / error)
- Ensemble spread maps
- Power spectra per variable
- Difference maps with colorbars in physical units

---

## Phase 5: Comprehensive Evaluation

### 5a. `src/evaluation/metrics.py`
- **CRPS** — ensemble probabilistic skill (32-member)
- **Per-variable RMSE/MAE** — denormalized, physical units
- **Power spectra** — per variable, Hann-windowed, wavelength in km
- **Rank histograms** — ensemble calibration
- **Spread-skill ratio**
- **Q-Q plots** — per variable, especially precipitation tails
- **PDFs** — kernel density, distribution tails
- **Multi-variable correlation** — joint distributions

### 5b. `src/evaluation/test_evaluation.py`
- Full test set evaluation (2018-2020)
- Generate 32-member ensembles per sample
- Compute all metrics, save to CSV/JSON
- Generate publication figures

### 5c. Ablation: Latent vs Pixel-Space CorrDiff
- Pixel-space variant: diffusion on 256×256 residuals directly (no VAE)
- Compare: quality (CRPS, RMSE), speed (wall-clock), memory (GPU peak)

---

## Phase 6: Case Studies

Select 3-4 events from test set (2018-2020):
1. **Heat wave** — temperature gradients, terrain effects
2. **Cold front** — multi-variable consistency (T, wind, pressure)
3. **Heavy precipitation** — distribution tails, spatial structure
4. **Seasonal mean** — long-term statistics

Full-domain maps, ensemble spread, per-variable panels, power spectra for each.

---

## Phase 7: Documentation (4 markdown files in `docs/`)

### 7a. `docs/ARCHITECTURE.md`
- Three-stage pipeline diagram (ASCII or description for figure)
- DRN architecture: layers, channel counts, parameter count
- VAE architecture: encoder/decoder, latent dimensions, compression ratio
- Diffusion UNet: conditioning channels, time embedding, EDM schedule
- Comparison table vs CorrDiff / R2-D2 / Sha et al. architectures
- Latent-space innovation explanation

### 7b. `docs/PREPROCESSING.md`
- ERA5 data format: dimensions, variables, time structure (12-month indexing)
- CONUS404 data format: WRF Lambert conformal grid
- Regridding: xESMF bilinear + nearest-neighbor extrapolation
- Static fields: terrain, orographic variance, lat, lon, LAI, land-sea mask
- Normalization: pretransforms (log1p, sqrt), z-scoring, per-variable stats
- Land masking: CONUS bounds, min_land_frac=0.8, 87 valid patch origins
- NaN day handling: 63 days across 4 years
- Cache format: .npy files per year, static_fields.npy

### 7c. `docs/DATA_SPECIFICATION.md`
- Variable table: ERA5 name, CONUS404 name, units, transform, range
- Q2 synthesis variable specification
- Grid specifications: ERA5 (111×235, 0.25°), CONUS404 (1015×1367, 4km)
- Train/val/test split: 1980-2014 / 2015-2017 / 2018-2020
- Data sources and file locations
- Patch extraction: 256×256, stride, overlap

### 7d. `docs/RESULTS.md`
- Per-variable RMSE/CRPS tables
- Power spectra plots (reference)
- Ablation results: latent vs pixel-space
- Q2 synthesis quality assessment
- Training convergence curves
- Comparison with reference papers
- Case study summaries

---

## Implementation Order

```
Phase 0 (reorganize)         ██░░░░░░  1 day
Phase 1 (multi-var config)   ░██░░░░░  1 day
Phase 2 (verify scripts)     ░░██░░░░  1 day
  → regenerate cache (SLURM)  ░░██░░░░  runs overnight
Phase 3 (retrain)            ░░░████░  3-4 days (12h SLURM × stages)
Phase 4 (inference pipeline) ░░░░██░░  1-2 days (parallel with training)
Phase 5 (evaluation)         ░░░░░██░  2 days
Phase 6 (case studies)       ░░░░░░██  1 day
Phase 7 (documentation)      ░░░░░░██  1 day (parallel with Phase 6)
```

**Critical path:** Phase 0 → Phase 1 → cache regen → Phase 3 → Phase 5 → Phase 6
**Total:** ~2-3 weeks

---

## Files Summary

### Create (new)
| File | Purpose |
|------|---------|
| `src/preprocessing/__init__.py` | Package init |
| `src/preprocessing/verify_cache.py` | Cache integrity checks |
| `src/evaluation/__init__.py` | Package init |
| `src/evaluation/metrics.py` | CRPS, rank histograms, etc. |
| `src/evaluation/test_evaluation.py` | Full test set eval |
| `src/inference/sample_nc.py` | .nc → .nc inference |
| `src/inference/plot_results.py` | Publication figures |
| `scripts/verify_preprocessing.py` | Preprocessing verification |
| `scripts/verify_normalization.py` | Normalization verification |
| `scripts/run_evaluation.sh` | SLURM eval job |
| `scripts/compare_baselines.py` | Baseline comparisons |
| `docs/ARCHITECTURE.md` | Model architecture docs |
| `docs/PREPROCESSING.md` | Data pipeline docs |
| `docs/DATA_SPECIFICATION.md` | Variable/grid specs |
| `docs/RESULTS.md` | Results and analysis |

### Move/Modify
| File | Action |
|------|--------|
| `config.py` | Uncomment vars, add SYNTHESIS_VARS, bump drn_base_ch |
| `preprocess_cache.py` → `src/preprocessing/cache_builder.py` | Move + update for multi-var |
| `src/data/regrid.py` → `src/preprocessing/regrid.py` | Move |
| `src/data/normalization.py` → `src/preprocessing/normalization.py` | Move |
| `src/data/land_mask.py` → `src/preprocessing/land_mask.py` | Move |
| `src/training/evaluation.py` → `src/evaluation/plots.py` | Move + expand per-variable |
| `sample_checkpoint.py` → `src/inference/sample_nc.py` | Move + rewrite for .nc I/O |
| `train.py` | Update imports, add resume, fix precip_channel |
| `src/training/train_*.py` | Add checkpoint resume support |
| `src/inference/pipeline.py` | Add Hann tiling, fix hardcoded channels |
| All files with moved imports | Update import paths |

---

## Verification
- After Phase 0: `python train.py --stage drn --help` works (imports resolve)
- After Phase 1: `python scripts/verify_preprocessing.py` passes
- After Phase 2: `python scripts/verify_normalization.py` shows mean≈0, std≈1
- After Phase 3: per-variable RMSE < bicubic for all vars; training plots generated
- After Phase 4: `python -m src.inference.sample_nc --era5 data/era5_2019.nc --output test.nc` produces valid .nc
- After Phase 5: all metrics computed, publication figures in results/
- After Phase 7: all 4 docs/*.md files complete and accurate
