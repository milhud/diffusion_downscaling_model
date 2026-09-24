# Results and Analysis

## Current Results (Single-Variable: Temperature Only)

### Stage 1: DRN
- **Epochs:** 7 (converged at epoch 4)
- **Best val loss:** 0.015732
- **Eval RMSE (normalized):** 0.118
- **Eval RMSE (physical):** 1.0-3.8 K

The DRN quickly learns the mapping from bilinearly-regridded ERA5 to CONUS404 temperature. Fast convergence is expected because ERA5 already contains ~90% of the temperature pattern; the DRN primarily learns terrain-induced corrections.

### Stage 2a: VAE
- **Epochs:** 20
- **Best val loss:** 0.000862
- **KL divergence:** ~0.4
- **Beta (final):** 1e-3

The VAE successfully compresses DRN residuals to the 64x64 latent space. DRN+VAE reconstruction achieves 0.20-0.60 K RMSE vs DRN alone at 1.0-3.8 K.

### Stage 2b: Latent Diffusion
- **Epochs:** 20 (ongoing, 12h SLURM limit)
- **Best val loss:** 0.916
- **Parameters:** 142M

| Epoch | Train Loss | Val Loss | DRN RMSE | DRN+Diff RMSE | Improvement |
|-------|-----------|---------|----------|---------------|-------------|
| 1 | 1.024 | 0.997 | 0.118 | 0.246 | -108% (worse) |
| 3 | 0.963 | 0.950 | 0.118 | 0.132 | -12% |
| 6 | 0.941 | 0.932 | 0.118 | 0.146 | -24% |
| 9 | 0.932 | 0.927 | 0.118 | 0.085 | **+28%** |
| 12 | 0.928 | 0.923 | 0.118 | 0.083 | **+30%** |
| 15 | 0.925 | 0.916 | 0.118 | 0.114 | +3% |
| 18 | 0.923 | 0.916 | 0.118 | 0.070 | **+41%** |

Note: Eval RMSE varies between epochs because each eval uses a single stochastic diffusion sample. The trend is clearly improving.

## Training Convergence

Training loss decreased monotonically for all three stages:
- DRN: Converged in ~4 epochs (0.068 -> 0.016)
- VAE: Converged in ~15 epochs (0.003 -> 0.0009)
- Diffusion: Still decreasing at epoch 20 (1.024 -> 0.922)

See `train_plots/` for loss curves and per-epoch evaluation plots.

## Planned Multi-Variable Results

Results will be updated after multi-variable training with:
- Per-variable RMSE/MAE tables (denormalized, physical units)
- CRPS scores (32-member ensembles)
- Power spectra per variable
- Rank histograms for ensemble calibration
- Q-Q plots
- Q2 synthesis quality assessment
- Ablation: latent vs pixel-space CorrDiff
- Case studies from test set (2018-2020)

## Evaluation Metrics

| Metric | Description | Reference Papers |
|--------|-------------|-----------------|
| RMSE | Root mean squared error (deterministic) | All |
| MAE | Mean absolute error (deterministic) | CorrDiff |
| CRPS | Continuous ranked probability score (probabilistic) | CorrDiff, R2-D2 |
| Power spectra | Radially averaged FFT, Hann-windowed | All |
| Rank histogram | Ensemble calibration assessment | CorrDiff |
| Spread-skill ratio | Ensemble spread vs RMSE of mean | CorrDiff |
| Q-Q plots | Quantile-quantile comparison | R2-D2 |
| PDFs | Probability density functions, especially tails | CorrDiff |

## Weights

Checkpoints stored on HuggingFace: [mudhil/diffusion-downscaling-model](https://huggingface.co/mudhil/diffusion-downscaling-model)

- `drn_best.pt` (568 MB) - DRN, epoch 4
- `drn_latest.pt` (568 MB) - DRN, epoch 7
- `vae_best.pt` (454 MB) - VAE, epoch 20
- `diffusion_best.pt` (2.2 GB) - Diffusion, best val loss
- `diffusion_latest.pt` (2.2 GB) - Diffusion, latest epoch
