# Ablation Studies

## Overview

Two ablation studies isolate the contributions of our key design choices:

1. **Latent vs Pixel-Space Diffusion** — Does compressing to latent space help?
2. **Compression Ratio** — What is the optimal VAE downsampling factor?

---

## Ablation 1: Latent vs Pixel-Space CorrDiff

### Hypothesis
Running diffusion in the VAE's 64x64 latent space is faster and converges better than running diffusion directly on 256x256 pixel-space residuals, because:
- 16x fewer spatial positions per denoising step
- VAE compression removes imperceptible high-frequency detail (per Rombach et al.)
- Combined with CorrDiff's residual variance reduction, the latent target distribution is simpler

### Setup

| | Latent CorrDiff (ours) | Pixel CorrDiff (ablation) |
|---|---|---|
| Diffusion input | z_noisy (8, 64, 64) | residual (OUT_CH, 256, 256) |
| Conditioning | ERA5_down + mu_encoded + pos (30ch, 64x64) | ERA5 + DRN_pred + pos (IN_CH+OUT_CH+2, 256x256) |
| Denoising target | Latent residual code | Pixel-space residual |
| VAE required | Yes | No |
| Architecture | Same UNet, same hyperparameters | Same UNet, same hyperparameters |

Both models use the same DRN for the conditional mean. The only difference is whether VAE compression is applied before diffusion.

### How to Run

```bash
# Train pixel-space ablation (requires ~12h on A100)
sbatch scripts/run_analysis.sh --analysis ablation_pixel --epochs 50

# Compare results after training
# Loss curves: train_plots/ablation_pixel/pixel_diff_loss.png
# Timings: train_plots/ablation_pixel/pixel_diff_timings.npz
```

### Expected Results
From Rombach et al. (LDM paper), we expect:
- LDM-1 (pixel space): slower convergence, higher FID
- LDM-4 (64x64 latent): 2.7x training speedup, better final quality
- The atmospheric domain may show different trade-offs due to spatially structured data vs natural images

### Metrics to Compare
1. **Convergence speed**: epochs to reach target val loss
2. **Wall time per epoch**: latent should be faster (smaller spatial dims)
3. **Final RMSE**: DRN + pixel-diff vs DRN + latent-diff
4. **Power spectra**: does pixel-space recover more high-k variance?
5. **GPU memory**: peak allocation during sampling

---

## Ablation 2: VAE Compression Ratio

### Hypothesis
Moderate compression (LDM-4, 4x spatial downsampling) preserves essential atmospheric structure while providing sufficient compute savings. Too little compression (LDM-2) wastes compute; too much (LDM-8) loses fine-scale information critical for downscaling.

### Setup

| Variant | Latent Size | Spatial Reduction | Latent Channels |
|---------|-------------|-------------------|-----------------|
| LDM-2 | 128x128 | 2x | 4 |
| LDM-4 | 64x64 | 4x (default) | 4 |
| LDM-8 | 32x32 | 8x | 8 (increased to compensate) |

### How to Run

```bash
# Train each variant
sbatch scripts/run_analysis.sh --analysis ablation_compression --ratio 2
sbatch scripts/run_analysis.sh --analysis ablation_compression --ratio 4
sbatch scripts/run_analysis.sh --analysis ablation_compression --ratio 8

# After all complete, generate comparison bar chart
python -m src.evaluation.ablation_compression --compare_only \
    --plot_dir train_plots/ablation_compression
```

### Metrics to Compare
1. **VAE reconstruction RMSE**: how much information is lost at each ratio
2. **Downstream diffusion quality**: final DRN+Diff RMSE at each ratio
3. **Diffusion convergence speed**: epochs needed at each latent resolution
4. **Inference throughput**: samples per second at each latent resolution

### Connection to Rombach et al.
Their findings on ImageNet:
- LDM-1: too slow, poor FID after 2M steps
- LDM-4 to LDM-8: best trade-off (38-point FID gap vs LDM-1)
- LDM-32: information loss causes quality degradation

We expect atmospheric fields to favor LDM-4 because:
- Atmospheric fields have strong spatial correlations (smoother than natural images)
- Fine-scale terrain-induced gradients need preservation
- LDM-8 may over-compress terrain effects

---

## How to Present in Paper

### Ablation Table Format

```
Table N: Ablation study results

Configuration          RMSE↓   CRPS↓   Train Time↓   Inference↓
                                       (GPU-hours)   (ms/sample)
──────────────────────────────────────────────────────────────────
DRN only (no diff)     0.118    --       X              Y
+ Pixel CorrDiff       A.AAA   B.BBB    XX             YY
+ Latent CorrDiff      C.CCC   D.DDD    ZZ             WW
  (LDM-2, 128x128)    E.EEE   F.FFF    ...            ...
  (LDM-4, 64x64)      C.CCC   D.DDD    ...            ...
  (LDM-8, 32x32)      G.GGG   H.HHH    ...            ...
```

### Key Claims to Support
1. Latent CorrDiff converges N× faster than pixel CorrDiff
2. Latent CorrDiff uses M× less GPU memory during sampling
3. LDM-4 achieves best quality-efficiency trade-off for atmospheric fields
4. Residual + latent compression are complementary (not redundant)
