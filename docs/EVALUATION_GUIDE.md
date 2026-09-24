# Evaluation Guide

## Quick Start

After training completes, run all post-training evaluations in one job:

```bash
sbatch scripts/run_analysis.sh --analysis all_eval
```

This runs: ensemble eval, power spectra, step ablation, latent analysis, and compute benchmark.

---

## Prerequisites

Before running evaluations, ensure:

1. **Trained checkpoints exist:**
   ```
   checkpoints/drn_best.pt
   checkpoints/vae_best.pt
   checkpoints/diffusion_best.pt
   ```

2. **Cached data exists:** `cached_data/` with all years' `.npy` files

3. **Normalization stats exist:** `norm_stats.npz`

---

## Individual Analyses

### Ensemble Evaluation
```bash
sbatch scripts/run_analysis.sh --analysis ensemble
# Custom ensemble size:
sbatch scripts/run_analysis.sh --analysis ensemble --num_members 16 --num_steps 16
```

### Power Spectra
```bash
sbatch scripts/run_analysis.sh --analysis spectra
# More averaging:
sbatch scripts/run_analysis.sh --analysis spectra --max_batches 100
```

### Step Count Ablation
```bash
sbatch scripts/run_analysis.sh --analysis step_ablation
# Custom steps:
sbatch scripts/run_analysis.sh --analysis step_ablation --steps 4 8 16 32
```

### Compute Benchmark
```bash
sbatch scripts/run_analysis.sh --analysis benchmark
```

### Latent Space Analysis
```bash
sbatch scripts/run_analysis.sh --analysis latent
```

### Climate Signal
```bash
sbatch scripts/run_analysis.sh --analysis climate
```

### Ablation: Pixel CorrDiff
```bash
sbatch scripts/run_analysis.sh --analysis ablation_pixel --epochs 50
```

### Ablation: Compression Ratio
```bash
sbatch scripts/run_analysis.sh --analysis ablation_compression --ratio 2
sbatch scripts/run_analysis.sh --analysis ablation_compression --ratio 4
sbatch scripts/run_analysis.sh --analysis ablation_compression --ratio 8
# Then compare:
python -m src.evaluation.ablation_compression --compare_only --plot_dir train_plots/ablation_compression
```

---

## Using Custom Checkpoints

All analysis scripts accept checkpoint path overrides:

```bash
sbatch scripts/run_analysis.sh --analysis ensemble \
    --drn_checkpoint checkpoints/drn_epoch10.pt \
    --vae_checkpoint checkpoints/vae_epoch15.pt \
    --diff_checkpoint checkpoints/diffusion_epoch30.pt
```

---

## Interpreting Results

### Rank Histograms
- **Uniform distribution** = well-calibrated ensemble
- **U-shaped** = under-dispersive (ensemble too narrow)
- **Dome-shaped** = over-dispersive (ensemble too wide)

### Spread-Skill Ratio (SSR)
- SSR = 1.0: perfectly calibrated
- SSR < 1.0: under-dispersive (common for diffusion models)
- SSR > 1.0: over-dispersive

### Power Spectra
- Target line should be matched at all wavenumbers
- DRN typically falls off at high-k (smooths small scales)
- DRN+Diff should recover power at high-k
- If DRN+Diff exceeds target at high-k: over-generation of noise

### CRPS vs RMSE
- CRPS rewards both accuracy AND calibration
- CRPS < RMSE always (CRPS is a proper scoring rule for ensembles)
- Large gap between CRPS and RMSE suggests ensemble spread is useful

### Step Ablation
- Quality typically plateaus around 8-16 steps
- Diminishing returns after 32 steps
- If quality is constant from 4+ steps: latent space is very smooth

---

## Collecting Results for Paper

After all analyses complete:

```bash
# Check all results exist
ls results/*/summary.txt results/*/*.png 2>/dev/null

# Key figures for paper:
#   results/spectra/stage_spectra.png          -> Figure 3 (spectral analysis)
#   results/ensemble/rank_histograms.png       -> Figure 4 (calibration)
#   results/ensemble/qq_plots.png              -> Figure 5 (distribution)
#   results/step_ablation/step_ablation.png    -> Figure 6 (efficiency)
#   results/latent_analysis/latent_*.png       -> Figure 7 (latent space)
#   results/climate_signal/climate_signal_*.png -> Figure 8 (climate trends)
#   results/benchmark/benchmark_summary.txt    -> Table 2 (compute costs)

# Key tables for paper:
#   results/ensemble/summary.txt               -> Table 1 (main results)
#   results/benchmark/benchmark_summary.txt    -> Table 2 (compute)
```
