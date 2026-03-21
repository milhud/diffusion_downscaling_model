# Experiments and Analyses

All analyses are run through a single SLURM entry point:

```bash
sbatch scripts/run_analysis.sh --analysis <NAME> [extra flags]
```

Results are saved to `results/<analysis_name>/` with plots, raw `.npz` data, and text summaries.

---

## 1. Ensemble Evaluation (32-member)

**What:** Evaluate the full pipeline with 32 stochastic diffusion samples per input. Computes CRPS, RMSE, MAE, spread-skill ratio, rank histograms, Q-Q plots, and power spectra.

**Why:** CRPS is the standard probabilistic metric used by CorrDiff and R2-D2. Rank histograms assess calibration. SSR=1 means well-calibrated; <1 means under-dispersive.

```bash
sbatch scripts/run_analysis.sh --analysis ensemble
```

**Flags:**
- `--num_members 32` — ensemble size (default 32, match CorrDiff/R2-D2)
- `--num_steps 32` — denoising steps
- `--guidance_scale 0.2` — classifier-free guidance weight

**Output:** `results/ensemble/`
- `rank_histograms.png` — per-variable rank histogram (uniform = well-calibrated)
- `qq_plots.png` — quantile-quantile plots
- `power_spectra.png` — ensemble mean vs target spectra
- `summary.txt` — per-variable CRPS, RMSE, MAE, SSR table
- `ensemble_results.npz` — raw metric arrays

---

## 2. Stage-by-Stage Power Spectra

**What:** Computes radially-averaged power spectra at each pipeline stage: ERA5 interpolation, DRN, DRN+VAE, DRN+Diffusion, and CONUS404 target.

**Why:** Shows how each stage progressively recovers small-scale variance. This is the main visual evidence that diffusion adds value over the deterministic baseline.

```bash
sbatch scripts/run_analysis.sh --analysis spectra
```

**Flags:**
- `--max_batches 50` — number of test batches to average over
- `--num_ensemble 4` — diffusion samples averaged for DRN+Diff spectrum

**Output:** `results/spectra/`
- `stage_spectra.png` — per-variable spectra with all 5 stages overlaid
- `stage_spectra_data.npz` — raw wavenumber/power arrays

---

## 3. Denoising Step Ablation

**What:** Evaluates quality (RMSE, CRPS) at 2, 4, 8, 16, 32, 64 denoising steps.

**Why:** CorrDiff uses ~12 steps (down from ~1000) thanks to residual learning. Our latent space may enable even fewer steps. Finding the minimum viable step count has direct implications for inference speed.

```bash
sbatch scripts/run_analysis.sh --analysis step_ablation
```

**Flags:**
- `--steps 2 4 8 16 32 64` — step counts to test
- `--num_ensemble 8` — ensemble size per step count
- `--max_batches 20` — test batches

**Output:** `results/step_ablation/`
- `step_ablation.png` — RMSE and CRPS vs step count (log-x scale)
- `step_ablation_data.npz` — raw data

---

## 4. Computational Benchmark

**What:** Measures wall-clock time, GPU memory, and throughput for each pipeline component. Compares latent (64x64) vs pixel (256x256) diffusion at matched step counts.

**Why:** Quantifies the core efficiency claim: latent diffusion is N× faster than pixel diffusion per denoising step.

```bash
sbatch scripts/run_analysis.sh --analysis benchmark
```

**Flags:**
- `--batch_size 1` — batch size for timing
- `--num_steps 4 8 16 32` — step counts to benchmark

**Output:** `results/benchmark/`
- `benchmark_summary.txt` — timing and memory table
- `benchmark_results.npz` — raw measurements

---

## 5. Latent Space Analysis

**What:** Visualizes VAE latent channel activations, inter-channel correlations, spatial patterns, and posterior uncertainty.

**Why:** Novel analysis not done in any prior atmospheric downscaling paper. Shows whether latent channels specialize in different variables or spatial scales.

```bash
sbatch scripts/run_analysis.sh --analysis latent
```

**Flags:**
- `--max_batches 30` — batches for statistics

**Output:** `results/latent_analysis/`
- `latent_distributions.png` — per-channel activation histograms
- `latent_spatial_sample*.png` — spatial maps of latent channels vs residual fields
- `latent_correlation_matrix.png` — inter-channel correlation heatmap
- `latent_uncertainty.png` — posterior std per channel

---

## 6. Climate Signal Preservation

**What:** Compares model predictions for early (1980-1989) vs late (2011-2020) periods to check whether the warming/moistening trend is preserved.

**Why:** Critical for climate applications. Sha et al. showed their model preserves SSP370 warming signals; this is our lightweight analog using observed trends within the training data.

```bash
sbatch scripts/run_analysis.sh --analysis climate
```

**Flags:**
- `--early_years 1980 1981 ... 1989` — early period
- `--late_years 2011 2012 ... 2020` — late period
- `--samples_per_year 30` — samples per year for averaging

**Output:** `results/climate_signal/`
- `climate_signal_<var>.png` — per-variable target vs DRN vs DRN+Diff change maps
- `climate_signal_summary.txt` — domain-averaged signal comparison table

---

## 7. Ablation: Pixel-Space CorrDiff

**What:** Trains the same diffusion UNet architecture directly on 256x256 pixel-space residuals (no VAE), then compares convergence speed and final quality.

**Why:** This is the single most important ablation — it isolates the benefit of latent-space compression. If latent CorrDiff converges faster with comparable quality, the method is validated.

```bash
sbatch scripts/run_analysis.sh --analysis ablation_pixel
```

**Flags:**
- `--epochs 50` — training epochs
- `--checkpoint_dir checkpoints/ablation_pixel`
- `--plot_dir train_plots/ablation_pixel`

**Output:**
- `checkpoints/ablation_pixel/pixel_diff_best.pt` — best checkpoint
- `train_plots/ablation_pixel/pixel_diff_loss.png` — loss curve
- `train_plots/ablation_pixel/pixel_diff_timings.npz` — per-epoch times

**Post-analysis:** After training, compare:
1. Final val loss: latent vs pixel
2. Convergence speed: epochs to reach same quality
3. Per-epoch wall time: latent should be faster due to smaller spatial dims
4. Per-step denoising time: from benchmark results

---

## 8. Ablation: VAE Compression Ratio

**What:** Trains VAEs with different spatial compression factors (LDM-2: 128x128, LDM-4: 64x64, LDM-8: 32x32) and compares reconstruction quality.

**Why:** Rombach et al. showed LDM-4 to LDM-8 is optimal for images. This tests whether the same holds for atmospheric fields.

```bash
# Train each ratio separately:
sbatch scripts/run_analysis.sh --analysis ablation_compression --ratio 2
sbatch scripts/run_analysis.sh --analysis ablation_compression --ratio 4
sbatch scripts/run_analysis.sh --analysis ablation_compression --ratio 8

# After all three complete, generate comparison plot:
python -m src.evaluation.ablation_compression --compare_only --plot_dir train_plots/ablation_compression
```

**Output:**
- `train_plots/ablation_compression/ablation_ldm{2,4,8}.npz` — per-ratio results
- `train_plots/ablation_compression/ablation_compression_comparison.png` — bar chart

---

## Running All Evaluations

To run ensemble, spectra, step ablation, latent analysis, and benchmark in one job:

```bash
sbatch scripts/run_analysis.sh --analysis all_eval
```

This runs all five post-training analyses sequentially in a single 12-hour A100 job. The ablation experiments (pixel, compression) require separate training runs and should be submitted individually.

---

## Results Directory Structure

After all analyses complete:

```
results/
  ensemble/
    rank_histograms.png
    qq_plots.png
    power_spectra.png
    summary.txt
    ensemble_results.npz
  spectra/
    stage_spectra.png
    stage_spectra_data.npz
  step_ablation/
    step_ablation.png
    step_ablation_data.npz
  benchmark/
    benchmark_summary.txt
    benchmark_results.npz
  latent_analysis/
    latent_distributions.png
    latent_spatial_sample*.png
    latent_correlation_matrix.png
    latent_uncertainty.png
  climate_signal/
    climate_signal_<var>.png
    climate_signal_summary.txt
```
