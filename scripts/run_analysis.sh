#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --partition=columbia
#SBATCH --qos=columbia
#SBATCH --account=columbia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=12:00:00
#SBATCH --output=analysis_output.%j
#SBATCH --error=analysis_error.%j

# ============================================================================
# Unified analysis runner. Pass --analysis flag to select which analysis to run.
#
# Usage:
#   sbatch scripts/run_analysis.sh --analysis ensemble
#   sbatch scripts/run_analysis.sh --analysis spectra
#   sbatch scripts/run_analysis.sh --analysis step_ablation
#   sbatch scripts/run_analysis.sh --analysis benchmark
#   sbatch scripts/run_analysis.sh --analysis latent
#   sbatch scripts/run_analysis.sh --analysis climate
#   sbatch scripts/run_analysis.sh --analysis ablation_pixel
#   sbatch scripts/run_analysis.sh --analysis ablation_compression --ratio 4
#   sbatch scripts/run_analysis.sh --analysis all_eval  (ensemble + spectra + step)
# ============================================================================

set -euo pipefail
cd /mnt/home/hmiller/diffusion_downscaling_model

# Parse arguments (passed after sbatch via -- or as SLURM comment)
ANALYSIS="${1:---analysis}"
shift || true
ANALYSIS="${1:-ensemble}"
shift || true

# Remaining args passed to the analysis script
EXTRA_ARGS="$@"

echo "============================================"
echo "Analysis: $ANALYSIS"
echo "Extra args: $EXTRA_ARGS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "============================================"

case "$ANALYSIS" in

  ensemble)
    echo "Running 32-member ensemble evaluation..."
    python -m src.evaluation.ensemble_eval \
      --num_members 32 \
      --num_steps 32 \
      --output_dir results/ensemble \
      $EXTRA_ARGS
    ;;

  spectra)
    echo "Running stage-by-stage power spectra analysis..."
    python -m src.evaluation.stage_spectra \
      --output_dir results/spectra \
      --num_steps 32 \
      --num_ensemble 4 \
      --max_batches 50 \
      $EXTRA_ARGS
    ;;

  step_ablation)
    echo "Running denoising step count ablation..."
    python -m src.evaluation.step_ablation \
      --output_dir results/step_ablation \
      --steps 2 4 8 16 32 64 \
      --num_ensemble 8 \
      --max_batches 20 \
      $EXTRA_ARGS
    ;;

  benchmark)
    echo "Running computational benchmark..."
    python -m src.evaluation.compute_benchmark \
      --output_dir results/benchmark \
      --num_steps 4 8 16 32 \
      $EXTRA_ARGS
    ;;

  latent)
    echo "Running latent space analysis..."
    python -m src.evaluation.latent_analysis \
      --output_dir results/latent_analysis \
      --max_batches 30 \
      $EXTRA_ARGS
    ;;

  climate)
    echo "Running climate signal preservation test..."
    python -m src.evaluation.climate_signal \
      --output_dir results/climate_signal \
      --early_years 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 \
      --late_years 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 \
      --samples_per_year 30 \
      $EXTRA_ARGS
    ;;

  ablation_pixel)
    echo "Running pixel-space CorrDiff ablation..."
    python -m src.evaluation.ablation_pixel \
      --checkpoint_dir checkpoints/ablation_pixel \
      --plot_dir train_plots/ablation_pixel \
      --epochs 50 \
      $EXTRA_ARGS
    ;;

  ablation_compression)
    echo "Running VAE compression ratio ablation..."
    python -m src.evaluation.ablation_compression \
      --checkpoint_dir checkpoints/ablation_compression \
      --plot_dir train_plots/ablation_compression \
      $EXTRA_ARGS
    ;;

  all_eval)
    echo "Running all evaluation analyses (ensemble + spectra + step ablation)..."
    python -m src.evaluation.ensemble_eval --num_members 32 --num_steps 32 --output_dir results/ensemble $EXTRA_ARGS
    python -m src.evaluation.stage_spectra --output_dir results/spectra --max_batches 50 $EXTRA_ARGS
    python -m src.evaluation.step_ablation --output_dir results/step_ablation --max_batches 20 $EXTRA_ARGS
    python -m src.evaluation.latent_analysis --output_dir results/latent_analysis $EXTRA_ARGS
    python -m src.evaluation.compute_benchmark --output_dir results/benchmark $EXTRA_ARGS
    ;;

  *)
    echo "Unknown analysis: $ANALYSIS"
    echo "Available: ensemble, spectra, step_ablation, benchmark, latent, climate,"
    echo "           ablation_pixel, ablation_compression, all_eval"
    exit 1
    ;;
esac

echo ""
echo "============================================"
echo "Analysis '$ANALYSIS' completed at $(date)"
echo "============================================"
