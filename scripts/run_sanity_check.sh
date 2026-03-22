#!/bin/bash
#SBATCH --job-name=sanity_check
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=rome
#SBATCH --time=03:00:00
#SBATCH --output=sanity_check_output.%j
#SBATCH --error=sanity_check_error.%j

# ============================================================================
# Comprehensive sanity check: trains DRN, VAE, and Diffusion for 1000 steps
# each, then runs ALL evaluation analyses (15 tests total).
#
# Usage:
#   sbatch scripts/run_sanity_check.sh
#
# Expected runtime: ~30-60 minutes on A100
# Output: sanity_plots/ directory with 16 diagnostic PNG files
# ============================================================================

set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

echo "============================================"
echo "Comprehensive Sanity Check"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "============================================"

python sanity_check.py --device cuda --plot_dir sanity_plots

echo ""
echo "============================================"
echo "Sanity check completed at $(date)"
echo "Plots saved to: sanity_plots/"
echo "============================================"
