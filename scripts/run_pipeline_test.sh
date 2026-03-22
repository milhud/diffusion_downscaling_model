#!/bin/bash
#SBATCH --job-name=pipeline_test
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=rome
#SBATCH --time=03:00:00
#SBATCH --output=pipeline_test_output.%j
#SBATCH --error=pipeline_test_error.%j

# Quick 2-epoch pipeline test: DRN → VAE → Diffusion
# Uses separate checkpoint/plot dirs to avoid overwriting real training

set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

echo "============================================"
echo "Pipeline Test (2 epochs per stage)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "============================================"

python -u train.py --stage all --data_dir data --cache_dir cached_data \
    --checkpoint_dir checkpoints_test --plot_dir train_plots_test --device cuda

echo ""
echo "============================================"
echo "Pipeline test completed at $(date)"
echo "============================================"
