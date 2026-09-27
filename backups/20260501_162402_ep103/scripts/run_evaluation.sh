#!/bin/bash
#SBATCH --job-name=eval_downscale
#SBATCH --output=eval_output.%j
#SBATCH --error=eval_error.%j
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --qos=alla100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

module load python
source activate diffusion_env 2>/dev/null || true

cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

echo "=== Evaluation: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

python -u -m src.evaluation.test_evaluation "$@"

echo "=== Done: $(date) ==="
