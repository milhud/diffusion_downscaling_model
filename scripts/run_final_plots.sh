#!/bin/bash
#SBATCH --job-name=final_plots
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=rome
#SBATCH --time=00:45:00
#SBATCH --output=fast_eval_output.%j
#SBATCH --error=fast_eval_error.%j

set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

CACHE_DIR="/discover/nobackup/sduan/.data"

echo "--- [1/2] 3-var inference plots (scan 150, 5-col, with axes) ---"
python -m src.evaluation.quick_inference_plots \
  --num_samples 6 \
  --num_members 4 \
  --num_steps 16 \
  --scan 150 \
  --vars T2 U10 TD2 \
  --output_dir results/inference_plots_3var \
  --cache_dir "$CACHE_DIR"

echo "Done: $(date)"
