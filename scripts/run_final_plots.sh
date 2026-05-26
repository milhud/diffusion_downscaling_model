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

echo "--- [1/2] 3-var inference plots (scan 80, 5-col, with axes) ---"
python -m src.evaluation.quick_inference_plots \
  --num_samples 6 \
  --num_members 4 \
  --num_steps 16 \
  --scan 80 \
  --vars T2 U10 PREC_ACC_NC \
  --output_dir results/inference_plots_3var \
  --cache_dir "$CACHE_DIR"

echo ""
echo "--- [2/2] Power spectra (2x3 layout) ---"
python -m src.evaluation.stage_spectra \
  --output_dir results/spectra \
  --num_steps 16 \
  --num_ensemble 2 \
  --max_batches 10 \
  --cache_dir "$CACHE_DIR"

echo "Done: $(date)"
