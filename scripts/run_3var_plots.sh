#!/bin/bash
#SBATCH --job-name=3var_plots
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=rome
#SBATCH --time=00:30:00
#SBATCH --output=fast_eval_output.%j
#SBATCH --error=fast_eval_error.%j

set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

echo "Scanning 40 patches, keeping 6 lowest-RMSE, plotting T2 + U10 + PREC..."
python -m src.evaluation.quick_inference_plots \
  --num_samples 6 \
  --num_members 4 \
  --num_steps 16 \
  --scan 40 \
  --vars T2 U10 PREC_ACC_NC \
  --output_dir results/inference_plots_3var \
  --cache_dir "/discover/nobackup/sduan/.data"

echo "Done: $(date)"
