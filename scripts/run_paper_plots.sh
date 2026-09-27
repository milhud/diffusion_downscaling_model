#!/bin/bash
#SBATCH --job-name=paper_plots
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=rome
#SBATCH --time=01:00:00
#SBATCH --output=paper_plots_output.%j
#SBATCH --error=paper_plots_error.%j

set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

CACHE_DIR="/discover/nobackup/sduan/.data"
EXT_REPO="/gpfsm/dnb33/hpmille1/diffusion_paper_external"

echo "--- [1/2] Inference plots (scan 150, Target|ERA5|DRN|Ens|Error) ---"
python -m src.evaluation.quick_inference_plots \
  --num_samples 6 \
  --num_members 4 \
  --num_steps 16 \
  --scan 150 \
  --vars T2 U10 TD2 \
  --output_dir "${EXT_REPO}/results/inference_plots_3var" \
  --cache_dir "$CACHE_DIR"

echo "--- [2/2] Power spectra (ERA5 Interp | DRN | Ensemble mean | Target) ---"
python -m src.evaluation.stage_spectra \
  --output_dir "${EXT_REPO}/results/spectra" \
  --num_steps 16 \
  --num_ensemble 4 \
  --max_batches 50 \
  --cache_dir "$CACHE_DIR"

echo "Done: $(date)"
