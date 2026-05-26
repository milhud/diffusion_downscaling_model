#!/bin/bash
#SBATCH --job-name=fast_eval
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

echo "============================================"
echo "Fast eval suite (target: <=30 min)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "============================================"

echo ""
echo "--- [1/5] Latent space analysis (~2 min) ---"
python -m src.evaluation.latent_analysis \
  --output_dir results/latent_analysis \
  --max_batches 10 \
  --cache_dir "$CACHE_DIR"

echo ""
echo "--- [2/5] Compute benchmark (~3 min) ---"
python -m src.evaluation.compute_benchmark \
  --output_dir results/benchmark \
  --num_steps 4 8 16 32 \
  --cache_dir "$CACHE_DIR"

echo ""
echo "--- [3/5] Power spectra (~5 min) ---"
python -m src.evaluation.stage_spectra \
  --output_dir results/spectra \
  --num_steps 16 \
  --num_ensemble 2 \
  --max_batches 10 \
  --cache_dir "$CACHE_DIR"

echo ""
echo "--- [4/5] Step count ablation (~5 min) ---"
python -m src.evaluation.step_ablation \
  --output_dir results/step_ablation \
  --steps 2 4 8 16 32 \
  --num_ensemble 2 \
  --max_batches 5 \
  --cache_dir "$CACHE_DIR"

echo ""
echo "--- [5/5] Quick inference plots (~10 min) ---"
python -m src.evaluation.quick_inference_plots \
  --num_samples 6 \
  --num_members 4 \
  --num_steps 16 \
  --output_dir results/inference_plots \
  --cache_dir "$CACHE_DIR"

echo ""
echo "============================================"
echo "Fast eval complete at $(date)"
echo "Results in: results/{latent_analysis,benchmark,spectra,step_ablation,inference_plots}"
echo "============================================"
