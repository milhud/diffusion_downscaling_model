#!/bin/bash
#SBATCH --job-name=baselines_eval
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --account=s1001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=rome
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=event_benchmark_output/logs/baselines_eval_%j.out
#SBATCH --error=event_benchmark_output/logs/baselines_eval_%j.err
set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model
module purge
module load python/GEOSpyD/24.3.0-0/3.12
python -u -m src.evaluation.time_models
python -u -m src.evaluation.event_predict_baselines --variant corrdiff
python -u -m src.evaluation.event_predict_baselines --variant r2d2
