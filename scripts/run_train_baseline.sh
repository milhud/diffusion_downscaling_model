#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --account=s1001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=rome
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=09:30:00
#SBATCH --output=event_benchmark_output/logs/train_%x_%j.out
#SBATCH --error=event_benchmark_output/logs/train_%x_%j.err
# usage: sbatch --job-name=corrdiff scripts/run_train_baseline.sh corrdiff 8
set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model
module purge
module load python/GEOSpyD/24.3.0-0/3.12
python -u -m src.evaluation.train_baselines --variant "$1" --max_hours "${2:-8}" --workers 10 --total_steps "${3:-38000}"
