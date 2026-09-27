#!/bin/bash
#SBATCH --job-name=event_predict
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --account=s1001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=rome
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=event_benchmark_output/logs/event_predict_%j.out
#SBATCH --error=event_benchmark_output/logs/event_predict_%j.err

set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model
module purge
module load python/GEOSpyD/24.3.0-0/3.12
echo "Start $(date) on $(hostname)"
python -u -m src.evaluation.event_predict --members 8 --steps 16 "$@"
echo "Done $(date)"
