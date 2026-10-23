#!/bin/bash
#SBATCH --job-name=time_infer
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --account=s1001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=rome
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH --output=event_benchmark_output/logs/time_infer_%j.out
#SBATCH --error=event_benchmark_output/logs/time_infer_%j.err
set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

module purge
module load python/GEOSpyD/24.3.0-0/3.12

export PYTHONPATH="/gpfsm/dnb33/hpmille1/diffusion_downscaling_model:${PYTHONPATH:-}"
python scripts/time_inference_example.py
