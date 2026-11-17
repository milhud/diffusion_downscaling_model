#!/bin/bash
#SBATCH --job-name=wildfire_cases
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --account=s1001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=rome
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=event_benchmark_output/logs/wildfire_cases_%j.out
#SBATCH --error=event_benchmark_output/logs/wildfire_cases_%j.err
#
# Runs the full trained pipeline (DRN + VAE + diffusion) on historical
# wildfire-weather case studies and saves visual comparison plots to
# results/wildfire_cases/.
set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

module purge
module load python/GEOSpyD/24.3.0-0/3.12

export PYTHONPATH="/gpfsm/dnb33/hpmille1/diffusion_downscaling_model:${PYTHONPATH:-}"
python scripts/run_wildfire_cases.py
