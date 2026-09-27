#!/bin/bash
#SBATCH --job-name=mrpd_events
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --account=s1001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=rome
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=event_benchmark_output/logs/mrpd_%j.out
#SBATCH --error=event_benchmark_output/logs/mrpd_%j.err
set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model
module purge
module load python/GEOSpyD/24.3.0-0/3.12
python -u "${1:-scripts/mrpd_event_baseline.py}"
