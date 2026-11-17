#!/bin/bash
#SBATCH --job-name=corrdiff_gen
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --account=s1001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=rome
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=event_benchmark_output/logs/corrdiff_gen_%j.out
#SBATCH --error=event_benchmark_output/logs/corrdiff_gen_%j.err
#
# Runs NVIDIA CorrDiff inference (both stages, now that training finished)
# on a handful of held-out test-year samples, for accuracy benchmarking
# against our own model. Output: external/corrdiff/output/era5conus404_generate/corrdiff_output.nc
set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

module purge
module load python/GEOSpyD/24.3.0-0/3.12
source external/corrdiff/venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True

CORRDIFF=external/corrdiff/physicsnemo_src/examples/weather/corrdiff
CONF_DIR=/gpfsm/dnb33/hpmille1/diffusion_downscaling_model/external/corrdiff/conf

cd "$CORRDIFF"
srun --ntasks=1 python generate.py \
    --config-path="$CONF_DIR" --config-name=config_generate_era5conus404
