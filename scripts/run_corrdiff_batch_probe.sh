#!/bin/bash
#SBATCH --job-name=corrdiff_probe
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --account=s1001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=rome
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --output=event_benchmark_output/logs/corrdiff_probe_%j.out
#SBATCH --error=event_benchmark_output/logs/corrdiff_probe_%j.err
#
# One-off: probes a larger batch_size_per_gpu on a single spare GPU (not the
# node running production training) to see if it fits in memory and how much
# it speeds up throughput, before changing the production config. Uses a
# throwaway checkpoint dir so it can't corrupt the real training's checkpoints.
#
# Usage:
#   sbatch scripts/run_corrdiff_batch_probe.sh <batch_size_per_gpu>
set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

module purge
module load python/GEOSpyD/24.3.0-0/3.12
source external/corrdiff/venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True

BS="${1:?Usage: sbatch run_corrdiff_batch_probe.sh batch_size_per_gpu}"
CORRDIFF=external/corrdiff/physicsnemo_src/examples/weather/corrdiff
CONF_DIR=/gpfsm/dnb33/hpmille1/diffusion_downscaling_model/external/corrdiff/conf
TRAIN_WRAPPER=/gpfsm/dnb33/hpmille1/diffusion_downscaling_model/external/corrdiff/train_wrapper.py
PROBE_CKPT_DIR=/gpfsm/dnb33/hpmille1/diffusion_downscaling_model/external/corrdiff/checkpoints_probe_bs${BS}

rm -rf "$PROBE_CKPT_DIR"
mkdir -p "$PROBE_CKPT_DIR"

cd "$CORRDIFF"
srun --ntasks=1 python "$TRAIN_WRAPPER" \
    --config-path="$CONF_DIR" --config-name=config_training_era5conus404_regression \
    ++training.hp.total_batch_size="$BS" \
    ++training.hp.batch_size_per_gpu="$BS" \
    ++training.hp.training_duration=512 \
    ++training.hp.lr_rampup=0 \
    ++training.io.checkpoint_dir="$PROBE_CKPT_DIR" \
    ++training.io.print_progress_freq=1 \
    ++training.io.save_checkpoint_freq=100000000 \
    ++training.io.validation_freq=1000000000

echo "=== Probe complete (batch_size_per_gpu=$BS) ==="
