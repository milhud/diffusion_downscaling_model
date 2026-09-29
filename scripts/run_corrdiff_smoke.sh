#!/bin/bash
#SBATCH --job-name=corrdiff_smoke
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --account=s1001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=rome
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=event_benchmark_output/logs/corrdiff_smoke_%j.out
#SBATCH --error=event_benchmark_output/logs/corrdiff_smoke_%j.err
#
# Smoke test: runs a handful of Stage-1 (regression) steps, then a handful of
# Stage-2 (diffusion) steps using the Stage-1 checkpoint it just produced.
# Verifies the dataset adapter loads real data, both nets forward/backward
# cleanly, and losses are finite -- before committing to the ~days-long full
# run (scripts/run_corrdiff_training.sh).
set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

module purge
module load python/GEOSpyD/24.3.0-0/3.12
source external/corrdiff/venv/bin/activate

export PYTORCH_ALLOC_CONF=expandable_segments:True

CORRDIFF=external/corrdiff/physicsnemo_src/examples/weather/corrdiff
CONF_DIR=/gpfsm/dnb33/hpmille1/diffusion_downscaling_model/external/corrdiff/conf
TRAIN_WRAPPER=/gpfsm/dnb33/hpmille1/diffusion_downscaling_model/external/corrdiff/train_wrapper.py

echo "=== Stage 1: regression smoke (tiny training_duration) ==="
cd "$CORRDIFF"
srun --ntasks=1 python "$TRAIN_WRAPPER" \
    --config-path="$CONF_DIR" --config-name=config_training_era5conus404_regression \
    ++training.hp.training_duration=64 \
    ++training.hp.total_batch_size=1 \
    ++training.hp.batch_size_per_gpu=1 \
    ++training.hp.lr_rampup=0 \
    ++training.io.print_progress_freq=1 \
    ++training.io.save_checkpoint_freq=64 \
    ++training.io.validation_freq=1000000 \
    ++training.perf.dataloader_workers=2

echo "=== locating regression checkpoint ==="
REG_CKPT=$(find /gpfsm/dnb33/hpmille1/diffusion_downscaling_model/external/corrdiff/checkpoints_regression -name '*.mdlus' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
echo "Using regression checkpoint: $REG_CKPT"

echo "=== Stage 2: diffusion smoke (tiny training_duration) ==="
srun --ntasks=1 python "$TRAIN_WRAPPER" \
    --config-path="$CONF_DIR" --config-name=config_training_era5conus404_diffusion \
    ++training.hp.training_duration=64 \
    ++training.hp.total_batch_size=1 \
    ++training.hp.batch_size_per_gpu=1 \
    ++training.hp.lr_rampup=0 \
    ++training.io.print_progress_freq=1 \
    ++training.io.save_checkpoint_freq=64 \
    ++training.io.validation_freq=1000000 \
    ++training.io.regression_checkpoint_path="$REG_CKPT" \
    ++training.perf.dataloader_workers=2

echo "=== Smoke test complete ==="
