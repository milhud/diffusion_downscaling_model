#!/bin/bash
#SBATCH --job-name=corrdiff_train
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --account=s1001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=rome
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=event_benchmark_output/logs/corrdiff_train_%j.out
#SBATCH --error=event_benchmark_output/logs/corrdiff_train_%j.err
#
# NOT auto-submitted -- review external/corrdiff/README.md (esp. the smoke
# test results and patch-size rationale) before running.
#
# Full CorrDiff baseline training: Stage 1 (regression) then Stage 2
# (diffusion), each 5M samples (physicsnemo README's suggested 1M-30M range
# for a first full run). alla100 QOS caps walltime at 12h, same constraint
# our own run_training.sh hits -- this will very likely need multiple
# resubmissions with the same command (training.io.checkpoint_dir persists
# and train.py auto-resumes from the latest checkpoint there).
#
# batch_size_per_gpu is pinned to 2 below (total_batch_size stays 64, so
# physicsnemo's compute_num_accumulation_rounds() makes up the difference
# via gradient accumulation -- same effective batch size, ~8x less peak
# memory per step). This node's A100s are 40GB, not the 80GB the original
# crop_size=448 sizing in external/corrdiff/README.md assumed:
# batch_size_per_gpu="auto" resolves to total_batch_size/world_size=16 and
# OOMs at crop_size=448 (job 58647358 died with "CUDA out of memory. Tried
# to allocate 1.53 GiB. GPU 2 has a total capacity of 39.39 GiB").
#
# Usage:
#   sbatch scripts/run_corrdiff_training.sh regression
#   sbatch scripts/run_corrdiff_training.sh diffusion <path/to/regression/checkpoint.mdlus>
set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

module purge
module load python/GEOSpyD/24.3.0-0/3.12
source external/corrdiff/venv/bin/activate

export PYTORCH_ALLOC_CONF=expandable_segments:True

STAGE="${1:?Usage: sbatch run_corrdiff_training.sh regression-or-diffusion optional-reg-ckpt-path}"
CORRDIFF=external/corrdiff/physicsnemo_src/examples/weather/corrdiff
CONF_DIR=/gpfsm/dnb33/hpmille1/diffusion_downscaling_model/external/corrdiff/conf
TRAIN_WRAPPER=/gpfsm/dnb33/hpmille1/diffusion_downscaling_model/external/corrdiff/train_wrapper.py

cd "$CORRDIFF"

if [ "$STAGE" = "regression" ]; then
    python -m torch.distributed.run --standalone --nproc_per_node=4 "$TRAIN_WRAPPER" \
        --config-path="$CONF_DIR" --config-name=config_training_era5conus404_regression \
        ++training.hp.batch_size_per_gpu=2
elif [ "$STAGE" = "diffusion" ]; then
    REG_CKPT="${2:?Must pass the Stage-1 checkpoint path as the second argument}"
    python -m torch.distributed.run --standalone --nproc_per_node=4 "$TRAIN_WRAPPER" \
        --config-path="$CONF_DIR" --config-name=config_training_era5conus404_diffusion \
        ++training.io.regression_checkpoint_path="$REG_CKPT" \
        ++training.hp.batch_size_per_gpu=2
else
    echo "Unknown stage: $STAGE (expected 'regression' or 'diffusion')" >&2
    exit 1
fi
