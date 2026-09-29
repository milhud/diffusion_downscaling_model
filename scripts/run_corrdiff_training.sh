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
# Usage:
#   sbatch scripts/run_corrdiff_training.sh regression
#   sbatch scripts/run_corrdiff_training.sh diffusion <path/to/regression/checkpoint.mdlus>
set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

module purge
module load python/GEOSpyD/24.3.0-0/3.12
source external/corrdiff/venv/bin/activate

STAGE="${1:?Usage: sbatch run_corrdiff_training.sh {regression|diffusion} [reg_ckpt_path]}"
CORRDIFF=external/corrdiff/physicsnemo_src/examples/weather/corrdiff
CONF_DIR=/gpfsm/dnb33/hpmille1/diffusion_downscaling_model/external/corrdiff/conf
TRAIN_WRAPPER=/gpfsm/dnb33/hpmille1/diffusion_downscaling_model/external/corrdiff/train_wrapper.py

cd "$CORRDIFF"

if [ "$STAGE" = "regression" ]; then
    torchrun --standalone --nproc_per_node=4 "$TRAIN_WRAPPER" \
        --config-path="$CONF_DIR" --config-name=config_training_era5conus404_regression
elif [ "$STAGE" = "diffusion" ]; then
    REG_CKPT="${2:?Must pass the Stage-1 checkpoint path as the second argument}"
    torchrun --standalone --nproc_per_node=4 "$TRAIN_WRAPPER" \
        --config-path="$CONF_DIR" --config-name=config_training_era5conus404_diffusion \
        ++training.io.regression_checkpoint_path="$REG_CKPT"
else
    echo "Unknown stage: $STAGE (expected 'regression' or 'diffusion')" >&2
    exit 1
fi
