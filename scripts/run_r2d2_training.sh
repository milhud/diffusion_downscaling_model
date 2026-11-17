#!/usr/bin/bash
#SBATCH -J train_r2d2
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --constraint=rome
#SBATCH --time=12:00:00
#SBATCH --qos=alla100
#SBATCH -o r2d2_train_output.%j
#SBATCH -e r2d2_train_error.%j
#SBATCH --account=s1001

# ──────────────────────────────────────────────────────────────────────
# Full training run for the R2-D2-style baseline (PyTorch reproduction —
# see docs/R2D2_BASELINE.md). Single stage, so no --stage flag needed.
#
# alla100 QOS caps at 12h; TRAIN["diff_epochs"] in config.py (300 epochs,
# same budget as our diffusion stage) will very likely need several
# --resume resubmissions, same as run_training.sh's diffusion stage. Use
# scripts/resume (adapted, or manually re-sbatch with --resume) to chain
# jobs automatically.
#
# NOT submitted automatically — review scripts/run_r2d2_smoke.sh's output
# first, then:
#   sbatch scripts/run_r2d2_training.sh
#   sbatch scripts/run_r2d2_training.sh --resume
# ──────────────────────────────────────────────────────────────────────

cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

mkdir -p sbatch_logs
for f in r2d2_train_output.* r2d2_train_error.*; do
    case "$f" in
        *."$SLURM_JOB_ID") ;;
        *) mv "$f" sbatch_logs/ 2>/dev/null ;;
    esac
done

module purge
module load python/GEOSpyD/24.3.0-0/3.12

export NCCL_DEBUG=WARN

torchrun --standalone --nproc_per_node=4 train_r2d2.py \
    --data_dir data --cache_dir cached_data \
    --checkpoint_dir checkpoints --plot_dir train_plots "$@"

exit 0
