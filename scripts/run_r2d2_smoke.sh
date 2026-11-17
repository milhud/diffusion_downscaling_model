#!/usr/bin/bash
#SBATCH -J r2d2_smoke
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --constraint=rome
#SBATCH --time=00:30:00
#SBATCH -o r2d2_smoke_output.%j
#SBATCH -e r2d2_smoke_error.%j
#SBATCH --account=s1001

# ──────────────────────────────────────────────────────────────────────
# Smoke test for the R2-D2-style baseline (single stage, single GPU).
# Runs a handful of optimizer steps to verify data loading, forward/backward,
# and checkpointing all work before committing to a full multi-day run.
#
# Usage:
#   sbatch scripts/run_r2d2_smoke.sh
# ──────────────────────────────────────────────────────────────────────

cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

mkdir -p sbatch_logs
for f in r2d2_smoke_output.* r2d2_smoke_error.*; do
    case "$f" in
        *."$SLURM_JOB_ID") ;;
        *) mv "$f" sbatch_logs/ 2>/dev/null ;;
    esac
done

module purge
module load python/GEOSpyD/24.3.0-0/3.12

python train_r2d2.py \
    --data_dir data --cache_dir /discover/nobackup/sduan/.data \
    --checkpoint_dir checkpoints_test --plot_dir train_plots_test \
    --max_steps 20 --epochs 1 --batch_size 2 "$@"

exit 0
