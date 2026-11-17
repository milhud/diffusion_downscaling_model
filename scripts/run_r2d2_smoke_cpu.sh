#!/usr/bin/bash
#SBATCH -J r2d2_smoke_cpu
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH -o r2d2_smoke_cpu_output.%j
#SBATCH -e r2d2_smoke_cpu_error.%j
#SBATCH --account=s1001

# ──────────────────────────────────────────────────────────────────────
# CPU variant of scripts/run_r2d2_smoke.sh -- runs while the GPU queue
# (corrdiff Stage 1 training) is occupied. train_r2d2.py falls back to
# CPU automatically when CUDA isn't available; same 20-step/1-epoch
# smoke config as the GPU version, just far slower per step.
#
# Usage:
#   sbatch scripts/run_r2d2_smoke_cpu.sh
# ──────────────────────────────────────────────────────────────────────

cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

mkdir -p sbatch_logs
for f in r2d2_smoke_cpu_output.* r2d2_smoke_cpu_error.*; do
    case "$f" in
        *."$SLURM_JOB_ID") ;;
        *) mv "$f" sbatch_logs/ 2>/dev/null ;;
    esac
done

module purge
module load python/GEOSpyD/24.3.0-0/3.12

CUDA_VISIBLE_DEVICES= python train_r2d2.py \
    --data_dir data --cache_dir /discover/nobackup/sduan/.data \
    --checkpoint_dir checkpoints_test --plot_dir train_plots_test \
    --max_steps 20 --epochs 1 --batch_size 2 "$@"

exit 0
