#!/usr/bin/bash
#SBATCH -J train_downscaling
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=rome
#SBATCH --time=12:00:00
#SBATCH --qos=alla100
#SBATCH -o train_output.%j
#SBATCH -e train_error.%j
#SBATCH --account=s1001

# ──────────────────────────────────────────────────────────────────────
# Full end-to-end training: DRN → VAE → Diffusion
#
# Time estimate (temperature only, 1 A100):
#   Sanity check extensive (16k steps) took ~1.5h
#   Full training: ~400k total steps → ~40h estimated
#   alla100 QOS caps at 12h; use --stage to split across jobs.
#
# Usage:
#   sbatch run_training.sh              # train all stages sequentially
#   sbatch run_training.sh --stage drn  # train DRN only
#   sbatch run_training.sh --resume     # resume from latest checkpoint
#   sbatch run_training.sh --stage diffusion --resume  # resume diffusion only
#   sbatch run_training.sh --no-cache                  # regrid on-the-fly (no disk cache needed)
# ──────────────────────────────────────────────────────────────────────

cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

# Archive old log files (exclude current job's files)
mkdir -p sbatch_logs
for f in train_output.* train_error.*; do
    case "$f" in
        *."$SLURM_JOB_ID") ;;  # skip current job
        *) mv "$f" sbatch_logs/ 2>/dev/null ;;
    esac
done

module purge
module load python/GEOSpyD/24.3.0-0/3.12

# Default: train all stages end-to-end.
# Override: sbatch run_training.sh --stage drn
python -u train.py --stage all --data_dir data --checkpoint_dir checkpoints \
    --plot_dir train_plots --device cuda "$@"

exit 0
