#!/usr/bin/bash
#SBATCH -J train_downscaling
#SBATCH --partition=columbia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=12:00:00
#SBATCH --qos=columbia
#SBATCH -o train_output.%j
#SBATCH -e train_error.%j
#SBATCH --account=columbia

# ──────────────────────────────────────────────────────────────────────
# Full end-to-end training: DRN → VAE → Diffusion
#
# Time estimate (temperature only, 1 A100):
#   Sanity check extensive (16k steps) took ~1.5h
#   Full training: ~400k total steps → ~40h estimated
#   columbia QOS allows up to 7d; 12h set here as default.
#
# Usage:
#   sbatch run_training.sh              # train all stages sequentially
#   sbatch run_training.sh --stage drn  # train DRN only
# ──────────────────────────────────────────────────────────────────────

cd /mnt/home/hmiller/diffusion_downscaling_model

# Archive old log files (exclude current job's files)
mkdir -p sbatch_logs
for f in train_output.* train_error.*; do
    case "$f" in
        *."$SLURM_JOB_ID") ;;  # skip current job
        *) mv "$f" sbatch_logs/ 2>/dev/null ;;
    esac
done

module purge
module load Python/3.10.15
source ~/venv/bin/activate

# Default: train all stages end-to-end.
# Override: sbatch run_training.sh --stage drn
python -u train.py --stage all --data_dir data --checkpoint_dir checkpoints \
    --plot_dir train_plots --device cuda "$@"

exit 0
