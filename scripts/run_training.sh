#!/usr/bin/bash
#SBATCH -J train_do
#SBATCH --partition=columbia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --time=12:00:00
#SBATCH --qos=columbia
#SBATCH -o train_output.%j
#SBATCH -e train_error.%j
#SBATCH --account=columbia

# ──────────────────────────────────────────────────────────────────────
# Multi-GPU training: DRN → VAE → Diffusion
#
# Allocation: 1 node × 4 H100s = 4 GPUs total.
# torchrun spawns SLURM_NTASKS_PER_NODE processes per node (one per GPU).
# World size and GPU counts are inferred automatically from SLURM env vars.
# No auto-resubmit — single 12h run.
#
# Usage (via sbatch):
#   sbatch scripts/run_training.sh --stage drn
#   sbatch scripts/run_training.sh --stage drn --resume
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

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

echo "Node: $(hostname)  GPUs/node: $SLURM_NTASKS_PER_NODE"

torchrun \
    --nproc_per_node="$SLURM_NTASKS_PER_NODE" \
    train.py \
    --data_dir data \
    --checkpoint_dir checkpoints \
    --plot_dir train_plots \
    --cache_dir /mnt/home/hmiller/cached_data \
    "$@"

exit 0
