#!/usr/bin/bash
#SBATCH -J train_do
#SBATCH --partition=gpu_a100
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --constraint=rome
#SBATCH --time=12:00:00
#SBATCH --qos=alla100
#SBATCH -o train_output.%j
#SBATCH -e train_error.%j
#SBATCH --account=s1001

# ──────────────────────────────────────────────────────────────────────
# Multi-GPU training: DRN → VAE → Diffusion
#
# Allocation: 3 nodes × 4 A100s = 12 GPUs total.
# torchrun spawns SLURM_NTASKS_PER_NODE processes per node (one per GPU).
# World size and GPU counts are inferred automatically from SLURM env vars.
#
# Usage (via sbatch):
#   sbatch scripts/run_training.sh --stage drn
#   sbatch scripts/run_training.sh --stage drn --resume
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

# Multi-node NCCL rendezvous
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

echo "Master: $MASTER_ADDR  Nodes: $SLURM_NNODES  GPUs/node: $SLURM_NTASKS_PER_NODE"

torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_NTASKS_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="$SLURM_JOB_ID" \
    train.py \
    --data_dir data \
    --checkpoint_dir checkpoints \
    --plot_dir train_plots \
    --cache_dir /discover/nobackup/sduan/.data \
    "$@"

exit 0
