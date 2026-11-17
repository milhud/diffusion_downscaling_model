#!/bin/bash
# ============================================================================
# Auto-resubmitting loop for CorrDiff baseline training (external/corrdiff/).
#
# physicsnemo's train.py auto-resumes from training.io.checkpoint_dir, but
# does not resubmit itself when SLURM kills it at the 12h wall-clock limit.
# This polls squeue for the job and resubmits (same command, which resumes
# from the latest checkpoint) whenever it's no longer running, until the
# target sample count is reached or --max-resubmits is hit.
#
# Does NOT kill a healthy running job -- only reacts to it having already
# ended (TIMEOUT/COMPLETED/FAILED), since checkpoints are saved every
# 100k samples (~90min at current throughput), not continuously.
#
# Usage:
#   nohup bash scripts/run_corrdiff_until_done.sh regression \
#       > corrdiff_regression_loop.log 2>&1 &
#   nohup bash scripts/run_corrdiff_until_done.sh diffusion <reg_ckpt.mdlus> \
#       > corrdiff_diffusion_loop.log 2>&1 &
#
# Options (env vars):
#   POLL_INTERVAL   seconds between squeue checks (default: 600)
#   MAX_RESUBMITS   max resubmissions before giving up (default: 20)
#   TARGET_SAMPLES  stop once this sample count appears in the log (default: 5000000)
#   ATTACH_JOB_ID   watch this already-running job instead of submitting a
#                   new one for the first iteration (use when a job you
#                   already sbatch'd manually is still healthy)
# ============================================================================
set -uo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

STAGE="${1:?Usage: run_corrdiff_until_done.sh regression-or-diffusion [reg_ckpt_path]}"
REG_CKPT="${2:-}"
POLL_INTERVAL="${POLL_INTERVAL:-600}"
MAX_RESUBMITS="${MAX_RESUBMITS:-20}"
TARGET_SAMPLES="${TARGET_SAMPLES:-5000000}"
ATTACH_JOB_ID="${ATTACH_JOB_ID:-}"

JOB_NAME="corrdiff_train"
LOG_DIR="event_benchmark_output/logs"

submit() {
    if [ "$STAGE" = "diffusion" ]; then
        sbatch scripts/run_corrdiff_training.sh diffusion "$REG_CKPT" | awk '{print $NF}'
    else
        sbatch scripts/run_corrdiff_training.sh regression | awk '{print $NF}'
    fi
}

is_running() {
    local jobid="$1"
    squeue -j "$jobid" -h -o "%T" 2>/dev/null | grep -qE "RUNNING|PENDING|CONFIGURING"
}

training_done() {
    local jobid="$1"
    local out="$LOG_DIR/${JOB_NAME}_${jobid}.out"
    [ -f "$out" ] && grep -qE "Training Completed|samples ${TARGET_SAMPLES}\.0" "$out" 2>/dev/null
}

echo "[$(date)] Starting CorrDiff auto-resubmit loop (stage=$STAGE, poll=${POLL_INTERVAL}s, max_resubmits=${MAX_RESUBMITS})"

if [ -n "$ATTACH_JOB_ID" ]; then
    CURRENT_JOB="$ATTACH_JOB_ID"
    echo "[$(date)] Attaching to already-running job $CURRENT_JOB"
else
    CURRENT_JOB=$(submit)
    echo "[$(date)] Submitted job $CURRENT_JOB"
fi
RESUBMITS=0

while true; do
    sleep "$POLL_INTERVAL"

    if training_done "$CURRENT_JOB"; then
        echo "[$(date)] Target sample count reached in job $CURRENT_JOB's log -- done."
        exit 0
    fi

    if is_running "$CURRENT_JOB"; then
        continue
    fi

    STATE=$(sacct -j "$CURRENT_JOB" --format=State -X -n 2>/dev/null | tr -d ' ')
    echo "[$(date)] Job $CURRENT_JOB ended (state: $STATE)."

    if training_done "$CURRENT_JOB"; then
        echo "[$(date)] Target sample count reached -- done."
        exit 0
    fi

    if [ "$RESUBMITS" -ge "$MAX_RESUBMITS" ]; then
        echo "[$(date)] Hit --max-resubmits ($MAX_RESUBMITS) without reaching target -- giving up. Check logs."
        exit 1
    fi

    RESUBMITS=$((RESUBMITS + 1))
    CURRENT_JOB=$(submit)
    echo "[$(date)] Resubmitted (attempt $RESUBMITS/$MAX_RESUBMITS): job $CURRENT_JOB"
done
