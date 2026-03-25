#!/bin/bash
# ============================================================================
# Auto-resubmitting training loop.
#
# Submits training jobs with --resume, waits for completion, checks if all
# epochs finished, and resubmits if not. Runs until training completes or
# max resubmissions reached.
#
# Usage:
#   # Run in foreground:
#   bash scripts/train_until_done.sh --stage diffusion
#
#   # Run in background (survives terminal exit):
#   nohup bash scripts/train_until_done.sh --stage diffusion \
#       > train_loop.log 2>&1 &
#
#   # Or use the companion daemon script:
#   bash scripts/start_training_daemon.sh --stage diffusion
#
# Options:
#   --stage <drn|vae|diffusion|all>  Training stage (required)
#   --max-resubmits <N>              Max resubmissions (default: 20)
#   --poll-interval <secs>           Seconds between job status checks (default: 120)
#   --checkpoint-dir <dir>           Checkpoint directory (default: checkpoints)
# ============================================================================

set -euo pipefail
cd /mnt/home/hmiller/diffusion_downscaling_model

# ── Parse arguments ──
STAGE=""
MAX_RESUBMITS=20
POLL_INTERVAL=120
CHECKPOINT_DIR="checkpoints"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --max-resubmits) MAX_RESUBMITS="$2"; shift 2 ;;
        --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [[ -z "$STAGE" ]]; then
    echo "ERROR: --stage is required (drn|vae|diffusion|all)"
    exit 1
fi

# ── Target epoch counts from config.py ──
get_target_epochs() {
    python3 -c "
from config import TRAIN
stage_map = {'drn': 'drn_epochs', 'vae': 'vae_epochs', 'diffusion': 'diff_epochs'}
if '$1' == 'all':
    # For 'all', we only care about the last stage (diffusion)
    print(TRAIN['diff_epochs'])
else:
    print(TRAIN[stage_map['$1']])
"
}

# ── Check if training is complete ──
is_training_complete() {
    local stage="$1"
    local ckpt_dir="$2"

    python3 -c "
import torch, sys
from config import TRAIN

stage_map = {'drn': ('drn_latest.pt', 'drn_epochs'),
             'vae': ('vae_latest.pt', 'vae_epochs'),
             'diffusion': ('diffusion_latest.pt', 'diff_epochs')}

if '$stage' == 'all':
    # Check the last stage that would run
    check_stages = ['drn', 'vae', 'diffusion']
else:
    check_stages = ['$stage']

for s in check_stages:
    ckpt_file, epoch_key = stage_map[s]
    target = TRAIN[epoch_key]
    ckpt_path = f'$ckpt_dir/{ckpt_file}'
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        completed = ckpt['epoch'] + 1  # epoch is 0-indexed
        if completed < target:
            print(f'{s}: {completed}/{target} epochs', file=sys.stderr)
            sys.exit(1)
        print(f'{s}: {completed}/{target} epochs (DONE)', file=sys.stderr)
    except FileNotFoundError:
        print(f'{s}: no checkpoint found', file=sys.stderr)
        sys.exit(1)
sys.exit(0)
" 2>&1
}

# ── Main loop ──
TARGET_EPOCHS=$(get_target_epochs "$STAGE")
echo "============================================"
echo "Auto-resubmitting training loop"
echo "  Stage: $STAGE"
echo "  Target epochs: $TARGET_EPOCHS"
echo "  Max resubmits: $MAX_RESUBMITS"
echo "  Poll interval: ${POLL_INTERVAL}s"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Extra args: $EXTRA_ARGS"
echo "  Started: $(date)"
echo "  PID: $$"
echo "============================================"

for attempt in $(seq 1 $MAX_RESUBMITS); do
    echo ""
    echo "[Attempt $attempt/$MAX_RESUBMITS] $(date)"

    # Check if already complete
    if is_training_complete "$STAGE" "$CHECKPOINT_DIR"; then
        echo "[DONE] Training complete!"
        exit 0
    fi

    # Submit job with --resume (first attempt may not have a checkpoint yet)
    RESUME_FLAG=""
    if [[ $attempt -gt 1 ]] || [[ -f "$CHECKPOINT_DIR/diffusion_latest.pt" ]] || \
       [[ -f "$CHECKPOINT_DIR/drn_latest.pt" ]] || [[ -f "$CHECKPOINT_DIR/vae_latest.pt" ]]; then
        RESUME_FLAG="--resume"
    fi

    JOB_OUTPUT=$(sbatch scripts/run_training.sh --stage "$STAGE" $RESUME_FLAG $EXTRA_ARGS)
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$')
    echo "[SUBMITTED] Job $JOB_ID (stage=$STAGE $RESUME_FLAG)"

    # Wait for job to finish
    while true; do
        sleep "$POLL_INTERVAL"

        # Check if job is still running
        JOB_STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || echo "")

        if [[ -z "$JOB_STATE" ]]; then
            echo "[FINISHED] Job $JOB_ID completed at $(date)"

            # Show last few lines of output
            OUTPUT_FILE="train_output.$JOB_ID"
            ERROR_FILE="train_error.$JOB_ID"
            if [[ -f "$OUTPUT_FILE" ]]; then
                echo "--- Last 5 lines of output ---"
                tail -5 "$OUTPUT_FILE"
            fi
            if [[ -f "$ERROR_FILE" ]] && [[ -s "$ERROR_FILE" ]]; then
                echo "--- Last 5 lines of errors ---"
                tail -5 "$ERROR_FILE"
            fi
            break
        else
            # Show progress
            LATEST_EPOCH=""
            OUTPUT_FILE="train_output.$JOB_ID"
            if [[ -f "$OUTPUT_FILE" ]]; then
                LATEST_EPOCH=$(grep -oP 'Epoch \d+/\d+' "$OUTPUT_FILE" | tail -1)
            fi
            echo "  [RUNNING] $JOB_STATE ${LATEST_EPOCH:+($LATEST_EPOCH)} ($(date '+%H:%M'))"
        fi
    done

    # Check if training completed
    if is_training_complete "$STAGE" "$CHECKPOINT_DIR"; then
        echo ""
        echo "============================================"
        echo "[DONE] Training complete after $attempt submission(s)!"
        echo "  Completed: $(date)"
        echo "============================================"
        exit 0
    fi

    echo "[INCOMPLETE] Will resubmit with --resume..."
done

echo ""
echo "============================================"
echo "[FAILED] Max resubmissions ($MAX_RESUBMITS) reached without completing."
echo "  Check logs and rerun manually."
echo "============================================"
exit 1
