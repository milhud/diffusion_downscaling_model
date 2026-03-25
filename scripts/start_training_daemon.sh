#!/bin/bash
# ============================================================================
# Start the auto-resubmitting training loop as a background daemon.
# Survives terminal exit (uses nohup + disown).
#
# Usage:
#   bash scripts/start_training_daemon.sh --stage diffusion
#   bash scripts/start_training_daemon.sh --stage all
#   bash scripts/start_training_daemon.sh --stage diffusion --max-resubmits 10
#
# Logs:  train_loop.log  (append mode)
# Stop:  kill $(cat .train_daemon.pid)
# ============================================================================

set -euo pipefail
cd /mnt/home/hmiller/diffusion_downscaling_model

# Check if daemon is already running
if [[ -f .train_daemon.pid ]]; then
    OLD_PID=$(cat .train_daemon.pid)
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Training daemon already running (PID $OLD_PID)"
        echo "Stop it first: kill $OLD_PID"
        exit 1
    else
        rm -f .train_daemon.pid
    fi
fi

# Launch in background with nohup
nohup bash scripts/train_until_done.sh "$@" >> train_loop.log 2>&1 &
DAEMON_PID=$!
echo "$DAEMON_PID" > .train_daemon.pid
disown "$DAEMON_PID"

echo "============================================"
echo "Training daemon started"
echo "  PID: $DAEMON_PID"
echo "  Args: $@"
echo "  Log: train_loop.log"
echo "  Stop: kill $DAEMON_PID"
echo "============================================"
echo ""
echo "Monitor with:"
echo "  tail -f train_loop.log"
