# Agent Context: Recent Engineering Work

This file documents the current state of the codebase, recent bug fixes, and known issues for an agent continuing this project.

---

## What This Project Is

A three-stage latent CorrDiff pipeline for ERA5 → CONUS404 atmospheric downscaling over CONUS:

1. **DRN** (`src/models/drn.py`, `src/training/train_drn.py`) — Deterministic Regression Network. UNet that predicts the conditional mean E[x|y]. Takes 12-channel input (6 ERA5 vars + 6 static fields) and outputs 6 CONUS404 vars. ~50M params.
2. **VAE** (`src/models/vae.py`, `src/training/train_vae.py`) — Encodes the residual `CONUS404 - DRN(ERA5)` into a compressed latent space.
3. **Latent Diffusion** (`src/models/diffusion.py`, `src/training/train_diffusion.py`) — EDM-style diffusion model in VAE latent space. Generates the stochastic residual.

The stages are **sequentially dependent** — VAE needs a trained DRN to compute residuals; diffusion needs both. You cannot train them simultaneously.

Entry point: `train.py`. Run via `sbatch scripts/run_training.sh --stage drn`.

---

## Current Training State

- DRN is **not currently running** (job was manually cancelled).
- No valid checkpoints exist in `checkpoints/` (they were deleted before the last cancelled run).
- DRN needs to train for 50 epochs before VAE can begin.
- After DRN: `sbatch scripts/run_training.sh --stage vae`
- After VAE: `sbatch scripts/run_training.sh --stage diffusion`

The daemon (`scripts/train_until_done.sh`) auto-resubmits with `--resume` when the 12h SLURM window expires. Start it with:
```bash
nohup bash scripts/start_training_daemon.sh --stage drn > train_loop.log 2>&1 &
```

---

## Recent Bugs Fixed (Critical)

### 1. Wrong LR Scaling for AdamW with DDP (just fixed, commit `e27dd92`)

**Bug:** `lr=lr * world_size` was used for AdamW optimizer across all three trainers. The linear LR scaling rule is designed for SGD. Adam normalizes gradients by their running variance estimate, so multiplying LR by 4 causes overshooting and ~3-4× worse RMSE convergence. Epoch 15 DDP had RMSE ~0.36 vs the expected ~0.10 at equivalent training progress.

**Fix:** Changed to `lr=lr` (no scaling) in `train_drn.py`, `train_vae.py`, and `train_diffusion.py`.

### 2. `criterion.log_var` Not Synced Across GPUs (just fixed, commit `e27dd92`)

**Bug:** `PerVariableMSE` has a learnable `log_var: nn.Parameter` (one per output variable) that was NOT wrapped in DDP. Each GPU updated it independently from its data shard. It was only broadcast from rank 0 once per epoch (at epoch start), so all GPUs diverged mid-epoch on the precision weights.

**Fix:** Now broadcasts `criterion.log_var.data` from rank 0 after every `optimizer.step()`. Only 6 floats — negligible overhead.

### 3. `criterion.log_var` Not Saved in Checkpoints (fixed earlier, commit `d42c695`)

**Bug:** `PerVariableMSE.log_var` was never included in checkpoint saves, so on every job resume it reset to 0 (uniform weighting). This caused loss to revert to ~+0.07 for 2-3 epochs before re-learning the precision weights.

**Fix:** `criterion.state_dict()` is now saved as `"criterion_state_dict"` in both `drn_best.pt` and `drn_latest.pt`, and restored on resume.

---

## Multi-GPU Setup (DDP)

- **Config:** 1 node × 4 A100 GPUs via `torchrun --nproc_per_node=4`
- **QOS limit:** `alla100` caps at `gres/gpu=4` — requesting more causes SLURM to silently downgrade
- **Confirmed working:** NCCL version line + "World size: 4 GPU(s)" in output
- `DistributedSampler` is used for training; each GPU sees 1/4 of the data → 4× fewer steps per epoch per GPU than single-GPU (but same total data processed)
- All I/O (prints, checkpoints, plots, git pushes) gated behind `is_main = rank == 0`
- Val loss aggregated via `dist.all_reduce` across all ranks before logging

---

## Loss Function: `PerVariableMSE` (Gaussian NLL)

```
L = 0.5 * exp(w) * MSE - 0.5 * w
```

Where `w = log_var[i]` is a learnable log-precision per variable. This auto-weights variables with different physical scales (temperature in K, precipitation in mm, etc.). Loss goes **negative** when precision is high and MSE is low — this is correct and expected behavior, not a bug.

---

## Data

- ERA5 files: `data/era5_{year}.nc` (1980–2020)
- CONUS404 files: `data/conus404_yearly_{year}.nc` (1980–2020)
- Pre-computed numpy cache: `/discover/nobackup/sduan/.data/` — use this, it's fast
- Train years: 1980–2014; Val years: 2015–2017; Test: 2018–2020
- **Do NOT modify the `.nc` files** — they are read-only source data

---

## Key File Locations

| File | Purpose |
|------|---------|
| `train.py` | Entry point, parses args, calls stage trainers |
| `src/training/train_drn.py` | DRN training loop (DDP-aware) |
| `src/training/train_vae.py` | VAE training loop (DDP-aware) |
| `src/training/train_diffusion.py` | Diffusion training loop (DDP-aware) |
| `src/data/dataset.py` | Dataset + DataLoader with DistributedSampler |
| `src/training/losses.py` | `PerVariableMSE` (Gaussian NLL) |
| `src/evaluation/plots.py` | `evaluate_drn`, `plot_loss_curves` |
| `scripts/run_training.sh` | SLURM sbatch script (4 GPU, 12h) |
| `scripts/train_until_done.sh` | Daemon that auto-resubmits with --resume |
| `docs/paper_points.md` | Academic paper scaffolding (Results TBD) |
| `checkpoints/` | Saved model weights (drn_best.pt, drn_latest.pt, etc.) |
| `train_plots/` | Evaluation plots generated every 3 epochs |

---

## How to Submit

```bash
# Fresh start
sbatch scripts/run_training.sh --stage drn

# Resume (after SLURM timeout)
sbatch scripts/run_training.sh --stage drn --resume

# Daemon (recommended — handles auto-resume)
nohup bash scripts/start_training_daemon.sh --stage drn > train_loop.log 2>&1 &
echo $! > .train_daemon.pid
```

Note: the daemon PID file is at `.train_daemon.pid`. Kill it with `kill $(cat .train_daemon.pid)` before manually resubmitting to avoid double-submission.

---

## Branch

Active development is on `paper-ready`. Push plots and fixes there.
