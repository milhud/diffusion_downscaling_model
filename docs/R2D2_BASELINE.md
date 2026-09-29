# R2-D2-style baseline (PyTorch reproduction)

**This is NOT the official R2-D2 checkpoint.** R2-D2 (Lopez-Gomez et al., "Dynamical-generative
downscaling of climate model ensembles," PNAS 2025 / arXiv:2410.01776) is trained in JAX/Flax,
checkpointed via Orbax, and its public weights (GCS bucket `dynamical_generative_downscaling`)
are fixed to a 352x288 Western-US 9km grid with a variable set (T2, Q2, U10, V10, PSFC, RAIN_*,
plus radiation/snow/runoff fields) that only partially overlaps ours — we don't have ERA5
radiation, snow, or runoff fields, and use dewpoint (d2m) rather than specific humidity (Q2).
Loading those weights would require installing an entirely separate JAX/Orbax/Gin stack for a
model that can't run on our grid or our variable set anyway.

Instead, this reproduces R2-D2's **architecture** in PyTorch (`src/models/r2d2_unet.py`) and
trains it from scratch on our own ERA5->CONUS404 cache, so it's a fair third baseline alongside
our in-house Latent CorrDiff model and NVIDIA's CorrDiff (`external/corrdiff/`).

## What R2-D2 actually is

A **single-stage residual diffusion model** — no VAE, no separate regression network (unlike
both our Latent CorrDiff and NVIDIA CorrDiff, which both have a two-stage mean-predictor +
diffusion-residual design). The published denoiser (`PreconditionedDenoiserUNet` in
`swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/backbones.py`) is trained
to predict the residual between a cubic-interpolated coarse conditioning field and the true
high-res target, via EDM-style denoising score matching under a variance-exploding (VE) SDE.

## Matched exactly

- **Single-stage residual diffusion** (no VAE, no regression net) — `src/training/train_r2d2.py`
  computes `residual = conus_target - era5_baseline` and trains a denoiser on it directly.
- **UNet channel widths**: `(256, 512, 768)`, 3 resolution levels.
- **`num_blocks=4`** residual blocks per level.
- **`use_attention=False`** — no self-attention anywhere in the network (a deliberate deviation
  from our own DRN/DiffusionUNet, which both use attention blocks).
- **`dropout_rate=0.1`**.
- **`noise_embed_dim=192`** for the noise/time embedding.
- **EDM/VE-style preconditioning** (`c_skip`/`c_out`/`c_in`/`c_noise`) — Karras et al. 2022's EDM
  formulation is itself a variance-exploding parameterization, so we reuse this repo's existing
  `src/models/edm.py` `EDMSchedule` unmodified rather than reimplementing R2-D2's schedule.

## Approximated / deviated (and why)

- **Noise schedule**: R2-D2 uses a custom "tangent" schedule (`sigma = tan(...)`, `clip_max=80`,
  `start=-1.5`, `end=1.5`) with log-uniform sigma sampling during training. We use our existing
  `EDMSchedule`'s log-normal sigma sampling (`p_mean`, `p_std`) instead — both are VE-style EDM
  parameterizations, differing in the training-time sigma distribution, not in the fundamental
  SDE. `sigma_data` is estimated empirically from a training batch's residual std at the start
  of each run (logged) rather than fixed at `1.0`, since our physical-space residual has a very
  different scale than R2-D2's pre-standardized target.
- **Conditioning merge**: R2-D2 uses a learned `InterpConvMerge` module to combine the
  interpolated LR field and static features. We approximate this with a plain channel-concat of
  the full ERA5+static conditioning stack (already regridded onto the CONUS404 grid by this
  repo's preprocessing pipeline) at the first conv layer — simpler, and consistent with how this
  repo's DRN already handles LR+static conditioning.
- **Downsampling depth**: R2-D2 downsamples after all 3 levels (`downsample_ratio=(2,2,2)`,
  8x total reduction). We downsample between levels only (2 downsamples, 4x reduction),
  matching this repo's DRN/DiffusionUNet convention, so the model operates at 64x64 (not 32x32)
  at the bottleneck for a 256x256 patch.
- **Residual baseline**: R2-D2's residual is against a *cubic-interpolated* coarse field. Our
  baseline is the matching ERA5 channel already regridded onto the CONUS404 grid by this repo's
  existing preprocessing (bilinear/conservative, not cubic) — the same conditioning field the
  rest of this repo's models already consume, so no new regridding code was needed.
- **Missing input variables**: no radiation (LWDNB/LWUPB/SWDNB/SWUPB), snow, or runoff fields —
  we only condition on the 6 ERA5 vars + 6 static fields already in this repo's cache.

## Status: smoke test PASSED (job 58640761)

Two things had to be fixed before it passed, both worth flagging:

1. **The local `cached_data/` in this repo checkout is stale (1 variable, not 6).** Its
   `era5_1980.npy`/`conus_1980.npy` are shape `(366, 1, 1015, 1367)` — leftover from before
   `config.py`'s `VARIABLE_PAIRS` was expanded from temperature-only to the current 6-variable
   set. This affects *any* model reading from local `cached_data/`, not just this one. The
   production cache actually matching the current 6-variable config lives at
   `/discover/nobackup/sduan/.data` (the path `run_training.sh` already points `--cache_dir` at)
   — confirmed by checking `checkpoints/drn_best.pt`'s `input_conv` shape (`[64, 12, 3, 3]`,
   i.e. `IN_CH=12`) against the stale local one (`checkpoints_test/drn_best.pt`, `[64, 7, 3, 3]`,
   `IN_CH=7`). `scripts/run_r2d2_smoke.sh` now points at the correct shared cache dir.
2. **OOM at the DRN/DiffusionUNet's usual `batch_size=8`.** This model runs entirely in pixel
   space at the full 256x256 patch resolution (no VAE latent compression like our other
   diffusion model, which only ever sees 64x64 latents) — the 226M-param, 3-level
   (256/512/768-channel) UNet simply needs much more activation memory per sample. OOM'd a
   single A100 (40GB) at `batch_size=8`; `batch_size=2` (effective batch 8 with the existing
   `grad_accum=4`) fits comfortably (12.3GB peak RSS reported by SLURM, plenty of headroom).
   Added a `--batch_size` override to `train_r2d2.py` for this (defaults to `config.py`'s
   `TRAIN["batch_size"]` if unset, same as every other model in this repo). Worth revisiting
   before the full run — `batch_size=2` per GPU x 4 GPUs = effective batch 8 before grad accum,
   noticeably smaller than the latent model's usual batch 32; either raise `grad_accum` further
   or accept slower convergence per wall-clock hour. This also means, at matched batch size,
   R2-D2-style pixel-space diffusion is meaningfully more memory-hungry per sample than our
   latent-space model — a real architectural tradeoff worth noting in the benchmark writeup,
   not an artifact of this port.

Actual passing log (`sbatch scripts/run_r2d2_smoke.sh`, job 58640761, single A100, 2m46s):

```
[R2D2] Params: 226,043,398
  [R2D2] Estimated sigma_data=0.3001 from training batch residual std
  [R2D2] sigma_data=0.3001, p_mean=0.0, p_std=1.2
  [R2D2] Grad accumulation: 4 (effective batch=8)
  [R2D2] Epoch 1, Step 0, Loss: 0.892265, LR: 2.00e-07
  [R2D2] Epoch 1, Step 50, Loss: 1.444060, LR: 2.00e-07
[R2D2] Epoch 1/1 | Train: 1.537008 | Val: 0.986755
  [R2D2] Epoch 1 eval — Baseline RMSE: 0.2566, R2D2-style RMSE: 0.4557, CRPS: 0.3652
```

Checkpointing, EMA, and the Heun sampler (used for the eval-time ensemble/CRPS) all ran
successfully — `checkpoints_test/r2d2_{best,latest}.pt` and
`train_plots_test/r2d2_epoch001.png` / `r2d2_loss_curves.png` were written. RMSE/CRPS numbers
are meaningless after only 20 steps (worse than the interpolation baseline, as expected for a
near-random-init model) — this run verified the pipeline works end-to-end, not model quality.

## Commands

Smoke test (single GPU, 20 steps) — passing as of job 58640761:
```bash
sbatch scripts/run_r2d2_smoke.sh
```

Full training (4x A100, single stage — no `--stage` flag needed unlike the DRN/VAE/diffusion
pipeline). `scripts/run_r2d2_training.sh` does not yet pass `--batch_size`; based on the smoke
test, add `--batch_size 2` (or tune upward with gradient checkpointing) before submitting, or
it will likely OOM at the default `config.py` batch size of 8:
```bash
sbatch scripts/run_r2d2_training.sh --batch_size 2
sbatch scripts/run_r2d2_training.sh --batch_size 2 --resume   # repeat until convergence; 12h QOS cap
```
