# CorrDiff baseline (NVIDIA PhysicsNeMo, deprecated `examples/weather/corrdiff` recipe)

A second baseline for comparison against our in-house Latent CorrDiff model, trained on
the **same** ERA5 -> CONUS404 cache (`cached_data/era5_{year}.npy`, `conus_{year}.npy`,
`static_fields.npy`, `norm_stats.npz`), using NVIDIA's own CorrDiff implementation rather
than a reimplementation, so the comparison isolates architecture/training-recipe choices
rather than any accidental adapter bugs.

## What's here

- `dataset.py` -- `Era5Conus404Dataset(DownscalingDataset)` adapter wrapping our cache in
  physicsnemo's expected interface. Reuses `src.preprocessing.normalization` for z-scoring
  so both models see identical normalized inputs. ERA5 is already regridded onto the
  CONUS404 grid at cache-build time, so no interpolation happens in this adapter.
- `conf/` -- Hydra configs for Stage 1 (regression) and Stage 2 (diffusion) training, and
  generation, adapted from physicsnemo's `config_training_custom.yaml` template. Variable
  lists match `config.py`'s `ERA5_VARS`/`CONUS404_VARS`; `crop_size: 448` (patched
  training via `patch_shape_x/y` inside that crop) chosen for A100 80GB memory headroom --
  the smoke test peaked at 23.6 GB with `total_batch_size=1`.
- `train_wrapper.py` -- thin wrapper around physicsnemo's `train.py` that blocks the
  system GEOSpyD TensorFlow install (see **TensorFlow bug** below).
- `venv/` (gitignored) -- Python 3.12 venv with `--system-site-packages` (to reuse the
  `python/GEOSpyD/24.3.0-0/3.12` module's torch 2.10.0+cu128), plus physicsnemo and its
  corrdiff example's extra deps installed on top.
- `physicsnemo_src/` (gitignored) -- shallow clone of `NVIDIA/physicsnemo` (pip's wheel
  doesn't ship the `examples/` tree, so we clone it directly and run `examples/weather/
  corrdiff/train.py` in place).
- `checkpoints_regression/`, `checkpoints_diffusion/` (gitignored) -- training output.

## Setup (already done in this venv; for a from-scratch rebuild)

```bash
module purge && module load python/GEOSpyD/24.3.0-0/3.12
python3 -m venv --system-site-packages external/corrdiff/venv
source external/corrdiff/venv/bin/activate
pip install --user nvidia-physicsnemo   # or from physicsnemo_src if example scripts needed
git clone --depth 1 https://github.com/NVIDIA/physicsnemo.git external/corrdiff/physicsnemo_src
pip install -r external/corrdiff/physicsnemo_src/examples/weather/corrdiff/requirements.txt
```

## TensorFlow bug (found and fixed during smoke testing)

The venv's `--system-site-packages` flag pulls in the GEOSpyD module's TensorFlow 2.17.0,
which is ABI-incompatible with the newer NumPy 2.2.6 that physicsnemo resolves in this
venv -- importing it raises a `SystemError` deep inside a compiled extension
(`_pywrap_checkpoint_reader`). That's harmless on its own (`torch.utils.tensorboard`
catches the failure and falls back to its own writer), *except* that `einops`'s backend
auto-detection (used inside physicsnemo's `GroupNorm`) checks `"tensorflow" in
sys.modules` without checking the value isn't `None`. A `None` sentinel in `sys.modules`
(deliberately set by `torch.utils.tensorboard` after a failed optional import, as a
"don't retry" cache) makes einops try `import tensorflow` again during every `GroupNorm`
forward pass, which Python refuses immediately with `ModuleNotFoundError: import of
tensorflow halted; None in sys.modules` -- crashing the diffusion stage's very first
training step (the regression stage doesn't hit this path).

**Fix**: `venv/lib/python3.12/site-packages/tensorflow/__init__.py` is a one-line stub
(`raise ImportError(...)`) that shadows the broken system install (venv's own
site-packages precedes system site-packages on `sys.path` even with
`--system-site-packages`). This makes the *first* `import tensorflow` attempt anywhere in
the process fail cleanly, and Python removes the failed entry from `sys.modules`
afterward -- so nothing downstream (including einops) ever sees a poisoned `None`
sentinel. `train_wrapper.py` originally tried to achieve the same thing by setting
`sys.modules["tensorflow"] = None` directly, which is exactly the poisoned-sentinel
pattern that broke einops -- that line was removed once the venv-level stub package was
in place; see the docstring in `train_wrapper.py` for the full explanation. Confirmed
fixed by re-running the smoke test (job 58644637, below) after applying the fix.

Note: this stub is package-local to `external/corrdiff/venv/`, not committed to git
(the whole `venv/` dir is gitignored) -- rebuilding the venv from scratch needs this file
recreated, or an equivalent numpy/tensorflow version pin.

## Smoke test result (job 58644637, 2026-09-29)

Single A100, `training_duration=64` (regression) then `64` (diffusion), `total_batch_size=1`.

Stage 1 (regression) -- completed cleanly, checkpoint saved:
```
[main][INFO] - Training for 64 images...
[main][INFO] - samples 67.0  training_loss 245718.16  ... peak_gpu_mem_gb 23.58
[main][INFO] - Training Completed.
```

Stage 2 (diffusion) -- loaded the Stage-1 checkpoint, ran all 64 steps over real batches,
finite (noisy, as expected for a freshly-initialized diffusion net) losses throughout, no
crash:
```
[main][INFO] - Loaded the pre-trained regression model
[main][INFO] - Training for 64 images...
[main][INFO] - samples 1.0   training_loss 1234214.25 ... peak_gpu_mem_gb 23.29
...
[main][INFO] - samples 64.0  training_loss 642200.25  ... peak_gpu_mem_gb 14.18
[main][INFO] - Training Completed.
```

## Launching full training

Not auto-submitted. `scripts/run_corrdiff_training.sh` runs Stage 1 then Stage 2 (4x A100,
`training_duration=5,000,000` each -- physicsnemo's README suggests 1M-30M samples for a
first full run; `alla100` QOS caps walltime at 12h so this will need several resubmissions,
which is safe since `train.py` auto-resumes from `training.io.checkpoint_dir`):

```bash
sbatch scripts/run_corrdiff_training.sh regression
# after it (repeatedly) completes / auto-resumes to convergence:
sbatch scripts/run_corrdiff_training.sh diffusion \
    external/corrdiff/checkpoints_regression/checkpoints_regression/CorrDiffRegressionUNet.<latest>.mdlus
```
