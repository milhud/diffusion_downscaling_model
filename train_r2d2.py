"""Entry point for the R2-D2-style baseline: python train_r2d2.py

PyTorch reproduction of R2-D2's published architecture (Lopez-Gomez et al., PNAS 2025 /
arXiv:2410.01776) — see docs/R2D2_BASELINE.md for exactly what is matched vs. approximated.
This is a standalone, single-stage model (no DRN, no VAE) trained on the same
ERA5->CONUS404 cache as the rest of this repo. Supports single-GPU and multi-GPU
(DDP via torchrun) transparently, mirroring train.py's conventions.

Usage:
    python train_r2d2.py --data_dir data --cache_dir cached_data \
        --checkpoint_dir checkpoints --plot_dir train_plots
    python train_r2d2.py --max_steps 20   # smoke test
    torchrun --standalone --nproc_per_node=4 train_r2d2.py --resume
"""

import argparse
import os
import time
import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path

from config import (
    ERA5_VARS, CONUS404_VARS, IN_CH, OUT_CH, PATCH_SIZE, TRAIN,
)
from src.preprocessing.land_mask import get_valid_patch_origins
from src.data.dataset import build_dataloaders
from src.models.r2d2_unet import R2D2UNet
from src.training.train_r2d2 import train_r2d2

from train import compute_norm_stats, setup_ddp, cleanup_ddp, setup_regridder_and_mask


def main():
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--plot_dir", type=str, default="train_plots")
    parser.add_argument("--cache_dir", type=str, default="cached_data")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after this many optimizer steps (smoke testing).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config.py TRAIN['diff_epochs'] for a short run.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override config.py TRAIN['batch_size']. This model runs at full "
                             "patch resolution (no VAE latent compression like our other "
                             "diffusion model), so it needs a much smaller batch size at the "
                             "same patch size — OOMs on a single A100 at batch_size=8/256px.")
    args = parser.parse_args()

    if is_main:
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_years = TRAIN["train_years"]
    val_years = TRAIN["val_years"]

    if is_main:
        print("=" * 70)
        print("TRAINING — R2-D2-style baseline (PyTorch reproduction)")
        print(f"  Variables: {ERA5_VARS} -> {CONUS404_VARS}")
        print(f"  IN_CH={IN_CH}, OUT_CH={OUT_CH}")
        print(f"  World size: {world_size} GPU(s)")
        if args.max_steps:
            print(f"  SMOKE TEST: max_steps={args.max_steps}")
        print("=" * 70)

    t0 = time.time()
    stats = compute_norm_stats(args.data_dir, train_years)
    cache_dir = None if args.no_cache else args.cache_dir

    min_land_frac = TRAIN.get("min_land_frac", 0.5)
    cache_static = (Path(cache_dir) / "static_fields.npy") if cache_dir else None
    if cache_static and cache_static.exists():
        if rank == 0:
            static = np.load(cache_static)
            land_mask = static[5] > 0.5
            valid_origins = get_valid_patch_origins(land_mask, PATCH_SIZE, min_land_frac)
            print(f"[LandMask] {len(valid_origins)} valid patch origins "
                  f"(min_land_frac={min_land_frac})")
            _setup = [(None, None, None, land_mask, valid_origins)]
        else:
            _setup = [None]
        if world_size > 1:
            dist.broadcast_object_list(_setup, src=0)
        _, _, _, land_mask, valid_origins = _setup[0]
        regridder = conus_lat = conus_lon = None
    else:
        if rank == 0:
            _setup = [setup_regridder_and_mask(args.data_dir)]
        else:
            _setup = [None]
        if world_size > 1:
            dist.broadcast_object_list(_setup, src=0)
        regridder, conus_lat, conus_lon, land_mask, valid_origins = _setup[0]
    num_workers = 4 if args.no_cache else 2

    batch_size = args.batch_size if args.batch_size is not None else TRAIN["batch_size"]
    if is_main:
        print(f"  Batch size: {batch_size} (per GPU)")

    train_dl, val_dl, train_sampler = build_dataloaders(
        args.data_dir, stats,
        batch_size=batch_size,
        patches_per_day=TRAIN["patches_per_day"],
        num_workers=num_workers,
        train_years=train_years, val_years=val_years,
        land_mask=land_mask, valid_origins=valid_origins,
        era5_vars=ERA5_VARS, conus_vars=CONUS404_VARS,
        cache_dir=cache_dir,
        regridder=regridder, conus_lat=conus_lat, conus_lon=conus_lon,
        rank=rank, world_size=world_size,
    )
    if is_main:
        print(f"[Data] Setup done in {time.time()-t0:.0f}s")

    model = R2D2UNet(in_ch=OUT_CH, out_ch=OUT_CH, cond_ch=IN_CH).to(device)
    if is_main:
        print(f"[R2D2] Params: {sum(p.numel() for p in model.parameters()):,}")

    epochs = args.epochs if args.epochs is not None else TRAIN["diff_epochs"]

    train_r2d2(
        model, train_dl, val_dl, out_ch=OUT_CH,
        epochs=epochs,
        lr=TRAIN["diff_lr"],
        warmup_epochs=TRAIN["diff_warmup_epochs"],
        ema_decay=TRAIN["ema_decay"],
        p_uncond=TRAIN["p_uncond"],
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        plot_dir=args.plot_dir,
        eval_every=3,
        resume=args.resume,
        grad_accum=TRAIN.get("diff_grad_accum", 1),
        p_mean=0.0,   # R2-D2 uses log-uniform sampling; 0.0/1.2 log-normal is our approximation
        p_std=1.2,
        rank=rank, local_rank=local_rank, world_size=world_size,
        train_sampler=train_sampler,
        max_steps=args.max_steps,
    )

    cleanup_ddp()

    if is_main:
        total_time = time.time() - t0
        print(f"\n{'='*70}")
        print(f"R2-D2-STYLE TRAINING COMPLETE — {total_time/3600:.2f}h")
        print(f"  Checkpoints: {args.checkpoint_dir}/")
        print(f"  Plots: {args.plot_dir}/")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
