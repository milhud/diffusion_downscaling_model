"""Entry point for training: python train.py --stage [drn|vae|diffusion|all]

Uses config.py for variable selection, model architecture, and training defaults.
Supports single-GPU and multi-GPU (DDP via torchrun) training transparently.
"""

import argparse
import os
import subprocess
import time
import torch
import torch.distributed as dist


def setup_ddp():
    """Initialize distributed training if launched via torchrun, else no-op."""
    if "RANK" not in os.environ:
        return 0, 0, 1  # rank, local_rank, world_size
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
import numpy as np
import xarray as xr
from pathlib import Path

from config import (
    ERA5_VARS, CONUS404_VARS, PRETRANSFORMS, VARIABLE_PAIRS,
    IN_CH, OUT_CH, PATCH_SIZE, LATENT_CH, LATENT_H,
    MODEL, TRAIN, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
)
from src.preprocessing.normalization import NormalizationStats, apply_pretransform
from src.preprocessing.regrid import ERA5Regridder
from src.preprocessing.land_mask import build_conus404_land_mask, get_valid_patch_origins
from src.data.dataset import build_dataloaders
from src.models.drn import DRN
from src.models.vae import VAE
from src.models.diffusion_unet import DiffusionUNet
from src.models.edm import EDMSchedule
from src.training.train_drn import train_drn
from src.training.train_vae import train_vae
from src.training.train_diffusion import train_diffusion
from src.evaluation.plots import (
    plot_loss_curves, evaluate_drn, evaluate_full_pipeline,
)
from src.training.ema import EMA


def compute_norm_stats(data_dir: str, years: list, cache_path: str = "norm_stats.npz"):
    """Compute normalization statistics from training years (config-aware)."""
    cache = Path(cache_path)
    stats = NormalizationStats()

    # Override stats class to use config variables
    stats_era5_vars = ERA5_VARS
    stats_conus_vars = CONUS404_VARS

    if cache.exists():
        print(f"Loading cached norm stats from {cache}")
        stats.load(str(cache))
        return stats

    print(f"Computing normalization stats from {len(years)} years "
          f"({len(stats_era5_vars)} ERA5 vars, {len(stats_conus_vars)} CONUS404 vars)...")
    era5_accum = {v: [] for v in stats_era5_vars}
    conus_accum = {v: [] for v in stats_conus_vars}

    for y in years[:5]:  # Use first 5 years for stats (sufficient)
        sample_days = list(range(0, 366, 30))

        with xr.open_dataset(f"{data_dir}/era5_{y}.nc") as ds:
            for d in sample_days:
                try:
                    for m in range(12):
                        # Check first var to see if this month has valid data for this day
                        test = ds[stats_era5_vars[0]].isel(time=m, valid_time=d).values
                        if np.all(np.isnan(test)):
                            continue
                        # Valid month found — accumulate all vars
                        for var in stats_era5_vars:
                            vals = ds[var].isel(time=m, valid_time=d).values.astype(np.float32)
                            era5_accum[var].append(apply_pretransform(vals.flatten()[::100], var))
                        break
                except (IndexError, ValueError):
                    continue

        with xr.open_dataset(f"{data_dir}/conus404_yearly_{y}.nc") as ds:
            for d in sample_days:
                try:
                    for var in stats_conus_vars:
                        vals = ds[var].isel(time=d).values.astype(np.float32)
                        conus_accum[var].append(apply_pretransform(vals.flatten()[::100], var))
                except (IndexError, ValueError):
                    continue

    era5_samples = {v: np.concatenate(era5_accum[v]) for v in stats_era5_vars}
    conus_samples = {v: np.concatenate(conus_accum[v]) for v in stats_conus_vars}

    stats.compute_from_data(era5_samples, conus_samples)
    stats.save(str(cache))
    print(f"Saved norm stats to {cache}")
    return stats


def setup_regridder_and_mask(data_dir: str):
    """Build ERA5→CONUS404 regridder and land mask."""
    with xr.open_dataset(f"{data_dir}/era5_1980.nc") as ds:
        era5_lat = ds["latitude"].values
        era5_lon = ds["longitude"].values
        # Build land mask from ERA5 lsm
        land_mask = build_conus404_land_mask(
            xr.open_dataset(f"{data_dir}/conus404_yearly_1980.nc")["lat"].values,
            xr.open_dataset(f"{data_dir}/conus404_yearly_1980.nc")["lon"].values,
            ds,
        )

    with xr.open_dataset(f"{data_dir}/conus404_yearly_1980.nc") as ds:
        conus_lat = ds["lat"].values
        conus_lon = ds["lon"].values

    print("Building ERA5→CONUS404 regridder...")
    regridder = ERA5Regridder(era5_lat, era5_lon, conus_lat, conus_lon)

    # Find valid land-only patch origins
    min_land_frac = TRAIN.get("min_land_frac", 0.5)
    valid_origins = get_valid_patch_origins(land_mask, PATCH_SIZE, min_land_frac)
    print(f"[LandMask] {len(valid_origins)} valid patch origins "
          f"(min_land_frac={min_land_frac})")

    return regridder, conus_lat, conus_lon, land_mask, valid_origins


def main():
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True,
                        choices=["drn", "vae", "diffusion", "all"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--plot_dir", type=str, default="train_plots")
    parser.add_argument("--drn_checkpoint", type=str, default="checkpoints/drn_best.pt")
    parser.add_argument("--vae_checkpoint", type=str, default="checkpoints/vae_best.pt")
    parser.add_argument("--cache_dir", type=str, default="cached_data")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip disk cache, regrid on-the-fly (saves disk space)")
    args = parser.parse_args()

    if is_main:
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_years = TRAIN["train_years"]
    val_years = TRAIN["val_years"]

    if is_main:
        print("=" * 70)
        print(f"TRAINING — Stage: {args.stage}")
        print(f"  Variables: {ERA5_VARS} -> {CONUS404_VARS}")
        print(f"  IN_CH={IN_CH}, OUT_CH={OUT_CH}, LATENT_CH={LATENT_CH}")
        print(f"  Train years: {train_years[0]}-{train_years[-1]} ({len(train_years)}y)")
        print(f"  Val years: {val_years[0]}-{val_years[-1]} ({len(val_years)}y)")
        print(f"  World size: {world_size} GPU(s)")
        print("=" * 70)

    t0 = time.time()
    stats = compute_norm_stats(args.data_dir, train_years)
    regridder, conus_lat, conus_lon, land_mask, valid_origins = \
        setup_regridder_and_mask(args.data_dir)

    cache_dir = None if args.no_cache else args.cache_dir
    num_workers = 4 if args.no_cache else 2

    train_dl, val_dl, train_sampler = build_dataloaders(
        args.data_dir, stats,
        batch_size=TRAIN["batch_size"],
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

    # Fixed eval batch for visualization (rank 0 only)
    eval_era5, eval_conus = next(iter(val_dl))

    stages = [args.stage] if args.stage != "all" else ["drn", "vae", "diffusion"]

    # ── DRN ──
    if "drn" in stages:
        if is_main:
            print("\n" + "=" * 70)
            print("STAGE 1: DRN")
            print("=" * 70)
        drn = DRN(in_ch=IN_CH, out_ch=OUT_CH, base_ch=MODEL["drn_base_ch"],
                  ch_mults=MODEL["drn_ch_mults"],
                  num_res_blocks=MODEL["drn_num_res_blocks"],
                  attn_resolutions=MODEL["drn_attn_resolutions"]).to(device)
        if is_main:
            print(f"[DRN] Params: {sum(p.numel() for p in drn.parameters()):,}")

        drn = train_drn(drn, train_dl, val_dl,
                        epochs=TRAIN["drn_epochs"],
                        lr=TRAIN["drn_lr"],
                        warmup_epochs=TRAIN["drn_warmup_epochs"],
                        device=device,
                        checkpoint_dir=args.checkpoint_dir,
                        plot_dir=args.plot_dir,
                        eval_every=3,
                        num_output_vars=OUT_CH,
                        precip_channel=-1,
                        resume=args.resume,
                        rank=rank, local_rank=local_rank, world_size=world_size,
                        train_sampler=train_sampler)

    # ── VAE ──
    if "vae" in stages:
        if is_main:
            print("\n" + "=" * 70)
            print("STAGE 2a: VAE")
            print("=" * 70)
        if "drn" not in stages:
            drn = DRN(in_ch=IN_CH, out_ch=OUT_CH, base_ch=MODEL["drn_base_ch"],
                      ch_mults=MODEL["drn_ch_mults"],
                      num_res_blocks=MODEL["drn_num_res_blocks"],
                      attn_resolutions=MODEL["drn_attn_resolutions"]).to(device)
            ckpt = torch.load(args.drn_checkpoint, map_location=device)
            drn.load_state_dict(ckpt["model_state_dict"])

        vae = VAE(in_ch=OUT_CH, latent_ch=LATENT_CH, base_ch=MODEL["vae_base_ch"]).to(device)
        if is_main:
            print(f"[VAE] Params: {sum(p.numel() for p in vae.parameters()):,}")

        vae = train_vae(vae, drn, train_dl, val_dl,
                        epochs=TRAIN["vae_epochs"],
                        lr=TRAIN["vae_lr"],
                        beta_max=TRAIN["vae_beta_max"],
                        beta_anneal_frac=TRAIN["vae_beta_anneal_frac"],
                        warmup_epochs=TRAIN["vae_warmup_epochs"],
                        device=device,
                        checkpoint_dir=args.checkpoint_dir,
                        resume=args.resume,
                        rank=rank, local_rank=local_rank, world_size=world_size,
                        train_sampler=train_sampler)

    # ── Diffusion ──
    if "diffusion" in stages:
        if is_main:
            print("\n" + "=" * 70)
            print("STAGE 2b: Diffusion")
            print("=" * 70)
        if "drn" not in stages:
            drn = DRN(in_ch=IN_CH, out_ch=OUT_CH, base_ch=MODEL["drn_base_ch"],
                      ch_mults=MODEL["drn_ch_mults"],
                      num_res_blocks=MODEL["drn_num_res_blocks"],
                      attn_resolutions=MODEL["drn_attn_resolutions"]).to(device)
            ckpt = torch.load(args.drn_checkpoint, map_location=device)
            drn.load_state_dict(ckpt["model_state_dict"])
        if "vae" not in stages:
            vae = VAE(in_ch=OUT_CH, latent_ch=LATENT_CH, base_ch=MODEL["vae_base_ch"]).to(device)
            ckpt = torch.load(args.vae_checkpoint, map_location=device)
            vae.load_state_dict(ckpt["model_state_dict"])

        diff_in_ch = LATENT_CH + IN_CH + LATENT_CH + 2
        diff_model = DiffusionUNet(
            in_ch=diff_in_ch, out_ch=LATENT_CH,
            base_ch=MODEL["diff_base_ch"],
            ch_mults=MODEL["diff_ch_mults"],
            num_res_blocks=MODEL["diff_num_res_blocks"],
            attn_resolutions=MODEL["diff_attn_resolutions"],
            time_dim=MODEL["diff_time_dim"],
        ).to(device)
        if is_main:
            print(f"[Diff] Params: {sum(p.numel() for p in diff_model.parameters()):,}")

        diff_model, ema = train_diffusion(
            diff_model, drn, vae, train_dl, val_dl,
            epochs=TRAIN["diff_epochs"],
            lr=TRAIN["diff_lr"],
            warmup_epochs=TRAIN["diff_warmup_epochs"],
            ema_decay=TRAIN["ema_decay"],
            p_uncond=TRAIN["p_uncond"],
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            plot_dir=args.plot_dir,
            eval_every=3,
            latent_ch=LATENT_CH,
            resume=args.resume,
            grad_accum=TRAIN.get("diff_grad_accum", 1),
            cosine_restart_period=TRAIN.get("diff_cosine_restart_period", 0),
            p_mean=TRAIN.get("diff_p_mean", -1.2),
            p_std=TRAIN.get("diff_p_std", 1.2),
            rank=rank, local_rank=local_rank, world_size=world_size,
            train_sampler=train_sampler,
        )

        if is_main:
            schedule = EDMSchedule()
            drn_rmse, final_rmse = evaluate_full_pipeline(
                drn, vae, diff_model, ema, schedule,
                eval_era5, eval_conus, args.plot_dir,
                epoch=TRAIN["diff_epochs"],
                latent_ch=LATENT_CH,
                num_sampling_steps=16, num_ensemble=4,
                device=device,
            )
            print(f"[Pipeline] DRN RMSE: {drn_rmse:.4f}, Full RMSE: {final_rmse:.4f}")

    cleanup_ddp()

    if is_main:
        total_time = time.time() - t0
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE — {total_time/3600:.1f}h")
        print(f"  Checkpoints: {args.checkpoint_dir}/")
        print(f"  Plots: {args.plot_dir}/")
        print(f"{'='*70}")

        repo_dir = Path(__file__).resolve().parent
        print("\n[Git] Committing training plots...")
        try:
            subprocess.run(["git", "add", args.plot_dir + "/"], cwd=repo_dir, check=True)
            subprocess.run(["git", "commit", "-m", "auto-commit plots"], cwd=repo_dir, check=True)
            subprocess.run(["git", "push"], cwd=repo_dir, check=True)
            print("[Git] Pushed plots successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[Git] Warning: git command failed: {e}")


if __name__ == "__main__":
    main()
