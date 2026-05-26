"""Computational cost benchmarking.

Measures GPU time, memory, and throughput for each pipeline stage
and compares latent vs pixel-space diffusion.

Usage:
    python -m src.evaluation.compute_benchmark --device cuda
"""

import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from config import IN_CH, OUT_CH, PATCH_SIZE, LATENT_CH, MODEL, TRAIN


def benchmark_stage(fn, name, warmup=3, repeats=20):
    """Time a function with CUDA synchronization."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    print(f"  {name}: {times.mean()*1000:.1f} +/- {times.std()*1000:.1f} ms "
          f"(min={times.min()*1000:.1f}, max={times.max()*1000:.1f})")
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", default="results/benchmark")
    parser.add_argument("--num_steps", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--drn_checkpoint",  default="checkpoints/drn_best.pt")
    parser.add_argument("--vae_checkpoint",  default="checkpoints/vae_best.pt")
    parser.add_argument("--diff_checkpoint", default="checkpoints/diffusion_best.pt")
    parser.add_argument("--data_dir",  default="data")
    parser.add_argument("--cache_dir", default="cached_data")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    B = args.batch_size

    from src.models.drn import DRN
    from src.models.vae import VAE
    from src.models.diffusion_unet import DiffusionUNet
    from src.models.edm import EDMSchedule, heun_sampler
    from src.evaluation._eval_setup import build_test_dataloader, load_models

    # Load trained models
    drn, vae, latent_diff, ema, schedule = load_models(
        args.drn_checkpoint, args.vae_checkpoint, args.diff_checkpoint, device)
    drn.eval(); vae.eval(); latent_diff.eval()

    # Pixel diffusion (same architecture but different I/O sizes — random weights for timing)
    pixel_diff_in_ch = OUT_CH + IN_CH + OUT_CH + 2
    pixel_diff = DiffusionUNet(
        in_ch=pixel_diff_in_ch, out_ch=OUT_CH, base_ch=MODEL["diff_base_ch"],
        ch_mults=MODEL["diff_ch_mults"], num_res_blocks=MODEL["diff_num_res_blocks"],
        attn_resolutions=MODEL["diff_attn_resolutions"],
        time_dim=MODEL["diff_time_dim"]).to(device).eval()

    # Real inputs from test set (one batch)
    test_dl, _, _, _ = build_test_dataloader(
        data_dir=args.data_dir, cache_dir=args.cache_dir, batch_size=B, num_workers=2)
    era5_input, conus_target = next(iter(test_dl))
    era5_input = era5_input[:B].to(device)
    conus_target = conus_target[:B].to(device)
    print(f"Using real test data: era5={tuple(era5_input.shape)}, conus={tuple(conus_target.shape)}")

    print(f"\nBenchmark: batch_size={B}, patch={PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  DRN params: {sum(p.numel() for p in drn.parameters()):,}")
    print(f"  VAE params: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"  Latent Diff params: {sum(p.numel() for p in latent_diff.parameters()):,}")
    print(f"  Pixel Diff params: {sum(p.numel() for p in pixel_diff.parameters()):,}")

    results = {}

    # DRN forward
    print("\n--- DRN ---")
    with torch.no_grad():
        t = benchmark_stage(lambda: drn(era5_input), "DRN forward")
        results["drn_ms"] = t.mean() * 1000

    # VAE encode + decode
    print("\n--- VAE ---")
    with torch.no_grad():
        drn_pred = drn(era5_input)
        residual = conus_target - drn_pred
        t_enc = benchmark_stage(lambda: vae.encode(residual), "VAE encode")
        mu, _ = vae.encode(residual)
        t_dec = benchmark_stage(lambda: vae.decode(mu), "VAE decode")
        results["vae_encode_ms"] = t_enc.mean() * 1000
        results["vae_decode_ms"] = t_dec.mean() * 1000

    # Latent diffusion sampling at various step counts
    print("\n--- Latent Diffusion (64x64) ---")
    with torch.no_grad():
        era5_down = F.interpolate(era5_input, size=(64, 64), mode="bilinear", align_corners=False)
        mu_drn, _ = vae.encode(drn_pred)
        ys = torch.linspace(-1, 1, 64, device=device)
        xs = torch.linspace(-1, 1, 64, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([yy, xx], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        latent_cond = torch.cat([era5_down, mu_drn, pos], dim=1)

        for ns in args.num_steps:
            t = benchmark_stage(
                lambda: heun_sampler(latent_diff, schedule, latent_cond,
                                     shape=(B, LATENT_CH, 64, 64), num_steps=ns),
                f"Latent sample ({ns} steps)", warmup=2, repeats=5)
            results[f"latent_{ns}steps_ms"] = t.mean() * 1000

    # Pixel diffusion sampling at various step counts
    print("\n--- Pixel Diffusion (256x256) ---")
    with torch.no_grad():
        pos256 = torch.stack([
            torch.linspace(-1, 1, 256, device=device).unsqueeze(1).expand(256, 256),
            torch.linspace(-1, 1, 256, device=device).unsqueeze(0).expand(256, 256),
        ], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        pixel_cond = torch.cat([era5_input, drn_pred, pos256], dim=1)

        for ns in args.num_steps:
            t = benchmark_stage(
                lambda: heun_sampler(pixel_diff, schedule, pixel_cond,
                                     shape=(B, OUT_CH, 256, 256), num_steps=ns),
                f"Pixel sample ({ns} steps)", warmup=2, repeats=5)
            results[f"pixel_{ns}steps_ms"] = t.mean() * 1000

    # GPU memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        heun_sampler(latent_diff, schedule, latent_cond,
                     shape=(B, LATENT_CH, 64, 64), num_steps=32)
    latent_mem = torch.cuda.max_memory_allocated() / 1e9

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        heun_sampler(pixel_diff, schedule, pixel_cond,
                     shape=(B, OUT_CH, 256, 256), num_steps=32)
    pixel_mem = torch.cuda.max_memory_allocated() / 1e9

    results["latent_peak_mem_gb"] = latent_mem
    results["pixel_peak_mem_gb"] = pixel_mem

    print(f"\n--- Memory ---")
    print(f"  Latent (64x64): {latent_mem:.2f} GB peak")
    print(f"  Pixel (256x256): {pixel_mem:.2f} GB peak")

    # Speedup summary
    print(f"\n--- Speedup Summary (32 steps) ---")
    if "latent_32steps_ms" in results and "pixel_32steps_ms" in results:
        speedup = results["pixel_32steps_ms"] / results["latent_32steps_ms"]
        print(f"  Latent vs Pixel speedup: {speedup:.1f}x")
        results["speedup_32steps"] = speedup

    # Save
    np.savez(out / "benchmark_results.npz", **results)

    with open(out / "benchmark_summary.txt", "w") as f:
        f.write("Computational Benchmark\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Batch size: {B}\n")
        f.write(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}\n\n")
        for k, v in sorted(results.items()):
            f.write(f"{k}: {v:.2f}\n")

    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
