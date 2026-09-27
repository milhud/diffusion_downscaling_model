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
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    B = args.batch_size

    from src.models.drn import DRN
    from src.models.vae import VAE
    from src.models.diffusion_unet import DiffusionUNet
    from src.models.edm import EDMSchedule, heun_sampler

    # Build models
    drn = DRN(in_ch=IN_CH, out_ch=OUT_CH, base_ch=MODEL["drn_base_ch"],
              ch_mults=MODEL["drn_ch_mults"], num_res_blocks=MODEL["drn_num_res_blocks"],
              attn_resolutions=MODEL["drn_attn_resolutions"]).to(device).eval()

    vae = VAE(in_ch=OUT_CH, latent_ch=LATENT_CH, base_ch=MODEL["vae_base_ch"]).to(device).eval()

    # Latent diffusion
    diff_in_ch = LATENT_CH + IN_CH + LATENT_CH + 2
    latent_diff = DiffusionUNet(
        in_ch=diff_in_ch, out_ch=LATENT_CH, base_ch=MODEL["diff_base_ch"],
        ch_mults=MODEL["diff_ch_mults"], num_res_blocks=MODEL["diff_num_res_blocks"],
        attn_resolutions=MODEL["diff_attn_resolutions"],
        time_dim=MODEL["diff_time_dim"]).to(device).eval()

    # Pixel diffusion (same architecture but different I/O sizes)
    pixel_diff_in_ch = OUT_CH + IN_CH + OUT_CH + 2
    pixel_diff = DiffusionUNet(
        in_ch=pixel_diff_in_ch, out_ch=OUT_CH, base_ch=MODEL["diff_base_ch"],
        ch_mults=MODEL["diff_ch_mults"], num_res_blocks=MODEL["diff_num_res_blocks"],
        attn_resolutions=MODEL["diff_attn_resolutions"],
        time_dim=MODEL["diff_time_dim"]).to(device).eval()

    schedule = EDMSchedule()

    # Dummy inputs
    era5_input = torch.randn(B, IN_CH, PATCH_SIZE, PATCH_SIZE, device=device)
    conus_target = torch.randn(B, OUT_CH, PATCH_SIZE, PATCH_SIZE, device=device)

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
