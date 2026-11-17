"""Live wall-clock timing example for our own pipeline's inference, to set next to
CorrDiff's own logged timing (event_benchmark_output/logs/corrdiff_gen_58824533.out:
"Total time to run 4 steps and 4 members = 65.4s", "Average time per batch element =
4.04s", on a 448x448 crop). Not a perfectly matched comparison: different resolution
(256x256 here vs CorrDiff's 448x448) and CorrDiff's number is amortized over however
many batch elements it ran in that job. Reports both 4-step (CorrDiff-matched step
count) and 16-step (our usual setting) timings, 4 members each, single A100, after a
warmup call so CUDA kernel compile time doesn't pollute the measurement.

Usage: python scripts/time_inference_example.py
"""
import time
import numpy as np
import torch

from config import ERA5_VARS, CONUS404_VARS, PATCH_SIZE
from src.inference.pipeline import run_pipeline
from src.evaluation._eval_setup import load_models, load_land_mask_from_cache
from src.preprocessing.normalization import NormalizationStats

CACHE = "/discover/nobackup/sduan/.data"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    stats = NormalizationStats(); stats.load("norm_stats.npz")
    em, es = stats.era5_mean.numpy(), stats.era5_std.numpy()
    static = np.load(f"{CACHE}/static_fields.npy")
    drn, vae, diff, ema, sched = load_models(device=device)

    year, day_idx, y0, x0 = 2019, 150, 512, 960
    era5_all = np.load(f"{CACHE}/era5_{year}.npy", mmap_mode="r")
    e_raw = np.asarray(era5_all[day_idx, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]).copy()
    static_patch = static[:, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
    e_n = (e_raw - em[:, None, None]) / es[:, None, None]
    inp = np.concatenate([e_n, static_patch], axis=0)[None]
    inp_t = torch.from_numpy(inp).float().to(device)

    def timed_run(num_steps, num_samples):
        with torch.no_grad(), ema.apply():
            run_pipeline(inp_t, drn, vae, diff, sched, num_steps=num_steps,
                         guidance_scale=0.2, num_samples=num_samples, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), ema.apply():
            run_pipeline(inp_t, drn, vae, diff, sched, num_steps=num_steps,
                         guidance_scale=0.2, num_samples=num_samples, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        return time.time() - t0

    for steps in [4, 16]:
        elapsed = timed_run(steps, 4)
        print(f"Ours, 256x256 patch, {steps} steps, 4 members (1 sample's full ensemble): "
              f"{elapsed:.2f}s wall clock on one A100")

    print("\nFor reference, CorrDiff's own logged timing (job 58824533, 448x448 crop): "
          "total 64.6s for 4 steps and 4 members, average 4.04s per batch element.")


if __name__ == "__main__":
    main()
