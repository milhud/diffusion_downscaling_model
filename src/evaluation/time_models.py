"""Inference cost per 256x256 tile per ensemble member, 16 Heun steps, A100: latent (ours) vs pixel-space baselines."""
import json, time
import torch
from config import OUT_CH, IN_CH, PATCH_SIZE
from src.evaluation._eval_setup import load_models
from src.inference.pipeline import run_pipeline
from src.models.edm import EDMSchedule, heun_sampler
from src.evaluation.train_baselines import build_model, COND_CH


def bench(fn, n=5):
    fn(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000, torch.cuda.max_memory_allocated() / 1e9


dev = "cuda"; out = {}
drn, vae, diff, ema, sched = load_models(device=dev)
x = torch.randn(1, IN_CH, PATCH_SIZE, PATCH_SIZE, device=dev)
with torch.no_grad():
    ms, gb = bench(lambda: run_pipeline(x, drn, vae, diff, sched, num_steps=16, num_samples=1, device=dev))
out["ours_latent_fp32_incl_DRN_VAE"] = dict(ms_per_member=ms, peak_gb=gb)
pix = build_model().to(dev).eval(); sc = EDMSchedule()
cond = torch.randn(1, COND_CH, PATCH_SIZE, PATCH_SIZE, device=dev)
with torch.no_grad():
    for name, ac in [("pixel_baseline_fp32", False), ("pixel_baseline_bf16", True)]:
        def f():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ac):
                heun_sampler(pix, sc, cond, (1, OUT_CH, PATCH_SIZE, PATCH_SIZE), num_steps=16, guidance_scale=0.0)
        ms, gb = bench(f); out[name] = dict(ms_per_member=ms, peak_gb=gb)
out["note"] = "pixel baselines = 41.5M-param U-Net (base 96), no CFG (1 network eval per step); ours = 142M latent U-Net with CFG (2 evals/step). Excludes the DRN for CorrDiff-style (~19 ms)."
json.dump(out, open("event_benchmark_output/data/timing_models.json", "w"), indent=1)
print(json.dumps(out, indent=1))
