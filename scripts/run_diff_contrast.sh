#!/bin/bash
#SBATCH --job-name=diff_contrast
#SBATCH --partition=gpu_a100
#SBATCH --qos=alla100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=rome
#SBATCH --time=00:30:00
#SBATCH --output=fast_eval_output.%j
#SBATCH --error=fast_eval_error.%j

set -euo pipefail
cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

python - <<'EOF'
import torch, numpy as np
from pathlib import Path
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CONUS404_VARS, VARIABLE_NAMES, VARIABLE_UNITS, OUT_CH
from src.inference.pipeline import run_pipeline
from src.evaluation._eval_setup import build_test_dataloader, load_models

CACHE_DIR = "/discover/nobackup/sduan/.data"
VARS = ["T2", "U10", "TD2"]
VAR_IDX = [CONUS404_VARS.index(v) for v in VARS]
CMAPS = {"T2": "RdBu_r", "TD2": "BrBG", "U10": "PuOr"}

test_dl, stats, _, _ = build_test_dataloader(cache_dir=CACHE_DIR, batch_size=1, num_workers=2)
drn, vae, diff_model, ema, schedule = load_models(device="cuda")

candidates = []
with torch.no_grad(), ema.apply():
    for i, (era5, conus) in enumerate(test_dl):
        if i >= 150: break
        era5, conus = era5.cuda(), conus.cuda()
        drn_pred, samples = run_pipeline(era5, drn, vae, diff_model, schedule,
                                         num_steps=16, num_samples=4, device="cuda")
        ens_mean = samples[0].mean(dim=0)
        # score = mean absolute diff between diffusion output and DRN (how much diffusion added)
        diff_contribution = float((ens_mean - drn_pred[0]).abs().mean().cpu())
        era5_up = F.interpolate(era5[:, :OUT_CH], (256,256), mode="bilinear", align_corners=False)
        candidates.append((diff_contribution,
                           era5_up[0].cpu().numpy(),
                           conus[0].cpu().numpy(),
                           drn_pred[0].cpu().numpy(),
                           samples[0].cpu().numpy()))
        print(f"  [{i+1}/150] diff_contrib={diff_contribution:.4f}")

candidates.sort(key=lambda x: -x[0])  # highest diff contribution first
best = candidates[:6]
print("\nTop 6 by diffusion contribution:")
for r, (s, *_) in enumerate(best): print(f"  rank {r+1}: {s:.4f}")

out = Path("results/inference_plots_diff_contrast"); out.mkdir(parents=True, exist_ok=True)
col_titles = ["ERA5 (interp)", "Target", "DRN", "Ensemble mean", "Error (Ens−Target)"]

for si, (score, era5_np, conus_np, drn_np, ens_np) in enumerate(best):
    n_vars = len(VARS)
    fig, axes = plt.subplots(n_vars, 5, figsize=(19, 3.5*n_vars))
    ens_mean = ens_np.mean(axis=0)
    for vi, v in enumerate(VARS):
        idx = CONUS404_VARS.index(v)
        cm = float(stats.conus_mean[idx]); cs = float(stats.conus_std[idx])
        em = float(stats.era5_mean[idx]) if idx < len(stats.era5_mean) else cm
        es = float(stats.era5_std[idx])  if idx < len(stats.era5_std)  else cs
        e_phys   = era5_np[VAR_IDX[vi]] * es + em
        tgt_phys = conus_np[VAR_IDX[vi]] * cs + cm
        drn_phys = drn_np[VAR_IDX[vi]]   * cs + cm
        dif_phys = ens_mean[VAR_IDX[vi]] * cs + cm
        err_phys = dif_phys - tgt_phys
        cmap = CMAPS.get(v, "viridis")
        unit = VARIABLE_UNITS.get(v, "")
        panels = [e_phys, tgt_phys, drn_phys, dif_phys, err_phys]
        for ci, (ax, data) in enumerate(zip(axes[vi], panels)):
            vmin2, vmax2 = np.nanpercentile(tgt_phys, 2), np.nanpercentile(tgt_phys, 98)
            if ci == 4:
                ev = np.nanpercentile(np.abs(err_phys), 95)
                im = ax.imshow(data, origin="lower", cmap="RdBu_r", vmin=-ev, vmax=ev, aspect="auto")
            else:
                im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin2, vmax=vmax2, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=unit)
            if vi == 0: ax.set_title(col_titles[ci], fontsize=10, fontweight="bold")
            if ci == 0: ax.set_ylabel(VARIABLE_NAMES.get(v, v), fontsize=9)
    fig.suptitle(f"Diffusion contribution score: {score:.4f}", fontsize=11)
    fig.tight_layout()
    path = out / f"sample_{si+1:02d}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {path}")

print("Done.")
EOF

echo "Finished: $(date)"
