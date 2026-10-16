"""Left-right, date-matched comparison: NVIDIA CorrDiff (official physicsnemo code,
trained to completion) vs our own full pipeline (DRN + diffusion, single full-pipeline
member -- not the ensemble mean), run on the SAME 6 calendar dates CorrDiff's own
generate.py used.

IMPORTANT CAVEAT: corrdiff_output.nc's lat/lon are all-NaN (physicsnemo's dataset
adapter picks a random CONUS crop per sample with no recorded coordinates), so this is
NOT a pixel-identical comparison -- it is both pipelines' real output on the same date,
at each pipeline's own (different, unknown-vs-known) crop location. Same caveat as
results/corrdiff_comparison/ours_vs_nvidia_corrdiff_rmse.png.

Produces one PNG per date: left column = NVIDIA CorrDiff (ERA5 input / truth / its own
prediction, ensemble member 0), right column = ours (ERA5 interp / truth / DRN /
full-pipeline ensemble member 0), for T2 and PREC_ACC_NC.

Usage: python scripts/corrdiff_vs_ours_sidebyside.py
"""
import datetime as dt
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ERA5_VARS, CONUS404_VARS, PATCH_SIZE, TRAIN
from src.inference.pipeline import run_pipeline
from src.evaluation._eval_setup import load_models, load_land_mask_from_cache
from src.preprocessing.normalization import NormalizationStats
from src.preprocessing.land_mask import get_valid_patch_origins

CACHE = "/discover/nobackup/sduan/.data"
NC_PATH = "external/corrdiff/physicsnemo_src/examples/weather/corrdiff/corrdiff_output.nc"
OUT_DIR = Path("results/corrdiff_comparison/side_by_side")
PS = PATCH_SIZE
PLOT_VARS = ["T2", "PREC_ACC_NC"]


def to_phys(x, mean, std, is_era5, var_list):
    m = mean.reshape(-1, 1, 1)
    s = std.reshape(-1, 1, 1)
    p = x * s + m
    pi = var_list.index("PREC_ACC_NC") if "PREC_ACC_NC" in var_list else var_list.index("tp")
    p[pi] = np.expm1(p[pi])
    if is_era5:
        p[pi] *= 1000.0
    return p


def parse_tag(tag):
    year, day = tag.split("-day")
    year = int(year); day_idx = int(day) - 1  # "day232" = day-of-year 232 (1-indexed)
    date = dt.date(year, 1, 1) + dt.timedelta(days=day_idx)
    return year, day_idx, date


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    f = nc.Dataset(NC_PATH)
    tags = [str(t) for t in f["time"][:]]
    print("CorrDiff generation dates:", tags)

    stats = NormalizationStats(); stats.load("norm_stats.npz")
    em, es = stats.era5_mean.numpy(), stats.era5_std.numpy()
    cm, cs = stats.conus_mean.numpy(), stats.conus_std.numpy()
    static = np.load(f"{CACHE}/static_fields.npy")
    land = load_land_mask_from_cache(CACHE)
    origins = get_valid_patch_origins(land, PS, TRAIN["min_land_frac"])
    models = load_models(device=device)
    drn, vae, diff, ema, sched = models

    # Fixed patch (same crop every date, picked once) so the 6 figures are directly
    # comparable to each other on our side; NVIDIA's crop is unknown and differs per date.
    rng = np.random.default_rng(0)
    y0, x0 = origins[int(rng.integers(0, len(origins)))]
    print(f"Our fixed patch origin: ({y0},{x0})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#86847c"

    for i, tag in enumerate(tags):
        year, day_idx, date = parse_tag(tag)
        print(f"[{i}] {tag} -> {date}")

        # --- Ours: run the full pipeline (single member, not ensemble mean) ---
        era5_all = np.load(f"{CACHE}/era5_{year}.npy", mmap_mode="r")
        conus_all = np.load(f"{CACHE}/conus_{year}.npy", mmap_mode="r")
        e_raw = np.asarray(era5_all[day_idx, :, y0:y0 + PS, x0:x0 + PS]).copy()
        c_raw = np.asarray(conus_all[day_idx, :, y0:y0 + PS, x0:x0 + PS]).copy()
        static_patch = static[:, y0:y0 + PS, x0:x0 + PS]
        e_n = (e_raw - em[:, None, None]) / es[:, None, None]
        c_n = (c_raw - cm[:, None, None]) / cs[:, None, None]
        inp = np.concatenate([e_n, static_patch], axis=0)[None]
        inp_t = torch.from_numpy(inp).float().to(device)
        with torch.no_grad(), ema.apply():
            drn_pred, ens = run_pipeline(inp_t, drn, vae, diff, sched, num_steps=16,
                                          guidance_scale=0.2, num_samples=4, device=device)
        drn_pred = drn_pred[0].cpu().numpy()
        ens_member0 = ens[0][0].cpu().numpy()   # single full-pipeline member, not the mean

        drn_phys = to_phys(drn_pred.copy(), cm, cs, False, CONUS404_VARS)
        ens_phys = to_phys(ens_member0.copy(), cm, cs, False, CONUS404_VARS)
        truth_phys_ours = to_phys(c_n.astype(np.float32), cm, cs, False, CONUS404_VARS)
        era5_phys_ours = to_phys(e_n.astype(np.float32), em, es, True, ERA5_VARS)

        # --- NVIDIA CorrDiff: already-generated output for this date ---
        fig, axes = plt.subplots(len(PLOT_VARS), 6, figsize=(20, 3.6 * len(PLOT_VARS) + 0.8), dpi=130)
        for row, var in enumerate(PLOT_VARS):
            ci = CONUS404_VARS.index(var)
            ei = ERA5_VARS.index({"T2": "t2m", "PREC_ACC_NC": "tp"}[var])

            nv_truth = np.asarray(f["truth"][var][i])
            nv_era5 = np.asarray(f["input"][{"T2": "t2m", "PREC_ACC_NC": "tp"}[var]][i])
            nv_pred = np.asarray(f["prediction"][var][0, i])   # member 0, "full pipeline" not mean

            our_truth = truth_phys_ours[ci]
            our_era5 = era5_phys_ours[ei]
            our_pipe = ens_phys[ci]

            cmap = "RdYlBu_r" if var == "T2" else "Blues"
            vmin = min(nv_truth.min(), our_truth.min())
            vmax = max(nv_truth.max(), our_truth.max())

            panels = [
                (nv_era5, "NVIDIA: ERA5 input"), (nv_truth, "NVIDIA: truth"), (nv_pred, "NVIDIA: prediction"),
                (our_era5, "Ours: ERA5 interp"), (our_truth, "Ours: truth"), (our_pipe, "Ours: full pipeline"),
            ]
            for col, (field, title) in enumerate(panels):
                ax = axes[row, col]
                im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
                ax.set_title(title, fontsize=9); ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            axes[row, 0].set_ylabel(var, fontsize=11)

        fig.suptitle(f"NVIDIA CorrDiff (left 3) vs. Ours (right 3) -- {date.isoformat()}\n"
                     f"Different, unrecorded crop locations for NVIDIA's panels -- same-date, not same-pixel comparison",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"sidebyside_{date.isoformat()}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote sidebyside_{date.isoformat()}.png")

    print("Done.")


if __name__ == "__main__":
    main()
