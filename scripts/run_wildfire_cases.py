"""
Visual case studies for historical wildfire-weather events + general ERA5->CONUS404
downscaling samples, using the trained Latent CorrDiff pipeline (DRN + VAE + diffusion).

Wildfire events (dates/locations approximate -- from Wikipedia's List of wildfires,
coordinates are the fire's general region, not precise ignition points). All fall in
our held-out val/test years (2015-2020), never seen during training (1980-2014):
  - Okanogan Complex, WA, 2015-08-15
  - Anderson Creek Fire, KS/OK, 2016-03-22
  - 2017 Montana wildfires (Lolo Peak area), 2017-08-15
  - Spring Creek Fire, CO, 2018-06-27
  - Mendocino Complex Fire, CA, 2018-07-27
  - Bush Fire, AZ, 2020-06-13

Usage (on a GPU node):
  python scripts/run_wildfire_cases.py
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ERA5_VARS, CONUS404_VARS, PATCH_SIZE
from src.inference.pipeline import run_pipeline
from src.evaluation._eval_setup import load_models, load_land_mask_from_cache
from src.preprocessing.normalization import NormalizationStats

CACHE = "/discover/nobackup/sduan/.data"
OUT_DIR = Path("results/wildfire_cases")
PS = PATCH_SIZE

WILDFIRES = [
    # name, date, lat, lon, note
    ("Okanogan Complex (WA, 2015)", "2015-08-15", 48.6, -119.8,
     "Largest wildfire complex in WA state history, 302,224 acres"),
    ("Anderson Creek Fire (KS/OK, 2016)", "2016-03-22", 37.1, -98.8,
     "Largest wildfire in Kansas history, 367,620 acres"),
    ("2017 Montana Wildfires", "2017-08-15", 46.75, -114.2,
     "1,295,000 acres, contained by rain/snow mid-September"),
    ("Spring Creek Fire (CO, 2018)", "2018-06-27", 37.4, -105.1,
     "108,043 acres"),
    ("Mendocino Complex Fire (CA, 2018)", "2018-07-27", 39.0, -122.8,
     "459,102 acres, 229 structures destroyed"),
    ("Bush Fire (AZ, 2020)", "2020-06-13", 33.8, -111.2,
     "193,455 acres, near Theodore Roosevelt Lake"),
]

PLOT_VARS = ["T2", "PREC_ACC_NC"]  # temperature, precip -- most relevant to fire weather


def to_phys(x, mean, std, is_era5, var_list):
    m = mean.reshape(-1, 1, 1)
    s = std.reshape(-1, 1, 1)
    p = x * s + m
    pi = var_list.index("PREC_ACC_NC") if "PREC_ACC_NC" in var_list else var_list.index("tp")
    p[pi] = np.expm1(p[pi])
    if is_era5:
        p[pi] *= 1000.0
    return p


def load_grid():
    g = np.load("event_benchmark_output/data/grid_latlon.npz")
    return g["lat"], g["lon"]


def pixel_for_latlon(lat_grid, lon_grid, lat, lon):
    dist = (lat_grid - lat) ** 2 + ((lon_grid - lon) * np.cos(np.deg2rad(lat))) ** 2
    cy, cx = np.unravel_index(np.argmin(dist), dist.shape)
    H, W = lat_grid.shape
    y0 = int(np.clip(cy - PS // 2, 0, H - PS))
    x0 = int(np.clip(cx - PS // 2, 0, W - PS))
    return y0, x0


def run_case(name, date_str, lat, lon, note, models, stats, static, land, lat_grid, lon_grid, device):
    drn, vae, diff, ema, sched = models
    d = dt.date.fromisoformat(date_str)
    day_idx = (d - dt.date(d.year, 1, 1)).days
    y0, x0 = pixel_for_latlon(lat_grid, lon_grid, lat, lon)

    era5_all = np.load(f"{CACHE}/era5_{d.year}.npy", mmap_mode="r")
    conus_all = np.load(f"{CACHE}/conus_{d.year}.npy", mmap_mode="r")
    e_raw = np.asarray(era5_all[day_idx, :, y0:y0 + PS, x0:x0 + PS]).copy()
    c_raw = np.asarray(conus_all[day_idx, :, y0:y0 + PS, x0:x0 + PS]).copy()
    static_patch = static[:, y0:y0 + PS, x0:x0 + PS]

    em, es = stats.era5_mean.numpy(), stats.era5_std.numpy()
    cm, cs = stats.conus_mean.numpy(), stats.conus_std.numpy()
    e_n = (e_raw - em[:, None, None]) / es[:, None, None]
    c_n = (c_raw - cm[:, None, None]) / cs[:, None, None]
    inp = np.concatenate([e_n, static_patch], axis=0)[None]  # (1, IN_CH, H, W)
    inp_t = torch.from_numpy(inp).float().to(device)

    with torch.no_grad(), ema.apply():
        drn_pred, ens = run_pipeline(inp_t, drn, vae, diff, sched, num_steps=16,
                                      guidance_scale=0.2, num_samples=4, device=device)
    drn_pred = drn_pred[0].cpu().numpy()
    ens_mean = ens[0].mean(axis=0).cpu().numpy()

    drn_phys = to_phys(drn_pred.copy(), cm, cs, False, CONUS404_VARS)
    ens_phys = to_phys(ens_mean.copy(), cm, cs, False, CONUS404_VARS)
    truth_phys = to_phys(c_n.astype(np.float32), cm, cs, False, CONUS404_VARS)
    era5_phys = to_phys(e_n.astype(np.float32), em, es, True, ERA5_VARS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = name.split(" (")[0].replace(" ", "_")

    fig, axes = plt.subplots(len(PLOT_VARS), 4, figsize=(15, 3.6 * len(PLOT_VARS) + 0.6), dpi=140)
    if len(PLOT_VARS) == 1:
        axes = axes[None, :]
    rmse_lines = []
    for row, var in enumerate(PLOT_VARS):
        ci = CONUS404_VARS.index(var)
        ei = ERA5_VARS.index({"T2": "t2m", "PREC_ACC_NC": "tp"}[var])
        truth_field = truth_phys[ci]
        drn_field = drn_phys[ci]
        ens_field = ens_phys[ci]
        rmse_drn = float(np.sqrt(np.mean((drn_field - truth_field) ** 2)))
        rmse_ens = float(np.sqrt(np.mean((ens_field - truth_field) ** 2)))
        unit = "K" if var == "T2" else "mm"
        rmse_lines.append(
            f"{var} vs CONUS404 target: DRN RMSE = {rmse_drn:.2f}{unit}, "
            f"full pipeline RMSE = {rmse_ens:.2f}{unit}"
        )
        panels = [
            (era5_phys[ei], f"ERA5 input (27km)\n{var}"),
            (truth_field, f"CONUS404 truth (4km)\n{var}"),
            (drn_field, f"DRN mean prediction\n{var}"),
            (ens_field, f"Full pipeline (DRN+diffusion)\n{var}, 4-member mean"),
        ]
        vmin = min(p[0].min() for p in panels)
        vmax = max(p[0].max() for p in panels)
        cmap = "RdYlBu_r" if var == "T2" else "Blues"
        for col, (field, title) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"{name}, {date_str}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.text(0.5, 0.01, "   |   ".join(rmse_lines), ha="center", va="bottom", fontsize=10)
    fig.savefig(OUT_DIR / f"{safe}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {name}: wrote {OUT_DIR}/{safe}.png  ({'; '.join(rmse_lines)})")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    stats = NormalizationStats()
    stats.load("norm_stats.npz")
    static = np.load(f"{CACHE}/static_fields.npy")
    land = load_land_mask_from_cache(CACHE)
    lat_grid, lon_grid = load_grid()
    models = load_models(device=device)

    for name, date_str, lat, lon, note in WILDFIRES:
        try:
            run_case(name, date_str, lat, lon, note, models, stats, static, land,
                      lat_grid, lon_grid, device)
        except Exception as e:
            print(f"[FAILED] {name}: {e}")


if __name__ == "__main__":
    main()
