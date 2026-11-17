"""
Generic (non-wildfire) random held-out samples, to check whether the full
diffusion pipeline's RMSE-vs-DRN pattern seen in the wildfire case studies
(diffusion sometimes *increases* RMSE relative to DRN alone) generalizes, or
was specific to those 6 events.

Picks N random (year, day, patch) samples from held-out val/test years
(2015-2020, never trained on), runs the same DRN vs full-pipeline comparison,
and saves results/wildfire_cases/baseline{i}.png plus a console summary of
whether full-pipeline RMSE > DRN RMSE on average.

Usage: python scripts/run_generic_baselines.py [--n 8] [--seed 0]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ERA5_VARS, CONUS404_VARS, PATCH_SIZE, TRAIN
from src.inference.pipeline import run_pipeline
from src.evaluation._eval_setup import load_models, load_land_mask_from_cache
from src.evaluation.metrics import power_spectrum_2d
from src.preprocessing.normalization import NormalizationStats
from src.preprocessing.land_mask import get_valid_patch_origins

CACHE = "/discover/nobackup/sduan/.data"
OUT_DIR = Path("results/wildfire_cases")
PS = PATCH_SIZE
HELD_OUT_YEARS = list(range(2015, 2021))  # val (2015-17) + test (2018-20), never trained on
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


def run_case(idx, year, day_idx, y0, x0, models, stats, static, device):
    drn, vae, diff, ema, sched = models
    era5_all = np.load(f"{CACHE}/era5_{year}.npy", mmap_mode="r")
    conus_all = np.load(f"{CACHE}/conus_{year}.npy", mmap_mode="r")
    e_raw = np.asarray(era5_all[day_idx, :, y0:y0 + PS, x0:x0 + PS]).copy()
    c_raw = np.asarray(conus_all[day_idx, :, y0:y0 + PS, x0:x0 + PS]).copy()
    static_patch = static[:, y0:y0 + PS, x0:x0 + PS]

    em, es = stats.era5_mean.numpy(), stats.era5_std.numpy()
    cm, cs = stats.conus_mean.numpy(), stats.conus_std.numpy()
    e_n = (e_raw - em[:, None, None]) / es[:, None, None]
    c_n = (c_raw - cm[:, None, None]) / cs[:, None, None]
    inp = np.concatenate([e_n, static_patch], axis=0)[None]
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

    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(PLOT_VARS), 5, figsize=(18, 3.6 * len(PLOT_VARS) + 0.6), dpi=140,
                              gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.8]})
    results = {}
    for row, var in enumerate(PLOT_VARS):
        ci = CONUS404_VARS.index(var)
        ei = ERA5_VARS.index({"T2": "t2m", "PREC_ACC_NC": "tp"}[var])
        truth_field = truth_phys[ci]
        era5_field = era5_phys[ei]
        drn_field = drn_phys[ci]
        ens_field = ens_phys[ci]
        rmse_era5 = float(np.sqrt(np.mean((era5_field - truth_field) ** 2)))
        rmse_drn = float(np.sqrt(np.mean((drn_field - truth_field) ** 2)))
        rmse_ens = float(np.sqrt(np.mean((ens_field - truth_field) ** 2)))
        results[var] = (rmse_era5, rmse_drn, rmse_ens)

        panels = [
            (era5_field, f"ERA5 input (27km)\n{var}"),
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

        bax = axes[row, 4]
        wl_truth, pw_truth = power_spectrum_2d(truth_field)
        wl_era5, pw_era5 = power_spectrum_2d(era5_field)
        wl_drn, pw_drn = power_spectrum_2d(drn_field)
        wl_ens, pw_ens = power_spectrum_2d(ens_field)
        bax.loglog(wl_truth, pw_truth, color="black", linewidth=2, label="CONUS404 truth")
        bax.loglog(wl_era5, pw_era5, color="#1baf7a", linewidth=1.5, linestyle="--", label="ERA5 (no downscale)")
        bax.loglog(wl_drn, pw_drn, color="#eb6834", linewidth=1.5, label="DRN")
        bax.loglog(wl_ens, pw_ens, color="#2a78d6", linewidth=1.5, label="Full pipeline")
        bax.set_title(f"Power spectrum\n{var}", fontsize=10)
        bax.set_xlabel("wavelength (km)")
        bax.set_ylabel("power")
        bax.invert_xaxis()
        bax.spines[["top", "right"]].set_visible(False)
        bax.legend(fontsize=7, loc="lower left", frameon=False)

    import datetime as dt
    date_str = (dt.date(year, 1, 1) + dt.timedelta(days=day_idx)).isoformat()
    fig.suptitle(f"Generic held-out sample #{idx} -- {date_str}, patch ({y0},{x0})", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"baseline{idx}.png", bbox_inches="tight")
    plt.close(fig)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    rng = np.random.default_rng(a.seed)

    stats = NormalizationStats()
    stats.load("norm_stats.npz")
    static = np.load(f"{CACHE}/static_fields.npy")
    land = load_land_mask_from_cache(CACHE)
    origins = get_valid_patch_origins(land, PS, TRAIN["min_land_frac"])
    print(f"{len(origins)} valid land patch origins available")
    models = load_models(device=device)

    all_results = {v: {"era5": [], "drn": [], "pipeline": []} for v in PLOT_VARS}
    i = 0
    tries = 0
    while i < a.n and tries < a.n * 5:
        tries += 1
        year = int(rng.choice(HELD_OUT_YEARS))
        day_idx = int(rng.integers(0, 365))
        y0, x0 = origins[int(rng.integers(0, len(origins)))]
        try:
            res = run_case(i + 1, year, day_idx, y0, x0, models, stats, static, device)
        except Exception as e:
            print(f"[skip] year={year} day={day_idx} patch=({y0},{x0}): {e}")
            plt.close("all")
            continue
        for var, (r_era5, r_drn, r_ens) in res.items():
            all_results[var]["era5"].append(r_era5)
            all_results[var]["drn"].append(r_drn)
            all_results[var]["pipeline"].append(r_ens)
        print(f"[done] baseline{i+1}: year={year} day={day_idx} patch=({y0},{x0}) -> "
              + ", ".join(f"{v}: ERA5={r[0]:.2f} DRN={r[1]:.2f} pipeline={r[2]:.2f}" for v, r in res.items()))
        i += 1

    print("\n=== Summary over", i, "generic held-out samples ===")
    for var in PLOT_VARS:
        era5_mean = np.mean(all_results[var]["era5"])
        drn_mean = np.mean(all_results[var]["drn"])
        pipe_mean = np.mean(all_results[var]["pipeline"])
        n_pipe_worse = sum(1 for d, p in zip(all_results[var]["drn"], all_results[var]["pipeline"]) if p > d)
        print(f"{var}: mean RMSE  ERA5={era5_mean:.3f}  DRN={drn_mean:.3f}  full_pipeline={pipe_mean:.3f}  "
              f"| pipeline worse than DRN in {n_pipe_worse}/{i} samples")


if __name__ == "__main__":
    main()
