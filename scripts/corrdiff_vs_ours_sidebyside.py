"""Left-right, date-matched comparison: CorrDiff (official physicsnemo code,
trained to completion) vs our own full pipeline (DRN + diffusion, single full-pipeline
member -- not the ensemble mean), run on the SAME 6 calendar dates CorrDiff's own
generate.py used.

PIXEL-ALIGNED: both pipelines use the per-date crops in shared_patches.json
(scripts/select_shared_patches.py). CorrDiff ran on the full 448x448 crop (its dataset reads
the manifest); ours runs on the central 256x256 of that same crop, and CorrDiff's output is
center-cropped to those same pixels. A truth-vs-truth check per date verifies alignment.

Produces one PNG per date: left column = CorrDiff (ERA5 input / truth / its own
prediction, ensemble member 0), right column = ours (ERA5 interp / truth / DRN /
full-pipeline ensemble member 0), for T2 and PREC_ACC_NC.

Usage: python scripts/corrdiff_vs_ours_sidebyside.py
"""
import datetime as dt
import json
from pathlib import Path

import netCDF4 as nc
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
MANIFEST = "results/corrdiff_comparison/shared_patches.json"
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
    year = int(year); day_idx = int(day)  # 0-based cache index, as Era5Conus404Dataset.time() emits
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
    models = load_models(device=device)
    drn, vae, diff, ema, sched = models

    manifest = json.load(open(MANIFEST))
    off = manifest["_meta"]["ours_offset"]
    assert manifest["_meta"]["ours_size"] == PS

    def cc(a):  # center-crop CorrDiff's 448 window to our 256 patch
        return np.asarray(a)[..., off:off + PS, off:off + PS]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#86847c"
    rmse_rows = []

    for i, tag in enumerate(tags):
        year, day_idx, date = parse_tag(tag)
        print(f"[{i}] {tag} -> {date}")
        y0, x0 = manifest[tag]["ours_y0"], manifest[tag]["ours_x0"]

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

        # --- Alignment check: CorrDiff's truth must equal our truth on the shared pixels ---
        for v in ("T2", "U10"):
            ours_t = truth_phys_ours[CONUS404_VARS.index(v)]
            d = np.abs(cc(f["truth"][v][i]) - ours_t).max()
            print(f"  truth alignment max|diff| {v}: {d:.3g}")
            assert d < 1e-2 * max(1.0, np.abs(ours_t).max()), \
                f"CorrDiff and our truth differ for {tag} -- crops are not aligned"

        # --- CorrDiff: already-generated output for this date ---
        fig, axes = plt.subplots(len(PLOT_VARS), 6, figsize=(20, 3.6 * len(PLOT_VARS) + 0.8), dpi=130)
        for row, var in enumerate(PLOT_VARS):
            ci = CONUS404_VARS.index(var)
            ei = ERA5_VARS.index({"T2": "t2m", "PREC_ACC_NC": "tp"}[var])

            cd_truth = cc(f["truth"][var][i])
            cd_era5 = cc(f["input"][{"T2": "t2m", "PREC_ACC_NC": "tp"}[var]][i])
            cd_pred = cc(f["prediction"][var][0, i])   # member 0, "full pipeline" not mean

            our_truth = truth_phys_ours[ci]
            our_era5 = era5_phys_ours[ei]
            our_pipe = ens_phys[ci]

            cmap = "RdYlBu_r" if var == "T2" else "Blues"
            vmin = min(cd_truth.min(), our_truth.min())
            vmax = max(cd_truth.max(), our_truth.max())

            panels = [
                (cd_era5, "CorrDiff: ERA5 input"), (cd_truth, "CorrDiff: truth"), (cd_pred, "CorrDiff: prediction"),
                (our_era5, "Ours: ERA5 interp"), (our_truth, "Ours: truth"), (our_pipe, "Ours: full pipeline"),
            ]
            for col, (field, title) in enumerate(panels):
                ax = axes[row, col]
                im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
                ax.set_title(title, fontsize=9); ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            axes[row, 0].set_ylabel(var, fontsize=11)

        fig.suptitle(f"CorrDiff (left 3) vs. Ours (right 3) -- {date.isoformat()}\n"
                     f"Same pixels: shared crop origin ({y0},{x0}), {PS}x{PS} patch",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"sidebyside_{date.isoformat()}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote sidebyside_{date.isoformat()}.png")

        # --- RMSE for ALL 6 variables (not just the 2 plotted above) ---
        for var in CONUS404_VARS:
            ci = CONUS404_VARS.index(var)
            era5_key = {"T2": "t2m", "TD2": "d2m", "U10": "u10", "V10": "v10",
                        "PSFC": "sp", "PREC_ACC_NC": "tp"}[var]
            ei = ERA5_VARS.index(era5_key)

            cd_truth = cc(f["truth"][var][i])
            cd_era5 = cc(f["input"][era5_key][i])
            cd_pred = cc(f["prediction"][var][0, i])
            our_truth = truth_phys_ours[ci]
            our_era5 = era5_phys_ours[ei]
            our_pipe = ens_phys[ci]

            rmse_rows.append(dict(
                date=date.isoformat(), var=var,
                corrdiff_era5_rmse=float(np.sqrt(np.mean((cd_era5 - cd_truth) ** 2))),
                corrdiff_pipeline_rmse=float(np.sqrt(np.mean((cd_pred - cd_truth) ** 2))),
                ours_era5_rmse=float(np.sqrt(np.mean((our_era5 - our_truth) ** 2))),
                ours_pipeline_rmse=float(np.sqrt(np.mean((our_pipe - our_truth) ** 2))),
            ))

    import csv
    with open(OUT_DIR.parent / "allvars_rmse.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rmse_rows[0])); w.writeheader(); w.writerows(rmse_rows)
    print(f"wrote {OUT_DIR.parent / 'allvars_rmse.csv'}")

    print("Done.")


if __name__ == "__main__":
    main()
