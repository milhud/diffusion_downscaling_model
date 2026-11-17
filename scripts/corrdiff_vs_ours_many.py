"""Pixel-aligned CorrDiff vs ours on many random test dates, all 6 variables, plus best-case figures.

Both pipelines use the same shared 448x448 crops (scripts/select_shared_patches.py); CorrDiff's
output is center-cropped to the 256x256 patch ours runs on. Metrics are in physical units
(precip in mm; CorrDiff's log1p-space precip is expm1'd, its ERA5 tp is expm1(.)*1000; predicted
precip clipped at 0 for BOTH models). Both models use 4 ensemble members.

Per sample/variable: RMSE of member 0, RMSE of ensemble mean, ensemble CRPS (for ERA5-interp: RMSE).
Writes: metrics CSV, summary markdown (mean/median/win-rate over ALL samples), and example
figures where ours beats CorrDiff (selected from all samples -- see summary for the base rate).

Usage (GPU): PYTHONPATH=. python scripts/corrdiff_vs_ours_many.py
Re-plot only: PYTHONPATH=. python scripts/corrdiff_vs_ours_many.py --plot_only
"""
import argparse
import csv
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ERA5_VARS, CONUS404_VARS, PATCH_SIZE

CACHE = "/discover/nobackup/sduan/.data"
RES = Path("results/corrdiff_comparison")
MANIFEST = RES / "shared_patches_many.json"
NC_PATH = RES / "corrdiff_output_many.nc"
OUT = RES / "many"
PS = PATCH_SIZE
E5KEY = {"T2": "t2m", "TD2": "d2m", "U10": "u10", "V10": "v10", "PSFC": "sp", "PREC_ACC_NC": "tp"}
LABEL = {"T2": "T2 (K)", "TD2": "TD2 (K)", "U10": "U10 (m/s)", "V10": "V10 (m/s)", "PSFC": "PSFC (Pa)", "PREC_ACC_NC": "Precip (mm/day)"}


def crps(ens, y):
    """ens (N,H,W), y (H,W): ensemble CRPS = E|X-y| - 0.5 E|X-X'| (mean over pixels)."""
    t1 = np.abs(ens - y[None]).mean()
    t2 = np.abs(ens[:, None] - ens[None]).mean()
    return float(t1 - 0.5 * t2)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def to_phys(x, mean, std, is_era5, var_list):
    """x (..., C, H, W) normalized -> physical (precip in mm)."""
    m = mean.reshape([1] * (x.ndim - 3) + [-1, 1, 1]); s = std.reshape(m.shape)
    p = x * s + m
    pi = var_list.index("PREC_ACC_NC") if "PREC_ACC_NC" in var_list else var_list.index("tp")
    p[..., pi, :, :] = np.expm1(p[..., pi, :, :])
    if is_era5:
        p[..., pi, :, :] *= 1000.0
    return p


def run_ours(tags, manifest):
    import torch
    from src.inference.pipeline import run_pipeline
    from src.evaluation._eval_setup import load_models
    from src.preprocessing.normalization import NormalizationStats

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = NormalizationStats(); stats.load("norm_stats.npz")
    em, es = stats.era5_mean.numpy(), stats.era5_std.numpy()
    cm, cs = stats.conus_mean.numpy(), stats.conus_std.numpy()
    static = np.load(f"{CACHE}/static_fields.npy")
    drn, vae, diff, ema, sched = load_models(device=dev)
    torch.manual_seed(0)
    ens_all, truth_all, era5_all = [], [], []
    for k, tag in enumerate(tags):
        m = manifest[tag]; y0, x0, d, yr = m["ours_y0"], m["ours_x0"], m["day_idx"], m["year"]
        e = np.asarray(np.load(f"{CACHE}/era5_{yr}.npy", mmap_mode="r")[d, :, y0:y0 + PS, x0:x0 + PS]).copy()
        c = np.asarray(np.load(f"{CACHE}/conus_{yr}.npy", mmap_mode="r")[d, :, y0:y0 + PS, x0:x0 + PS]).copy()
        e_n = (e - em[:, None, None]) / es[:, None, None]
        c_n = (c - cm[:, None, None]) / cs[:, None, None]
        inp = torch.from_numpy(np.concatenate([e_n, static[:, y0:y0 + PS, x0:x0 + PS]], 0)[None]).float().to(dev)
        with torch.no_grad(), ema.apply():
            _, ens = run_pipeline(inp, drn, vae, diff, sched, num_steps=16, guidance_scale=0.2, num_samples=4, device=dev)
        ens_all.append(to_phys(ens[0].cpu().numpy(), cm, cs, False, CONUS404_VARS).astype(np.float32))
        truth_all.append(to_phys(c_n.astype(np.float32), cm, cs, False, CONUS404_VARS))
        era5_all.append(to_phys(e_n.astype(np.float32), em, es, True, ERA5_VARS))
        print(f"[{k + 1}/{len(tags)}] {tag} done", flush=True)
    return np.stack(ens_all), np.stack(truth_all), np.stack(era5_all)


def load_corrdiff(tags):
    f = nc.Dataset(NC_PATH)
    ctags = [str(t) for t in f["time"][:]]
    off = (448 - PS) // 2
    cc = lambda a: np.asarray(a)[..., off:off + PS, off:off + PS]
    pi = CONUS404_VARS.index("PREC_ACC_NC")
    pred, truth, era5 = [], [], []
    for tag in tags:
        i = ctags.index(tag)
        p = np.stack([cc(f["prediction"][v][:, i]) for v in CONUS404_VARS], axis=1)   # (4,6,H,W)
        t = np.stack([cc(f["truth"][v][i]) for v in CONUS404_VARS])
        e = np.stack([cc(f["input"][E5KEY[v]][i]) for v in CONUS404_VARS])
        p[:, pi] = np.expm1(p[:, pi]); t[pi] = np.expm1(t[pi]); e[pi] = np.expm1(e[pi]) * 1000.0
        pred.append(p); truth.append(t); era5.append(e)
    return np.stack(pred), np.stack(truth), np.stack(era5)


def compute(tags, o_ens, o_truth, o_era5, c_ens, c_truth, c_era5):
    pi = CONUS404_VARS.index("PREC_ACC_NC")
    o_ens = o_ens.copy(); c_ens = c_ens.copy()
    o_ens[:, :, pi] = np.maximum(o_ens[:, :, pi], 0); c_ens[:, :, pi] = np.maximum(c_ens[:, :, pi], 0)
    rows = []
    for k, tag in enumerate(tags):
        for vi, v in enumerate(CONUS404_VARS):
            y = o_truth[k, vi]
            assert np.abs(c_truth[k, vi] - y).max() < 1e-2 * max(1.0, np.abs(y).max()), f"truth mismatch {tag} {v}"
            rows.append(dict(
                tag=tag, var=v,
                era5_rmse=rmse(o_era5[k, vi], y),
                cd_rmse_m0=rmse(c_ens[k, :, vi][0], y), ours_rmse_m0=rmse(o_ens[k, :, vi][0], y),
                cd_rmse_mean=rmse(c_ens[k, :, vi].mean(0), y), ours_rmse_mean=rmse(o_ens[k, :, vi].mean(0), y),
                cd_crps=crps(c_ens[k, :, vi], y), ours_crps=crps(o_ens[k, :, vi], y),
                truth_std=float(y.std())))
    return rows, o_ens, c_ens


def summarize(rows, path):
    lines = ["# CorrDiff vs ours: pixel-aligned, all random test samples (not cherry-picked)\n",
             f"N samples = {len(set(r['tag'] for r in rows))}, 4 members each, {PS}x{PS} shared patch.\n",
             "| var | CorrDiff RMSE (m0) | Ours RMSE (m0) | ours win % (RMSE m0) | CorrDiff CRPS | Ours CRPS | ours win % (CRPS) |",
             "|---|---|---|---|---|---|---|"]
    for v in CONUS404_VARS:
        r = [x for x in rows if x["var"] == v]
        g = lambda k: np.array([x[k] for x in r])
        lines.append(f"| {v} | {g('cd_rmse_m0').mean():.4g} | {g('ours_rmse_m0').mean():.4g} | "
                     f"{100 * (g('ours_rmse_m0') < g('cd_rmse_m0')).mean():.0f}% | {g('cd_crps').mean():.4g} | "
                     f"{g('ours_crps').mean():.4g} | {100 * (g('ours_crps') < g('cd_crps')).mean():.0f}% |")
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_example(tag, var_list, k, o_ens, o_truth, o_era5, c_ens, c_truth, c_era5, rows, path, stat="m0"):
    fig, axes = plt.subplots(len(var_list), 6, figsize=(21, 3.7 * len(var_list) + 0.9), dpi=130, squeeze=False)
    for r, v in enumerate(var_list):
        vi = CONUS404_VARS.index(v)
        y = o_truth[k, vi]; e5 = o_era5[k, vi]
        cd, ou = (c_ens[k, 0, vi], o_ens[k, 0, vi]) if stat == "m0" else (c_ens[k, :, vi].mean(0), o_ens[k, :, vi].mean(0))
        sfx = "m0" if stat == "m0" else "mean"
        row = next(x for x in rows if x["tag"] == tag and x["var"] == v)
        lo, hi = float(y.min()), float(y.max())
        if v == "PREC_ACC_NC":
            hi = max(hi, 1.0); lo = 0.0
        cmap = "Blues" if v == "PREC_ACC_NC" else "RdYlBu_r"
        err_hi = max(np.abs(cd - y).max(), np.abs(ou - y).max())
        panels = [(e5, "ERA5 input (interp)", cmap, lo, hi),
                  (y, "Truth (CONUS404)", cmap, lo, hi),
                  (cd, f"CorrDiff  RMSE {row['cd_rmse_' + sfx]:.3g}", cmap, lo, hi),
                  (ou, f"Ours  RMSE {row['ours_rmse_' + sfx]:.3g}", cmap, lo, hi),
                  (np.abs(cd - y), "|CorrDiff - truth|", "magma_r", 0, err_hi),
                  (np.abs(ou - y), "|Ours - truth|", "magma_r", 0, err_hi)]
        for c, (fld, title, cm_, a, b) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(fld, cmap=cm_, vmin=a, vmax=b, origin="upper")
            ax.set_title(title, fontsize=10); ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        axes[r, 0].text(-0.02, 0.5, LABEL[v], transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=11)
    y_, d_ = tag.split("-day")
    fig.suptitle(f"{tag} (year {y_}, cache day {int(d_)})  --  same 256x256 pixels, {'ensemble member 0' if stat == 'm0' else '4-member ensemble mean'} of each model\n"
                 f"(selected from all samples as a case where ours wins; see summary.md for the overall win rate)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot_only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(MANIFEST)); manifest.pop("_meta")
    tags = list(manifest)
    cache = OUT / "ours_cache.npz"
    if a.plot_only and cache.exists():
        z = np.load(cache); o_ens, o_truth, o_era5 = z["ens"], z["truth"], z["era5"]
    else:
        o_ens, o_truth, o_era5 = run_ours(tags, manifest)
        np.savez(cache, ens=o_ens, truth=o_truth, era5=o_era5)
    c_ens, c_truth, c_era5 = load_corrdiff(tags)
    rows, o_ens, c_ens = compute(tags, o_ens, o_truth, o_era5, c_ens, c_truth, c_era5)
    with open(OUT / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summarize(rows, OUT / "summary.md")

    # Examples: ours beats CorrDiff on member-0 RMSE, ensemble-mean RMSE AND CRPS; rank by relative RMSE gain.
    def wins(v):
        out = []
        for x in rows:
            if x["var"] == v and x["ours_rmse_m0"] < x["cd_rmse_m0"] and x["ours_rmse_mean"] < x["cd_rmse_mean"] and x["ours_crps"] < x["cd_crps"]:
                out.append((1 - x["ours_rmse_m0"] / x["cd_rmse_m0"], x["tag"]))
        return sorted(out, reverse=True)
    best = {v: wins(v) for v in ("TD2", "PREC_ACC_NC")}
    for v, lst in best.items():
        print(f"{v}: {len(lst)} samples where ours wins on all 3 metrics; top:", [(f'{g:.0%}', t) for g, t in lst[:5]])
    # one figure per variable's top win, plus one combined (TD2+TP) figure for the best date winning both
    both = {t: g for g, t in best["TD2"]}
    both = sorted([(min(both[t], g2), t) for g2, t in best["PREC_ACC_NC"] if t in both], reverse=True)
    for v, lst in best.items():
        for rank, (g, t) in enumerate(lst[:3]):
            plot_example(t, [v], tags.index(t), o_ens, o_truth, o_era5, c_ens, c_truth, c_era5, rows,
                         OUT / f"win_{v}_{rank + 1}_{t}.png")
    for rank, (g, t) in enumerate(both[:3]):
        plot_example(t, ["TD2", "PREC_ACC_NC"], tags.index(t), o_ens, o_truth, o_era5, c_ens, c_truth, c_era5, rows,
                     OUT / f"win_both_{rank + 1}_{t}.png")
    print("both-win dates:", [(f"{g:.0%}", t) for g, t in both[:5]])
    # Relaxed criterion for precip (no date wins member-0 RMSE too): ens-mean RMSE + CRPS, TD2 wins all 3.
    for t in ("2019-day136",):
        if t in tags:
            plot_example(t, ["TD2", "PREC_ACC_NC"], tags.index(t), o_ens, o_truth, o_era5, c_ens, c_truth, c_era5, rows,
                         OUT / f"win_combo_ensmean_{t}.png", stat="mean")


if __name__ == "__main__":
    main()
