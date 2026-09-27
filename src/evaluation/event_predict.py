"""Run ERA5-interp / DRN / latent-diffusion ensemble on event windows (GPU).

DRN is blended over 3x3 overlapping tiles; diffusion residuals use 2x2 non-overlapping tiles.
For each event, processes days [d-2, d+2] (temporal analysis) over a 512x512 window
(2x2 tiles of 256). Saves per-event npz with physical-unit arrays:
  era5 (D,6,H,W)  truth (D,6,H,W)  drn (D,6,H,W)  ens (D,N,6,H,W)  land (H,W)
Precip is stored in mm (expm1 of the log1p cache/model space); ERA5 tp m->mm.

Usage: python -m src.evaluation.event_predict --members 8 --steps 16
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from config import ERA5_VARS, CONUS404_VARS, OUT_CH
from src.inference.pipeline import run_pipeline
from src.data.normalization import invert_pretransform
from src.evaluation._eval_setup import load_models, load_land_mask_from_cache
from src.preprocessing.normalization import NormalizationStats

CACHE = "/discover/nobackup/sduan/.data"
PS = 256
OFFSETS = [-2, -1, 0, 1, 2]
_w = np.sin(np.pi * (np.arange(PS) + 0.5) / PS) ** 2 + 1e-3
WIN2D = np.outer(_w, _w).astype(np.float32)       # separable Hann-type blend weight


def to_phys(x, stats_mean, stats_std, is_era5):
    """x: (..., 6, H, W) normalized -> physical units (precip in mm)."""
    m = stats_mean.reshape([1] * (x.ndim - 3) + [-1, 1, 1])
    s = stats_std.reshape([1] * (x.ndim - 3) + [-1, 1, 1])
    p = x * s + m
    pi = CONUS404_VARS.index("PREC_ACC_NC")
    p[..., pi, :, :] = np.expm1(p[..., pi, :, :])
    if is_era5:
        p[..., pi, :, :] *= 1000.0
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--out", default="event_benchmark_output/data")
    ap.add_argument("--events", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    cat = json.load(open(Path(a.out) / "event_catalog.json"))
    if a.events:
        cat = [e for e in cat if e["name"] in a.events]

    stats = NormalizationStats(); stats.load("norm_stats.npz")
    em, es = stats.era5_mean.numpy(), stats.era5_std.numpy()
    cm, cs = stats.conus_mean.numpy(), stats.conus_std.numpy()
    static = np.load(f"{CACHE}/static_fields.npy")          # (6,H,W), z-scored, ch5=lsm
    land_full = load_land_mask_from_cache(CACHE)
    drn, vae, diff, ema, sched = load_models(device=dev)

    for ev in cat:
        t0 = time.time()
        y0, x0, S = ev["y0"], ev["x0"], ev["size"]
        yr = np.load(f"{CACHE}/era5_{ev['year']}.npy", mmap_mode="r")
        yc = np.load(f"{CACHE}/conus_{ev['year']}.npy", mmap_mode="r")
        land = land_full[y0:y0 + S, x0:x0 + S]
        days = [ev["day_index"] + o for o in OFFSETS if 0 <= ev["day_index"] + o < yr.shape[0]]
        E5, TR, DR, EN = [], [], [], []
        for d in days:
            e_raw = np.asarray(yr[d, :, y0:y0 + S, x0:x0 + S]).copy()    # (6,S,S)
            c_raw = np.asarray(yc[d, :, y0:y0 + S, x0:x0 + S]).copy()
            # normalize (stats are on pretransformed cache values)
            e_n = (e_raw - em[:, None, None]) / es[:, None, None]
            c_n = (c_raw - cm[:, None, None]) / cs[:, None, None]
            drn_o = np.zeros_like(c_n); ens_o = np.zeros((a.members,) + c_n.shape, np.float32)

            def prep(ty, tx):
                sl = (slice(ty, ty + PS), slice(tx, tx + PS))
                pl = land[sl]
                ep = np.concatenate([e_n[:, sl[0], sl[1]],
                                     static[:, y0 + ty:y0 + ty + PS, x0 + tx:x0 + tx + PS]], 0)
                if not pl.all() and pl.any():   # training-style fill of non-land
                    for c in range(OUT_CH):
                        ep[c][~pl] = ep[c][pl].mean()
                return sl, torch.from_numpy(ep[None]).float().to(dev)

            # (1) DRN: 3x3 overlapping tiles (stride 128) with smooth blending -> no tile seams.
            acc = np.zeros_like(c_n); wsum = np.zeros((S, S), np.float32)
            for ty in (0, PS // 2, PS):
                for tx in (0, PS // 2, PS):
                    sl, xin = prep(ty, tx)
                    with torch.no_grad():
                        dp = drn(xin)[0].cpu().numpy()
                    acc[:, sl[0], sl[1]] += dp * WIN2D
                    wsum[sl] += WIN2D
            drn_o = acc / wsum
            # (2) Diffusion residual on NON-overlapping tiles (blending independent samples
            #     would artificially shrink ensemble variance). final = blended DRN + residual.
            for ty in (0, PS):
                for tx in (0, PS):
                    sl, xin = prep(ty, tx)
                    with torch.no_grad(), ema.apply():
                        dp, sm = run_pipeline(xin, drn, vae, diff, sched, num_steps=a.steps,
                                              num_samples=a.members, device=dev)
                    resid = sm[0].cpu().numpy() - dp[0].cpu().numpy()[None]    # (N,C,PS,PS)
                    ens_o[:, :, sl[0], sl[1]] = drn_o[None, :, sl[0], sl[1]] + resid
            E5.append(to_phys(e_n.astype(np.float32), em, es, True))
            TR.append(to_phys(c_n.astype(np.float32), cm, cs, False))
            DR.append(to_phys(drn_o, cm, cs, False))
            EN.append(to_phys(ens_o, cm, cs, False))
        np.savez_compressed(Path(a.out) / f"pred_{ev['name']}.npz",
                            era5=np.stack(E5).astype(np.float32), truth=np.stack(TR).astype(np.float32),
                            drn=np.stack(DR).astype(np.float32), ens=np.stack(EN).astype(np.float32),
                            land=land, days=np.array(days), vars=np.array(CONUS404_VARS))
        print(f"[{ev['name']}] {len(days)} days x {a.members} members done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
