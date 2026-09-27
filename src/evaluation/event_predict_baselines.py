"""Run the CorrDiff-style / R2-D2-style baselines on the same event windows as event_predict.py.

Identical protocol to our latent model: same days, same 2x2 non-overlapping residual tiles, 8 members,
16 Heun steps; mean field (DRN blended over 3x3 tiles for corrdiff; interpolated ERA5 for r2d2).
Writes predb_<variant>_<event>.npz with ens (D,N,6,H,W) and mu (D,6,H,W) in physical units, plus timing.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from config import CONUS404_VARS, OUT_CH
from src.models.edm import EDMSchedule, heun_sampler
from src.training.ema import EMA
from src.preprocessing.normalization import NormalizationStats
from src.evaluation._eval_setup import load_land_mask_from_cache
from src.evaluation.event_predict import to_phys, PS, OFFSETS, WIN2D, CACHE
from src.evaluation.train_baselines import (build_model, load_drn, mean_field, pos_embedding)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["corrdiff", "r2d2"], required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4, help="members sampled per forward batch")
    ap.add_argument("--out", default="event_benchmark_output/data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--events", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    ck = torch.load(a.ckpt or f"checkpoints/baselines/{a.variant}_best.pt", map_location=dev, weights_only=False)
    model = build_model().to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    ema = EMA(model, decay=0.999); ema.load_state_dict(ck["ema_state_dict"])
    rstd = ck["rstd"].to(dev)
    print(f"[{a.variant}] checkpoint step {ck['step']} val {ck['val_loss']:.4f} trained {ck['hours']:.1f} h", flush=True)
    drn = load_drn(dev) if a.variant == "corrdiff" else None
    sched = EDMSchedule()
    pos = pos_embedding(PS, PS, dev)

    stats = NormalizationStats(); stats.load("norm_stats.npz")
    em, es = stats.era5_mean.numpy(), stats.era5_std.numpy()
    cm, cs = stats.conus_mean.numpy(), stats.conus_std.numpy()
    static = np.load(f"{CACHE}/static_fields.npy")
    land_full = load_land_mask_from_cache(CACHE)
    cat = json.load(open(Path(a.out) / "event_catalog.json"))
    if a.events:
        cat = [e for e in cat if e["name"] in a.events]
    timing = []

    for ev in cat:
        t0 = time.time()
        y0, x0, S = ev["y0"], ev["x0"], ev["size"]
        yr = np.load(f"{CACHE}/era5_{ev['year']}.npy", mmap_mode="r")
        land = land_full[y0:y0 + S, x0:x0 + S]
        days = [ev["day_index"] + o for o in OFFSETS if 0 <= ev["day_index"] + o < yr.shape[0]]
        MU, EN = [], []
        for d in days:
            e_n = (np.asarray(yr[d, :, y0:y0 + S, x0:x0 + S]) - em[:, None, None]) / es[:, None, None]

            def prep(ty, tx):
                sl = (slice(ty, ty + PS), slice(tx, tx + PS)); pl = land[sl]
                ep = np.concatenate([e_n[:, sl[0], sl[1]], static[:, y0 + ty:y0 + ty + PS, x0 + tx:x0 + tx + PS]], 0)
                if not pl.all() and pl.any():
                    for c in range(OUT_CH):
                        ep[c][~pl] = ep[c][pl].mean()
                return sl, torch.from_numpy(ep[None]).float().to(dev)

            # mean field over the whole window (same construction as the latent model's DRN blend)
            if a.variant == "corrdiff":
                acc = np.zeros((OUT_CH, S, S), np.float32); wsum = np.zeros((S, S), np.float32)
                for ty in (0, PS // 2, PS):
                    for tx in (0, PS // 2, PS):
                        sl, xin = prep(ty, tx)
                        with torch.no_grad():
                            acc[:, sl[0], sl[1]] += drn(xin)[0].cpu().numpy() * WIN2D
                        wsum[sl] += WIN2D
                mu_w = acc / wsum
            else:
                with torch.no_grad():
                    from src.evaluation.train_baselines import era5_to_conus_norm
                    mu_w = era5_to_conus_norm(torch.from_numpy(e_n[None, :OUT_CH]).float().to(dev), stats)[0].cpu().numpy()

            ens = np.zeros((a.members, OUT_CH, S, S), np.float32)
            for ty in (0, PS):
                for tx in (0, PS):
                    sl, xin = prep(ty, tx)
                    mu_t = torch.from_numpy(mu_w[None, :, sl[0], sl[1]]).float().to(dev)
                    cond1 = torch.cat([xin, mu_t, pos[None]], 1)
                    outs = []
                    for c0 in range(0, a.members, a.chunk):
                        n = min(a.chunk, a.members - c0)
                        torch.cuda.synchronize(); ts = time.time()
                        with torch.no_grad(), ema.apply(), torch.autocast("cuda", dtype=torch.bfloat16):
                            r = heun_sampler(model, sched, cond1.expand(n, -1, -1, -1).contiguous(),
                                             (n, OUT_CH, PS, PS), num_steps=a.steps, guidance_scale=0.0)
                        torch.cuda.synchronize(); timing.append((time.time() - ts) / n)
                        outs.append((r.float() * rstd).cpu().numpy())
                    ens[:, :, sl[0], sl[1]] = mu_w[None, :, sl[0], sl[1]] + np.concatenate(outs, 0)
            MU.append(to_phys(mu_w.astype(np.float32), cm, cs, False))
            EN.append(to_phys(ens, cm, cs, False))
        np.savez_compressed(Path(a.out) / f"predb{a.tag}_{a.variant}_{ev['name']}.npz",
                            mu=np.stack(MU).astype(np.float32), ens=np.stack(EN).astype(np.float32),
                            land=land, days=np.array(days))
        print(f"[{a.variant}:{ev['name']}] done in {time.time()-t0:.0f}s", flush=True)
    json.dump(dict(variant=a.variant, sec_per_member_tile_256=float(np.mean(timing)), steps=a.steps),
              open(Path(a.out) / f"timing_{a.variant}.json", "w"))


if __name__ == "__main__":
    main()
