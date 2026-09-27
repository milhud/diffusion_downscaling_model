"""Train re-implementations of the two published designs on OUR data, for benchmarking.

These are NOT the authors' released models (none are available, both are region-specific);
they are faithful-in-structure, budget-limited re-implementations on ERA5 -> CONUS404:

  corrdiff : Mardani et al. 2025. Regression UNet mean (our DRN, frozen) + PIXEL-space EDM diffusion
             on the residual  r = x - mu(y).  Conditioned on [ERA5+static, mu, pos]. No VAE, no CFG.
  r2d2     : Lopez-Gomez et al. 2025. PIXEL-space EDM diffusion on the residual between the fine field
             and the interpolated coarse field, r = x - interp(coarse). No learned regression stage
             (R2-D2's physics-based WRF 45 km stage is replaced by the ERA5 field itself, which is the
             only coarse product available). Conditioned on [ERA5+static, coarse, pos].

Both use the same EDM formulation, U-Net family, sampler (16 Heun steps), tiles and ensemble size as our
latent model, so differences come from (a) pixel vs latent space and (b) DRN mean vs interpolated-coarse mean.

Usage: python -m src.evaluation.train_baselines --variant corrdiff --max_hours 8
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch

from config import ERA5_VARS, CONUS404_VARS, IN_CH, OUT_CH, PATCH_SIZE, MODEL, TRAIN
from src.models.drn import DRN
from src.models.diffusion_unet import DiffusionUNet
from src.models.edm import EDMSchedule, edm_training_loss
from src.training.ema import EMA
from src.preprocessing.normalization import NormalizationStats

CACHE = "/discover/nobackup/sduan/.data"
BASELINE = dict(base_ch=96, ch_mults=(1, 2, 2, 4), num_res_blocks=2, attn_resolutions=(2, 3), time_dim=256)
COND_CH = IN_CH + OUT_CH + 2       # ERA5(+static) + mean field + (y,x) position


def pos_embedding(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device); xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy, xx], 0)


def era5_to_conus_norm(era5_n, stats):
    """ERA5 vars (era5-normalised, (B,6,H,W)) -> CONUS404-normalised space (same physical field).
    ERA5 tp is log1p(metres); CONUS404 precip is log1p(mm)."""
    dev = era5_n.device
    em, es = stats.era5_mean.to(dev).view(1, -1, 1, 1), stats.era5_std.to(dev).view(1, -1, 1, 1)
    cm, cs = stats.conus_mean.to(dev).view(1, -1, 1, 1), stats.conus_std.to(dev).view(1, -1, 1, 1)
    phys = era5_n * es + em
    pi = CONUS404_VARS.index("PREC_ACC_NC")
    tp_mm = torch.expm1(phys[:, pi]).clamp(min=0) * 1000.0
    phys = phys.clone()
    phys[:, pi] = torch.log1p(tp_mm)
    return (phys - cm) / cs


def mean_field(variant, era5, drn, stats):
    """era5: (B, IN_CH, H, W) normalised inputs incl. static. Returns mu in CONUS-normalised space."""
    if variant == "corrdiff":
        return drn(era5)
    return era5_to_conus_norm(era5[:, :OUT_CH], stats)


def build_model():
    return DiffusionUNet(in_ch=OUT_CH + COND_CH, out_ch=OUT_CH, **BASELINE)


def load_drn(device, path="checkpoints/drn_best.pt"):
    drn = DRN(in_ch=IN_CH, out_ch=OUT_CH, base_ch=MODEL["drn_base_ch"], ch_mults=MODEL["drn_ch_mults"],
              num_res_blocks=MODEL["drn_num_res_blocks"], attn_resolutions=MODEL["drn_attn_resolutions"])
    drn.load_state_dict(torch.load(path, map_location=device)["model_state_dict"])
    return drn.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["corrdiff", "r2d2"], required=True)
    ap.add_argument("--max_hours", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--total_steps", type=int, default=200000, help="cosine horizon (training also stops at max_hours)")
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/baselines")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    from src.evaluation._eval_setup import load_land_mask_from_cache
    from src.preprocessing.land_mask import get_valid_patch_origins
    from src.data.dataset import build_dataloaders
    stats = NormalizationStats(); stats.load("norm_stats.npz")
    land = load_land_mask_from_cache(CACHE)
    origins = get_valid_patch_origins(land, PATCH_SIZE, TRAIN["min_land_frac"])
    train_dl, val_dl, _ = build_dataloaders(
        "data", stats, batch_size=a.batch, patches_per_day=TRAIN["patches_per_day"], num_workers=a.workers,
        train_years=TRAIN["train_years"], val_years=TRAIN["val_years"], land_mask=land, valid_origins=origins,
        era5_vars=ERA5_VARS, conus_vars=CONUS404_VARS, cache_dir=CACHE)

    drn = load_drn(dev) if a.variant == "corrdiff" else None
    model = build_model().to(dev)
    print(f"[{a.variant}] params {sum(p.numel() for p in model.parameters())/1e6:.1f} M", flush=True)
    sched = EDMSchedule()
    ema = EMA(model, decay=a.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
    pos = pos_embedding(PATCH_SIZE, PATCH_SIZE, dev)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def make_batch(era5, conus, rstd):
        with torch.no_grad():
            mu = mean_field(a.variant, era5, drn, stats)
            resid = (conus - mu) / rstd
        cond = torch.cat([era5, mu, pos.unsqueeze(0).expand(era5.shape[0], -1, -1, -1)], 1)
        return resid, cond

    # per-channel residual scale so the EDM data std is ~1 (stored with the checkpoint)
    acc = []
    for i, (era5, conus) in enumerate(train_dl):
        era5, conus = era5.to(dev), conus.to(dev)
        with torch.no_grad():
            mu = mean_field(a.variant, era5, drn, stats)
            acc.append(((conus - mu) ** 2).mean((0, 2, 3)).cpu())
        if i >= 60:
            break
    rstd = torch.stack(acc).mean(0).sqrt().to(dev).view(1, -1, 1, 1).clamp(min=1e-3)
    print(f"[{a.variant}] residual std per channel (normalised units): {rstd.flatten().tolist()}", flush=True)

    t_start = time.time(); step = 0; best = float("inf"); hist = []
    done = False
    while not done:
        for era5, conus in train_dl:
            era5, conus = era5.to(dev, non_blocking=True), conus.to(dev, non_blocking=True)
            lr = a.lr * min(1.0, (step + 1) / a.warmup_steps) * (0.5 * (1 + math.cos(math.pi * min(step / a.total_steps, 1.0))) * 0.95 + 0.05)
            for g in opt.param_groups:
                g["lr"] = lr
            resid, cond = make_batch(era5, conus, rstd)
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = edm_training_loss(model, sched, resid, cond)
            if not torch.isfinite(loss):
                print("non-finite loss, skipping step", flush=True); opt.zero_grad(); continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(); step += 1
            hist.append(loss.item())
            el = (time.time() - t_start) / 3600
            if step % 100 == 0:
                print(f"[{a.variant}] step {step} loss {np.mean(hist[-100:]):.4f} lr {lr:.2e} {step/(el*3600):.2f} it/s  {el:.2f} h", flush=True)
            stop = el >= a.max_hours
            if step % a.val_every == 0 or stop:
                model.eval(); vl = []
                with torch.no_grad(), ema.apply(), torch.autocast("cuda", dtype=torch.bfloat16):
                    torch.manual_seed(1234)
                    for j, (e5, cn) in enumerate(val_dl):
                        r, c = make_batch(e5.to(dev), cn.to(dev), rstd)
                        vl.append(edm_training_loss(model, sched, r, c).item())
                        if j >= 30:
                            break
                v = float(np.mean(vl)); torch.manual_seed(step)
                print(f"[{a.variant}] === step {step} val(EMA) {v:.4f} (best {best:.4f})", flush=True)
                state = dict(model_state_dict=model.state_dict(), ema_state_dict=ema.state_dict(), rstd=rstd.cpu(),
                             variant=a.variant, step=step, val_loss=v, hours=el, baseline_cfg=BASELINE)
                torch.save(state, out / f"{a.variant}_latest.pt")
                if v < best:
                    best = v; torch.save(state, out / f"{a.variant}_best.pt")
            if stop:
                done = True; break
    json.dump(dict(variant=a.variant, steps=step, hours=(time.time() - t_start) / 3600, best_val=best,
                   final_train_loss=float(np.mean(hist[-200:])), params=sum(p.numel() for p in model.parameters())),
              open(out / f"{a.variant}_summary.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
