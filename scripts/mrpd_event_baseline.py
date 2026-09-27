"""T2-only benchmark baseline: the local MRPD cascade (25->12->4 km, /gpfsm/.../model).

MRPD is our only local *other* trained downscaler and is the closest local analogue to
R2-D2's multi-stage (coarse -> intermediate -> fine) design. Caveats (also in README):
  * temperature only, 900 km (225 px) domain, trained 2013,14,15,17,18 (val 2019-20)
    -> 2018 events are in its TRAINING years (leakage in its favour).
  * its raw 25 km ERA5 .nc inputs are unavailable; the 36x36 input is emulated by
    area-averaging our 4 km-regridded ERA5 T2 down to 36x36.
  * only the regression (mean) cascade is evaluated - the repo has no diffusion sampler
    (its own test_eval.py also uses regression means only).
Run from any cwd:  python scripts/mrpd_event_baseline.py
"""
import sys, json
from pathlib import Path
MODEL_DIR = "/gpfsm/dnb33/hpmille1/model"
sys.path.insert(0, MODEL_DIR)
import numpy as np
import torch
import torch.nn.functional as F
import train as mrpd            # noqa: E402  (MRPD's train.py; its config.py is picked from MODEL_DIR)
from config import DataConfig, PathConfig   # noqa: E402

REPO = Path("/gpfsm/dnb33/hpmille1/diffusion_downscaling_model")
DATA = REPO / "event_benchmark_output" / "data"
CK = Path(MODEL_DIR) / "experiments/mrpd_era5_conus404_temp/checkpoints"
T = 225
OFFS = [0, 512 - T]
MEAN, STD = DataConfig.TEMP_MEAN, DataConfig.TEMP_STD


def load(model, stage, ep):
    """Load ONLY stage{stage}_* keys from that stage's checkpoint.

    The stage-3 checkpoints contain all-NaN stage2_reg/stage2_diff weights (found while
    building this baseline), so loading a whole checkpoint with strict=False would
    silently poison the earlier stages.
    """
    ck = torch.load(CK / f"stage{stage}_epoch{ep}.pt", map_location="cuda", weights_only=False)
    sd = {k: v for k, v in ck["model"].items() if k.startswith(f"stage{stage}_")}
    bad = [k for k, v in sd.items() if torch.is_floating_point(v) and not torch.isfinite(v).all()]
    assert not bad, f"non-finite weights in stage {stage}: {bad[:3]}"
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"stage {stage}: loaded {len(sd)} tensors from epoch {ep}", flush=True)


def main():
    dev = "cuda"
    model = mrpd.MRPDModel().to(dev).eval()
    load(model, 1, 15); load(model, 2, 15); load(model, 3, 20)
    for f in sorted(DATA.glob("pred_*.npz")):
        z = np.load(f)
        era = z["era5"][:, 0]                        # (D,512,512) T2 K
        D = era.shape[0]
        s2 = np.full((D, 512, 512), np.nan, np.float32); s3 = s2.copy()
        for d in range(D):
            for oy in OFFS:
                for ox in OFFS:
                    crop = (era[d, oy:oy + T, ox:ox + T] - MEAN) / STD
                    x = torch.from_numpy(crop[None, None].astype(np.float32)).to(dev)
                    x36 = F.adaptive_avg_pool2d(x, 36)
                    with torch.no_grad():
                        s1, _ = model.forward_stage1(x36)
                        m2, _ = model.forward_stage2(s1)
                        m3, _ = model.forward_stage3(m2)
                    s2[d, oy:oy + T, ox:ox + T] = m2[0, 0].cpu().numpy() * STD + MEAN
                    s3[d, oy:oy + T, ox:ox + T] = m3[0, 0].cpu().numpy() * STD + MEAN
        if not (np.isfinite(s2[~np.isnan(s2)]).all() and np.isfinite(s3[~np.isnan(s3)]).all()):
            print("WARNING: non-finite MRPD output", f.stem, flush=True)
        name = f.stem.replace("pred_", "")
        np.savez_compressed(DATA / f"mrpd_T2_{name}.npz", s2=s2, s3=s3)
        print(name, "done", flush=True)


if __name__ == "__main__":
    main()
