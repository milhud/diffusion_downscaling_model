"""Pick ONE crop per benchmark date and write it to a manifest that BOTH pipelines read.

CorrDiff's Era5Conus404Dataset normally draws an unseeded random 448x448 crop per sample and
records no coordinates, so its output can't be compared pixel-for-pixel with ours. This script
fixes that: it draws a deterministic, land-valid 448x448 crop origin for every tag in the
CorrDiff generate config, and writes them to shared_patches.json.

  - CorrDiff: dataset.patch_manifest in config_generate_era5conus404.yaml points at the JSON;
    the dataset then crops exactly that 448x448 window (full window is fed to CorrDiff).
  - Ours: scripts/corrdiff_vs_ours_sidebyside.py runs on the central 256x256 of the same window
    (our model's trained patch size); CorrDiff's output is center-cropped to the same pixels.

Tags are "{year}-day{idx:03d}" where idx is the 0-based cache index (exactly what
Era5Conus404Dataset.time() emits -- NOT 1-based day-of-year).

Usage: python scripts/select_shared_patches.py [--seed 0] [--out results/corrdiff_comparison/shared_patches.json]

Larger benchmark (random test dates, separate config/manifest/output so the 6-date results stay intact):
  python scripts/select_shared_patches.py --n_random 72 --out results/corrdiff_comparison/shared_patches_many.json \
      --write_config external/corrdiff/conf/config_generate_era5conus404_many.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from config import PATCH_SIZE
from src.preprocessing.land_mask import get_valid_patch_origins

CACHE = "/discover/nobackup/sduan/.data"
GEN_CFG = "external/corrdiff/conf/config_generate_era5conus404.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=GEN_CFG)
    ap.add_argument("--out", default="results/corrdiff_comparison/shared_patches.json")
    ap.add_argument("--n_random", type=int, default=0, help="sample this many random (year, day) tags instead of using the config's times")
    ap.add_argument("--write_config", default=None, help="with --n_random: write a derived generate config (times, manifest, output file replaced)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    crop = cfg["dataset"]["crop_size"]
    min_land = cfg["dataset"]["min_land_frac"]
    tags = cfg["generation"]["times"]
    if a.n_random:
        nan_days = {}
        nd = Path(CACHE) / "nan_days.json"
        if nd.exists():
            nan_days = {int(k): set(v) for k, v in json.load(open(nd)).items()}
        pool = []
        for y in cfg["dataset"]["years"]:
            n = np.load(f"{CACHE}/era5_{y}.npy", mmap_mode="r").shape[0]
            pool += [f"{y}-day{d:03d}" for d in range(n) if d not in nan_days.get(y, set())]
        pick = np.random.default_rng(a.seed + 12345).choice(len(pool), size=a.n_random, replace=False)
        tags = [pool[i] for i in sorted(pick)]
    inner = PATCH_SIZE
    assert crop >= inner, f"crop_size {crop} < our patch size {inner}"
    off = (crop - inner) // 2

    # Same land-mask definition and origin set the CorrDiff dataset itself uses.
    static = np.load(f"{CACHE}/static_fields.npy")
    origins = get_valid_patch_origins(static[5] > 0.5, crop, min_land)
    assert origins, "no valid crop origins at this crop size / land fraction"
    print(f"{len(origins)} valid {crop}x{crop} origins (min_land_frac={min_land})")

    manifest = {"_meta": dict(seed=a.seed, crop_size=crop, ours_size=inner, ours_offset=off,
                              origin_convention="(y0, x0) top-left in the full CONUS404 cache grid")}
    for tag in tags:
        year, day = tag.split("-day")
        rng = np.random.default_rng([a.seed, int(year), int(day)])
        y0, x0 = origins[int(rng.integers(0, len(origins)))]
        manifest[tag] = dict(year=int(year), day_idx=int(day), y0=int(y0), x0=int(x0),
                             ours_y0=int(y0) + off, ours_x0=int(x0) + off)
        print(f"{tag}: crop origin ({y0},{x0}) -> ours 256 patch ({y0 + off},{x0 + off})")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(out, "w"), indent=2)
    print(f"wrote {out}")

    if a.n_random and a.write_config:
        cfg["generation"]["times"] = tags
        cfg["dataset"]["patch_manifest"] = str(out.resolve())
        cfg["generation"]["io"]["output_filename"] = str(Path(a.out).resolve().with_name("corrdiff_output_many.nc"))
        cfg["hydra"]["job"]["name"] = "era5conus404_generate_many"
        cfg["generation"]["seed_batch_size"] = cfg["generation"].get("seed_batch_size", 4)
        yaml.safe_dump(cfg, open(a.write_config, "w"), sort_keys=False)
        print(f"wrote {a.write_config}")


if __name__ == "__main__":
    main()
