"""T2-only comparison incl. the local MRPD cascade (25->12->4 km), per event.

MRPD trained on 2013,14,15,17,18 (val 2019-20): 2018 events are in ITS TRAINING years
(marked with *). Only its regression cascade is run (no sampler in that repo); its 36x36
input is emulated from our 4 km-regridded ERA5. Scored on pixels covered by its 225 px tiles.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DAT = Path("event_benchmark_output/data"); FIG = Path("event_benchmark_output/figures")
cat = json.load(open(DAT / "event_catalog.json"))
M = ["ERA5 interp", "MRPD s2 (12→4km reg.)", "MRPD s3 (final reg.)", "DRN (ours)", "Ens mean (ours)", "Ens member (ours)"]
COL = ["#888888", "#B5838D", "#6D597A", "#F4A261", "#2A9D8F", "#264653"]
rows = []
for ev in cat:
    z = np.load(DAT / f"pred_{ev['name']}.npz"); r = np.load(DAT / f"mrpd_T2_{ev['name']}.npz")
    land = z["land"].astype(bool); i = 2
    cov = ~np.isnan(r["s3"][i]); msk = land & cov
    t = z["truth"][i, 0]
    preds = [z["era5"][i, 0], r["s2"][i], r["s3"][i], z["drn"][i, 0], z["ens"][i, :, 0].mean(0), z["ens"][i, 0, 0]]
    if msk.sum() < 5000:      # too little land inside MRPD tile coverage -> not scored
        print(f"SKIP {ev['name']}: only {int(msk.sum())} land px covered by MRPD tiles")
        continue
    for m, p in zip(M, preds):
        e = (p - t)[msk]
        rows.append(dict(event=ev["name"], year=ev["year"], method=m, rmse=float(np.sqrt(np.mean(e ** 2))),
                         bias=float(e.mean()), mae=float(np.abs(e).mean()), npx=int(msk.sum())))
import csv
with open(DAT / "mrpd_T2_comparison.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

names = sorted({r["event"] for r in rows}, key=lambda n: [e["name"] for e in cat].index(n))
fig, axs = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw=dict(width_ratios=[3, 1]))
for j, m in enumerate(M):
    v = [[r["rmse"] for r in rows if r["event"] == n and r["method"] == m][0] for n in names]
    axs[0].bar(np.arange(len(names)) + j * 0.13 - 0.33, v, 0.13, color=COL[j], label=m)
    ag = [np.mean([r["rmse"] for r in rows if r["method"] == m and r["year"] == y] or [np.nan]) for y in (2018, 2019, 2020)]
    axs[1].bar(np.arange(3) + j * 0.13 - 0.33, ag, 0.13, color=COL[j])
axs[0].set_xticks(range(len(names))); axs[0].set_xticklabels([n + ("*" if [e for e in cat if e["name"] == n][0]["year"] == 2018 else "") for n in names], rotation=30)
axs[0].set_ylabel("T2 RMSE (K), land"); axs[0].legend(fontsize=8, ncol=2)
axs[0].set_title("T2 event-day RMSE. CAUTION: MRPD checkpoints are non-functional (worse than ERA5 interp even on its own\n training-domain crop: ~7-13 K vs 1.9 K) -> shown for completeness, NOT a meaningful benchmark. * = MRPD training year")
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["2018*", "2019", "2020"]); axs[1].set_title("Mean by year")
fig.tight_layout(); fig.savefig(FIG / "mrpd_T2_benchmark.png", dpi=140); plt.close(fig)

# map for one event
ev = [e for e in cat if e["name"] == "IowaDerecho"][0]
z = np.load(DAT / f"pred_{ev['name']}.npz"); r = np.load(DAT / f"mrpd_T2_{ev['name']}.npz"); land = z["land"].astype(bool)
t = z["truth"][2, 0]
fs = [("CONUS404", t), ("ERA5 interp", z["era5"][2, 0]), ("MRPD final", r["s3"][2]), ("DRN (ours)", z["drn"][2, 0]), ("Ens mean (ours)", z["ens"][2, :, 0].mean(0))]
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
vmin, vmax = np.nanpercentile(t[land], 1), np.nanpercentile(t[land], 99)
for c, (n, f) in enumerate(fs):
    axs[0, c].imshow(np.where(land, f, np.nan), origin="lower", cmap="RdYlBu_r", vmin=vmin, vmax=vmax); axs[0, c].set_title(n)
    if c:
        e = np.where(land, f - t, np.nan); axs[1, c].imshow(e, origin="lower", cmap="RdBu_r", vmin=-4, vmax=4)
        axs[1, c].set_title(f"{n} − truth")
    axs[0, c].set_xticks([]); axs[0, c].set_yticks([]); axs[1, c].set_xticks([]); axs[1, c].set_yticks([])
axs[1, 0].axis("off")
fig.suptitle("Iowa derecho window, T2 (K): MRPD covers only its two 225px tile rows/cols (blank centre band)")
fig.tight_layout(); fig.savefig(FIG / "mrpd_T2_map_IowaDerecho.png", dpi=110); plt.close(fig)
for m in M:
    print(f"{m:26s} mean RMSE all={np.mean([r['rmse'] for r in rows if r['method']==m]):.3f}  2019-20={np.mean([r['rmse'] for r in rows if r['method']==m and r['year']>2018]):.3f}  bias={np.mean([r['bias'] for r in rows if r['method']==m]):+.3f}")
