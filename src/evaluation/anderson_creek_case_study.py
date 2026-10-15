"""Anderson Creek Fire (2016) case study -- kept separate from the main 10-event,
test-year (2018-2020) benchmark in event_analyze.py.

Why separate: 2016 is a VALIDATION year (2015-2017 per docs/DATA_SPECIFICATION.md),
not a held-out TEST year, so mixing it into event_analyze.py's aggregate figures
(scorecard_vs_era5.png, fire_extreme_detection.png, etc.) would silently change
numbers that event_benchmark_output/README.md's prose already quotes for the
10-event test-year set. This script reuses event_analyze.py's helper functions but
writes its own anderson_creek_*.png / .json / .csv outputs and never touches cat
beyond the 3 AndersonCreek_* entries.

Outputs (event_benchmark_output/):
  figures/anderson_creek_fire_weather_{start,peak,end}.png  - same panel layout as
      event_analyze.fig_fire: truth/ERA5/DRN/Ens-mean x RH/VPD/WS/FFWI/HDW
  figures/anderson_creek_timeline.png   - start -> peak -> end progression of
      derived fire-weather fields (spatial mean over the fire-perimeter box) per method
  figures/anderson_creek_perimeter_overlay.png - MTBS burn perimeter polygon over
      the peak-day truth/DRN/Ens-mean FFWI maps
  figures/anderson_creek_sensitivity.png - how differences between methods' T2/TD2/
      U10/V10 (the regression target) propagate into the derived FFWI/VPD/HDW index
      ("plug in different outputs into one model" -- same fire_fields() derivation,
      fed different upstream fields, to see how much the downstream index moves)
  data/anderson_creek_summary.json  - numeric summary for all of the above

Usage: python -m src.evaluation.anderson_creek_case_study
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.event_analyze import (
    load_event, event_fields, fire_fields, FIG, DAT, COL, UNITS,
)

EVENTS = ["AndersonCreek_start", "AndersonCreek_peak", "AndersonCreek_end"]
STAGE_LABEL = {"AndersonCreek_start": "day before ignition (2016-03-21)",
               "AndersonCreek_peak": "day of max spread (2016-03-23)",
               "AndersonCreek_end": "near-full containment (2016-03-29)"}
PERIM_GEOJSON = DAT / "anderson_creek_mtbs_perimeter.geojson"
FIELDS_SHOW = ["RH", "VPD", "WS", "FFWI", "HDW"]


def load_perimeter():
    if not PERIM_GEOJSON.exists():
        return None
    gj = json.loads(PERIM_GEOJSON.read_text())
    geom = gj["features"][0]["geometry"]
    rings = geom["coordinates"] if geom["type"] == "Polygon" else [r for poly in geom["coordinates"] for r in poly]
    return [np.array(r) for r in rings]  # list of (N,2) lon,lat rings


def latlon_grid_for(cat_entry):
    g = np.load(DAT / "grid_latlon.npz")
    y0, x0, S = cat_entry["y0"], cat_entry["x0"], cat_entry["size"]
    return g["lat"][y0:y0 + S, x0:x0 + S], g["lon"][y0:y0 + S, x0:x0 + S]


def fig_fire_panel(cat_by_name):
    """Same layout as event_analyze.fig_fire, scoped to the 3 Anderson Creek dates."""
    for name in EVENTS:
        ev = cat_by_name[name]
        d = load_event(name); land = d["land"].astype(bool)
        di, tr, m = event_fields(d)
        fig, axs = plt.subplots(4, 5, figsize=(17, 13.5))
        rows = [("CONUS404 (truth)", lambda v: tr[v]), ("ERA5 interp", lambda v: m["ERA5 interp"][v]),
                ("DRN", lambda v: m["DRN"][v]), ("Ens mean (diff)", lambda v: m["Ens"][v].mean(0))]
        for c, v in enumerate(FIELDS_SHOW):
            vmin = np.percentile(tr[v][land], 1); vmax = np.percentile(tr[v][land], 99.5)
            cm = "YlOrRd_r" if v == "RH" else "YlOrRd"
            for r, (nm, fn) in enumerate(rows):
                im = axs[r, c].imshow(np.where(land, fn(v), np.nan), origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
                axs[r, c].set_title(f"{nm}: {v}", fontsize=9); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            fig.colorbar(im, ax=axs[:, c], shrink=0.6, orientation="horizontal", pad=0.02, label=UNITS[v])
        fig.suptitle(f"Anderson Creek Fire -- {STAGE_LABEL[name]}  (derived from daily-mean T2/TD2/U10/V10)", y=0.92)
        fig.savefig(FIG / f"anderson_creek_fire_weather_{name.split('_')[1]}.png", dpi=105, bbox_inches="tight")
        plt.close(fig)


def fig_timeline(cat_by_name):
    """Spatial-mean (land, fire-perimeter bbox if available) derived fields across the
    3 key dates, per method -- does the model track the fire's actual weather arc?"""
    rings = load_perimeter()
    stats = {v: {m: [] for m in ["CONUS404", "ERA5 interp", "DRN", "Ens mean"]} for v in FIELDS_SHOW}
    for name in EVENTS:
        ev = cat_by_name[name]
        d = load_event(name); land = d["land"].astype(bool)
        lat, lon = latlon_grid_for(ev)
        if rings is not None:
            from matplotlib.path import Path as MplPath
            pts = np.stack([lon.ravel(), lat.ravel()], -1)
            inside = np.zeros(pts.shape[0], bool)
            for r in rings:
                inside |= MplPath(r).contains_points(pts)
            mask = (inside.reshape(land.shape)) & land
            if mask.sum() < 50:        # perimeter too small relative to grid res -> fall back
                mask = land
        else:
            mask = land
        di, tr, m = event_fields(d)
        for v in FIELDS_SHOW:
            stats[v]["CONUS404"].append(float(tr[v][mask].mean()))
            stats[v]["ERA5 interp"].append(float(m["ERA5 interp"][v][mask].mean()))
            stats[v]["DRN"].append(float(m["DRN"][v][mask].mean()))
            stats[v]["Ens mean"].append(float(m["Ens"][v].mean(0)[mask].mean()))

    fig, axs = plt.subplots(1, len(FIELDS_SHOW), figsize=(4 * len(FIELDS_SHOW), 4))
    x = np.arange(3)
    for a, v in zip(axs, FIELDS_SHOW):
        for meth in ["CONUS404", "ERA5 interp", "DRN", "Ens mean"]:
            a.plot(x, stats[v][meth], "o-", color=COL[meth], label=meth)
        a.set_xticks(x); a.set_xticklabels(["start\n(-1d)", "peak\n(max spread)", "end\n(contain.)"], fontsize=8)
        a.set_title(f"{v} [{UNITS[v]}]"); a.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    fig.suptitle("Anderson Creek Fire: derived fire-weather fields over the burn-perimeter mask, start/peak/end")
    fig.tight_layout(); fig.savefig(FIG / "anderson_creek_timeline.png", dpi=140); plt.close(fig)
    return stats


def fig_perimeter_overlay(cat_by_name):
    rings = load_perimeter()
    ev = cat_by_name["AndersonCreek_peak"]
    d = load_event("AndersonCreek_peak"); land = d["land"].astype(bool)
    lat, lon = latlon_grid_for(ev)
    di, tr, m = event_fields(d)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    panels = [("CONUS404 (truth) FFWI", tr["FFWI"]), ("DRN FFWI", m["DRN"]["FFWI"]),
              ("Ens mean FFWI", m["Ens"]["FFWI"].mean(0))]
    for a, (ttl, field) in zip(axs, panels):
        im = a.pcolormesh(lon, lat, np.where(land, field, np.nan), cmap="YlOrRd",
                           vmin=np.percentile(tr["FFWI"][land], 1), vmax=np.percentile(tr["FFWI"][land], 99.5))
        if rings is not None:
            for r in rings:
                a.plot(r[:, 0], r[:, 1], color="blue", lw=1.2)
        a.set_title(ttl, fontsize=10); a.set_xticks([]); a.set_yticks([])
    if rings is not None:
        axs[0].plot([], [], color="blue", lw=1.2, label="MTBS burn perimeter (post-fire, final)")
        axs[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Anderson Creek Fire -- peak-spread day FFWI vs. actual MTBS burn perimeter")
    fig.tight_layout(); fig.savefig(FIG / "anderson_creek_perimeter_overlay.png", dpi=140); plt.close(fig)

    # Tight zoom on the burn scar itself (same 4 km data, just cropped -- no rerun needed):
    # the full 2048 km window above makes the ~65x45 km fire scar nearly invisible.
    if rings is not None:
        all_pts = np.concatenate(rings, axis=0)
        lon_lo, lon_hi = all_pts[:, 0].min() - 0.5, all_pts[:, 0].max() + 0.5
        lat_lo, lat_hi = all_pts[:, 1].min() - 0.5, all_pts[:, 1].max() + 0.5
        in_box = (lon >= lon_lo) & (lon <= lon_hi) & (lat >= lat_lo) & (lat <= lat_hi)
        ys, xs = np.where(in_box)
        if len(ys) > 0:
            y0b, y1b, x0b, x1b = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            fig, axs = plt.subplots(1, 3, figsize=(15, 5.5))
            for a, (ttl, field) in zip(axs, panels):
                sub = field[y0b:y1b, x0b:x1b]
                sublat, sublon = lat[y0b:y1b, x0b:x1b], lon[y0b:y1b, x0b:x1b]
                subland = land[y0b:y1b, x0b:x1b]
                im = a.pcolormesh(sublon, sublat, np.where(subland, sub, np.nan), cmap="YlOrRd",
                                   vmin=np.percentile(tr["FFWI"][land], 1), vmax=np.percentile(tr["FFWI"][land], 99.5))
                for r in rings:
                    a.plot(r[:, 0], r[:, 1], color="blue", lw=1.5)
                a.set_title(ttl, fontsize=10); a.set_xticks([]); a.set_yticks([])
            axs[0].plot([], [], color="blue", lw=1.5, label="MTBS burn perimeter")
            axs[0].legend(fontsize=8, loc="lower left")
            fig.suptitle("Anderson Creek Fire -- zoomed on the actual burn scar (4 km native resolution, no rerun)")
            fig.tight_layout(); fig.savefig(FIG / "anderson_creek_perimeter_overlay_zoom.png", dpi=140)
            plt.close(fig)


def fig_sensitivity(cat_by_name):
    """Sensitivity analysis: feed each method's own T2/TD2/U10/V10 into the SAME
    fire_fields() formula and see how much the derived indices move -- i.e. how much
    of the downstream FFWI/VPD/HDW error is inherited from upstream T2/TD2/wind error
    vs. amplified/dampened by the (nonlinear) derivation itself."""
    rows = []
    for name in EVENTS:
        d = load_event(name); land = d["land"].astype(bool)
        di, tr, m = event_fields(d)
        methods = {"ERA5 interp": m["ERA5 interp"], "DRN": m["DRN"],
                   "Ens mean": {v: m["Ens"][v].mean(0) for v in m["Ens"]}}
        for v in ["T2", "TD2", "WS", "VPD", "FFWI", "HDW"]:
            tstd = float(tr[v][land].std())
            for meth, fields_dict in methods.items():
                p = fields_dict[v]
                out_rmse = float(np.sqrt(np.mean((p[land] - tr[v][land]) ** 2)))
                rows.append(dict(stage=name, var=v, method=meth,
                                 out_rmse=out_rmse, out_rmse_norm=out_rmse / max(tstd, 1e-9)))

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    up_vars, down_vars = ["T2", "TD2", "WS"], ["VPD", "FFWI", "HDW"]
    for a, show_vars, ttl in zip(axs, [up_vars, down_vars],
                                 ["Upstream inputs (direct model outputs)",
                                  "Downstream derived fire-weather indices"]):
        x = np.arange(len(show_vars))
        for j, meth in enumerate(["ERA5 interp", "DRN", "Ens mean"]):
            vals = [np.mean([r["out_rmse_norm"] for r in rows if r["method"] == meth and r["var"] == v])
                    for v in show_vars]
            a.bar(x + j * 0.25 - 0.25, vals, 0.25, label=meth, color=COL[meth])
        a.set_xticks(x); a.set_xticklabels(show_vars); a.set_title(ttl, fontsize=10)
        a.set_ylabel("RMSE / truth sigma")
    axs[0].legend(fontsize=8)
    fig.suptitle("Sensitivity: does the nonlinear fire_fields() derivation amplify or dampen\n"
                 "upstream T2/TD2/wind error when it propagates into VPD/FFWI/HDW? (Anderson Creek, 3 dates, mean)")
    fig.tight_layout(); fig.savefig(FIG / "anderson_creek_sensitivity.png", dpi=140); plt.close(fig)

    import csv
    with open(DAT / "anderson_creek_sensitivity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def main():
    cat = json.load(open(DAT / "event_catalog.json"))
    cat_by_name = {e["name"]: e for e in cat if e["name"] in EVENTS}
    missing = [n for n in EVENTS if not (DAT / f"pred_{n}.npz").exists()]
    if missing:
        print("Missing predictions, run event_predict first:", missing)
        return
    fig_fire_panel(cat_by_name)
    timeline = fig_timeline(cat_by_name)
    fig_perimeter_overlay(cat_by_name)
    sens = fig_sensitivity(cat_by_name)
    summary = dict(events=cat_by_name, timeline=timeline,
                   perimeter_available=PERIM_GEOJSON.exists(),
                   note="2016 is a VALIDATION year (2015-2017), not a held-out TEST year "
                        "(2018-2020) -- kept separate from the main 10-event scorecard.")
    (DAT / "anderson_creek_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("Anderson Creek case study done.")


if __name__ == "__main__":
    main()
