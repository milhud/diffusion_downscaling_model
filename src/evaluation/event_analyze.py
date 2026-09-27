"""Event-based benchmarking analysis (CPU). Consumes pred_<event>.npz from event_predict.

Methods compared (all vs CONUS404 truth, land pixels only):
  ERA5 interp  - bilinear-regridded ERA5 (the coarse input; the "uncorrected" baseline)
  DRN          - regression stage (CorrDiff's UNet analogue: bias/mean correction)
  Ens mean     - mean of N latent-diffusion members (DRN + sampled residual)
  Ens member   - a single member (what a user of the generative model actually gets)

Outputs (figures/, data/*.csv, data/summary.json) go under event_benchmark_output/.
"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(os.environ.get("EVENT_OUT", "event_benchmark_output"))
FIG = OUT / "figures"
DAT = OUT / "data"
VARS = ["T2", "TD2", "U10", "V10", "PSFC", "PREC"]
UNITS = {"T2": "K", "TD2": "K", "U10": "m/s", "V10": "m/s", "PSFC": "Pa", "PREC": "mm/day",
         "WS": "m/s", "RH": "%", "VPD": "hPa", "FFWI": "-", "HDW": "hPa·m/s"}
DERIVED = ["WS", "RH", "VPD", "FFWI", "HDW"]
METHODS = ["ERA5 interp", "DRN", "Ens mean", "Ens member"]
COL = {"ERA5 interp": "#888888", "DRN": "#F4A261", "Ens mean": "#2A9D8F",
       "Ens member": "#264653", "CONUS404": "k"}


# ────────────────────────── derived fire-weather fields ──────────────────────────
def sat_vp_hpa(tk):
    tc = tk - 273.15
    return 6.112 * np.exp(17.62 * tc / (243.12 + tc))


def fire_fields(f):
    """f: dict of arrays (..., H, W) with keys T2,TD2,U10,V10 -> derived dict.

    Inputs are DAILY MEANS. RH/VPD/wind derived from daily means understate the
    afternoon minimum-RH / peak-wind conditions that drive real fire behaviour.
    """
    t, td = f["T2"], f["TD2"]
    es_t, es_d = sat_vp_hpa(t), sat_vp_hpa(td)
    rh_raw = 100.0 * es_d / es_t
    rh = np.clip(rh_raw, 0, 100)
    vpd = np.maximum(es_t - es_d, 0)
    ws = np.sqrt(f["U10"] ** 2 + f["V10"] ** 2)
    tf = (t - 273.15) * 9 / 5 + 32
    emc = np.where(rh < 10, 0.03229 + 0.281073 * rh - 0.000578 * rh * tf,
          np.where(rh <= 50, 2.22749 + 0.160107 * rh - 0.01478 * tf,
                   21.0606 + 0.005565 * rh ** 2 - 0.00035 * rh * tf - 0.483199 * rh))
    m = np.clip(emc, 0, 40) / 30.0
    eta = 1 - 2 * m + 1.5 * m ** 2 - 0.5 * m ** 3
    u_mph = ws * 2.23694
    ffwi = np.clip(eta * np.sqrt(1 + u_mph ** 2) / 0.3002, 0, None)
    return dict(WS=ws, RH=rh, VPD=vpd, FFWI=ffwi, HDW=vpd * ws), rh_raw


# ────────────────────────────────── helpers ──────────────────────────────────
def load_event(name):
    z = np.load(DAT / f"pred_{name}.npz")
    d = {k: z[k] for k in z.files}
    d["ens"][:, :, 5] = d["ens"][:, :, 5]  # precip kept raw; clipped later
    return d


def fields(arr):
    """(...,6,H,W) -> dict var->(...,H,W)"""
    return {v: arr[..., i, :, :] for i, v in enumerate(VARS)}


def all_fields(arr):
    f = fields(arr)
    f["PREC"] = np.clip(f["PREC"], 0, None)
    der, rh_raw = fire_fields(f)
    f.update(der)
    f["_rh_raw"] = rh_raw
    return f


def coarse(x, k):
    """block-average last two dims by k (crops remainder)."""
    if k == 1:
        return x
    H, W = x.shape[-2:]
    H2, W2 = H // k * k, W // k * k
    x = x[..., :H2, :W2]
    return x.reshape(x.shape[:-2] + (H2 // k, k, W2 // k, k)).mean(axis=(-3, -1))


def land_stats(err, land):
    e = err[..., land]
    return dict(bias=float(np.mean(e)), rmse=float(np.sqrt(np.mean(e ** 2))),
                mae=float(np.mean(np.abs(e))))


def fair_crps(ens, obs):
    """ens (N,...), obs (...). Fair CRPS estimator."""
    n = ens.shape[0]
    t1 = np.mean(np.abs(ens - obs[None]), axis=0)
    t2 = np.zeros_like(t1)
    for i in range(n):
        for j in range(i + 1, n):
            t2 += np.abs(ens[i] - ens[j])
    t2 *= 2.0 / (n * (n - 1)) * 0.5
    return t1 - t2


def event_fields(d):
    """returns dict method -> fields dict for the event day (offset 0), + truth."""
    di = list(d["days"]).index(d["days"][len(d["days"]) // 2]) if len(d["days"]) < 5 else 2
    tr = all_fields(d["truth"][di])
    era = all_fields(d["era5"][di])
    drn = all_fields(d["drn"][di])
    ensf = all_fields(d["ens"][di])                    # each (N,H,W)
    return di, tr, dict(**{"ERA5 interp": era, "DRN": drn, "Ens": ensf})


# ────────────────────────────── per-event metrics ──────────────────────────────
def compute_metrics(events, cat):
    rows = []
    for ev in cat:
        d = load_event(ev["name"])
        land = d["land"].astype(bool)
        di, tr, m = event_fields(d)
        for v in VARS + DERIVED:
            t = tr[v]
            ens = m["Ens"][v]
            preds = {"ERA5 interp": m["ERA5 interp"][v], "DRN": m["DRN"][v],
                     "Ens mean": ens.mean(0), "Ens member": ens[0]}
            for meth, p in preds.items():
                s = land_stats(p - t, land)
                rows.append(dict(event=ev["name"], kind=ev["kind"], var=v, method=meth, **s,
                                 tstd=float(t[land].std())))
            crps = fair_crps(ens[:, land], t[land]).mean()
            spread = float(np.sqrt(ens[:, land].var(0, ddof=1).mean()))
            skill = float(np.sqrt(np.mean((ens.mean(0)[land] - t[land]) ** 2)))
            rows.append(dict(event=ev["name"], kind=ev["kind"], var=v, method="CRPS",
                             bias=np.nan, rmse=np.nan, mae=float(crps), tstd=float(t[land].std()),
                             spread=spread, skill=skill))
    import csv
    keys = sorted({k for r in rows for k in r})
    with open(DAT / "event_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    return rows


# ───────────────────────────────── figures ─────────────────────────────────
def fig_overview(cat):
    g = np.load(DAT / "grid_latlon.npz")
    lat, lon = g["lat"], g["lon"]
    static = np.load("/discover/nobackup/sduan/.data/static_fields.npy", mmap_mode="r")
    lsm = np.asarray(static[5][::8, ::8])
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.contour(lon[::8, ::8], lat[::8, ::8], lsm, levels=[0.5], colors="0.4", linewidths=0.8)
    kinds = {"tropical_cyclone": "tab:blue", "fire_weather": "tab:red", "cold_outbreak": "tab:cyan",
             "extratropical_storm": "tab:purple", "convective_wind": "tab:orange"}
    for e in cat:
        y0, x0, S = e["y0"], e["x0"], e["size"]
        cs = [(y0, x0), (y0, x0 + S - 1), (y0 + S - 1, x0 + S - 1), (y0 + S - 1, x0), (y0, x0)]
        ax.plot([lon[y, x] for y, x in cs], [lat[y, x] for y, x in cs], color=kinds[e["kind"]], lw=1.4)
        ax.text(lon[e["cy"], e["cx"]], lat[e["cy"], e["cx"]], f"{e['name']}\n{e['date']}", fontsize=7,
                ha="center", va="center", color=kinds[e["kind"]], weight="bold")
    for k, c in kinds.items():
        ax.plot([], [], color=c, label=k)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_title("Event windows (512x512 px = ~2048 km at 4 km), test years 2018-2020")
    ax.set_aspect(1.25)
    fig.tight_layout(); fig.savefig(FIG / "00_event_overview.png", dpi=150); plt.close(fig)


def fig_event_maps(cat):
    for ev in cat:
        d = load_event(ev["name"]); land = d["land"].astype(bool)
        di, tr, m = event_fields(d)
        show = ["T2", "WS", "PREC"] if ev["kind"] != "fire_weather" else ["T2", "WS", "FFWI"]
        cols = ["ERA5 interp", "DRN", "Ens mean", "CONUS404", "Ens mean − truth"]
        cmaps = {"T2": "RdYlBu_r", "WS": "viridis", "PREC": "Blues", "FFWI": "YlOrRd"}
        fig, axs = plt.subplots(len(show), 5, figsize=(17, 3.3 * len(show)))
        for r, v in enumerate(show):
            fs = [m["ERA5 interp"][v], m["DRN"][v], m["Ens"][v].mean(0), tr[v]]
            vmin = np.percentile(tr[v][land], 1); vmax = np.percentile(tr[v][land], 99.5)
            if v == "PREC":
                vmin = 0
            for c in range(4):
                im = axs[r, c].imshow(np.where(land, fs[c], np.nan), origin="lower", cmap=cmaps[v],
                                      vmin=vmin, vmax=vmax)
                axs[r, c].set_title(f"{cols[c]}  ({v})", fontsize=9)
            fig.colorbar(im, ax=axs[r, :4], shrink=0.8, label=UNITS[v], pad=0.01)
            err = np.where(land, fs[2] - tr[v], np.nan)
            lim = np.nanpercentile(np.abs(err), 99) or 1
            im2 = axs[r, 4].imshow(err, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            axs[r, 4].set_title(cols[4], fontsize=9)
            fig.colorbar(im2, ax=axs[r, 4], shrink=0.8, label=UNITS[v])
            for a in axs[r]:
                a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{ev['name']} — {ev['date']} ({ev['kind']}); daily mean, land pixels only (ocean masked)", y=0.995)
        fig.savefig(FIG / f"event_{ev['name']}.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def fig_fire(cat):
    fire = [e for e in cat if e["kind"] == "fire_weather"]
    if not fire:
        return
    fields_show = ["RH", "VPD", "WS", "FFWI", "HDW"]
    for ev in fire:
        d = load_event(ev["name"]); land = d["land"].astype(bool)
        di, tr, m = event_fields(d)
        fig, axs = plt.subplots(4, 5, figsize=(17, 13.5))
        rows = [("CONUS404 (truth)", lambda v: tr[v]), ("ERA5 interp", lambda v: m["ERA5 interp"][v]),
                ("DRN", lambda v: m["DRN"][v]), ("Ens mean (diff)", lambda v: m["Ens"][v].mean(0))]
        for c, v in enumerate(fields_show):
            vmin = np.percentile(tr[v][land], 1); vmax = np.percentile(tr[v][land], 99.5)
            cm = "YlOrRd_r" if v == "RH" else "YlOrRd"
            for r, (nm, fn) in enumerate(rows):
                im = axs[r, c].imshow(np.where(land, fn(v), np.nan), origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
                axs[r, c].set_title(f"{nm}: {v}", fontsize=9); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            fig.colorbar(im, ax=axs[:, c], shrink=0.6, orientation="horizontal", pad=0.02, label=UNITS[v])
        fig.suptitle(f"Fire-weather fields — {ev['name']} {ev['date']}  (derived from daily-mean T2/TD2/U10/V10)", y=0.92)
        fig.savefig(FIG / f"fire_weather_{ev['name']}.png", dpi=105, bbox_inches="tight"); plt.close(fig)


def fire_skill(cat):
    """Detection of extreme fire weather: pixels above truth's 95th pct (per event, land)."""
    rows = []
    for ev in cat:
        d = load_event(ev["name"]); land = d["land"].astype(bool)
        di, tr, m = event_fields(d)
        for v in ["FFWI", "HDW", "WS", "VPD"]:
            thr = np.percentile(tr[v][land], 95)
            truth = (tr[v] > thr) & land
            for meth, p in {"ERA5 interp": m["ERA5 interp"][v], "DRN": m["DRN"][v],
                            "Ens mean": m["Ens"][v].mean(0), "Ens member": m["Ens"][v][0]}.items():
                pred = (p > thr) & land
                tp = (truth & pred).sum(); fp = (~truth & pred & land).sum(); fn = (truth & ~pred).sum()
                rows.append(dict(event=ev["name"], var=v, method=meth,
                                 csi=float(tp / max(tp + fp + fn, 1)), hit=float(tp / max(tp + fn, 1)),
                                 far=float(fp / max(tp + fp, 1)),
                                 pred_p95_over_truth=float(np.percentile(p[land], 95) / max(thr, 1e-9)),
                                 pred_max_over_truth=float(p[land].max() / max(tr[v][land].max(), 1e-9))))
    import csv
    with open(DAT / "extreme_detection.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, key, ttl in zip(axs, ["csi", "pred_p95_over_truth", "pred_max_over_truth"],
                           ["CSI for top-5% (higher better)", "Predicted p95 / truth p95", "Predicted max / truth max"]):
        for j, meth in enumerate(METHODS):
            vals = []
            for v in ["FFWI", "HDW", "WS", "VPD"]:
                vals.append(np.mean([r[key] for r in rows if r["var"] == v and r["method"] == meth]))
            a.bar(np.arange(4) + j * 0.2 - 0.3, vals, 0.2, label=meth, color=COL[meth])
        a.set_xticks(range(4)); a.set_xticklabels(["FFWI", "HDW", "WS", "VPD"]); a.set_title(ttl, fontsize=10)
        if key != "csi":
            a.axhline(1, color="k", lw=0.8, ls="--")
    axs[0].legend(fontsize=8)
    fig.suptitle("Extreme fire-weather / wind detection, all events (mean)")
    fig.tight_layout(); fig.savefig(FIG / "fire_extreme_detection.png", dpi=140); plt.close(fig)
    return rows


def fig_bias_vs_scale(cat):
    scales = [1, 2, 4, 8, 16, 32]
    show = VARS + ["WS", "RH", "FFWI"]
    res = {v: {m: [] for m in METHODS} for v in show}
    for ev in cat:
        d = load_event(ev["name"]); land = d["land"].astype(bool)
        di, tr, m = event_fields(d)
        lc = coarse(land.astype(float), 1) > 0.5
        for v in show:
            row = {mm: [] for mm in METHODS}
            for k in scales:
                lk = coarse(land.astype(float), k) > 0.99   # coarse cells fully on land
                t = coarse(tr[v], k)
                for meth, p in {"ERA5 interp": m["ERA5 interp"][v], "DRN": m["DRN"][v],
                                "Ens mean": m["Ens"][v].mean(0), "Ens member": m["Ens"][v][0]}.items():
                    e = (coarse(p, k) - t)[lk]
                    row[meth].append(float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan)
            for meth in METHODS:
                res[v][meth].append(row[meth])
    fig, axs = plt.subplots(3, 3, figsize=(15, 11))
    for a, v in zip(axs.ravel(), show):
        for meth in METHODS:
            arr = np.nanmean(np.array(res[v][meth]), axis=0)
            a.plot([4 * s for s in scales], arr, "o-", color=COL[meth], label=meth)
        a.set_xscale("log", base=2); a.set_title(f"{v} [{UNITS[v]}]"); a.set_xlabel("averaging scale (km)")
        a.set_ylabel("RMSE"); a.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Error vs spatial resolution: RMSE of coarse-grained fields (mean over events, land)")
    fig.tight_layout(); fig.savefig(FIG / "bias_vs_resolution.png", dpi=140); plt.close(fig)
    json.dump({v: {m: np.nanmean(np.array(res[v][m]), axis=0).tolist() for m in METHODS} for v in show},
              open(DAT / "rmse_vs_scale.json", "w"), indent=1)
    return res


def fig_bias_heatmap(rows):
    events = sorted({r["event"] for r in rows}, key=lambda e: [x["event"] for x in rows].index(e))
    show = VARS + DERIVED
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.2), sharey=True)
    for a, meth in zip(axs, ["ERA5 interp", "DRN", "Ens mean"]):
        M = np.full((len(events), len(show)), np.nan)
        for i, e in enumerate(events):
            for j, v in enumerate(show):
                r = [x for x in rows if x["event"] == e and x["var"] == v and x["method"] == meth]
                if r:
                    M[i, j] = r[0]["bias"] / max(r[0]["tstd"], 1e-9)
        im = a.imshow(M, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        a.set_xticks(range(len(show))); a.set_xticklabels(show, rotation=45)
        a.set_yticks(range(len(events))); a.set_yticklabels(events); a.set_title(f"{meth}: bias / truth σ")
        for i in range(len(events)):
            for j in range(len(show)):
                if not np.isnan(M[i, j]):
                    a.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axs, shrink=0.8)
    fig.suptitle("Mean bias (pred − CONUS404) normalised by truth spatial σ; event day, land")
    fig.savefig(FIG / "bias_heatmap_events_x_variables.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_metric_bars(rows):
    show = VARS + DERIVED
    fig, axs = plt.subplots(1, 2, figsize=(17, 4.8))
    for a, key, ttl in zip(axs, ["rmse", "mae"], ["RMSE / truth σ (mean over events)", "CRPS-or-MAE / truth σ"]):
        for j, meth in enumerate(METHODS if key == "rmse" else ["ERA5 interp", "DRN", "Ens mean", "CRPS"]):
            vals = []
            for v in show:
                rr = [x[key] / max(x["tstd"], 1e-9) for x in rows if x["var"] == v and x["method"] == meth]
                vals.append(np.mean(rr))
            a.bar(np.arange(len(show)) + j * 0.2 - 0.3, vals, 0.2, label=meth,
                  color=COL.get(meth, "#E76F51"))
        a.set_xticks(range(len(show))); a.set_xticklabels(show, rotation=30); a.set_title(ttl); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "summary_error_by_variable.png", dpi=140); plt.close(fig)


def spectrum(x):
    f = np.fft.fftshift(np.fft.fft2(x - x.mean()))
    p = np.abs(f) ** 2
    H, W = x.shape; cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    mr = min(cy, cx)
    return np.bincount(r.ravel(), p.ravel(), minlength=mr + 1)[:mr] / np.maximum(np.bincount(r.ravel(), minlength=mr + 1)[:mr], 1)


def fig_spectra(cat):
    show = ["T2", "WS", "PREC", "RH"]
    S = {v: {m: [] for m in ["CONUS404"] + METHODS[:3]} for v in show}
    for ev in cat:
        d = load_event(ev["name"]); land = d["land"].astype(bool)
        if land.mean() < 0.85:      # spectra need mostly-land windows (ocean fill is artificial)
            continue
        di, tr, m = event_fields(d)
        for v in show:
            S[v]["CONUS404"].append(spectrum(tr[v]))
            S[v]["ERA5 interp"].append(spectrum(m["ERA5 interp"][v]))
            S[v]["DRN"].append(spectrum(m["DRN"][v]))
            S[v]["Ens mean"].append(spectrum(m["Ens"][v][0]))   # single member (ens mean is smoothed)
    n = len(S["T2"]["CONUS404"])
    if n == 0:
        return
    fig, axs = plt.subplots(2, 4, figsize=(18, 7))
    for j, v in enumerate(show):
        k = np.arange(1, len(S[v]["CONUS404"][0]))
        wl = 2048.0 / k
        for meth, c in [("CONUS404", "k"), ("ERA5 interp", COL["ERA5 interp"]), ("DRN", COL["DRN"]),
                        ("Ens mean", COL["Ens member"])]:
            avg = np.mean(S[v][meth], axis=0)[1:]
            lab = "Ens member" if meth == "Ens mean" else meth
            axs[0, j].loglog(wl, avg, color=c, label=lab)
            if meth != "CONUS404":
                ref = np.mean(S[v]["CONUS404"], axis=0)[1:]
                axs[1, j].semilogx(wl, avg / ref, color=c, label=lab)
        axs[0, j].invert_xaxis(); axs[1, j].invert_xaxis()
        axs[0, j].set_title(f"{v} power spectrum ({n} land-dominated windows)")
        axs[1, j].axhline(1, color="k", lw=0.8); axs[1, j].set_ylim(0, 2.2)
        axs[1, j].set_xlabel("wavelength (km)"); axs[1, j].set_ylabel("pred / truth power")
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "spectra_ratio_events.png", dpi=140); plt.close(fig)


def fig_tc_wind(cat):
    """CorrDiff/Haikui-style: max wind, wind PDF tail, radius of max wind proxy for cyclones."""
    tcs = [e for e in cat if e["kind"] in ("tropical_cyclone", "extratropical_storm", "convective_wind")]
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for ev in tcs:
        d = load_event(ev["name"]); land = d["land"].astype(bool)
        di, tr, m = event_fields(d)
        ws = {"CONUS404": tr["WS"], "ERA5 interp": m["ERA5 interp"]["WS"], "DRN": m["DRN"]["WS"],
              "Ens mean": m["Ens"]["WS"].mean(0), "Ens member": m["Ens"]["WS"][0]}
        # RMW proxy: distance from the truth max-wind pixel to pixel-of-max in each method (km)
        ty, tx = np.unravel_index(np.argmax(np.where(land, tr["WS"], -1)), land.shape)
        for k, w in ws.items():
            wl = np.where(land, w, -1)
            py, px = np.unravel_index(np.argmax(wl), land.shape)
            rows.append(dict(event=ev["name"], kind=ev["kind"], method=k, max_ws=float(wl.max()),
                             p99_ws=float(np.percentile(w[land], 99)),
                             maxloc_err_km=float(4 * np.hypot(py - ty, px - tx))))
    import csv
    with open(DAT / "wind_extremes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    names = [e["name"] for e in tcs]
    meths = ["CONUS404", "ERA5 interp", "DRN", "Ens mean", "Ens member"]
    for j, meth in enumerate(meths):
        for a, key in zip(axs, ["max_ws", "p99_ws"]):
            vals = [[r[key] for r in rows if r["event"] == n and r["method"] == meth][0] for n in names]
            a.bar(np.arange(len(names)) + j * 0.16 - 0.32, vals, 0.16, label=meth, color=COL[meth])
    for a, t in zip(axs, ["Max daily-mean 10 m wind (m/s), event window (land)", "99th percentile wind (m/s)"]):
        a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=30); a.set_title(t)
    axs[0].legend(fontsize=8)
    fig.suptitle("Storm wind extremes: does the model recover the intense tail? (cf. CorrDiff Typhoon Haikui: ERA5 22 m/s -> 33 m/s vs WRF 45)")
    fig.tight_layout(); fig.savefig(FIG / "storm_wind_extremes.png", dpi=140); plt.close(fig)
    return rows


def fig_temporal(cat):
    """Day-to-day behaviour over the 5-day event window (daily temporal resolution only)."""
    rows = []
    for ev in cat:
        d = load_event(ev["name"]); land = d["land"].astype(bool)
        if d["truth"].shape[0] < 4:
            continue
        T = all_fields(d["truth"]); D = all_fields(d["drn"]); E = all_fields(d["ens"])   # E: (D,N,H,W)
        for v in VARS[:4] + ["WS"]:
            tr = T[v][:, land]                        # (D,P)
            for meth, arr in {"ERA5 interp": all_fields(d["era5"])[v][:, land], "DRN": D[v][:, land],
                              "Ens mean": E[v].mean(1)[:, land], "Ens member": E[v][:, 0][:, land]}.items():
                dt_t = np.diff(tr, axis=0); dt_p = np.diff(arr, axis=0)
                tend_rmse = float(np.sqrt(np.mean((dt_p - dt_t) ** 2)))
                tcorr = float(np.mean([np.corrcoef(arr[:, i], tr[:, i])[0, 1] for i in range(0, tr.shape[1], 97)
                                       if arr[:, i].std() > 0 and tr[:, i].std() > 0]))
                rows.append(dict(event=ev["name"], var=v, method=meth, tendency_rmse=tend_rmse,
                                 pixel_series_corr=tcorr, tstd=float(dt_t.std())))
            # residual coherence: lag-1 corr of (member - DRN) vs (truth - DRN), high-freq part
            res_m = (E[v][:, 0] - D[v])[:, land]
            res_t = (T[v] - D[v])[:, land]
            def lag1(x):
                a, b = x[:-1].ravel(), x[1:].ravel()
                return float(np.corrcoef(a, b)[0, 1])
            rows.append(dict(event=ev["name"], var=v, method="resid_lag1_member", tendency_rmse=np.nan,
                             pixel_series_corr=lag1(res_m), tstd=np.nan))
            rows.append(dict(event=ev["name"], var=v, method="resid_lag1_truth", tendency_rmse=np.nan,
                             pixel_series_corr=lag1(res_t), tstd=np.nan))
    import csv
    with open(DAT / "temporal_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    show = VARS[:4] + ["WS"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.6))
    for j, meth in enumerate(METHODS):
        vals = [np.mean([r["tendency_rmse"] / r["tstd"] for r in rows if r["var"] == v and r["method"] == meth]) for v in show]
        axs[0].bar(np.arange(len(show)) + j * 0.2 - 0.3, vals, 0.2, label=meth, color=COL[meth])
        vals = [np.mean([r["pixel_series_corr"] for r in rows if r["var"] == v and r["method"] == meth]) for v in show]
        axs[1].bar(np.arange(len(show)) + j * 0.2 - 0.3, vals, 0.2, label=meth, color=COL[meth])
    for j, (meth, c) in enumerate([("resid_lag1_truth", "k"), ("resid_lag1_member", COL["Ens member"])]):
        vals = [np.mean([r["pixel_series_corr"] for r in rows if r["var"] == v and r["method"] == meth]) for v in show]
        axs[2].bar(np.arange(len(show)) + j * 0.35 - 0.175, vals, 0.35, color=c,
                   label="truth − DRN" if "truth" in meth else "member − DRN")
    axs[0].set_title("Day-to-day tendency error / truth tendency σ (lower better)")
    axs[1].set_title("Per-pixel correlation with truth across the 5 days")
    axs[2].set_title("Lag-1 (day) autocorrelation of the fine-scale residual")
    for a in axs:
        a.set_xticks(range(len(show))); a.set_xticklabels(show)
    axs[0].legend(fontsize=8); axs[2].legend(fontsize=8)
    fig.suptitle("Temporal behaviour (daily fields; samples drawn independently per day -> no built-in temporal coherence)")
    fig.tight_layout(); fig.savefig(FIG / "temporal_coherence.png", dpi=140); plt.close(fig)
    return rows


def fig_timeseries(cat):
    """Event-window area-mean / max time series over the 5 days for T2, WS, PREC, HDW."""
    fig, axs = plt.subplots(len(cat), 4, figsize=(16, 2.6 * len(cat)), sharex=True)
    for i, ev in enumerate(cat):
        d = load_event(ev["name"]); land = d["land"].astype(bool)
        T = all_fields(d["truth"]); D = all_fields(d["drn"]); E = all_fields(d["ens"]); R = all_fields(d["era5"])
        x = np.arange(-2, -2 + d["truth"].shape[0])
        for j, (v, red) in enumerate([("T2", "mean"), ("WS", "max"), ("PREC", "max"), ("HDW", "max")]):
            f = (lambda a: a[..., land].mean(-1)) if red == "mean" else (lambda a: a[..., land].max(-1))
            ax = axs[i, j]
            ax.plot(x, f(T[v]), "k-o", label="CONUS404"); ax.plot(x, f(R[v]), color=COL["ERA5 interp"], marker="s", label="ERA5")
            ax.plot(x, f(D[v]), color=COL["DRN"], marker="^", label="DRN")
            em = f(E[v])            # (D,N)
            ax.plot(x, em.mean(1), color=COL["Ens mean"], marker="v", label="Ens mean")
            ax.fill_between(x, em.min(1), em.max(1), color=COL["Ens mean"], alpha=0.2)
            if j == 0:
                ax.set_ylabel(ev["name"], fontsize=8)
            if i == 0:
                ax.set_title(f"{v} ({red} over land) [{UNITS[v]}]", fontsize=9)
    axs[0, 0].legend(fontsize=7)
    for a in axs[-1]:
        a.set_xlabel("days from event")
    fig.tight_layout(); fig.savefig(FIG / "event_timeseries.png", dpi=110); plt.close(fig)


def fig_bias_decomp(cat):
    """MSE = mean-bias^2 + error variance; and large-scale (>=64 km) vs small-scale error share."""
    show = VARS[:5] + ["PREC", "WS"]
    show = ["T2", "TD2", "U10", "V10", "PSFC", "PREC", "WS"]
    stages = ["ERA5 interp", "DRN", "Ens mean", "Ens member"]
    acc = {v: {s: dict(b2=[], var=[], large=[], small=[]) for s in stages} for v in show}
    for ev in cat:
        d = load_event(ev["name"]); land = d["land"].astype(bool)
        di, tr, m = event_fields(d)
        for v in show:
            for s, p in {"ERA5 interp": m["ERA5 interp"][v], "DRN": m["DRN"][v], "Ens mean": m["Ens"][v].mean(0),
                         "Ens member": m["Ens"][v][0]}.items():
                e = (p - tr[v])
                el = e[land]
                acc[v][s]["b2"].append(el.mean() ** 2 / tr[v][land].var())
                acc[v][s]["var"].append(el.var() / tr[v][land].var())
                k = 16   # 64 km
                lk = coarse(land.astype(float), k) > 0.99
                ec = coarse(e, k)[lk]
                up = np.kron(coarse(e, k), np.ones((k, k)))
                H, W = up.shape
                small = (e[:H, :W] - up)[land[:H, :W]]
                acc[v][s]["large"].append(np.mean(ec ** 2) / tr[v][land].var())
                acc[v][s]["small"].append(np.mean(small ** 2) / tr[v][land].var())
    fig, axs = plt.subplots(1, 2, figsize=(17, 5))
    w = 0.2
    for j, s in enumerate(stages):
        b = [np.mean(acc[v][s]["b2"]) for v in show]; r = [np.mean(acc[v][s]["var"]) for v in show]
        axs[0].bar(np.arange(len(show)) + j * w - 0.3, b, w, color=COL[s], label=f"{s}: bias²")
        axs[0].bar(np.arange(len(show)) + j * w - 0.3, r, w, bottom=b, color=COL[s], alpha=0.4)
        lg = [np.mean(acc[v][s]["large"]) for v in show]; sm = [np.mean(acc[v][s]["small"]) for v in show]
        axs[1].bar(np.arange(len(show)) + j * w - 0.3, lg, w, color=COL[s], label=s)
        axs[1].bar(np.arange(len(show)) + j * w - 0.3, sm, w, bottom=lg, color=COL[s], alpha=0.4)
    for a, t in zip(axs, ["MSE / truth variance: solid = region-mean bias², faint = remaining error variance",
                          "MSE / truth variance: solid = large-scale (>=64 km) error, faint = small-scale (<64 km)"]):
        a.set_xticks(range(len(show))); a.set_xticklabels(show); a.set_title(t, fontsize=10)
    axs[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Where does the correction happen? Bias removal (DRN, CorrDiff's regression stage) vs fine-scale residual (diffusion)")
    fig.tight_layout(); fig.savefig(FIG / "bias_decomposition.png", dpi=140); plt.close(fig)
    return {v: {s: {k: float(np.mean(x)) for k, x in acc[v][s].items()} for s in stages} for v in show}


def fig_physical(cat):
    rows = []
    for ev in cat:
        d = load_event(ev["name"]); land = d["land"].astype(bool)
        di = 2 if d["truth"].shape[0] >= 5 else 0
        for meth, arr in {"CONUS404": d["truth"][di], "ERA5 interp": d["era5"][di], "DRN": d["drn"][di],
                          "Ens member": d["ens"][di][0]}.items():
            f = fields(arr)
            rh_raw = 100 * sat_vp_hpa(f["TD2"]) / sat_vp_hpa(f["T2"])
            rows.append(dict(event=ev["name"], method=meth,
                             supersat_frac=float(np.mean((f["TD2"] > f["T2"] + 0.01)[land])),
                             rh_gt100_frac=float(np.mean((rh_raw > 100.5)[land])),
                             neg_precip_frac=float(np.mean((f["PREC"] < -0.05)[land])),
                             psfc_err_pa=float(np.mean(np.abs(f["PSFC"] - d["truth"][di][4])[land])),
                             ))
    import csv
    with open(DAT / "physical_consistency.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    meths = ["CONUS404", "ERA5 interp", "DRN", "Ens member"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    for a, key, t in zip(axs, ["supersat_frac", "rh_gt100_frac", "neg_precip_frac"],
                         ["Fraction of pixels with TD2 > T2", "Fraction with derived RH > 100%", "Fraction with negative precip"]):
        for j, mth in enumerate(meths):
            a.bar(j, np.mean([r[key] for r in rows if r["method"] == mth]),
                  color=COL.get(mth, "k"), label=mth)
        a.set_xticks(range(len(meths))); a.set_xticklabels(meths, rotation=20); a.set_title(t, fontsize=10)
    fig.suptitle("Physical-consistency checks (mean over events; daily means)")
    fig.tight_layout(); fig.savefig(FIG / "physical_consistency.png", dpi=140); plt.close(fig)
    return rows


def fig_precip_diag(cat):
    """Precip is the weak variable: spikes from expm1 of log-space noise, and log-space skill."""
    rows = []
    l = lambda x: np.log1p(np.clip(x, 0, None))
    for ev in cat:
        d = load_event(ev["name"]); land = d["land"].astype(bool); di = 2 if d["truth"].shape[0] >= 5 else 0
        t = d["truth"][di, 5][land]; era = d["era5"][di, 5][land]; dr = d["drn"][di, 5][land]
        en = d["ens"][di, :, 5][:, land]
        rows.append(dict(event=ev["name"], truth_max=float(t.max()), era5_max=float(era.max()), drn_max=float(dr.max()),
                         ens_max=float(en.max()), frac_gt_1000mm=float(np.mean(en > 1000)),
                         logrmse_era5=float(np.sqrt(np.mean((l(era) - l(t)) ** 2))),
                         logrmse_drn=float(np.sqrt(np.mean((l(dr) - l(t)) ** 2))),
                         logrmse_member=float(np.sqrt(np.mean((l(en[0]) - l(t)) ** 2))),
                         logrmse_ensmean=float(np.sqrt(np.mean((l(en.mean(0)) - l(t)) ** 2))),
                         neg_frac_member=float(np.mean(d["ens"][di, 0, 5][land] < -0.05))))
    import csv
    with open(DAT / "precip_diagnostics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    names = [r["event"] for r in rows]; x = np.arange(len(names))
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.8))
    for j, (k, lab, c) in enumerate([("truth_max", "CONUS404", "k"), ("era5_max", "ERA5 interp", COL["ERA5 interp"]),
                                      ("drn_max", "DRN", COL["DRN"]), ("ens_max", "Ens (max over 8 members)", COL["Ens member"])]):
        axs[0].bar(x + j * 0.2 - 0.3, [r[k] for r in rows], 0.2, label=lab, color=c)
    axs[0].set_yscale("log"); axs[0].set_xticks(x); axs[0].set_xticklabels(names, rotation=30)
    axs[0].set_title("Max daily precip in window (mm/day, log axis): ensemble spikes"); axs[0].legend(fontsize=8)
    for j, (k, lab, c) in enumerate([("logrmse_era5", "ERA5 interp", COL["ERA5 interp"]), ("logrmse_drn", "DRN", COL["DRN"]),
                                      ("logrmse_ensmean", "Ens mean", COL["Ens mean"]), ("logrmse_member", "Ens member", COL["Ens member"])]):
        axs[1].bar(x + j * 0.2 - 0.3, [r[k] for r in rows], 0.2, label=lab, color=c)
    axs[1].set_xticks(x); axs[1].set_xticklabels(names, rotation=30); axs[1].legend(fontsize=8)
    axs[1].set_title("Precip RMSE in log1p(mm) space (robust to spikes/displacement scaling)")
    fig.tight_layout(); fig.savefig(FIG / "precip_diagnostics.png", dpi=140); plt.close(fig)


def fig_scorecard(rows):
    """Skill vs ERA5 interpolation: 1 - RMSE/RMSE_ERA5 (positive = better than the coarse input)."""
    show = VARS + DERIVED
    meths = ["DRN", "Ens mean", "Ens member"]
    M = np.zeros((len(meths), len(show)))
    for j, v in enumerate(show):
        base = np.mean([r["rmse"] for r in rows if r["var"] == v and r["method"] == "ERA5 interp"])
        for i, m in enumerate(meths):
            M[i, j] = 1 - np.mean([r["rmse"] for r in rows if r["var"] == v and r["method"] == m]) / base
    crps = []
    for v in show:
        base = np.mean([r["mae"] for r in rows if r["var"] == v and r["method"] == "ERA5 interp"])   # MAE == CRPS of a point forecast
        crps.append(1 - np.mean([r["mae"] for r in rows if r["var"] == v and r["method"] == "CRPS"]) / base)
    M = np.vstack([M, crps])
    fig, ax = plt.subplots(figsize=(13, 3.6))
    im = ax.imshow(M, cmap="RdYlGn", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(show))); ax.set_xticklabels(show)
    ax.set_yticks(range(4)); ax.set_yticklabels(["DRN RMSE", "Ens-mean RMSE", "Ens-member RMSE", "Ensemble CRPS (vs ERA5 MAE)"])
    for i in range(4):
        for j in range(len(show)):
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="fractional improvement over ERA5 interp")
    ax.set_title("Scorecard (mean over 10 events, event day, land): green = better than the coarse ERA5 input")
    fig.tight_layout(); fig.savefig(FIG / "scorecard_vs_era5.png", dpi=140); plt.close(fig)
    json.dump(dict(vars=show, rows=["DRN", "Ens mean", "Ens member", "CRPS"], values=M.tolist()),
              open(DAT / "scorecard.json", "w"), indent=1)


def fig_temporal_aggregation(cat):
    """RMSE (normalised by truth sigma) after averaging over 1, 3, 5 consecutive days around the event.
    If a method's error is day-to-day independent noise it should shrink under averaging faster than
    a temporally-coherent (systematic) error. Only DAILY data exist, so sub-daily behaviour is untestable."""
    show = ["T2", "TD2", "WS", "PREC", "FFWI"]
    wins = {1: slice(2, 3), 3: slice(1, 4), 5: slice(0, 5)}
    res = {v: {m: {k: [] for k in wins} for m in METHODS} for v in show}
    for ev in cat:
        d = load_event(ev["name"]); land = d["land"].astype(bool)
        if d["truth"].shape[0] < 5:
            continue
        T = all_fields(d["truth"]); R = all_fields(d["era5"]); D = all_fields(d["drn"]); E = all_fields(d["ens"])
        for v in show:
            for k, sl in wins.items():
                t = T[v][sl].mean(0)[land]; sd = t.std()
                for m, arr in {"ERA5 interp": R[v][sl].mean(0), "DRN": D[v][sl].mean(0),
                               "Ens mean": E[v][sl].mean(1).mean(0), "Ens member": E[v][sl][:, 0].mean(0)}.items():
                    res[v][m][k].append(float(np.sqrt(np.mean((arr[land] - t) ** 2)) / max(sd, 1e-9)))
    fig, axs = plt.subplots(1, len(show), figsize=(18, 3.8), sharey=False)
    out = {}
    for a, v in zip(axs, show):
        for m in METHODS:
            y = [np.mean(res[v][m][k]) for k in wins]
            a.plot(list(wins), y, "o-", color=COL[m], label=m)
            out.setdefault(v, {})[m] = y
        a.set_xticks(list(wins)); a.set_xlabel("days averaged"); a.set_title(v); a.grid(alpha=0.3)
    axs[0].set_ylabel("RMSE / truth σ"); axs[0].legend(fontsize=8)
    fig.suptitle("Temporal aggregation: does error shrink when averaging consecutive days? (steeper drop = more day-to-day-independent error)")
    fig.tight_layout(); fig.savefig(FIG / "temporal_aggregation.png", dpi=140); plt.close(fig)
    json.dump(out, open(DAT / "temporal_aggregation.json", "w"), indent=1)


def main():
    cat = json.load(open(DAT / "event_catalog.json"))
    cat = [e for e in cat if (DAT / f"pred_{e['name']}.npz").exists()]
    print("events with predictions:", [e["name"] for e in cat])
    FIG.mkdir(exist_ok=True)
    fig_overview(cat)
    rows = compute_metrics(None, cat)
    fig_event_maps(cat)
    fig_fire(cat)
    fs = fire_skill(cat)
    fig_bias_vs_scale(cat)
    fig_bias_heatmap(rows)
    fig_metric_bars(rows)
    fig_scorecard(rows)
    fig_precip_diag(cat)
    fig_spectra(cat)
    tc = fig_tc_wind(cat)
    tm = fig_temporal(cat)
    fig_timeseries(cat)
    fig_temporal_aggregation(cat)
    dec = fig_bias_decomp(cat)
    ph = fig_physical(cat)
    json.dump(dict(bias_decomposition=dec), open(DAT / "summary.json", "w"), indent=1)
    print("analysis complete")


if __name__ == "__main__":
    main()
