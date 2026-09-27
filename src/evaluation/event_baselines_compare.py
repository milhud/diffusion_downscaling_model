"""Head-to-head on the 10 events: ours (latent CorrDiff) vs CorrDiff-style vs R2-D2-style re-implementations.

All three sample 8 members with 16 Heun steps on identical tiles/days, scored on land pixels vs CONUS404.
Reuses the metric helpers of event_analyze.py (set EVENT_OUT to redirect for debugging).
"""
import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.event_analyze import (DAT, FIG, VARS, DERIVED, UNITS, all_fields, fair_crps, spectrum,
                                          load_event)

LAB = {"era5": "ERA5 interp", "drn": "DRN only", "r2d2": "R2-D2-style", "corr": "CorrDiff-style", "ours": "Ours (latent)"}
COL = {"ERA5 interp": "#888888", "DRN only": "#F4A261", "R2-D2-style": "#E76F51", "CorrDiff-style": "#457B9D",
       "Ours (latent)": "#2A9D8F", "CONUS404": "k"}
MEM = ["R2-D2-style", "CorrDiff-style", "Ours (latent)"]


def load_all(cat):
    out = {}
    for ev in cat:
        n = ev["name"]
        d = load_event(n)
        r = np.load(DAT / f"predb_r2d2_{n}.npz"); c = np.load(DAT / f"predb_corrdiff_{n}.npz")
        di = 2 if d["truth"].shape[0] >= 5 else 0
        F = dict(truth=all_fields(d["truth"]), era5=all_fields(d["era5"]), drn=all_fields(d["drn"]),
                 ours=all_fields(d["ens"]), corr=all_fields(c["ens"]), r2d2=all_fields(r["ens"]),
                 r2d2_mu=all_fields(r["mu"]))
        out[n] = dict(F=F, land=d["land"].astype(bool), di=di, ev=ev, ndays=d["truth"].shape[0])
    return out


def day(f, di):
    return {k: v[di] for k, v in f.items() if not k.startswith("_")}


def metrics(data):
    rows = []
    for n, D in data.items():
        di, land = D["di"], D["land"]; F = D["F"]
        tr = day(F["truth"], di)
        for v in VARS + DERIVED:
            t = tr[v]; sd = float(t[land].std())
            single = {"ERA5 interp": F["era5"][v][di], "DRN only": F["drn"][v][di]}
            for key in ("r2d2", "corr", "ours"):
                ens = F[key][v][di]           # (N,H,W)
                lab = LAB[key]
                single[lab + " mean"] = ens.mean(0); single[lab] = ens[0]
                crps = fair_crps(ens[:, land], t[land]).mean()
                spread = float(np.sqrt(ens[:, land].var(0, ddof=1).mean()))
                skill = float(np.sqrt(np.mean((ens.mean(0)[land] - t[land]) ** 2)))
                rows.append(dict(event=n, var=v, method=lab, kind="crps", value=float(crps) / sd, spread=spread, skill=skill))
            for m, p in single.items():
                e = (p - t)[land]
                rows.append(dict(event=n, var=v, method=m, kind="rmse", value=float(np.sqrt(np.mean(e ** 2))) / sd,
                                 bias=float(e.mean()) / sd))
    with open(DAT / "baselines_metrics.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r}); w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    return rows


def agg(rows, kind, method, v, key="value"):
    x = [r[key] for r in rows if r["kind"] == kind and r["method"] == method and r["var"] == v]
    return float(np.mean(x)) if x else np.nan


def fig_scorecard(rows):
    show = VARS + DERIVED
    meths = ["ERA5 interp", "DRN only"] + [m + " mean" for m in MEM] + MEM
    fig, axs = plt.subplots(1, 2, figsize=(19, 4.6), gridspec_kw=dict(width_ratios=[1, 0.75]))
    M = np.array([[agg(rows, "rmse", m, v) for v in show] for m in meths])
    im = axs[0].imshow(M, cmap="viridis_r", vmin=0.2, vmax=1.2, aspect="auto")
    axs[0].set_xticks(range(len(show))); axs[0].set_xticklabels(show); axs[0].set_yticks(range(len(meths)))
    axs[0].set_yticklabels(meths)
    for i in range(len(meths)):
        for j in range(len(show)):
            axs[0].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7, color="w" if M[i, j] < 0.7 else "k")
    axs[0].set_title("RMSE / truth σ (mean over 10 events; lower = better)")
    fig.colorbar(im, ax=axs[0], shrink=0.8)
    C = np.array([[agg(rows, "crps", m, v) for v in show] for m in MEM])
    im2 = axs[1].imshow(C, cmap="viridis_r", vmin=0.1, vmax=0.6, aspect="auto")
    axs[1].set_xticks(range(len(show))); axs[1].set_xticklabels(show, rotation=45); axs[1].set_yticks(range(3)); axs[1].set_yticklabels(MEM)
    for i in range(3):
        for j in range(len(show)):
            axs[1].text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=7, color="w" if C[i, j] < 0.3 else "k")
    axs[1].set_title("Ensemble CRPS / truth σ (8 members)")
    fig.colorbar(im2, ax=axs[1], shrink=0.8)
    fig.tight_layout(); fig.savefig(FIG / "baselines_scorecard.png", dpi=140); plt.close(fig)


def fig_bars(rows):
    show = VARS + DERIVED
    fig, axs = plt.subplots(1, 3, figsize=(21, 4.6))
    for a, (kind, lab, meths) in zip(axs, [("rmse", "single-member RMSE / σ", ["ERA5 interp"] + MEM),
                                           ("rmse", "ensemble-mean RMSE / σ", ["ERA5 interp", "DRN only"] + [m + " mean" for m in MEM]),
                                           ("crps", "CRPS / σ", MEM)]):
        w = 0.8 / len(meths)
        for j, m in enumerate(meths):
            a.bar(np.arange(len(show)) + j * w - 0.4 + w / 2, [agg(rows, kind, m, v) for v in show], w,
                  color=COL[m.replace(" mean", "")], label=m, alpha=0.6 if m.endswith(" mean") else 1)
        a.set_xticks(range(len(show))); a.set_xticklabels(show, rotation=30); a.set_title(lab); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "baselines_error_bars.png", dpi=140); plt.close(fig)
    # spread-skill
    fig, ax = plt.subplots(figsize=(12, 4))
    for j, m in enumerate(MEM):
        ssr = []
        for v in show:
            sp = np.mean([r["spread"] for r in rows if r["kind"] == "crps" and r["method"] == m and r["var"] == v])
            sk = np.mean([r["skill"] for r in rows if r["kind"] == "crps" and r["method"] == m and r["var"] == v])
            ssr.append(sp / sk)
        ax.bar(np.arange(len(show)) + j * 0.27 - 0.27, ssr, 0.27, color=COL[m], label=m)
    ax.axhline(1, color="k", lw=0.8, ls="--"); ax.set_xticks(range(len(show))); ax.set_xticklabels(show)
    ax.set_title("Spread-skill ratio (1 = calibrated, <1 under-dispersive)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "baselines_spread_skill.png", dpi=140); plt.close(fig)


def fig_spectra(data):
    show = ["T2", "WS", "PREC", "RH"]
    S = {v: {m: [] for m in ["CONUS404", "ERA5 interp"] + MEM} for v in show}
    for n, D in data.items():
        if D["land"].mean() < 0.85:
            continue
        di, F = D["di"], D["F"]
        for v in show:
            S[v]["CONUS404"].append(spectrum(F["truth"][v][di])); S[v]["ERA5 interp"].append(spectrum(F["era5"][v][di]))
            for key in ("r2d2", "corr", "ours"):
                S[v][LAB[key]].append(spectrum(F[key][v][di][0]))
    if not S["T2"]["CONUS404"]:
        return
    fig, axs = plt.subplots(2, 4, figsize=(19, 7))
    for j, v in enumerate(show):
        k = np.arange(1, len(S[v]["CONUS404"][0])); wl = 2048.0 / k
        ref = np.mean(S[v]["CONUS404"], axis=0)[1:]
        for m in ["CONUS404", "ERA5 interp"] + MEM:
            avg = np.mean(S[v][m], axis=0)[1:]
            axs[0, j].loglog(wl, avg, color=COL[m], label=m)
            if m != "CONUS404":
                axs[1, j].semilogx(wl, avg / ref, color=COL[m], label=m)
        for a in (axs[0, j], axs[1, j]):
            a.invert_xaxis()
        axs[0, j].set_title(f"{v} spectrum ({len(S[v]['CONUS404'])} land-dominated windows), single member")
        axs[1, j].axhline(1, color="k", lw=0.8); axs[1, j].set_ylim(0, 2.5); axs[1, j].set_xlabel("wavelength (km)")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "baselines_spectra.png", dpi=140); plt.close(fig)


def fig_fire_and_extremes(data):
    rows = []
    for n, D in data.items():
        di, land, F = D["di"], D["land"], D["F"]
        tr = day(F["truth"], di)
        for v in ["FFWI", "HDW", "WS", "VPD"]:
            thr = np.percentile(tr[v][land], 95); truth = (tr[v] > thr) & land
            preds = {"ERA5 interp": F["era5"][v][di], "DRN only": F["drn"][v][di]}
            for key in ("r2d2", "corr", "ours"):
                preds[LAB[key]] = F[key][v][di][0]; preds[LAB[key] + " mean"] = F[key][v][di].mean(0)
            for m, p in preds.items():
                pr = (p > thr) & land
                tp = (truth & pr).sum(); fp = (~truth & pr).sum(); fn = (truth & ~pr).sum()
                rows.append(dict(event=n, var=v, method=m, csi=float(tp / max(tp + fp + fn, 1)),
                                 max_ratio=float(p[land].max() / max(tr[v][land].max(), 1e-9))))
    with open(DAT / "baselines_fire.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.4))
    meths = ["ERA5 interp", "DRN only"] + MEM
    for a, key, t in zip(axs, ["csi", "max_ratio"], ["CSI of each event's top-5% pixels (single member)", "predicted max / truth max"]):
        for j, m in enumerate(meths):
            a.bar(np.arange(4) + j * 0.16 - 0.32, [np.mean([r[key] for r in rows if r["var"] == v and r["method"] == m]) for v in ["FFWI", "HDW", "WS", "VPD"]],
                  0.16, color=COL[m], label=m)
        a.set_xticks(range(4)); a.set_xticklabels(["FFWI", "HDW", "WS", "VPD"]); a.set_title(t)
    axs[1].axhline(1, color="k", lw=0.8, ls="--"); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "baselines_fire_extremes.png", dpi=140); plt.close(fig)
    # storm winds
    names = [n for n, D in data.items() if D["ev"]["kind"] in ("tropical_cyclone", "extratropical_storm", "convective_wind")]
    fig, ax = plt.subplots(figsize=(14, 4.6))
    meths = ["CONUS404", "ERA5 interp"] + MEM
    W = {}
    for j, m in enumerate(meths):
        vals = []
        for n in names:
            D = data[n]; di, land, F = D["di"], D["land"], D["F"]
            src = {"CONUS404": F["truth"]["WS"][di], "ERA5 interp": F["era5"]["WS"][di]}
            for key in ("r2d2", "corr", "ours"):
                src[LAB[key]] = F[key]["WS"][di][0]
            vals.append(float(src[m][land].max()))
        W[m] = vals
        ax.bar(np.arange(len(names)) + j * 0.16 - 0.32, vals, 0.16, color=COL[m], label=m)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=25); ax.set_ylabel("max daily-mean 10 m wind over land (m/s)")
    ax.legend(fontsize=8); ax.set_title("Storm peak wind: truth vs ERA5 vs the three downscalers (single member)")
    fig.tight_layout(); fig.savefig(FIG / "baselines_storm_wind.png", dpi=140); plt.close(fig)
    json.dump(dict(events=names, max_wind=W), open(DAT / "baselines_storm_wind.json", "w"), indent=1)


def fig_physical(data):
    rows = []
    for n, D in data.items():
        di, land, F = D["di"], D["land"], D["F"]
        for lab, f in [("CONUS404", F["truth"]), ("ERA5 interp", F["era5"]), ("DRN only", F["drn"])] + [(LAB[k], F[k]) for k in ("r2d2", "corr", "ours")]:
            g = lambda v: f[v][di] if f[v].ndim == 3 else f[v][di][0]
            t2, td, rh, pr = g("T2"), g("TD2"), g("_rh_raw"), g("PREC")
            raw_neg = None
            rows.append(dict(event=n, method=lab, supersat=float(np.mean((td > t2 + 0.01)[land])),
                             rh_gt100=float(np.mean((rh > 100.5)[land])),
                             prec_max_ratio=float(pr[land].max() / max(F["truth"]["PREC"][di][land].max(), 1e-6))))
    with open(DAT / "baselines_physical.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    meths = ["CONUS404", "ERA5 interp", "DRN only"] + MEM
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.2))
    for a, key, t in zip(axs, ["supersat", "rh_gt100", "prec_max_ratio"], ["fraction TD2 > T2", "fraction RH > 100 %", "max precip / truth max (median over events)"]):
        vals = [(np.median if key == "prec_max_ratio" else np.mean)([r[key] for r in rows if r["method"] == m]) for m in meths]
        a.bar(range(len(meths)), vals, color=[COL[m] for m in meths]); a.set_xticks(range(len(meths))); a.set_xticklabels(meths, rotation=30, fontsize=8)
        a.set_title(t, fontsize=10)
        if key == "prec_max_ratio":
            a.set_yscale("log"); a.axhline(1, color="k", lw=0.8, ls="--")
    fig.tight_layout(); fig.savefig(FIG / "baselines_physical.png", dpi=140); plt.close(fig)


def fig_temporal(data):
    rows = []
    for n, D in data.items():
        if D["ndays"] < 5:
            continue
        F, land = D["F"], D["land"]
        for v in ["T2", "TD2", "U10", "V10", "WS"]:
            tr = F["truth"][v][:, land]
            lag = lambda x: float(np.corrcoef(x[:-1].ravel(), x[1:].ravel())[0, 1])
            # common reference (our DRN) for ALL models so the residuals are comparable
            base = {"corr": F["drn"][v][:, land], "ours": F["drn"][v][:, land], "r2d2": F["drn"][v][:, land]}
            rows.append(dict(event=n, var=v, method="CONUS404 truth", lag1=lag(tr - base["corr"]), tend=np.nan))
            for key in ("r2d2", "corr", "ours"):
                mem = F[key][v][:, 0][:, land]
                rows.append(dict(event=n, var=v, method=LAB[key], lag1=lag(mem - base[key]),
                                 tend=float(np.sqrt(np.mean((np.diff(mem, axis=0) - np.diff(tr, axis=0)) ** 2)) / max(np.diff(tr, axis=0).std(), 1e-9))))
    with open(DAT / "baselines_temporal.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    show = ["T2", "TD2", "U10", "V10", "WS"]; meths = MEM
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.2))
    for j, m in enumerate(meths):
        axs[0].bar(np.arange(5) + j * 0.27 - 0.27, [np.mean([r["tend"] for r in rows if r["var"] == v and r["method"] == m]) for v in show], 0.27, color=COL[m], label=m)
    axs[0].set_title("Day-to-day tendency error / truth σ (single member; lower better)"); axs[0].legend(fontsize=8)
    for j, m in enumerate(["CONUS404 truth"] + meths):
        axs[1].bar(np.arange(5) + j * 0.2 - 0.3, [np.mean([r["lag1"] for r in rows if r["var"] == v and r["method"] == m]) for v in show], 0.2,
                   color=COL.get(m, "k"), label=m)
    axs[1].set_title("Lag-1 day autocorrelation of (member − DRN); truth: (CONUS404 − DRN)"); axs[1].legend(fontsize=8)
    for a in axs:
        a.set_xticks(range(5)); a.set_xticklabels(show)
    fig.tight_layout(); fig.savefig(FIG / "baselines_temporal.png", dpi=140); plt.close(fig)


def fig_maps(data):
    for name, vs in [("Michael", ["WS", "PREC"]), ("LaborDay_heat", ["FFWI", "T2"]), ("Florence", ["T2", "PREC"])]:
        if name not in data:
            continue
        D = data[name]; di, land, F = D["di"], D["land"], D["F"]
        cols = [("CONUS404", F["truth"]), ("ERA5 interp", F["era5"]), ("R2-D2-style", F["r2d2"]), ("CorrDiff-style", F["corr"]), ("Ours (latent)", F["ours"])]
        fig, axs = plt.subplots(len(vs), 5, figsize=(19, 3.6 * len(vs)))
        for r, v in enumerate(vs):
            t = F["truth"][v][di]
            vmin, vmax = (0, np.percentile(t[land], 99.5)) if v == "PREC" else (np.percentile(t[land], 1), np.percentile(t[land], 99.5))
            for c, (lab, f) in enumerate(cols):
                arr = f[v][di] if f[v].ndim == 3 else f[v][di][0]
                im = axs[r, c].imshow(np.where(land, arr, np.nan), origin="lower", cmap={"T2": "RdYlBu_r", "WS": "viridis", "PREC": "Blues", "FFWI": "YlOrRd"}[v],
                                      vmin=vmin, vmax=vmax)
                axs[r, c].set_title(f"{lab} ({v})" + (" [member 1]" if c > 1 else ""), fontsize=9); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            fig.colorbar(im, ax=axs[r, :], shrink=0.8, label=UNITS[v], pad=0.01)
        fig.suptitle(f"{name} {D['ev']['date']}: baselines vs ours (single ensemble member; land only)", y=0.995)
        fig.savefig(FIG / f"baselines_map_{name}.png", dpi=105, bbox_inches="tight"); plt.close(fig)


def main():
    cat = json.load(open(DAT / "event_catalog.json"))
    cat = [e for e in cat if all((DAT / f).exists() for f in (f"pred_{e['name']}.npz", f"predb_r2d2_{e['name']}.npz", f"predb_corrdiff_{e['name']}.npz"))]
    print("events:", [e["name"] for e in cat])
    data = load_all(cat)
    rows = metrics(data)
    fig_scorecard(rows); fig_bars(rows); fig_spectra(data); fig_fire_and_extremes(data); fig_physical(data); fig_temporal(data); fig_maps(data)
    print("baseline comparison complete")


if __name__ == "__main__":
    main()
