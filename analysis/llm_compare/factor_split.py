"""llm_compare step 18 (D567 follow-up): on gg -> t t~ h our-setup arms split the events into two groups about 3 units of
log|M|^2 below and above the truth (figs/shape_dists.png). Do the two groups follow the emission-scale target's
per-event factor F (the target is |M|^2 F; tools/llm_score.py --events-out writes log F per event)?
Per process, from d = pred - true of log|M|^2 over the CSV's events (first 5000 test events per process):
  groups: d split in two by 1D 2-means (lo/hi: share, mean d), the gap between the two means over the pooled sd of d
    within the groups (separation; >~2 is two visible groups);
  and for each per-event variable x of the CSV (log_factor = log F; offshell_<pdg> = the network's standardized
    signed-log off-shellness input (s_prop - M^2) of that internal particle, when the run has offshell_per_event):
  vars[x]: spearman of d with x; eta2, the share of var(d) explained by x, var(E[d | x]) / var(d), E over 20 quantile
    bins of x; auc = P(x of a hi event > x of a lo event) (0.5: x carries nothing about the group; 0 or 1: x decides
    it); for the gg -> t t~ X processes also the 2D histogram of d against x (counts) and the binned mean of d.
    python analysis/llm_compare/factor_split.py --csv <run>/llm_events_heldout.csv [--out data/<tag>_factor_split.json]
        [--pole-run <run>]   (each gg -> t t~ X / t b W process: d near the internal pole against the rest)
    python analysis/llm_compare/factor_split.py --fig data/<tag>_factor_split.json [--x offshell_6]   (figure, locally)
  --vars-from <other run>/llm_events_heldout.csv: the variables this CSV lacks (offshell_<pdg> of a run without the
    off-shellness input) taken from another run's CSV row by row, the same test events checked on true_logm2.
Without --out the JSON is printed on the last line (FACTOR_SPLIT ...), for a CPU job on the site that holds the CSV.
"""
import argparse, csv, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TTX = ("dc_gg_ttxh", "dc_gg_ttxa", "dc_gg_ttxz", "dc_gg_gttx")
# step 15 (D564): the pair carrying the pooled new-set gap (gg -> W qq' family), histogrammed like gg -> t t~ X
TBW = ("dc_gg_tbxwm", "dc_gg_btxwp")


def poles(run):
    """{offshell_<pdg>: the standardized input at s_prop = M^2}: the input is (sign(o) log1p|o| - mu) / sd (data_mmap,
    experiment._resolve_offshell_stats), so the pole sits at -mu / sd, read from the run's own data_stats.json"""
    import yaml
    mu, sd = json.load(open(os.path.join(run, "data_stats.json")))["offshell_stats"]
    pdgs = yaml.safe_load(open(os.path.join(run, "config.yaml")))["data"]["internal_mass_pdgs"]
    return {f"offshell_{g}": -m / v for g, m, v in zip(pdgs, mu, sd)}


def pole_split(d, x, x_pole, hi):
    """the fifth of events closest to the pole against the rest: mean d in each, their difference, its noise (the two
    means' standard errors in quadrature), and the share of the low 2-means group in each"""
    near = np.abs(x - x_pole) <= np.quantile(np.abs(x - x_pole), 0.2)
    se = np.sqrt(d[near].var() / near.sum() + d[~near].var() / (~near).sum())
    return {"x_pole": float(x_pole), "mean_d_near": float(d[near].mean()), "mean_d_far": float(d[~near].mean()),
            "contrast": float(d[near].mean() - d[~near].mean()), "contrast_se": float(se),
            "lo_share_near": float((~hi[near]).mean()), "lo_share_far": float((~hi[~near]).mean())}
D_EDGES = np.linspace(-10, 10, 81)


def two_means(d, it=50):
    c = np.quantile(d, [0.25, 0.75])
    for _ in range(it):
        hi = np.abs(d - c[1]) < np.abs(d - c[0])
        c = np.array([d[~hi].mean(), d[hi].mean()]) if hi.any() and (~hi).any() else c
    return hi, c


def auc(x_hi, x_lo):
    r = np.argsort(np.argsort(np.concatenate([x_hi, x_lo]))) + 1.0   # shortcut: ties broken by order; log F is continuous
    return float((r[:x_hi.size].sum() - x_hi.size * (x_hi.size + 1) / 2) / (x_hi.size * x_lo.size))


def spearman(x, y):
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def read(path):
    """{process: (true, pred, {variable: per-event values})}, the variables being every column after pred_logm2"""
    rows = list(csv.DictReader(open(path)))
    xs = [k for k in rows[0] if k not in ("process", "true_logm2", "pred_logm2")]
    ev = {}
    for r in rows:
        ev.setdefault(r["process"], []).append([float(r["true_logm2"]), float(r["pred_logm2"])] + [float(r[k]) for k in xs])
    return {n: (a[:, 0], a[:, 1], {k: a[:, 2 + i] for i, k in enumerate(xs)}) for n, a in
            ((n, np.array(v)) for n, v in ev.items())}


def join_vars(ev, other):
    """ev with the per-event variables it lacks taken from `other` (read() of another run's CSV over the same events)"""
    for n, (t, q, X) in ev.items():
        t2, _, X2 = other[n]
        assert t.size == t2.size and np.allclose(t, t2, rtol=0, atol=1e-4), f"{n}: not the same test events"
        X.update({k: v for k, v in X2.items() if k not in X})
    return ev


def against(d, x, hi, full):
    """d's dependence on one per-event variable x (a constant x: None)"""
    if x.std() == 0:
        return None
    qb = np.quantile(x, np.linspace(0, 1, 21))
    b = np.clip(np.searchsorted(qb, x, side="right") - 1, 0, 19)
    m = np.array([d[b == i].mean() if (b == i).any() else np.nan for i in range(20)])
    w = np.bincount(b, minlength=20) / d.size
    ok = w > 0
    r = {"sd": float(x.std()), "spearman": spearman(d, x),
         "eta2": float((w[ok] * (m[ok] - d.mean()) ** 2).sum() / d.var()),
         "auc": auc(x[hi], x[~hi]) if hi.any() and (~hi).any() else 0.5}
    if full:
        x_edges = np.linspace(*np.quantile(x, [0.0, 1.0]), 41)
        r["hist2d"] = {"d_edges": D_EDGES.tolist(), "x_edges": x_edges.tolist(),
                       "counts": np.histogram2d(np.clip(d, -10, 10), x, [D_EDGES, x_edges])[0].astype(int).tolist()}
        r["binned"] = {"x_mid": [float(np.median(x[b == i])) for i in range(20) if ok[i]], "mean_d": m[ok].tolist()}
    return r


def stats(t, q, X, full, pole=None):
    d = q - t
    hi, c = two_means(d)
    within = np.sqrt((((d[hi] - c[1]) ** 2).sum() + ((d[~hi] - c[0]) ** 2).sum()) / d.size)
    return {"n": int(d.size), "mse": float((d ** 2).mean()), "bias": float(d.mean()), "sd_d": float(d.std()),
            "groups": {"lo_share": float((~hi).mean()), "lo_mean_d": float(c[0]), "hi_mean_d": float(c[1]),
                       "separation": float((c[1] - c[0]) / within)},
            "vars": {k: against(d, x, hi, full) for k, x in X.items()}} | (
        {"pole": {k: pole_split(d, X[k], v, hi) for k, v in pole.items() if k in X and X[k].std() > 1e-12}} if pole else {})


XLAB = {"log_factor": r"$\log F$", "offshell_23": r"off-shellness input, internal $Z$",
        "offshell_6": r"off-shellness input, internal $t$", "offshell_25": r"off-shellness input, internal $H$",
        "offshell_24": r"off-shellness input, internal $W$"}


def fig(path, xv, panels="ttx"):
    sys.path.insert(0, ROOT)
    import plot_style as ps
    from matplotlib.colors import LogNorm
    sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_daniel"))
    from heldout_delta import proc_label
    meta = {"dc_" + p["folder"]: p["process"] for p in
            json.load(open(os.path.join(ROOT, "analysis", "catalog_daniel", "classes_default.json")))["processes"]}
    o = json.load(open(path))
    fig, axes = ps.figure(ncols=2, nrows=1 if panels == "tbw" else 2)
    for ax, n in zip(axes.ravel(), TBW if panels == "tbw" else TTX):
        v = o["processes"][n]["vars"][xv]
        h = v["hist2d"]
        im = ax.pcolormesh(h["x_edges"], h["d_edges"], np.ma.masked_equal(np.array(h["counts"]), 0), cmap="viridis",
                           norm=LogNorm(), rasterized=True)
        ps.colorbar(ax, im, "events")
        bb = v["binned"]
        ax.plot(bb["x_mid"], bb["mean_d"], "o-", color=ps.C.vermillion, ms=2, lw=0.8, label="mean in bins")
        ax.axhline(0, color="0.5", lw=0.5)
        if "pole" in o["processes"][n] and xv in o["processes"][n]["pole"]:
            ax.axvline(o["processes"][n]["pole"][xv]["x_pole"], color="k", ls=":", lw=0.8, label="pole")
        ax.text(0.03, 0.95, proc_label(meta[n]), transform=ax.transAxes, va="top", color="k")
        ax.set_xlabel(XLAB[xv])
        ax.set_ylabel(r"pred $-$ true of $\log|\mathcal{M}|^2$")
    ps.legend(axes.ravel()[0], "upper left" if panels == "tbw" else "lower right")
    base = os.path.splitext(os.path.basename(path))[0]
    ps.save(fig, os.path.join(HERE, "figs", (base if xv == "log_factor" else f"{base}_{xv}") + ("_tbw" if panels == "tbw" else "")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig", default=None, help="draw the figure from this JSON")
    ap.add_argument("--x", default="log_factor", help="the figure's per-event variable")
    ap.add_argument("--pole-run", default=None, help="the scored run's dir: its off-shellness statistics place each "
                    "pole on the input axis, and the gg -> t b W pair gets the near-pole fifth against the rest")
    ap.add_argument("--panels", choices=("ttx", "tbw"), default="ttx", help="the figure's four processes")
    ap.add_argument("--vars-from", default=None, help="another run's events CSV giving the variables this one lacks")
    a = ap.parse_args()
    if a.fig:
        return fig(a.fig, a.x, a.panels)
    ev = read(a.csv)
    if a.vars_from:
        ev = join_vars(ev, read(a.vars_from))
    pl = poles(a.pole_run) if a.pole_run else None
    out = {"csv": a.csv, "vars_from": a.vars_from, "poles": pl,
           "processes": {n: stats(t, q, X, n in TTX + TBW, pl if n in TTX + TBW else None) for n, (t, q, X) in ev.items()}}
    out["all"] = {}
    for k in next(iter(ev.values()))[2]:
        vs = [p["vars"][k] for p in out["processes"].values() if p["vars"][k]]
        out["all"][k] = {"n_processes": len(vs), "median_abs_spearman": float(np.median([abs(v["spearman"]) for v in vs])),
                         "median_eta2": float(np.median([v["eta2"] for v in vs]))} if vs else None
    for n in [m for m in TTX + TBW if m in out["processes"]]:
        p = out["processes"][n]
        print(n, p["mse"], p["bias"], p["groups"], {k: v and {j: round(w, 3) for j, w in v.items() if isinstance(w, float)}
                                                     for k, v in p["vars"].items()}, file=sys.stderr)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    else:
        print("FACTOR_SPLIT " + json.dumps(out))


if __name__ == "__main__":
    main()
