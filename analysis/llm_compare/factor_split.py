"""llm_compare step 18 (D567 follow-up): on gg -> t t~ h our-setup arms split the events into two groups about 3 units of
log|M|^2 below and above the truth (figs/shape_dists.png). Do the two groups follow the emission-scale target's
per-event factor F (the target is |M|^2 F; tools/llm_score.py --events-out writes log F per event)?
Per process, from d = pred - true of log|M|^2 over the CSV's events (first 5000 test events per process):
  spearman_d_logF: rank correlation of d with log F;
  eta2: share of var(d) explained by log F, var(E[d | log F]) / var(d), E over 20 quantile bins of log F;
  groups: d split in two by 1D 2-means (lo/hi: share, mean d), the gap between the two means over the pooled sd of d
    within the groups (separation; >~2 is two visible groups), and auc_logF = P(log F of a hi event > log F of a lo
    event) (0.5: log F carries nothing about the group; 0 or 1: log F decides it);
  for the gg -> t t~ X processes also the 2D histogram of d against log F (counts) and the binned mean of d.
    python analysis/llm_compare/factor_split.py --csv <run>/llm_events_heldout.csv [--out data/<tag>_factor_split.json]
    python analysis/llm_compare/factor_split.py --fig data/<tag>_factor_split.json   (figure from the JSON, locally)
Without --out the JSON is printed on the last line (FACTOR_SPLIT ...), for a CPU job on the site that holds the CSV.
"""
import argparse, csv, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TTX = ("dc_gg_ttxh", "dc_gg_ttxa", "dc_gg_ttxz", "dc_gg_gttx")
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
    ev = {}
    for r in csv.DictReader(open(path)):
        ev.setdefault(r["process"], []).append((float(r["true_logm2"]), float(r["pred_logm2"]), float(r["log_factor"])))
    return {n: np.array(v) for n, v in ev.items()}


def stats(t, q, lf, full):
    d = q - t
    qb = np.quantile(lf, np.linspace(0, 1, 21))
    b = np.clip(np.searchsorted(qb, lf, side="right") - 1, 0, 19)
    m = np.array([d[b == i].mean() if (b == i).any() else 0.0 for i in range(20)])
    w = np.bincount(b, minlength=20) / d.size
    hi, c = two_means(d)
    within = np.sqrt((((d[hi] - c[1]) ** 2).sum() + ((d[~hi] - c[0]) ** 2).sum()) / d.size)
    r = {"n": int(d.size), "mse": float((d ** 2).mean()), "bias": float(d.mean()), "sd_d": float(d.std()),
         "sd_logF": float(lf.std()), "spearman_d_logF": spearman(d, lf) if lf.std() > 0 else 0.0,
         "eta2": float((w * (m - d.mean()) ** 2).sum() / d.var()),
         "groups": {"lo_share": float((~hi).mean()), "lo_mean_d": float(c[0]), "hi_mean_d": float(c[1]),
                    "separation": float((c[1] - c[0]) / within),
                    "auc_logF": auc(lf[hi], lf[~hi]) if hi.any() and (~hi).any() else 0.5}}
    if full:
        f_edges = np.linspace(*np.quantile(lf, [0.0, 1.0]), 41)
        r["hist2d"] = {"d_edges": D_EDGES.tolist(), "logF_edges": f_edges.tolist(),
                       "counts": np.histogram2d(np.clip(d, -10, 10), lf, [D_EDGES, f_edges])[0].astype(int).tolist()}
        r["binned"] = {"logF_mid": [float(np.median(lf[b == i])) for i in range(20)], "mean_d": m.tolist()}
    return r


def fig(path):
    sys.path.insert(0, ROOT)
    import plot_style as ps
    from matplotlib.colors import LogNorm
    sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_daniel"))
    from heldout_delta import proc_label
    meta = {"dc_" + p["folder"]: p["process"] for p in
            json.load(open(os.path.join(ROOT, "analysis", "catalog_daniel", "classes_default.json")))["processes"]}
    o = json.load(open(path))
    fig, axes = ps.figure(ncols=2, nrows=2)
    for ax, n in zip(axes.ravel(), TTX):
        h = o["processes"][n]["hist2d"]
        im = ax.pcolormesh(h["logF_edges"], h["d_edges"], np.ma.masked_equal(np.array(h["counts"]), 0), cmap="viridis",
                           norm=LogNorm(), rasterized=True)
        ps.colorbar(ax, im, "events")
        bb = o["processes"][n]["binned"]
        ax.plot(bb["logF_mid"], bb["mean_d"], "o-", color=ps.C.vermillion, ms=2, lw=0.8, label="mean in bins of $\\log F$")
        ax.axhline(0, color="0.5", lw=0.5)
        ax.text(0.03, 0.95, proc_label(meta[n]), transform=ax.transAxes, va="top", color="k")
        ax.set_xlabel(r"$\log F$")
        ax.set_ylabel(r"pred $-$ true of $\log|\mathcal{M}|^2$")
    ps.legend(axes.ravel()[0], "lower right")
    ps.save(fig, os.path.join(HERE, "figs", os.path.splitext(os.path.basename(path))[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig", default=None, help="draw the figure from this JSON")
    a = ap.parse_args()
    if a.fig:
        return fig(a.fig)
    ev = read(a.csv)
    out = {"csv": a.csv, "processes": {n: stats(v[:, 0], v[:, 1], v[:, 2], n in TTX) for n, v in ev.items()}}
    s = np.array([p["spearman_d_logF"] for p in out["processes"].values()])
    e = np.array([p["eta2"] for p in out["processes"].values()])
    out["all"] = {"n_processes": len(s), "median_abs_spearman": float(np.median(np.abs(s))), "median_eta2": float(np.median(e))}
    for n in TTX:
        p = out["processes"][n]
        print(n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items() if k not in ("hist2d", "binned")},
              file=sys.stderr)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    else:
        print("FACTOR_SPLIT " + json.dumps(out))


if __name__ == "__main__":
    main()
