"""Figure of the r2 sampler check (plan catalog_coverage step 78; data/r2_check.json, data/r2_where.json and the per-job
reads in data/multichannel_check_reads/ of the r2 jobs, current+grid_r1_budget12+grid_r2_budget18).

(a) fill per 0.25-dex quota bin (kept / quota, summed over the chunks of a class and seed; mean of seeds 42 and 43,
    bars from one seed to the other) against log10|M|^2 minus the class's reach (the highest candidate any policy drew
    in the r1 and r2 jobs at either seed), for r1 and r2 on the three classes still short under r2, with the 0.95 bar.
(b) per class and seed, the sampling seconds of r2 over the old sampling's against r1's over the old sampling's, all
    in the same job on the same node; the 2x bar on both axes and the diagonal.

    python analysis/catalog_v4/r2_check_fig.py      -> analysis/catalog_v4/figs/r2_check.{png,pdf}
    python analysis/catalog_v4/r2_check_fig.py r3   -> analysis/catalog_v4/figs/r3_check.{png,pdf} (plan step 79: the
        r3 jobs, current+grid_r2_budget18+grid_r3_budget18, data/r3_check.json; r2 against r3, the reach of r3_check.json,
        taken over the r1, r2 and r3 jobs)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

DATA = os.path.join(HERE, "data")
READS = os.path.join(DATA, "multichannel_check_reads")
TAG = "current+grid_r1_budget12+grid_r2_budget18"
SEEDS = (42, 43)
SHORT = {"v4_bb_bb": (r"$b b\to b b$", ps.C.green), "v4_bbx_ccxh": (r"$b\bar b\to c\bar c h$", ps.C.orange),
         "v4_uux_bgtxwp": (r"$u\bar u\to b g \bar t W^+$", ps.C.purple)}
SHORT_R3 = dict(SHORT, v4_uu_bbxuu=(r"$u u\to b\bar b u u$", ps.C.sky))
POL = {"grid_r1_budget12": ("r1", "--", "o", "none"), "grid_r2_budget18": ("r2", "-", "o", None)}
WINDOW = 2.5          # dex below the reach shown in (a)
# per revision: (job tag, check file, {policy: (label, line, marker, face)}, axis labels of (b))
REV = {"r2": (TAG, "r2_check.json", POL, ("first revision", "second revision"), SHORT),
       "r3": ("current+grid_r2_budget18+grid_r3_budget18", "r3_check.json",
              {"grid_r2_budget18": ("r2", "--", "o", "none"), "grid_r3_budget18": ("r3", "-", "o", None)},
              ("second revision", "third revision"), SHORT_R3)}


def fill_per_bin(cls, pol, seed, tag=TAG):
    runs = json.load(open(os.path.join(READS, f"{cls}_{tag}_{seed}.json")))["runs"]
    qc = runs[f"{pol}_{seed}"]["quota_chunks"]
    e = np.asarray(qc[0]["edges_log10"])
    q = np.sum([x["quota"] for x in qc], 0).astype(float)
    k = np.sum([x["kept_per_bin"] for x in qc], 0).astype(float)
    return e, q, k


def main(rev="r2"):
    tag, cfile, pols, (lab_old, lab_new), short = REV[rev]
    chk = json.load(open(os.path.join(DATA, cfile)))
    where = json.load(open(os.path.join(DATA, "r2_where.json"))) if rev == "r2" else None
    fig, (axa, axb) = ps.figure(ncols=2)
    out = {}
    for cls, (lab, col) in short.items():
        reach = (max(where[f"{cls}/{s}"]["reach_all_jobs"] for s in SEEDS) if where else
                 max(v["class_reach_log10"] for v in chk["classes"][cls].values()))
        for pol, (plab, ls, mk, mfc) in pols.items():
            per = [fill_per_bin(cls, pol, s, tag) for s in SEEDS]
            e, q = per[0][0], per[0][1]
            f = np.array([np.where(p[1] > 0, p[2] / np.maximum(p[1], 1), np.nan) for p in per])
            x = 0.5 * (e[:-1] + e[1:]) - reach
            sel = (x > -WINDOW) & (q > 0)
            m = np.nanmean(f, 0)
            axa.errorbar(x[sel], m[sel], yerr=[m[sel] - np.nanmin(f, 0)[sel], np.nanmax(f, 0)[sel] - m[sel]],
                         color=col, ls=ls, marker=mk, mfc=mfc if mfc else col, capsize=2)
            out[f"{cls}/{pol}"] = {"x": x[sel].round(3).tolist(), "fill_mean": m[sel].round(3).tolist()}
    axa.axhline(0.95, color=ps.C.grey, ls=":")
    axa.axvline(0.0, color=ps.C.grey, ls="-.")
    axa.set_xlabel(r"$\log_{10}|\mathcal{M}|^2 - \log_{10}|\mathcal{M}|^2_{\rm reach}$")
    axa.set_ylabel(r"kept / quota per bin")
    # (b): every class and seed
    xs, ys, cs = [], [], []
    for cls, v in chk["classes"].items():
        for s in SEEDS:
            r1, r2 = (v.get(f"{p}_{s}") for p in pols)
            if r1 and r2:
                xs.append(r1["cost_vs_current"]); ys.append(r2["cost_vs_current"])
                cs.append(short[cls][1] if cls in short else ps.C.grey)
    axb.scatter(xs, ys, c=cs)
    lim = [0.8, 3.3]
    axb.plot(lim, lim, color=ps.C.blue, ls="-")
    axb.axhline(2.0, color=ps.C.vermillion, ls="--"); axb.axvline(2.0, color=ps.C.vermillion, ls="--")
    axb.set_xlim(lim)
    axb.set_xlabel(f"time, {lab_old} / old sampling")
    axb.set_ylabel(f"time, {lab_new} / old sampling")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, marker="o", ls="") for _, c in short.values()]
    h += [Line2D([], [], color=ps.C.grey, marker="o", ls=""),
          Line2D([], [], color="k", ls="--", marker="o", mfc="none"), Line2D([], [], color="k", ls="-", marker="o"),
          Line2D([], [], color=ps.C.grey, ls=":"), Line2D([], [], color=ps.C.grey, ls="-."),
          Line2D([], [], color=ps.C.vermillion, ls="--"), Line2D([], [], color=ps.C.blue)]
    labs = [l for l, _ in short.values()] + ["other classes", lab_old, lab_new, "fill 0.95", "reach",
                                             "cost 2x", "equal cost"]
    ps.legend(axa, "lower left", handles=h, labels=labs, ncol=2)
    base = os.path.join(HERE, "figs", f"{rev}_check")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    ps.save(fig, base)
    tot = chk["total_cost"]
    print(f"{rev.upper()}_FIG", base, json.dumps({"total": tot, "n_points": len(xs), "over2_new": int(sum(y > 2 for y in ys)),
                                      "over2_old": int(sum(x > 2 for x in xs)),
                                      "new_above_old": int(sum(y > x for x, y in zip(xs, ys))), "fill": out}))


if __name__ == "__main__":
    main(*sys.argv[1:2])
