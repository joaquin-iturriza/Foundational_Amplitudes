"""The pp study (D117, scope D119): does training on catalog_v3's shaped sqrt(s_hat) windows serve pp-sampled use, maybe
more than training on pp? Every catalog_v3 short arm read zero-shot on the 13 TeV MadEvent pools of the 1396 hadronic
tree classes (scripts/job_catalog_v3_pp_score.sh; recipes/catalog_v3_pp_test.yaml: 410 held-out classes of H1, H2, H3
and the 986 training classes; each pp13_<class> is its class's |M|^2 function on pp-distributed events, 2000 validation
events). Predictions copied home from $SCRATCH/zz_v3_pp_<arm>/<probe>.npz into data/pp_reads/<arm>/.
  window-trained   baseline, dimensionless, dimensionless_context, dimensionless_twin (the four short pretrainings,
                   every hadronic class on its shaped windows)
  pp-trained       dimensionless_pp: the dimensionless arm with every hadronic training tree class on 13 TeV pp events
                   (the other pools on their windows, D119)
Per probe, from the saved truth t and the arm's prediction q (log|M|^2), as analysis/catalog_v3/short_reads.py:
  mse    mean (q - t)^2;  level  (mean (q - t))^2;  shape  var(q - t)
Per arm and group (H1, H2, H3, and the training classes, in distribution for the pp-trained arm only on pp events):
the medians over probes, the 10% and 90% MSE quantiles, the count against the recipe, every missing probe listed by name.
For the held-out classes, the same arm's read of the same class on its window pool (data/short_reads.json) beside it.
Figure: (a) per probe, the pp-trained arm's MSE against the window-trained dimensionless arm's, on the pp pools;
(b) the window-trained dimensionless arm's MSE per class on the pp pool against its window pool.
    python3 analysis/catalog_v3/pp_compare.py   -> data/pp_compare.json, figs/pp_compare.{png,pdf}
"""
import glob, json, os, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
ARMS = ("baseline", "dimensionless", "dimensionless_context", "dimensionless_twin", "dimensionless_pp")
GROUPS = ("H1", "H2", "H3", "train")
NAMES = {"H1": "new combinations", "H2": r"$2\to4$ families", "H3": "the top", "train": "training classes"}


def groups():
    """pp13_<x> -> its class's role: the held-out axis, or 'train'; every probe of the pp test recipe"""
    T = json.load(open(os.path.join(ROOT, "recipes", "catalog_v3_roles.json")))["trees"]
    names = [e["name"] for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_pp_test.yaml")))["processes"]]
    out = {}
    for n in names:
        r = T["v3_" + n[len("pp13_"):]]
        out[n] = "train" if r["role"] == "train" else r["axis"]
    return out


def read(arm):
    rows, bad = {}, []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "pp_reads", arm, "*.npz"))):
        z = np.load(f)
        p = os.path.basename(f)[:-4]
        if "q" not in z.files or len(z["q"]) != len(z["t"]) or not len(z["t"]):
            bad.append(p)                      # no prediction in the parent's units: listed, never dropped silently
            continue
        d = z["q"].astype(float) - z["t"].astype(float)
        rows[p] = {"mse": float(np.mean(d ** 2)), "level": float(np.mean(d) ** 2), "shape": float(np.var(d))}
    return rows, bad


def summ(rows):
    m = np.array([v["mse"] for v in rows.values()])
    return {"n": len(rows), "mse": float(np.median(m)), "mse_q10_q90": [float(x) for x in np.quantile(m, [0.1, 0.9])],
            "level": float(np.median([v["level"] for v in rows.values()])),
            "shape": float(np.median([v["shape"] for v in rows.values()]))}


def main():
    g = groups()
    sr_path = os.path.join(HERE, "data", "short_reads.json")
    sr = json.load(open(sr_path)) if os.path.exists(sr_path) else {}
    res = {}
    for arm in ARMS:
        if not os.path.isdir(os.path.join(HERE, "data", "pp_reads", arm)):
            print(f"{arm}: no pp reads copied home")
            continue
        rows, bad = read(arm)
        res[arm] = {"no_prediction": bad}
        for grp in GROUPS:
            exp = sorted(p for p, a in g.items() if a == grp)
            r = {p: rows[p] for p in exp if p in rows}
            ent = {"n_expected": len(exp), "missing": sorted(set(exp) - set(r))}
            if r:
                ent.update(summ(r))
                ent["probes"] = r
                # the same arm on the same classes' window pools (the held-out axes only: the training classes were
                # not read on their windows)
                win = sr.get(arm, {}).get(grp, {}).get("probes", {})
                pair = [(r[p]["mse"], win["v3_" + p[len("pp13_"):]]["mse"]) for p in r if "v3_" + p[len("pp13_"):] in win]
                if pair:
                    ent["window_same_classes"] = {"n": len(pair), "mse_pp": float(np.median([a for a, _ in pair])),
                                                  "mse_window": float(np.median([b for _, b in pair]))}
            else:
                ent["n"] = 0
            res[arm][grp] = ent
            w = ent.get("window_same_classes")
            print(f"{arm:24s} {grp:5s} n={ent['n']:4d}/{len(exp):4d}" +
                  (f" mse {ent['mse']:.3g} level {ent['level']:.3g} shape {ent['shape']:.3g}" if ent["n"] else " NO READS") +
                  (f" | window {w['mse_window']:.3g} on the same {w['n']}" if w else ""))
    json.dump(res, open(os.path.join(HERE, "data", "pp_compare.json"), "w"), indent=1)
    plot(res, sr)


def plot(res, sr):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    marks = {"H1": ("o", ps.C.blue), "H2": ("s", ps.C.green), "H3": ("^", ps.C.vermillion), "train": ("D", ps.C.grey)}
    # (a) pp-trained against window-trained, both on the pp pools
    ax = axs[0]
    lo, hi = np.inf, 0
    for grp in GROUPS:
        a = res.get("dimensionless", {}).get(grp, {}).get("probes", {})
        b = res.get("dimensionless_pp", {}).get(grp, {}).get("probes", {})
        ps_ = sorted(set(a) & set(b))
        if ps_:
            x, y = [a[p]["mse"] for p in ps_], [b[p]["mse"] for p in ps_]
            ax.plot(x, y, marks[grp][0], mfc="none", color=marks[grp][1], label=NAMES[grp])
            lo, hi = min(lo, min(x), min(y)), max(hi, max(x), max(y))
    if hi > 0:
        ax.plot([lo, hi], [lo, hi], ls="--", color=ps.C.grey, label="equal")
    ax.set_xlabel("MSE, trained on windows"); ax.set_ylabel("MSE, trained on pp")
    # (b) window-trained: the same class on its pp pool against its window pool
    ax = axs[1]
    lo, hi = np.inf, 0
    for grp in ("H1", "H2", "H3"):
        a = res.get("dimensionless", {}).get(grp, {}).get("probes", {})
        w = sr.get("dimensionless", {}).get(grp, {}).get("probes", {})
        ps_ = sorted(p for p in a if "v3_" + p[len("pp13_"):] in w)
        if ps_:
            x = [w["v3_" + p[len("pp13_"):]]["mse"] for p in ps_]
            y = [a[p]["mse"] for p in ps_]
            ax.plot(x, y, marks[grp][0], mfc="none", color=marks[grp][1], label=NAMES[grp])
            lo, hi = min(lo, min(x), min(y)), max(hi, max(x), max(y))
    if hi > 0:
        ax.plot([lo, hi], [lo, hi], ls="--", color=ps.C.grey, label="equal")
    ax.set_xlabel("MSE, window pool"); ax.set_ylabel("MSE, pp pool")
    for ax in axs:
        ax.set_xscale("log"); ax.set_yscale("log")
    ps.legend(axs[0], "upper left")
    for ax in axs:
        ps.make_room(ax, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "pp_compare"))


if __name__ == "__main__":
    main()
