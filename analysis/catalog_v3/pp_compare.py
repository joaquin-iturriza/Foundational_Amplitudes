"""The pp study (D117, scope D119): does training on catalog_v3's shaped sqrt(s_hat) windows serve pp-sampled use, maybe
more than training on pp? Every catalog_v3 short arm read zero-shot on the 13 TeV MadEvent pools of the 1396 hadronic
tree classes (scripts/job_catalog_v3_pp_score.sh; recipes/catalog_v3_pp_test.yaml: 410 held-out classes of H1, H2, H3
and the 986 training classes; each pp13_<class> is its class's |M|^2 function on pp-distributed events, 2000 validation
events). The predictions stay on the site ($SCRATCH/zz_v3_pp_<arm>/<probe>.npz); `collect`, run there, reduces them to
one row per probe and adds each training class's window loss, read from the arm's own run at its best checkpoint
(result.json proc_val_losses_no_reg, 200 validation events of the class's shaped window, times the shared
standardization's variance: log|M|^2 units, EMA weights as the pp read), and prints the rows as JSON:
    site run --item pp_sampling ccin2p3 FA -- python analysis/catalog_v3/pp_compare.py collect > analysis/catalog_v3/data/pp_rows.json
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
Without the pp-trained arm's reads, panel (a) is the four window-trained arms' pp MSE against the window MSE on the
training classes (same arm, same class), panel (b) as above.
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


def collect():
    """on the site: every arm's pp predictions reduced to rows, and its training classes' window losses"""
    out = {}
    for arm in ARMS:
        pred = os.path.join(os.environ["SCRATCH"], f"zz_v3_pp_{arm}")
        files = sorted(glob.glob(os.path.join(pred, "*.npz")))
        if not files:
            continue
        rows, bad = {}, []
        for f in files:
            z = np.load(f)
            p = os.path.basename(f)[:-4]
            if "q" not in z.files or len(z["q"]) != len(z["t"]) or not len(z["t"]):
                bad.append(p)                  # no prediction in the parent's units: listed, never dropped silently
                continue
            d = z["q"].astype(float) - z["t"].astype(float)
            rows[p] = {"mse": float(np.mean(d ** 2)), "level": float(np.mean(d) ** 2), "shape": float(np.var(d)),
                       "var_t": float(np.var(z["t"].astype(float))), "n": int(len(d))}
        run = os.path.join(ROOT, "runs", "catalog_v3_short", arm)
        res = json.load(open(os.path.join(run, "result.json")))
        st = json.load(open(os.path.join(run, "data_stats.json")))
        sd = st["prepd_std"]
        assert st.get("shared_standardization") and max(sd) - min(sd) < 1e-9 * max(sd), f"{arm}: not one shared std"
        win = {k: float(v) * sd[0] ** 2 for k, v in res["proc_val_losses_no_reg"].items()}
        out[arm] = {"pp": rows, "no_prediction": bad, "window_train": win, "best_step": res.get("best_step"),
                    "shared_std": sd[0]}
    json.dump(out, sys.stdout)


def read(arm, rows_all):
    a = rows_all.get(arm)
    return (a["pp"], a["no_prediction"]) if a else (None, None)


def summ(rows):
    m = np.array([v["mse"] for v in rows.values()])
    return {"n": len(rows), "mse": float(np.median(m)), "mse_q10_q90": [float(x) for x in np.quantile(m, [0.1, 0.9])],
            "level": float(np.median([v["level"] for v in rows.values()])),
            "shape": float(np.median([v["shape"] for v in rows.values()]))}


def main():
    g = groups()
    sr_path = os.path.join(HERE, "data", "short_reads.json")
    sr = json.load(open(sr_path)) if os.path.exists(sr_path) else {}
    rows_all = json.load(open(os.path.join(HERE, "data", "pp_rows.json")))
    res = {}
    for arm in ARMS:
        rows, bad = read(arm, rows_all)
        if rows is None:
            print(f"{arm}: no pp reads")
            continue
        res[arm] = {"no_prediction": bad}
        for grp in GROUPS:
            exp = sorted(p for p, a in g.items() if a == grp)
            r = {p: rows[p] for p in exp if p in rows}
            ent = {"n_expected": len(exp), "missing": sorted(set(exp) - set(r))}
            if r:
                ent.update(summ(r))
                ent["probes"] = r
                # the same arm on the same classes' window pools: a held-out class's zero-shot window read
                # (data/short_reads.json, 2000 events), a training class's validation loss in the arm's own run
                # (200 events, its best checkpoint); a class the arm never trained on (the twin's removed ones) has
                # neither and is counted apart
                if grp == "train":
                    win = {k: {"mse": v} for k, v in rows_all[arm]["window_train"].items()}
                    ent["not_in_arm_training"] = sorted(p for p in r if "v3_" + p[len("pp13_"):] not in win)
                else:
                    win = sr.get(arm, {}).get(grp, {}).get("probes", {})
                pair = [(r[p]["mse"], win["v3_" + p[len("pp13_"):]]["mse"]) for p in r if "v3_" + p[len("pp13_"):] in win]
                ent["window_pairs"] = {p: win["v3_" + p[len("pp13_"):]]["mse"] for p in r if "v3_" + p[len("pp13_"):] in win}
                if pair:
                    lr = np.log10([a / b for a, b in pair])
                    ent["window_same_classes"] = {"n": len(pair), "mse_pp": float(np.median([a for a, _ in pair])),
                                                  "mse_window": float(np.median([b for _, b in pair])),
                                                  "ratio_pp_over_window_median": float(10 ** np.median(lr)),
                                                  "ratio_q10_q90": [float(10 ** x) for x in np.quantile(lr, [0.1, 0.9])],
                                                  "n_pp_worse": int((lr > 0).sum())}
            else:
                ent["n"] = 0
            res[arm][grp] = ent
            w = ent.get("window_same_classes")
            print(f"{arm:24s} {grp:5s} n={ent['n']:4d}/{len(exp):4d}" +
                  (f" mse {ent['mse']:.3g} level {ent['level']:.3g} shape {ent['shape']:.3g}" if ent["n"] else " NO READS") +
                  (f" | window {w['mse_window']:.3g} on the same {w['n']}, ratio {w['ratio_pp_over_window_median']:.3g}"
                   f" ({w['n_pp_worse']} worse on pp)" if w else ""))
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
    collect() if sys.argv[1:] == ["collect"] else main()
