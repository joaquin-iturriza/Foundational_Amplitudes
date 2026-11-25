"""llm_compare step 15 (D564): what sets a new family's level. Our arms' error on the 43 new held-out processes is mostly
the level, bias = mean(pred - true) of log|M|^2 (error_split.py). Here that bias is set beside the level error of the
small level networks of analysis/catalog_daniel/norm_explore3.py (2x64 MLP, 3 seeds, fitted on the 1352 training
processes' mean log|M|^2 from the process alone, no events): "process features" (multiplicity, coupling orders,
particle-kind counts, flavour flow; stats_net.py) and "+ diagrams" (the leading coupling product, the sum over diagrams,
their number, massive internal lines). Same level definition (target B: log|M|^2, pool mean of the train pool; ours:
mean over the first 1e4 test events, the two agree to 0.01). Per arm on the 43 new and 96 relabels: median bias^2 and
bias^2 / var_p (var_p the process's own variance of log|M|^2, the unit norm_explore3 reports in), how many processes the
level network places better, and the Pearson/Spearman correlation of our signed bias with the level network's signed
error (does our network err where a level read from the process alone errs, and in the same direction?). The LLM's
whole MSE is an upper bound on its own bias^2 (his figure gives no split).
Also writes data/split/lvl_feat_heldout.json and lvl_diag_heldout.json (bias_logm2 = the level network's signed error)
so heldout_per_process.py --part bias draws them under fixed names. And per arm, the MSE once its own predicted mean\nper process is replaced by a level network's (no held-out label used): MSE - bias^2 + (that network's error)^2.
    python analysis/llm_compare/level_source.py [--tag a_trial5 b_h16 c_control c_best]
Output: data/step15_level_source.json.
"""
import argparse, json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DAN = os.path.join(ROOT, "analysis", "catalog_daniel")
NETS = {"lvl_feat": "B|nodiag|sampled", "lvl_diag": "B|diag|sampled"}


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["a_trial5", "b_h16", "c_control", "c_best"])
    a = ap.parse_args()
    ne = json.load(open(os.path.join(DAN, "data", "norm_explore3.json")))
    draft = json.load(open(os.path.join(ROOT, "talks", "progress_oct2026b", "data", "draft_pid_by_process.json")))
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(DAN, "classes_default.json")))["processes"]}
    sets = {"new": sorted(draft["extrap"]), "relabels": sorted(draft["interp"])}
    assert (len(sets["new"]), len(sets["relabels"])) == (43, 96)
    ours = {t: json.load(open(os.path.join(HERE, "data", "split", f"{t}_heldout.json")))["processes"] for t in a.tag}
    sd = {n: ours[a.tag[0]][n]["true_std_logm2"] for n in ours[a.tag[0]]}
    for k, key in NETS.items():
        sg = ne[key]["signed"]
        assert set(sg) == set(sd), f"{key}: held-out sets differ"
        # level definitions agree: the pool's sd against the test events' sd
        assert max(abs(ne[key]["sd"][n] / sd[n] - 1) for n in sd) < 0.05
        json.dump({"source": f"analysis/catalog_daniel/norm_explore3.py {key}",
                   "processes": {n: {"bias_logm2": sg[n]} for n in sg}},
                  open(os.path.join(HERE, "data", "split", f"{k}_heldout.json"), "w"), indent=1)
    bias = {t: {n: q["bias_logm2"] for n, q in ours[t].items()} for t in a.tag}
    bias.update({k: ne[key]["signed"] for k, key in NETS.items()})
    llm = {n: v[1] for g in draft.values() for n, v in g.items()}
    out = {}
    for s, names in sets.items():
        v2 = np.array([sd[n] ** 2 for n in names])
        L = np.array([llm[n] for n in names])
        r = {"llm_mse_median": float(np.median(L)), "llm_mse_over_var_median": float(np.median(L / v2))}
        for t, b in bias.items():
            x = np.array([b[n] for n in names])
            r[t] = {"bias2_median": float(np.median(x ** 2)), "bias2_over_var_median": float(np.median(x ** 2 / v2)),
                    "bias2_below_llm_mse": int((x ** 2 < L).sum()), "mean_bias": float(x.mean())}
            for k in NETS:
                y = np.array([bias[k][n] for n in names])
                r[t][f"vs_{k}"] = {"pearson": float(np.corrcoef(x, y)[0, 1]), "spearman": spearman(x, y),
                                   "same_sign": int((np.sign(x) == np.sign(y)).sum()),
                                   "net_better": int((y ** 2 < x ** 2).sum())}
        # each arm's own predicted mean replaced by a level network's (no held-out label used):
        # MSE' = MSE - bias^2 + (level network's signed error)^2; pooled over equal 1e4 events per process
        for t in a.tag:
            mse = np.array([ours[t][n]["mse_logm2"] for n in names])
            x = np.array([bias[t][n] for n in names])
            for k in NETS:
                m2 = mse - x ** 2 + np.array([bias[k][n] for n in names]) ** 2
                r[t][f"level_from_{k}"] = {"median_mse": float(np.median(m2)), "pooled_rmse": float(np.sqrt(m2.mean())),
                                           "below_llm": int((m2 < L).sum())}
            r[t]["as_is"] = {"median_mse": float(np.median(mse)), "pooled_rmse": float(np.sqrt(mse.mean())),
                             "below_llm": int((mse < L).sum())}
        r["llm_pooled_rmse"] = float(np.sqrt(L.mean()))
        fams = {}
        for n in names:
            fams.setdefault(meta[n]["test_group"], []).append(n)
        r["families"] = {g: {"n": len(ns), "llm_mse_median": float(np.median([llm[n] for n in ns])),
                             **{t: float(np.median([bias[t][n] ** 2 for n in ns])) for t in bias}}
                         for g, ns in fams.items()}
        r["per_process"] = {meta[n]["process"]: {"true_mean": ours[a.tag[0]][n]["true_mean_logm2"], "sd": sd[n],
                                                 "llm_mse": llm[n], **{t: bias[t][n] for t in bias}} for n in names}
        out[s] = r
    p = os.path.join(HERE, "data", "step15_level_source.json")
    json.dump(out, open(p, "w"), indent=1)
    for s, r in out.items():
        print(f"== {s}: LLM MSE median {r['llm_mse_median']:.3g} (/var {r['llm_mse_over_var_median']:.3g})")
        for t in bias:
            q = r[t]
            print(f"  {t:10s} bias2 med {q['bias2_median']:.3g} /var {q['bias2_over_var_median']:.3g} "
                  f"below LLM MSE {q['bias2_below_llm_mse']} mean bias {q['mean_bias']:+.2f} | " +
                  " ".join(f"{k}: r {q[f'vs_{k}']['pearson']:+.2f} rho {q[f'vs_{k}']['spearman']:+.2f} "
                           f"same sign {q[f'vs_{k}']['same_sign']} net better {q[f'vs_{k}']['net_better']}"
                           for k in NETS))
        print(f"  LLM pooled {r['llm_pooled_rmse']:.3f}")
        for t in a.tag:
            print(f"  {t:10s} " + " | ".join(f"{v}: median {r[t][v]['median_mse']:.3g} pooled {r[t][v]['pooled_rmse']:.3f} "
                                             f"below LLM {r[t][v]['below_llm']}"
                                             for v in ["as_is"] + [f"level_from_{k}" for k in NETS]))
    print("wrote", p)


if __name__ == "__main__":
    main()
