"""The order axis of the dimensionless round (D93): the short catalog design runs (trained on the catalog's tree-level pp
processes) read zero-shot on our one-loop pools and the trees of the same processes (recipes/catalog_loop_probes.yaml:
the 57 one-loop pools positive in every split, every sqrt(s) window, and 42 trees; scripts/job_catalog_norm.sh <arm>
eval_short loop). Per probe the error d = q - t of log|M|^2 under the run's shared standardization: MSE, offset (mean of
d) and shape (variance of d), as catalog_2to4_heldout.py. Groups: the trees, the one-loop pools that have a tree (paired,
same process and window: the loop over tree MSE ratio isolates the change of order), the one-loop pools without one, and
the trees and one-loop pools by initial state (e+e- against quarks; the catalog has no e+e- initial state).
DlessInOrd (the coupling-order factor) scores only the pure-order probes (mixed-order entries are refused), so every arm
is also reported on that common subset.
Inputs: data/catalog_loop_short_<arm>.npz (collect_heldout_preds.py on $SCRATCH/zz_catalog_norm_short_<arm>_sigstage_loop).
Output: the table printed, data/catalog_loop_heldout.json.
    python analysis/catalog_daniel/catalog_loop_heldout.py
"""
import json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"DlessIn": "dimensionless inputs", "DlessInOrd": "dimensionless inputs + coupling orders"}
LOOP = ("_nlo_", "_loop_")


def stats(z, ps):
    d = [z[f"{p}__q"].astype(float) - z[f"{p}__t"].astype(float) for p in ps]
    return dict(n=len(ps), mse=float(np.median([np.mean(x ** 2) for x in d])) if d else None,
                offset=float(np.median([x.mean() for x in d])) if d else None,
                shape=float(np.median([x.var() for x in d])) if d else None)


def mse(z, p):
    return float(np.mean((z[f"{p}__q"].astype(float) - z[f"{p}__t"].astype(float)) ** 2))


def main():
    zs = {a: np.load(os.path.join(HERE, "data", f"catalog_loop_short_{a}.npz")) for a in ARMS
          if os.path.exists(os.path.join(HERE, "data", f"catalog_loop_short_{a}.npz"))}
    probes = {a: sorted({k.split("__")[0] for k in z.files}) for a, z in zs.items()}
    common = sorted(set.intersection(*map(set, probes.values())))
    res = {"common_n": len(common)}
    for a, z in zs.items():
        for sub, P in (("all", probes[a]), ("common", common)):
            loops = [p for p in P if any(s in p for s in LOOP)]
            trees = [p for p in P if p not in loops]
            tree_of = {p: p.replace("_nlo_", "_").replace("_loop_", "_") for p in loops}
            paired = [p for p in loops if tree_of[p] in trees]
            ee = lambda L: [p for p in L if p.startswith("ee_")]
            qq = lambda L: [p for p in L if not p.startswith("ee_")]
            g = {"trees": trees, "loops paired": paired, "their trees": [tree_of[p] for p in paired],
                 "loops unpaired": [p for p in loops if p not in paired],
                 "trees ee": ee(trees), "trees qq": qq(trees), "loops ee": ee(loops), "loops qq": qq(loops)}
            for k, L in g.items():
                res[f"{a}|{sub}|{k}"] = stats(z, L)
            r = [mse(z, p) / mse(z, tree_of[p]) for p in paired]
            res[f"{a}|{sub}|loop/tree"] = dict(n=len(r), q=[float(x) for x in np.quantile(r, [.1, .5, .9])] if r else None)
            print(f"{ARMS[a]:40s} {sub:6s} " + "  ".join(
                f"{k} n={res[f'{a}|{sub}|{k}']['n']} {res[f'{a}|{sub}|{k}']['mse'] or float('nan'):.3g}" for k in g)
                + f"  loop/tree {res[f'{a}|{sub}|loop/tree']['q']}")
    json.dump(res, open(os.path.join(HERE, "data", "catalog_loop_heldout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
