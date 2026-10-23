"""Where the zero-shot level errors of the dimensionless round (D93) sit (2026-10-23). Three reads, CPU only, on the saved
held-out predictions:
1. range: per arm, each process's mean true target (log of |M|^2 times the arm's factor) for the 96 relabels of
   training processes (a proxy of the training processes' levels) against the 2->4 +g probes', and the probes' mean
   prediction: are the 2->4 levels inside the range the training spans?
2. loops: the per-pool offset (mean of q - t) of every one-loop probe, loop-induced against stripped virtuals.
3. top: the per-process offset of the twin trained without any top, by initial state and by a top pair in the final state.
Inputs: data/catalog_short_<arm>_sigstage.npz, data/catalog_2to4_short_<arm>.npz, data/catalog_loop_short_<arm>.npz,
data/catalog_top_short_DlessInOrd_notop.npz. Output: the tables printed, data/level_range.json.
    .venv-driver/bin/python analysis/catalog_daniel/level_range.py
"""
import json, os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
ARMS = {"Adim": "catalog_short_Adim_sigstage", "Dless": "catalog_short_Dless_sigstage",
        "Dpair_seed1": "catalog_short_Dpair_seed1_sigstage", "DlessIn": "catalog_short_DlessIn_sigstage",
        "DlessInOrd": "catalog_short_DlessInOrd_sigstage"}


def load(f):
    z = np.load(os.path.join(D, f + ".npz"))
    return z, sorted({k.split("__")[0] for k in z.files if k.endswith("__q")})


def off(z, p):
    return float(np.mean(z[f"{p}__q"].astype(float) - z[f"{p}__t"].astype(float)))


def main():
    twins = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    res = {"range": {}, "loops": {}, "top": {}}
    for arm, f in ARMS.items():
        z, ps = load(f)
        r = np.array([z[f"{p}__t"].astype(float).mean() for p in ps if p in twins])
        z4, ps4 = load(f"catalog_2to4_short_{arm}")
        g = [p for p in ps4 if p.endswith("_g")]
        t4 = np.array([z4[f"{p}__t"].astype(float).mean() for p in g]); q4 = np.array([z4[f"{p}__q"].astype(float).mean() for p in g])
        res["range"][arm] = dict(train_proxy=[float(r.min()), float(np.median(r)), float(r.max())],
                                 true_2to4=[float(t4.min()), float(np.median(t4)), float(t4.max())],
                                 pred_2to4_median=float(np.median(q4)), frac_above=float(np.mean(t4 > r.max())))
        print(f"{arm:12s} relabels [{r.min():.1f}, {np.median(r):.1f}, {r.max():.1f}]  2->4 +g true [{t4.min():.1f}, "
              f"{np.median(t4):.1f}, {t4.max():.1f}] predicted median {np.median(q4):.1f}  above the relabels' max "
              f"{np.mean(t4 > r.max()):.2f}")
    for arm in ("DlessIn", "DlessInOrd"):
        z, ps = load(f"catalog_loop_short_{arm}")
        o = {p: off(z, p) for p in ps if "_loop_" in p or "_nlo_" in p}
        li = [v for p, v in o.items() if "_loop_" in p]; vi = [v for p, v in o.items() if "_nlo_" in p]
        res["loops"][arm] = dict(per_pool=o, loop_induced_median=float(np.median(li)), virtual_median=float(np.median(vi)),
                                 n_loop_induced=len(li), n_virtual=len(vi))
        print(f"{arm:12s} loop-induced ({len(li)}) offset median {np.median(li):+.2f} [{min(li):+.2f}, {max(li):+.2f}]; "
              f"stripped virtuals ({len(vi)}) {np.median(vi):+.2f} [{min(vi):+.2f}, {max(vi):+.2f}]; "
              f"log (16 pi^2)^2 = {2 * np.log(16 * np.pi ** 2):.2f}")
    z, ps = load("catalog_top_short_DlessInOrd_notop")
    g = defaultdict(list)
    for p in ps:
        ini, fin = p.split("_")[1], p.split("_")[2]
        k = ("gg" if ini == "gg" else "qg" if "g" in ini else "qq") + (", top pair" if "ttx" in fin else ", single top")
        g[k].append(off(z, p))
    for k, v in sorted(g.items()):
        res["top"][k] = dict(n=len(v), median=float(np.median(v)), sd=float(np.std(v)))
        print(f"top, {k:18s} n={len(v):3d} offset median {np.median(v):+.2f} sd {np.std(v):.2f}")
    json.dump(res, open(os.path.join(D, "level_range.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
