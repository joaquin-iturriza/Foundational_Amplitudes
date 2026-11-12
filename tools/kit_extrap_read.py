#!/usr/bin/env python3
"""How our transformer and Bakar's MLP extrapolate into the KIT limits (eval-only, existing predictions): per stratum
the median truth log10|M|^2, each model's median prediction and median signed error (prediction - truth), for the
softpair transformer runs and the D403 ladder's mlp and ours arms (seeds 0-2); and the transformer's validation curve
(train_steps.npz: its val and train losses) to read whether 8000 steps converged.

    python tools/kit_extrap_read.py collect --out analysis/kit_benchmark/data/kit_extrap.json    (reads CC)
    python tools/kit_extrap_read.py plot analysis/kit_benchmark/data/kit_extrap.json --figs analysis/kit_benchmark/figs
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RUNS = {"ee_uug_me2": "kit_A_ee_uug_me2_n100000_t8000_softpair",              # FA-ccin2p3-c15b3337
        "ee_uugg_me2": "kit_A_ee_uugg_me2_n100000_t8000_softpair",            # FA-ccin2p3-c364a7ae
        "ee_tt_virt_finite": "kit_A_ee_tt_virt_finite_n100000_t8000_softpair"}  # FA-ccin2p3-f21fa163

SITE = r'''
import glob, json, os, sys
import numpy as np, h5py
sys.path.insert(0, ".")
from tools import kit_hardps as kh
RUNS = json.loads(open(sys.argv[1]).read()) if sys.argv[1].endswith(".json") else dict(a.split("=") for a in sys.argv[1:])
out = {}
def read(pred_path, stem, target):
    src = kh.h5_path(stem); r = {}
    with h5py.File(pred_path, "r") as P, h5py.File(src, "r") as T:
        for split in ("bulk_test", "hard_test"):
            for st in P[split]:
                k = list(P[split][st])[0]
                p = P[split][st][k][:]; t = T[split][st][k][:]
                r[f"{split}/{st}"] = {"truth": float(np.median(t)), "pred": float(np.median(p)),
                                      "err": float(np.median(p - t)), "logmse": float(np.mean((p - t) ** 2))}
    return r
for proc, run in RUNS.items():
    stem, target = (proc[:-4], "me2") if proc.endswith("_me2") else ("ee_tt", "virt_finite")
    d = f"runs/kit/{run}"
    z = np.load(f"{d}/train_steps.npz")
    e = {"transformer": read(f"{d}/kit_benchmark/predictions.h5", stem, target),
         "curves": {k: [float(x) for x in np.ravel(z[k])] for k in z.files if "val" in k or "train_loss" in k}}
    for arm in ("mlp", "ours"):
        for s in range(3):
            g = glob.glob(f"runs/kit_mlp/kit_A_{proc}_n100000_t*_mlp{arm}_s{s}/kit_benchmark/predictions.h5")
            if g:
                e[f"{arm}_s{s}"] = read(g[0], stem, target)
    out[proc] = e
print("KITEXTRAP " + json.dumps(out))
'''


FAMS = {"ee_uug_me2": ["coll_y", "soft_x"], "ee_uugg_me2": ["coll_qg_y", "coll_gg_y"],
        "ee_tt_virt_finite": ["sudakov_sqrts", "threshold_beta"]}


def plot(path, figs):
    import plot_style as ps
    from tools.kit_figures import family, edges, XLAB, PROC
    d = json.load(open(path))
    for proc, e in d.items():
        panels = ps.panels(len(FAMS[proc]))
        for (fig, ax), fam in zip(panels, FAMS[proc]):
            ks = [k for k in e["transformer"] if k.startswith("hard_test/") and family(k.split("/")[1]) == fam]
            ks = sorted(ks, key=lambda k: edges(k.split("/")[1])[0])
            x = np.array([np.sqrt(np.prod([max(v, 1e-12) for v in edges(k.split("/")[1])])) for k in ks])
            ax.plot(x, [e["transformer"][k]["err"] for k in ks], "o-", color=ps.C.vermillion, label="transformer")
            for arm, mk, c in (("mlp", "s-", ps.C.blue), ("ours", "^--", ps.C.green)):
                v = [[e[f"{arm}_s{s}"][k]["err"] for k in ks] for s in range(3) if f"{arm}_s{s}" in e]
                ax.plot(x, np.median(v, 0), mk, color=c, label={"mlp": "MLP", "ours": "MLP, our choices"}[arm])
            ax.axhline(0, color=ps.C.grey, lw=0.8)
            ax.set_xscale("log")
            from matplotlib.ticker import NullFormatter
            ax.xaxis.set_minor_formatter(NullFormatter())   # the Sudakov axis spans one decade: minor labels overlap
            ax.set_xlabel(XLAB.get(fam, fam))
            ax.set_ylabel(r"median prediction $-$ truth, $\log_{10}|\mathcal{M}|^2$")
            e_, f2, tg = proc.split("_", 2)
            ps.process_label(ax, PROC.get((f"{e_}_{f2}", tg), proc))
            ps.legend(ax, "lower left" if e["transformer"][ks[-1]]["err"] > 0 else "upper left")
            ps.make_room(ax)
        ps.save_panels(panels, os.path.join(figs, f"kit_extrap_{proc}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--out", required=True)
    p = sub.add_parser("plot"); p.add_argument("data"); p.add_argument("--figs", required=True)
    a = ap.parse_args()
    if a.cmd == "collect":
        from analysis.common.readers import site_python
        d = site_python(SITE, [f"{k}={v}" for k, v in RUNS.items()], "KITEXTRAP")
        json.dump(d, open(a.out, "w"))
        print(f"-> {a.out}")
    else:
        plot(a.data, a.figs)


if __name__ == "__main__":
    main()
