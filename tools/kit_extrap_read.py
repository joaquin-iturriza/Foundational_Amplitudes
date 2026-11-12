#!/usr/bin/env python3
"""How our transformer and Bakar's MLP extrapolate into the KIT limits (eval-only, existing predictions): per stratum
the median truth log10|M|^2, each model's median prediction and median signed error (prediction - truth), for the
softpair transformer runs and the D403 ladder's mlp and ours arms (seeds 0-2); and the transformer's validation curve
(val_loss_no_reg per validation, from per_process_metrics.json) to read whether 8000 steps converged.

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
from analysis.common.readers import run_metrics
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
    m = run_metrics(d)
    e = {"transformer": read(f"{d}/kit_benchmark/predictions.h5", stem, target),
         "val_curve": [float(x) for x in m["val_loss_no_reg"]]}
    for arm in ("mlp", "ours"):
        for s in range(3):
            g = glob.glob(f"runs/kit_mlp/kit_A_{proc}_n100000_t*_mlp{arm}_s{s}/kit_benchmark/predictions.h5")
            if g:
                e[f"{arm}_s{s}"] = read(g[0], stem, target)
    out[proc] = e
print("KITEXTRAP " + json.dumps(out))
'''


def plot(path, figs):
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    from tools.kit_figures import family, edges
    import plot_style as ps
    d = json.load(open(path))
    for proc, e in d.items():
        fams = {}
        for k in e["transformer"]:
            if k.startswith("hard_test/"):
                fams.setdefault(family(k.split("/")[1]), []).append(k)
        for fam, ks in fams.items():
            ks = sorted(ks, key=lambda k: edges(k.split("/")[1])[0], reverse=fam != "sudakov_sqrts")
            x = np.arange(len(ks))
            fig, ax = ps.figure() if hasattr(ps, "figure") else plt.subplots()
            ax.plot(x, [e["transformer"][k]["err"] for k in ks], "o-", label="transformer")
            for arm, mk in (("mlp", "s-"), ("ours", "^-")):
                v = [[e[f"{arm}_s{s}"][k]["err"] for k in ks] for s in range(3) if f"{arm}_s{s}" in e]
                if v:
                    ax.plot(x, np.median(v, 0), mk, label={"mlp": "MLP", "ours": "MLP, our choices"}[arm])
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xticks(x, [k.split("/")[1].split("_", 2)[-1].replace("_", "-") for k in ks], rotation=30)
            ax.set_ylabel("median (pred - truth) [log10|M|^2]")
            ax.legend()
            base = os.path.join(figs, f"kit_extrap_{proc}_{fam}")
            fig.savefig(base + ".png", dpi=150, bbox_inches="tight"); fig.savefig(base + ".pdf", bbox_inches="tight")
            plt.close(fig)
            print(base + ".png")


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
