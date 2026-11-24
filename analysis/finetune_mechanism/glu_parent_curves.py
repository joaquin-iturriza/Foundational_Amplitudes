"""Plan finetune_mechanism step 2 (H2 dilution): the validation curves of the 9-process gluon/Z parent
(FA-ccin2p3-1f2d9ad5) beside the 28-process no-settings parent (FA-ccin2p3-9a73698a) on the 9 processes they share,
per validation of the main stage (val_loss_no_reg per process, HETEROSC NLL), with the training loss. Read before any
word on how the gluon parent trained.

    python3 analysis/finetune_mechanism/glu_parent_curves.py [--cached]   -> data/glu_parent_curves.json, figs/glu_parent_curves
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from analysis.common.readers import site_python  # noqa: E402
from parent_rerun_check import PARENTS  # noqa: E402

REMOTE = r'''
import json, os, sys
out = {}
for a in sys.argv[1:]:
    tag, d = a.split("#")
    m = json.load(open(os.path.join(d, "plots_0", "per_process_metrics.json")))
    out[tag] = {"combined": m["val_loss_no_reg"], "proc": m["proc_val_losses_no_reg"],
                "train": {k: v for k, v in m.items() if "train" in k.lower() and isinstance(v, list)}}
print("GLUC " + json.dumps(out))
'''


def main():
    cache = os.path.join(HERE, "data", "glu_parent_curves.json")
    if "--cached" in sys.argv:
        raw = json.load(open(cache))
    else:
        raw = site_python(REMOTE, [f"{t}#{PARENTS[t][1]}" for t in ("glu", "noset")], "GLUC")
        json.dump(raw, open(cache, "w"))
    procs = sorted(raw["glu"]["proc"])
    res = {}
    for t in ("glu", "noset"):
        c = np.array(raw[t]["combined"])
        b = int(c.argmin())   # best checkpoint: lowest aggregate val_loss_no_reg, every process read there
        res[t] = {"best_idx": b, "best_val": b + 1, "n_val": len(c), "combined_at_best": float(c[b]),
                  "proc": {p: {"first": raw[t]["proc"][p][0], "at_best": raw[t]["proc"][p][b],
                               "last": raw[t]["proc"][p][-1]}
                           for p in procs if p in raw[t]["proc"]},
                  "train_keys": sorted(raw[t]["train"])}
    print(json.dumps(res))
    plot(raw, procs)


LAB = {"ee_ZZ": r"$e^+e^-\to ZZ$", "ee_bbbarg": r"$e^+e^-\to b\bar bg$", "ee_bbbargg": r"$e^+e^-\to b\bar bgg$",
       "ee_uug": r"$e^+e^-\to u\bar ug$", "ee_uugg": r"$e^+e^-\to u\bar ugg$", "udbar_Wg": r"$u\bar d\to Wg$",
       "udbar_Wgg": r"$u\bar d\to Wgg$", "udbar_Wggg": r"$u\bar d\to Wggg$", "udbar_enug": r"$u\bar d\to e\nu g$"}


def plot(raw, procs):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2, nrows=2)
    (a, b), (h, cell) = axs
    cols = dict(zip(procs, ps.sequence(len(procs))))
    for ax, t, lab in ((a, "glu", "9-process parent"), (b, "noset", "28-process parent")):
        for p in procs:
            v = raw[t]["proc"].get(p)
            if v is not None:
                ax.plot(np.arange(1, len(v) + 1) / len(v), v, color=cols[p], label=LAB.get(p, p))
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("training progress (validation / 50)")
        ax.set_ylabel("val NLL per process")
        ps.process_label(ax, lab)
    # per_process_metrics.json holds no training-loss series: panel (c) is validation only
    for t, col, lab in (("glu", ps.C.vermillion, "9-process parent"), ("noset", ps.C.blue, "28-process parent")):
        c = np.mean([raw[t]["proc"][p] for p in procs], axis=0)
        h.plot(np.arange(1, len(c) + 1) / len(c), c, color=col, label=lab)
    h.set_yscale("symlog", linthresh=1.0)
    h.set_xlabel("training progress (validation / 50)"); h.set_ylabel("val NLL, mean of the 9")
    ps.legend(h)
    ps.legend_cell(cell, a)
    ps.save(fig, os.path.join(HERE, "figs", "glu_parent_curves"))


if __name__ == "__main__":
    main()
