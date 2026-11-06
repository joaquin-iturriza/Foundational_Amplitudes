#!/usr/bin/env python3
"""plan warm_start (d): the compared quantities themselves for three results, one row each:
  tuned    (D284, par:v4_warm_start_tuned): the k=533 warm start untuned (lr_scale 1) and tuned on the NLL (hp_1860),
           the parent A (het control) and the share-0.25 change from scratch B;
  ckpt     (D289): the tuned k=533 warm start from the parent rerun's step-2132, step-3198 and final models, and B;
  growth   (D285, par:v4_warm_start_grow): width 8 -> 16 grown and retrained 533 / 1066 / 2132 steps, w8 parent A,
           w16 from scratch B.
Columns: (1) the validation curve of the main training stage, val_loss_no_reg (het NLL, arithmetic mean over
processes, each run's own standardized target) against the run's own step, seed 42 (data/reads/<run>_val.json,
mixture_ab.py val); a warm start's first point is its parent's model at its first validation; (2, 3) each arm's
per-class zero-shot MSE of log|M|^2 on PP4 and EE4 (all classes) as an empirical CDF over classes: per probe the
geometric mean over the two seeds, per class the geometric mean over its probes (mixture_ab.class_of, the reads of
data/reads/<run>_<axis>.json, the runs as mixture_ab.WARM names them).
-> figs/warm_start_compare.{png,pdf}"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from mixture_ab import READS, TRAINED_BY, WARM, class_of  # noqa: E402


def pair(x, axis):
    return x[axis] if isinstance(x, dict) else x


GROUPS = {
    "tuned": [("parent A", WARM["C1"]["A"]), ("scratch B", WARM["C1"]["B"]),
              ("warm 533, untuned", WARM["C1"]["W"][533]), ("warm 533, NLL-tuned", WARM["C1t"]["W"][533])],
    "ckpt": [("scratch B", WARM["C1"]["B"]), ("from step 2132", WARM["C1ck2132"]["W"][533]),
             ("from step 3198", WARM["C1ck3198"]["W"][533]), ("from final", WARM["C1ckfinal"]["W"][533])],
    "growth": [("w8 parent A", WARM["C10w"]["A"]), ("w16 scratch B", WARM["C10w"]["B"])]
              + [(f"grown, {k} steps", WARM["C10w"]["W"][k]) for k in (533, 1066, 2132)],
}


def curve(run):
    f = os.path.join(READS, f"{TRAINED_BY.get(run, run)}_val.json")
    if not os.path.exists(f):
        return None
    v = np.array(json.load(open(f))["blocks"][0]["vals"], float)
    return v[:, 0], v[:, 1]


def class_mse(runs, axis, cls):
    reads = []
    for r in runs:
        f = os.path.join(READS, f"{r}_{axis}.json")
        if not os.path.exists(f):
            return None
        reads.append(json.load(open(f))["mse"])
    probes = set.intersection(*(set(m) for m in reads))
    per = {}
    for p in probes:
        x = [m[p] for m in reads]
        if all(np.isfinite(x)) and min(x) > 0:
            per.setdefault(cls(p), []).append(np.exp(np.mean(np.log(x))))
    return np.array(sorted(np.exp(np.mean(np.log(v))) for v in per.values()))


def main():
    cls = class_of()
    rows = list(GROUPS)
    fig, axs = plt.subplots(len(rows), 3, figsize=ps.figsize(3, len(rows)))
    out = {}
    for i, g in enumerate(rows):
        out[g] = {}
        lo = []
        for j, (lab, runs) in enumerate(GROUPS[g]):
            c = curve(pair(runs, "PP4")[0])
            col = f"C{j}"
            rec = {}
            if c is not None:
                axs[i, 0].plot(c[0], c[1], color=col, label=lab)
                rec["best_val"] = float(c[1].min()); lo.append(rec["best_val"])
            for k, axis in enumerate(("PP4", "EE4")):
                m = class_mse(pair(runs, axis), axis, cls)
                if m is None:
                    continue
                axs[i, k + 1].step(m, np.arange(1, len(m) + 1) / len(m), where="post", color=col, label=lab)
                rec[axis] = {"n_classes": len(m), "median": float(np.median(m)),
                             "q10": float(np.quantile(m, 0.1)), "q90": float(np.quantile(m, 0.9))}
            out[g][lab] = rec
        # the from-scratch runs' first validations sit at NLL up to 1e9 (sigma not yet fitted): cut at -1.5
        axs[i, 0].set_ylim(min(lo) - 0.1, -1.5)
        axs[i, 0].set_xscale("log"); axs[i, 0].set_xlabel("step"); axs[i, 0].set_ylabel("val NLL")
        axs[i, 0].legend(fontsize=6)
        for k, axis in enumerate(("PP4", "EE4")):
            a = axs[i, k + 1]
            a.set_xscale("log"); a.set_xlabel(axis + r" class MSE of $\log|M|^2$"); a.set_ylabel("fraction of classes")
    ps.layout(fig) if hasattr(ps, "layout") else fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    base = os.path.join(HERE, "figs", "warm_start_compare")
    fig.savefig(base + ".png", dpi=150); fig.savefig(base + ".pdf")
    json.dump(out, open(os.path.join(HERE, "data", "warm_start_compare.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
