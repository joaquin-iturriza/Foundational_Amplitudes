#!/usr/bin/env python3
"""Absolute zero-shot error on the held-out 2->4 classes against width and length (plan v3_prep_explore step 11,
catalog_v4; Joaquin 2026-11-04, D274: the absolute error, not the gap to training).

Runs (catalog_coverage's, het loss, share-0.25 default mixture, whole catalog_v4 training set): widths 8/16/32 at
4264 steps (1x), seeds 42 and 43; the 4x (17056 steps) and 16x (68224 steps) sweeps' best trials, width 8, seed 42,
read by eval jobs. Every read is of the run's final model (D276: mu at the best checkpoint by aggregate
val_loss_no_reg, then the sigma fit; the zero-shot parent is the run's sigma_stage).
Reads, MSE of log|M|^2 per probe (mse_logm2_real, data/reads/<run>_<axis>.json, mixture_ab.py extract):
  catalog-sampled  the 218 H2 pools (held-out 6-particle classes, 2000 validation events each)
  default-sampled  the PP4 and EE4 probes whose catalog_v4 class is an H2 class (D227: the held-out check on events
                   generated the default way)
Per read: the absolute MSE = median over classes of the class's geometric mean over its probes (catalog_coverage's
convention, mixture_ab.py _one_point), and R = median over classes of the class's geometric mean of run / reference
on the same probes. Width: w16, w32 over w8 per seed. Length (one seed): 4x best over 1x w8 (base HPs against tuned:
length and tuning together), 16x best over 4x best (the same 8-trial sweep's best each; both bests' lr at the window
top). No verdict rule: a read of existing runs, reported as numbers.
    python3 analysis/catalog_v4/abs_multiplicity.py  -> data/abs_multiplicity.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
from mixture_ab import WIDTH, class_of, class_ratios, gmean, read_of, _held_axes  # noqa: E402

LENGTH = {"x1_w8": WIDTH["w8"][0], "x4_best": "FA-ccin2p3-6f8a1eed", "x16_best": "FA-ccin2p3-1a46f3e0"}


def reads(run, cls, h2cls):
    cat = read_of(run, "H2")
    dflt = {}
    for ax in ("PP4", "EE4"):
        d = read_of(run, ax)
        if d is None:
            return None
        dflt.update({(ax, p): x for p, x in d.items() if cls(p) in h2cls})
    return {"catalog": {("H2", p): x for p, x in cat.items()}, "default": dflt}


def absolute(x, cl):
    by = {}
    for k, v in x.items():
        by.setdefault(cl(k), []).append(v)
    return float(np.median([gmean(v) for v in by.values()])), len(by), len(x)


def ratio(a, b, cl):
    ks = sorted(set(a) & set(b), key=str)
    r = class_ratios(a, b, ks, cl)
    v = np.array(list(r.values()))
    return {"R": round(float(np.median(v)), 4), "frac_classes_below_1": round(float(np.mean(v < 1)), 4),
            "n_classes": len(v), "n_probes": len(ks)}


def main():
    cls = class_of()
    ax_of = _held_axes(cls)
    h2cls = {c for c, ax in ax_of.items() if ax == "H2"}
    cl = lambda k: cls(k[1])
    runs = {f"{w}_s{s}": WIDTH[w][i] for w in WIDTH for i, s in enumerate((42, 43))}
    runs.update({k: v for k, v in LENGTH.items() if k != "x1_w8"})
    R = {name: reads(run, cls, h2cls) for name, run in runs.items()}
    miss = [n for n, v in R.items() if v is None]
    if miss:
        sys.exit("reads missing: " + ", ".join(miss))
    out = {"design": "plan v3_prep_explore step 11 (D274)", "runs": runs, "absolute": {}, "width": {}, "length": {}}
    for n, v in R.items():
        out["absolute"][n] = {}
        for side, x in v.items():
            m, nc, npr = absolute(x, cl)
            out["absolute"][n][side] = {"mse": round(m, 4), "n_classes": nc, "n_probes": npr}
    for side in ("catalog", "default"):
        out["width"][side] = {f"{w}_over_w8_s{s}": ratio(R[f"{w}_s{s}"][side], R[f"w8_s{s}"][side], cl)
                              for w in ("w16", "w32") for s in (42, 43)}
        out["length"][side] = {"x4_over_x1": ratio(R["x4_best"][side], R["w8_s42"][side], cl),
                               "x16_over_x4": ratio(R["x16_best"][side], R["x4_best"][side], cl),
                               "x16_over_x1": ratio(R["x16_best"][side], R["w8_s42"][side], cl)}
    json.dump(out, open(os.path.join(DATA, "abs_multiplicity.json"), "w"), indent=1)
    print(json.dumps({"absolute": {n: {s: d["mse"] for s, d in v.items()} for n, v in out["absolute"].items()},
                      "n": {s: (out["absolute"]["w8_s42"][s]["n_classes"], out["absolute"]["w8_s42"][s]["n_probes"])
                            for s in ("catalog", "default")},
                      "width": {s: {k: d["R"] for k, d in v.items()} for s, v in out["width"].items()},
                      "length": {s: {k: d["R"] for k, d in v.items()} for s, v in out["length"].items()}}))


if __name__ == "__main__":
    main()
