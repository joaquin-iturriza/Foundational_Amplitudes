#!/usr/bin/env python3
"""plan warm_start (d): the compared quantities themselves for three results, one row each:
  tuned    (D284, par:v4_warm_start_tuned): the k=533 warm start untuned (lr_scale 1) and tuned on the NLL (hp_1860),
           the parent A (het control, share 0) and the share-0.25 change from scratch B;
  ckpt     (D289): the tuned k=533 warm start from the parent rerun's step-2132, step-3198 and final models, A and B
           (the rerun parent itself, FA-ccin2p3-d521160b/b688a859, has no zero-shot reads: the design reads each
           checkpoint against the warm start from the final model, not against the rerun);
  growth   (D285, par:v4_warm_start_grow): width 8 -> 16 grown and retrained 533 / 1066 / 2132 steps, w8 parent A,
           w16 from scratch B.
Labels are words (every warm start of the tuned and ckpt rows retrains 533 steps).
Columns: (1) the validation curve of the main training stage, val_loss_no_reg (het NLL, arithmetic mean over
processes) against the run's own step, seed 42 (data/reads/<run>_val.json, mixture_ab.py val), a diagnostic only.
Only runs on the same validation data and the same target standardization share a panel: in the tuned and ckpt
rows the warm starts (share-0.25 validation, the parent's target_stats) and neither A (share-0 validation) nor B (its
own target_stats); in the growth row all five (one recipe, v4_het_default25; the grown runs take the w8 parent's
stats, computed on the same training data). The y range is cut at -1.5 (the from-scratch first validations sit up to
1e9 before sigma is fitted). (2, 3) each arm's per-class zero-shot MSE of log|M|^2 on PP4 and EE4 (all classes) as
an empirical CDF over classes: per probe the geometric mean over the two seeds, per class the geometric mean over
its probes (mixture_ab.class_of, the reads of data/reads/<run>_<axis>.json: the model after the sigma stage, D276,
the runs as mixture_ab.WARM names them). A missing read stops the script (as mixture_ab); probes read by one seed
only or non-finite are counted per arm in the JSON, with each run's blow-up check (mixture_ab.blowup_of).
-> figs/warm_start_compare_{a,b,c}.{png,pdf} ((a) column 1, (b) 2, (c) 3), data/warm_start_compare.json"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402
from mixture_ab import READS, TRAINED_BY, WARM, blowup_of, class_of  # noqa: E402


def pair(x, axis):
    return x[axis] if isinstance(x, dict) else x


AXIS_WORDS = {"PP4": "pp probes", "EE4": r"$e^+e^-$ probes"}
# (label, runs, on the NLL panel)
GROUPS = {
    "tuned": [("pretrained start", WARM["C1"]["A"], False),
              ("scratch with change", WARM["C1"]["B"], False),
              ("warm start, default lr", WARM["C1"]["W"][533], True),
              ("warm start, NLL-tuned lr", WARM["C1t"]["W"][533], True)],
    "ckpt": [("pretrained start", WARM["C1"]["A"], False),
             ("scratch with change", WARM["C1"]["B"], False),
             ("warm start from step 2132", WARM["C1ck2132"]["W"][533], True),
             ("warm start from step 3198", WARM["C1ck3198"]["W"][533], True),
             ("warm start from final", WARM["C1ckfinal"]["W"][533], True)],
    "growth": [("width 8 pretrained", WARM["C10w"]["A"], True), ("width 16 from scratch", WARM["C10w"]["B"], True)]
              + [(f"grown 8 to 16, {k} steps", WARM["C10w"]["W"][k], True) for k in (533, 1066, 2132)],
}


def need(f):
    if not os.path.exists(f):
        sys.exit(f"missing read {os.path.relpath(f, ROOT)} (mixture_ab.py)")
    return json.load(open(f))


def curve(run):
    v = np.array(need(os.path.join(READS, f"{TRAINED_BY.get(run, run)}_val.json"))["blocks"][0]["vals"], float)
    return v[:, 0], v[:, 1]


def class_mse(runs, axis, cls):
    reads = [need(os.path.join(READS, f"{r}_{axis}.json"))["mse"] for r in runs]
    every = set.union(*(set(m) for m in reads))
    probes = set.intersection(*(set(m) for m in reads))
    per, bad = {}, []
    for p in sorted(probes):
        x = [m[p] for m in reads]
        if all(np.isfinite(x)) and min(x) > 0:
            per.setdefault(cls(p), []).append(np.exp(np.mean(np.log(x))))
        else:
            bad.append(p)
    m = np.array(sorted(np.exp(np.mean(np.log(v))) for v in per.values()))
    return m, {"n_probes": len(probes) - len(bad), "unmatched": sorted(every - probes), "nonfinite": bad}


def main():
    cls = class_of()
    rows = list(GROUPS)
    # three one-column figures, (a) val NLL, (b) PP4, (c) EE4, rows tuned / ckpt / growth (three columns of
    # standard plot boxes do not fit the printed width)
    figs = [ps.figure(ncols=1, nrows=len(rows)) for _ in range(3)]
    axs = np.array([[f[1][i] for f in figs] for i in range(len(rows))])
    out = {}
    for i, g in enumerate(rows):
        out[g] = {}
        lo = []
        for j, (lab, runs, on_nll) in enumerate(GROUPS[g]):
            col = ps.CYCLE[j % len(ps.CYCLE)]
            rec = {"runs": {a: list(pair(runs, a)) for a in ("PP4", "EE4")},
                   "blowup": {r: blowup_of(r).get("blew_up", "unchecked") for r in
                              sorted(set(pair(runs, "PP4")) | set(pair(runs, "EE4")))}}
            if on_nll:
                c = curve(pair(runs, "PP4")[0])
                axs[i, 0].plot(c[0], c[1], color=col, label=lab)
                lo.append(c[1].min())
            for k, axis in enumerate(("PP4", "EE4")):
                m, cnt = class_mse(pair(runs, axis), axis, cls)
                axs[i, k + 1].step(m, np.arange(1, len(m) + 1) / len(m), where="post", color=col, label=lab)
                rec[axis] = dict(cnt, n_classes=len(m), median=float(np.median(m)),
                                 q10=float(np.quantile(m, 0.1)), q90=float(np.quantile(m, 0.9)))
            out[g][lab] = rec
        axs[i, 0].set_ylim(min(lo) - 0.1, -1.5)
        axs[i, 0].set_xscale("log"); axs[i, 0].set_xlabel("step"); axs[i, 0].set_ylabel("validation loss (het NLL)")
        for k in range(3):
            ps.legend(axs[i, k])
        for k, axis in enumerate(("PP4", "EE4")):
            a = axs[i, k + 1]
            a.set_xscale("log"); a.set_xlabel(AXIS_WORDS[axis] + r": class MSE of $\log|M|^2$"); a.set_ylim(0, 1)
            a.set_ylabel("fraction of classes")
    ps.save_panels([f[0] for f in figs], os.path.join(HERE, "figs", "warm_start_compare"))
    json.dump(out, open(os.path.join(HERE, "data", "warm_start_compare.json"), "w"), indent=1)
    for g, d in out.items():
        for lab, r in d.items():
            print(g, lab, {a: (r[a]["n_probes"], len(r[a]["unmatched"]), len(r[a]["nonfinite"]),
                               round(r[a]["median"], 3)) for a in ("PP4", "EE4")}, r["blowup"])


if __name__ == "__main__":
    main()
