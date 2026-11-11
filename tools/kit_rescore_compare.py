#!/usr/bin/env python3
"""KIT scores before and after the event-frame fix (adf1742b: kit_predict reads each scored event unaugmented, so its
prediction depends on the event alone; before, the augmentation was keyed on the scored pool). Two collected figure
files (tools/kit_figures.py collect) of the same runs, old and new:

    python tools/kit_rescore_compare.py <old.json> <new.json> [--fig <png path>]

prints per run the largest |log10(new/old)| of the per-stratum KIT log-MSE (and where), the bulk_test log-MSE old/new,
each hard family's deepest-bin log-MSE old/new and the fitted exponents old/new; --fig draws every (run, stratum)
log-MSE new against old (log-log, the diagonal = unchanged), softpair-input runs apart.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.kit_readout import deepest  # noqa: E402


def compare(old, new):
    """{run: {"strata": {stratum: (old, new)}, "slopes": {family: (old, new)}}} over the runs and strata of both"""
    out = {}
    for name, r in new["runs"].items():
        o = old["runs"].get(name)
        if o is None:
            continue
        st = {k: (o["strata"][k]["logmse"], v["logmse"]) for k, v in r["strata"].items() if k in o["strata"]}
        sl = {k: (o["slopes"][k]["pred"], v["pred"]) for k, v in (r.get("slopes") or {}).items()
              if k in (o.get("slopes") or {})}
        out[name] = {"strata": st, "slopes": sl, "deep_old": deepest(o["strata"]), "deep_new": deepest(r["strata"])}
    return out


def report(cmp):
    for name, c in sorted(cmp.items()):
        lr = {k: np.log10(n / o) for k, (o, n) in c["strata"].items()}
        worst = max(lr, key=lambda k: abs(lr[k]))
        bulk = next((v for k, v in c["strata"].items() if k.startswith("bulk_test/")), (np.nan, np.nan))
        print(f"== {name}: max |log10 new/old| {abs(lr[worst]):.3f} at {worst.split('/')[1]}; "
              f"bulk {bulk[0]:.3g} -> {bulk[1]:.3g}")
        for fam in sorted(c["deep_new"]):
            so, xo, mo = c["deep_old"].get(fam, ("", np.nan, np.nan))
            sn, xn, mn = c["deep_new"][fam]
            print(f"   {fam:14s} deepest {sn}: {mo:.3g} -> {mn:.3g}")
        for k, (o, n) in sorted(c["slopes"].items()):
            print(f"   exponent {k.split('/')[0]:14s} {o:+.2f} -> {n:+.2f}")


def figure(cmp, path):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import plot_style as ps
    ps.use()
    fig, ax = ps.figure(width="half")
    for soft, mk, col, lab in ((False, "o", ps.C.blue, "plain"), (True, "^", ps.C.vermillion, "softpair")):
        pts = [v for n, c in cmp.items() if ("softpair" in n) == soft for v in c["strata"].values()]
        ax.scatter([o for o, _ in pts], [n for _, n in pts], s=6, marker=mk, color=col, label=lab, alpha=0.7, lw=0)
    allv = [x for c in cmp.values() for v in c["strata"].values() for x in v]
    lo, hi = min(allv), max(allv)
    ax.plot([lo, hi], [lo, hi], color=ps.C.grey, lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("log-MSE old"); ax.set_ylabel("log-MSE new")
    ax.legend()
    ps.save(fig, os.path.splitext(path)[0])


if __name__ == "__main__":
    old, new = (json.load(open(p)) for p in sys.argv[1:3])
    c = compare(old, new)
    report(c)
    if "--fig" in sys.argv:
        figure(c, sys.argv[sys.argv.index("--fig") + 1])
