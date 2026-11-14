#!/usr/bin/env python3
"""Readout of a collected KIT figure file (tools/kit_figures.py collect): per process and arm, the bulk_test log-MSE,
each hard family's deepest-bin log-MSE (the stratum farthest into the limit: smallest singular variable, or the
largest for a family whose limit is at large values, sudakov) and the fitted exponent (KIT evaluate.py: pred vs
truth vs exact). Prints one table per process.

    python tools/kit_readout.py analysis/kit_benchmark/data/kit_fig_B.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.kit_figures import family  # noqa: E402

UP = ("sudakov",)       # families whose limit is at the large end of the singular variable


def deepest(strata):
    """{family: (stratum, x, logmse)} at the deepest stratum of each hard family"""
    out = {}
    for k, v in strata.items():
        if not k.startswith("hard_test/"):
            continue
        fam = family(k.split("/")[1])
        up = any(u in fam for u in UP)
        cur = out.get(fam)
        if cur is None or (v["x"] > cur[1] if up else v["x"] < cur[1]):
            out[fam] = (k.split("/")[1], v["x"], v["logmse"])
    return out


def main(path):
    d = json.load(open(path))
    by = {}
    for name, r in sorted(d["runs"].items()):
        by.setdefault((r["file"], r["target"]), []).append((name, r))
    for (stem, tgt), runs in sorted(by.items()):
        print(f"== {stem} {tgt}")
        for name, r in runs:
            bulk = [v["logmse"] for k, v in r["strata"].items() if k.startswith("bulk_test/")]
            deep = deepest(r["strata"])
            sl = r.get("slopes") or {}
            cells = [f"{fam} {v[0].split(fam + '_')[-1]}: {v[2]:.3g}" for fam, v in sorted(deep.items())]
            exps = [f"{k.split('/')[0]} {v['pred']:+.2f} (truth {v['truth']:+.2f})" for k, v in sorted(sl.items())]
            print(f"  {r['mode']:6s} {r['tag'] or '-':8s} n={r['n']} bulk {bulk[0] if bulk else float('nan'):.3g} | "
                  + "; ".join(cells) + (" | exp " + "; ".join(exps) if exps else ""))


def versus(path, ref):
    """Every stratum against the run `ref` (D376: bulk and every bin, not only the deepest): per arm and family,
    log10(arm log-MSE / ref log-MSE) per stratum (negative = arm better), its median and range, and how many strata are
    better (below 0) or worse."""
    import numpy as np
    d = json.load(open(path))
    R = d["runs"][ref]["strata"]
    for name, r in sorted(d["runs"].items()):
        if name == ref:
            continue
        fams = {}
        for k, v in r["strata"].items():
            if k in R:
                fam = "bulk" if k.startswith("bulk_test/") else family(k.split("/")[1])
                fams.setdefault(fam, []).append(np.log10(v["logmse"] / R[k]["logmse"]))
        print(f"== {name} vs {ref}")
        for fam, x in sorted(fams.items()):
            x = np.array(x)
            print(f"  {fam:14s} n={len(x):2d} median {np.median(x):+.3f} [{x.min():+.2f}, {x.max():+.2f}] "
                  f"better: {(x < 0).sum()} worse: {(x > 0).sum()}")


def verdict(path, arm, ref):
    """The passed design's outcome (kit_benchmark, 2026-11-13 10:47) per hard family, arm against control `ref`:
    FIXED  |exponent - truth's| <= 0.1, |deepest-bin median signed error| <= 0.3, bulk log-MSE <= 0.5 decades worse;
    MOVED  deepest |signed error| at least halved and the exponent closer to the truth's;
    STAYED deepest signed error within 0.3 of the control's; else inconclusive.
    Signed error = median of prediction - truth (log10|M|^2) in the deepest stratum (kit_figures collect's `med`).
    A family KIT fits no exponent for (sudakov) can only be STAYED or inconclusive: its signed errors are printed."""
    import numpy as np
    d = json.load(open(path))
    a, c = d["runs"][arm], d["runs"][ref]
    bulk = lambda r: next(v["logmse"] for k, v in r["strata"].items() if k.startswith("bulk_test/"))
    dbulk = float(np.log10(bulk(a) / bulk(c)))
    out = {"bulk_decades": dbulk, "families": {}}
    print(f"== {arm} vs {ref}: bulk log-MSE {bulk(a):.3g} vs {bulk(c):.3g} ({dbulk:+.2f} decades)")
    for fam, (st, _, lm) in sorted(deepest(a["strata"]).items()):
        k = next(k for k in a["strata"] if k.startswith(f"hard_test/{st}/"))
        ba, bc, lc = a["strata"][k]["med"], c["strata"][k]["med"], c["strata"][k]["logmse"]
        sk = next((k2 for k2 in a.get("slopes", {}) if k2.startswith(fam + "/")), None)
        ea, ec, et = (a["slopes"][sk]["pred"], c["slopes"][sk]["pred"], a["slopes"][sk]["truth"]) if sk else (None,) * 3
        if sk and abs(ea - et) <= 0.1 and abs(ba) <= 0.3 and dbulk <= 0.5:
            v = "FIXED"
        elif sk and abs(ba) <= 0.5 * abs(bc) and abs(ea - et) < abs(ec - et):
            v = "MOVED"
        elif abs(ba - bc) <= 0.3:
            v = "STAYED"
        else:
            v = "inconclusive"
        out["families"][fam] = dict(stratum=st, logmse=lm, logmse_ref=lc, bias=ba, bias_ref=bc, exp=ea, exp_ref=ec,
                                    exp_truth=et, verdict=v)
        e = f"exp {ea:+.2f} vs {ec:+.2f} (truth {et:+.2f})" if sk else "no exponent"
        print(f"  {fam:14s} {st}: log-MSE {lm:.3g} vs {lc:.3g}, signed {ba:+.2f} vs {bc:+.2f}, {e}: {v}")
    return out


if __name__ == "__main__":
    # python tools/kit_readout.py <json> [--vs <run name>] [--verdict <arm> <control> ...pairs]
    if "--verdict" in sys.argv:
        av = sys.argv[:]
        out = av.pop(av.index("--out") + 1) if "--out" in av else None
        av = [x for x in av if x != "--out"]
        pairs = av[av.index("--verdict") + 1:]
        res = {f"{a} vs {r}": verdict(av[1], a, r) for a, r in zip(pairs[::2], pairs[1::2])}
        if out:
            json.dump(res, open(out, "w"), indent=1)
    elif "--vs" in sys.argv:
        versus(sys.argv[1], sys.argv[sys.argv.index("--vs") + 1])
    else:
        main(sys.argv[1])
