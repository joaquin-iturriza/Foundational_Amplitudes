"""Generate data/solo_alpha.json: the per-dataset SOLO scaling exponent α.

α_solo = -slope of log(val_loss) vs log(compute) across the single-dataset
scaling_solo_full_* sweeps (best val_loss per compute level).  Batch size is a
constant factor on compute, so it cancels out of the slope — the exponent only
needs (t_steps, loss) pairs.

The ProcessBalancedSampler uses these as the *expected* exponent each process
should reach: a dataset whose live local α sits below its α_solo is under-scaling
(starved by the mixture) and is boosted.  Regenerate if the solo sweeps change.

Run from the project root:  python make_solo_alpha.py
"""
import json, glob, os, re, math

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEPS = os.path.join(ROOT, "sweeps")

DATASETS = [
    "ee_wwz_255-1000GeV_amplitudes", "ee_WW_162-1000GeV_amplitudes",
    "ee_ttbar_346-1000GeV_amplitudes", "ee_uug_91-1000GeV_amplitudes",
    "ee_uugg_91-1000GeV_amplitudes", "ee_aa_10-1000GeV_amplitudes",
    "ee_aaa_10-1000GeV_amplitudes", "ee_uu_91-1000GeV_amplitudes",
]
tag = lambda n: n.replace("_amplitudes", "").replace("_", "")


def _slope(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")


def solo_alpha(name):
    pts = []
    for d in glob.glob(os.path.join(SWEEPS, f"scaling_solo_full_{tag(name)}_t*")):
        best = None
        for f in glob.glob(os.path.join(d, "results", "*.json")):
            vl = json.load(open(f))["val_loss"]
            if best is None or vl < best:
                best = vl
        m = re.search(r"_t(\d+)$", d)
        if best and best > 0 and m:
            pts.append((math.log(int(m.group(1))), math.log(best)))
    if len(pts) < 2:
        return None, len(pts)
    return -_slope([p[0] for p in pts], [p[1] for p in pts]), len(pts)


if __name__ == "__main__":
    out = {}
    for n in DATASETS:
        a, npts = solo_alpha(n)
        if a is not None:
            out[n] = round(a, 4)
            print(f"{n:40s} alpha_solo={a:.3f}  ({npts} pts)")
        else:
            print(f"{n:40s} SKIP (only {npts} pts)")
    path = os.path.join(ROOT, "data", "solo_alpha.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nWrote {path}")
