"""Generate data/solo_scaling.json: per-dataset SOLO scaling fit.

For each process, fit log(val_loss) = a + b·log(compute) over the single-dataset
scaling_solo_full_* sweeps (best val_loss per compute level; compute = t_steps ×
solo batch).  Stores {dataset: [a, b]} so a solo loss curve L_solo(c)=exp(a+b·log c)
can be overlaid on the per-process loss-vs-compute plots — showing directly which
processes in a mixture follow their solo scaling and which fall short.

Run from the project root:  python make_solo_scaling.py
"""
import json, glob, os, re, math

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEPS = os.path.join(ROOT, "sweeps")
SOLO_BATCH = 16384      # training.batchsize in the scaling_solo_full_* sweeps

DATASETS = [
    "ee_wwz_255-1000GeV_amplitudes", "ee_WW_162-1000GeV_amplitudes",
    "ee_ttbar_346-1000GeV_amplitudes", "ee_uug_91-1000GeV_amplitudes",
    "ee_uugg_91-1000GeV_amplitudes", "ee_aa_10-1000GeV_amplitudes",
    "ee_aaa_10-1000GeV_amplitudes", "ee_uu_91-1000GeV_amplitudes",
]
tag = lambda n: n.replace("_amplitudes", "").replace("_", "")


def _fit(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - b * mx, b


def solo_fit(name):
    pts = []
    for d in glob.glob(os.path.join(SWEEPS, f"scaling_solo_full_{tag(name)}_t*")):
        best = None
        for f in glob.glob(os.path.join(d, "results", "*.json")):
            vl = json.load(open(f))["val_loss"]
            if best is None or vl < best:
                best = vl
        m = re.search(r"_t(\d+)$", d)
        if best and best > 0 and m:
            pts.append((math.log(int(m.group(1)) * SOLO_BATCH), math.log(best)))
    if len(pts) < 2:
        return None, len(pts)
    return _fit([p[0] for p in pts], [p[1] for p in pts]), len(pts)


if __name__ == "__main__":
    out = {}
    for n in DATASETS:
        fit, npts = solo_fit(n)
        if fit:
            a, b = fit
            out[n] = [round(a, 5), round(b, 5)]
            print(f"{n:40s} alpha={-b:5.2f}  L_solo(1e8)={math.exp(a + b*math.log(1e8)):.2e}  ({npts} pts)")
        else:
            print(f"{n:40s} SKIP ({npts} pts)")
    path = os.path.join(ROOT, "data", "solo_scaling.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nWrote {path}")
