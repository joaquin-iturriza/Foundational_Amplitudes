#!/usr/bin/env python3
"""The lr centre of a run at horizon t from the measured lr*(t,D) surface (HPO rule 2, .claude/context
fa-sweeps-and-hpo; sweep/sweep_config_jeanzay_template.yaml), at the high-D asymptote.

    python tools/lr_surface_center.py --t 17056 68224 [--json analysis/hpo_optima/hpo_optima.json]

Method (the one behind the template's "~2e-3 @30k, ~1.4e-3 @50k, ~9e-4 @100k"): the converged scaling_p sweeps of
analysis/hpo_optima/hpo_optima.json with D (n_train, summed over their 8 processes) at or above --dmin (22130: lr* is
saturated in D from there, the high-D rows 22130 and 70000) and t at or past the peak (t >= 3162, the decaying branch);
log lr* is fitted with the law's fixed slope -0.55 (lr ~ t^-0.55) and only its amplitude free, so
lr(t) = exp(mean(log lr_i + 0.55 log t_i)) t^-0.55. A free slope on these 32 points is steeper (-0.88) and driven by
the noisiest corner; it is printed as a diagnostic. The surface was measured with AdamW: for a MuonAdamW run the
centre is untested (no Muon optimum past 4264 steps exists). A run whose D is above the grid (the catalog: about 1e8
events) is taken at the saturated high-D value (an extrapolation in D and, past t = 56234, in t)."""
import argparse, json, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOPE = -0.55


def points(recs, dmin, tmin):
    return [(r["t_steps"], r["hp_best"]["training.lr"]["val"], r["n_train"]) for r in recs
            if r.get("family") == "scaling_p" and r.get("converged") and (r.get("hp_best") or {}).get("training.lr")
            and (r.get("n_train") or 0) >= dmin and (r.get("t_steps") or 0) >= tmin]


def fit(pts):
    lx = [math.log(t) for t, _, _ in pts]
    ly = [math.log(v) for _, v, _ in pts]
    amp = sum(y - SLOPE * x for x, y in zip(lx, ly)) / len(pts)
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    free = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sum((x - mx) ** 2 for x in lx)
    return amp, free


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t", type=int, nargs="+", required=True)
    ap.add_argument("--json", default=os.path.join(ROOT, "analysis/hpo_optima/hpo_optima.json"))
    ap.add_argument("--dmin", type=int, default=22130)
    ap.add_argument("--tmin", type=int, default=3162)
    a = ap.parse_args(argv)
    pts = points(json.load(open(a.json)), a.dmin, a.tmin)
    amp, free = fit(pts)
    out = {"n_points": len(pts), "dmin": a.dmin, "tmin": a.tmin, "slope": SLOPE, "free_slope_diagnostic": round(free, 3),
           "optimizer_measured": "AdamW",
           "centre": {str(t): float("%.4g" % math.exp(amp + SLOPE * math.log(t))) for t in a.t},
           "window_half_decade": {str(t): [float("%.4g" % (math.exp(amp + SLOPE * math.log(t)) / 3)),
                                           float("%.4g" % (math.exp(amp + SLOPE * math.log(t)) * 3))] for t in a.t}}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
