"""D88: the multiplicity-extrapolation round, read on the held-out processes (docs/results.tex, "Extrapolation in
multiplicity: the units of the target"). Two networks trained on the 236 2->2 catalog processes only
(scripts/job_catalog_norm.sh Bgev|Dless extrap): B in GeV (one shared standardization of log|M|^2) and the dimensionless
target log|M|^2 + (N-4) log s_hat (data.target_physics_factor, dim_only, s0 = 1). With 4-N = 0 on every training event,
B in TeV (T = log|M|^2 - (4-N) log 1e6) trains on B in GeV's targets and the s0 = 289 GeV^2 variant
(T = log|M|^2 - (4-N) log(s_hat / 289)) on the dimensionless one's, so their networks are the same and their predictions
of log|M|^2 differ by a constant at N = 5:
    B, TeV:   pred = pred_GeV + (4-N) log 1e6      = pred_GeV - log 1e6 at N = 5
    s0 = 289: pred = pred_dimless - (4-N) log 289  = pred_dimless + log 289 at N = 5
(a network output T_hat reads back as T_hat + (4-N) log Lambda^2, the same T_hat for both members of a pair). Per process:
MSE of log|M|^2 on its own, split into the squared mean error (level) and the error's variance (shape), and the MSE
after the oracle rescale and shift; then the dimensionless arm's 2->3 level offset against the leading coupling product.
Inputs: data/catalog_extrap_{Bgev,Dless}.npz (collect_heldout_preds.py on $SCRATCH/zz_catalog_norm_extrap_<arm>),
data/diagram_couplings.json. Output: the table printed, data/catalog_extrap_heldout.json.
    python analysis/catalog_daniel/catalog_extrap_heldout.py
"""
import json, math, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L6, L289 = math.log(1e6), math.log(289.0)
ARMS = {"B, GeV": ("Bgev", 0.0), "B, TeV": ("Bgev", -L6), "dimensionless": ("Dless", 0.0), "s0 = 289": ("Dless", +L289)}


def main():
    twins = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    dc = json.load(open(os.path.join(HERE, "data", "diagram_couplings.json")))
    npart = lambda p: len(meta[p]["process"].replace(">", " ").split())
    Z = {a: dict(np.load(os.path.join(HERE, "data", f"catalog_extrap_{a}.npz"))) for a in ("Bgev", "Dless")}
    probes = sorted({k.split("__")[0] for k in Z["Bgev"] if k.endswith("__t")})
    res = {}
    for N in (4, 5):
        for rel in (False, True):
            sel = [p for p in probes if npart(p) == N and (p in twins) == rel]
            for lab, (a, c5) in ARMS.items():
                z = Z[a]
                c = c5 if N == 5 else 0.0
                d = [z[f"{p}__q"].astype(float) + c - z[f"{p}__t"].astype(float) for p in sel]
                aff = [np.mean((z[f"{p}__q_aff"].astype(float) - z[f"{p}__t"].astype(float)) ** 2) for p in sel]
                r = dict(n=len(sel), mse=float(np.median([np.mean(x ** 2) for x in d])),
                         level=float(np.median([x.mean() ** 2 for x in d])),
                         shape=float(np.median([x.var() for x in d])), affine=float(np.median(aff)))
                key = f"2->{N - 2} {'relabels' if rel else 'new'}|{lab}"
                res[key] = r
                print(f"{key:30s} n={r['n']:3d} MSE {r['mse']:.3g} (level {r['level']:.3g}, shape {r['shape']:.3g}) "
                      f"after rescale and shift {r['affine']:.3g}")
    # the dimensionless arm's 2->3 level offset (pred - true), and how much of it the leading coupling product explains
    z = Z["Dless"]
    sel = [p for p in probes if npart(p) == 5]
    b = np.array([np.mean(z[f"{p}__q"].astype(float) - z[f"{p}__t"].astype(float)) for p in sel])
    lc = np.array([max(dc[p]["logc2"]) for p in sel])
    A = np.stack([np.ones_like(lc), lc], 1)
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    resid = b - A @ coef
    res["dimensionless 2->3 offset"] = dict(median=float(np.median(b)), p10=float(np.quantile(b, .1)),
                                            p90=float(np.quantile(b, .9)), fit=[float(x) for x in coef],
                                            median_sq=float(np.median(b ** 2)), median_sq_after_fit=float(np.median(resid ** 2)))
    v = res["dimensionless 2->3 offset"]
    print(f"dimensionless 2->3 offset: median {v['median']:.2f} (10-90% {v['p10']:.2f} to {v['p90']:.2f}); median square "
          f"{v['median_sq']:.3g}, after a linear fit on the leading log coupling product {v['median_sq_after_fit']:.3g}")
    json.dump(res, open(os.path.join(HERE, "data", "catalog_extrap_heldout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
