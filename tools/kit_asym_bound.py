#!/usr/bin/env python
"""The coefficient range model.input_asym_factor needs on the KIT targets (the basis of wrappers._ASYM_BOUND).

Per dataset, with the bulk_train target's mean and std of log|M|^2 (the run's standardization, first 1e5 events):
  z = (log|M|^2 - mean)/sd             what the regular readout r alone would carry (bulk_train)
  c = (log|M|^2 - log sum_t D_t - mean)/sd   the common coefficient if the factor alone carried |M|^2 (wrappers._asym_terms'
                                       D_t), on bulk_train and on every hard_test stratum (the limits, where it is exact)
printed as min / 0.1% / 99.9% / max. Runs on CC (the KIT files), the factor's code sent from this checkout.

    python tools/kit_asym_bound.py [--out analysis/kit_benchmark/data/kit_asym_bound.json]
"""
import argparse
import inspect
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REMOTE = r'''
import json, sys, numpy as np, torch
sys.path.insert(0, ".")
from tools import kit_hardps as K
from particle_ids import GLOBAL_PROPERTY_MATRIX as G, global_encode
_ETA = (1.0, -1.0, -1.0, -1.0)
_CF, _CA = 4.0 / 3.0, 3.0
%s
Gt = torch.as_tensor(G)
def stats(x):
    return [float(v) for v in (x.min(), np.percentile(x, 0.1), np.percentile(x, 99.9), x.max())]
out = {}
for stem, target, loops in (("ee_uug", "me2", 0), ("ee_uugg", "me2", 0), ("ee_tt", "virt_finite", 1)):
    def rows_logD(rows):
        n = (rows.shape[1] - 1) // 5
        mom, pdg, y = rows[:, :4 * n].reshape(-1, n, 4), rows[:, 4 * n:5 * n], rows[:, -1]
        tok = torch.tensor(global_encode(pdg[0].astype(int)))
        B = len(rows)
        p = torch.tensor(mom.reshape(-1, 4))
        ptr = torch.arange(B + 1) * n
        tk = tok.repeat(B)
        logD, on, _ = _asym_terms(p, ptr, n, Gt[tk, 7], Gt[tk, 2] >= 0, torch.full((B,), float(loops)))
        lse = torch.logsumexp(torch.where(on, logD, torch.full_like(logD, -torch.inf)), 1).numpy()
        return np.log(y), lse
    bulk = K.stratum_rows(stem, target, "bulk_train", "bulk")[:100000]
    ly, lse = rows_logD(bulk)
    m, sd = ly.mean(), ly.std()
    rec = {"mean": float(m), "sd": float(sd), "n_terms_on": None,
           "bulk_train": {"z": stats((ly - m) / sd), "c": stats((ly - lse - m) / sd)}}
    for st in K.strata(stem, "hard_test"):
        r = K.stratum_rows(stem, target, "hard_test", st)[:20000]
        ly2, lse2 = rows_logD(r)
        rec["hard_test/" + st] = {"c": stats((ly2 - lse2 - m) / sd)}
    out[f"{stem}__{target}"] = rec
print("ASYMBOUND " + json.dumps(out))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis/kit_benchmark/data/kit_asym_bound.json"))
    a = ap.parse_args()
    from analysis.common.readers import site_python
    import wrappers
    res = site_python(REMOTE % inspect.getsource(wrappers._asym_terms), [], "ASYMBOUND")
    for k, v in res.items():
        print(k, f"mean {v['mean']:.3f} sd {v['sd']:.3f}")
        for g, s in v.items():
            if isinstance(s, dict):
                print("  ", g, {q: [round(x, 2) for x in w] for q, w in s.items()})
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
