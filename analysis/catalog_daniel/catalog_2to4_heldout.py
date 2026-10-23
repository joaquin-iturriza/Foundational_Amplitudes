"""The 2->4 probes (recipes/gen_catalog_2to4.py: 40 catalog 2->3 training processes plus one gluon or photon, D90) read
by the short catalog design runs (scripts/job_catalog_norm.sh <arm> eval_short <set>, from each run's sigma stage), at
the catalog's cuts (set 2to4, pT >= 10 GeV) and with ptj = ptb = pta = 30 GeV (set 2to4c30, the cut test, D92). Per
probe the error d = q - t of log|M|^2 under the run's shared standardization: MSE, its offset (the mean of d: the level
the target puts the extra particle at) and its shape (the variance of d). Medians over the 20 +gluon and the 20 +photon
probes; the offset's shift between the two cut sets is what a cut change does to each target.
Inputs: data/catalog_2to4{,c30}_short_<arm>.npz (analysis/catalog_daniel/collect_heldout_preds.py on
$SCRATCH/zz_catalog_norm_short_<arm>_sigstage_<set>). Output: the table printed, data/catalog_2to4_heldout.json.
    python analysis/catalog_daniel/catalog_2to4_heldout.py
"""
import json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"B": "shared target (Lambda = 1 GeV)", "Adim": "s_hat only, s0 = 289 GeV^2",
        "Dless": "dimensionless (Lambda^2 = s_hat)", "DlessIn": "dimensionless, dimensionless inputs",
        "Dpair": "dimensionless (Lambda^2 = pairs)", "Dpair_seed1": "dimensionless (pairs), seed 1",
        "DlessInVirt": "dimless inputs + virtuality pooling", "DlessInNoDiag": "dimless inputs, no diagrams"}
SETS = {"2to4": "pT >= 10 GeV", "2to4c30": "pT >= 30 GeV", "2to4E100p": "2->4 at 100 TeV", "2to4E100": "2->3 at 100 TeV"}


def main():
    res = {}
    for arm, lab in ARMS.items():
        for st, cut in SETS.items():
            f = os.path.join(HERE, "data", f"catalog_{st}_short_{arm}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f)
            probes = sorted({k.split("__")[0] for k in z.files})
            # the 100 TeV parents (2to4E100) are the catalog's own 2->3 training processes: one group
            kinds = (("parent", ""),) if st == "2to4E100" else (("gluon", "_g"), ("photon", "_a"))
            for kind, suf in kinds:
                ps = [p for p in probes if p.endswith(suf)]
                d = [z[f"{p}__q"].astype(float) - z[f"{p}__t"].astype(float) for p in ps]
                r = dict(n=len(ps), mse=float(np.median([np.mean(x ** 2) for x in d])),
                         offset=float(np.median([x.mean() for x in d])),
                         shape=float(np.median([x.var() for x in d])))
                res[f"{arm}|{st}|{kind}"] = r
                print(f"{lab:38s} {cut:13s} +{kind:6s} n={r['n']:2d}  MSE {r['mse']:7.3g}  offset {r['offset']:+6.2f}  "
                      f"shape {r['shape']:.3g}")
    for arm in ARMS:
        for kind in ("gluon", "photon"):
            a, b = res.get(f"{arm}|2to4|{kind}"), res.get(f"{arm}|2to4c30|{kind}")
            if a and b:
                print(f"offset shift 10 -> 30 GeV, {ARMS[arm]:38s} +{kind}: {b['offset'] - a['offset']:+.2f}")
    json.dump(res, open(os.path.join(HERE, "data", "catalog_2to4_heldout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
