"""The particle-content axis of the dimensionless round (D93: "every process with ... a top held out of training"): the
coupling-order design run DlessInOrd trained on the catalog's 1352 training processes and its twin DlessInOrd_notop
trained on the same minus the 142 with a top quark (recipes/catalog_daniel_train_notop.yaml, 1210; same steps, HPs,
seed), both read on all 162 catalog processes with a top (recipes/catalog_daniel_top.yaml: the 142 training ones and the
20 held out by the catalog's split; scripts/job_catalog_norm.sh DlessInOrd eval_short top, FA_HOLDOUT=top for the twin).
Per process the error d = q - t of log|M|^2 under the run's shared standardization: MSE, offset (mean of d), shape
(variance of d); medians over the 142 and over the 20. For the 142, DlessInOrd has trained on them (its value is an
in-distribution error at 200 val events per process, here read on 2000), the twin has never seen a top.
Inputs: data/catalog_top_short_DlessInOrd{,_notop}.npz (collect_heldout_preds.py on
$SCRATCH/zz_catalog_norm_short_DlessInOrd{,_notop}_sigstage_top). Output: the table printed, data/catalog_top_heldout.json.
    python analysis/catalog_daniel/catalog_top_heldout.py
"""
import json, os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ARMS = {"DlessInOrd": "trained with the top processes", "DlessInOrd_notop": "trained without any top"}


def main():
    train = {p["name"] for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_daniel_train.yaml")))["processes"]}
    res = {}
    per = {}
    for a, lab in ARMS.items():
        z = np.load(os.path.join(HERE, "data", f"catalog_top_short_{a}.npz"))
        ps = sorted({k.split("__")[0] for k in z.files})
        per[a] = {p: float(np.mean((z[f"{p}__q"].astype(float) - z[f"{p}__t"].astype(float)) ** 2)) for p in ps}
        for grp, L in (("142 training", [p for p in ps if p in train]), ("20 held out", [p for p in ps if p not in train])):
            d = [z[f"{p}__q"].astype(float) - z[f"{p}__t"].astype(float) for p in L]
            r = dict(n=len(L), mse=float(np.median([np.mean(x ** 2) for x in d])),
                     offset=float(np.median([x.mean() for x in d])), abs_offset=float(np.median([abs(x.mean()) for x in d])),
                     shape=float(np.median([x.var() for x in d])))
            res[f"{a}|{grp}"] = r
            print(f"{lab:32s} {grp:13s} n={r['n']:3d}  MSE {r['mse']:.3g}  offset {r['offset']:+.2f} (|.| {r['abs_offset']:.2f})  shape {r['shape']:.3g}")
    common = sorted(set(per["DlessInOrd"]) & set(per["DlessInOrd_notop"]))
    for grp, L in (("142 training", [p for p in common if p in train]), ("20 held out", [p for p in common if p not in train])):
        r = np.array([per["DlessInOrd_notop"][p] / per["DlessInOrd"][p] for p in L])
        res[f"ratio|{grp}"] = dict(n=len(r), q=[float(x) for x in np.quantile(r, [.1, .5, .9])])
        print(f"no-top over with-top, per process, {grp}: 10/50/90% {np.round(np.quantile(r, [.1, .5, .9]), 2).tolist()}")
    json.dump(res, open(os.path.join(HERE, "data", "catalog_top_heldout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
