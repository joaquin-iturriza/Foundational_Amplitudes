"""D87: the three short catalog design runs (scripts/job_catalog_norm.sh <arm> short: hp58's configuration, 1e4 train and
200 val events per process, 4126 steps, a 150-step sigma stage) read on the 139 held-out processes on their own, next to
the full-length runs of D83 (A and B; B at its best checkpoint, it blew up) and hp58 read with g. Arms: B (one shared
standardization), Adim (the factor (s0/s_hat)^(4-N), s0 = 289 GeV^2, no coupling product), A (the same with the leading
coupling product). Per process: MSE of log|M|^2 on its own, its split into the squared mean error (level) and the
error's variance (shape), the median per-process level fraction, the MSE after the best rescale and shift (an oracle),
and, where the npz has sigma, the pulls (q - t) / sigma.
Inputs: data/catalog_short_{B,Adim,A}_sigstage.npz (scored from each run's sigma stage; mu is the run's, frozen), data/catalog_norm_{A,B}.npz, data/heldout_preds.npz + data/stats_net.json.
Output: the table printed, data/catalog_short_heldout.json.
    python analysis/catalog_daniel/catalog_short_heldout.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from heldout_delta import G  # noqa: E402


def main():
    twins = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    z58 = dict(np.load(os.path.join(HERE, "data", "heldout_preds.npz")))
    sn = json.load(open(os.path.join(HERE, "data", "stats_net.json")))
    for n, (mu, lsd) in zip(sn["heldout_names"], sn["heldout_pred"][G]):
        if f"{n}__h" in z58:
            z58[f"{n}__q"] = mu + np.exp(lsd) * z58[f"{n}__h"].astype(float)
    models = {"hp58 + g (full)": z58}
    for lab, f in (("A (full)", "catalog_norm_A"), ("B (full, blew up)", "catalog_norm_B"),
                   ("B (short)", "catalog_short_B_sigstage"), ("Adim (short)", "catalog_short_Adim_sigstage"), ("A (short)", "catalog_short_A_sigstage"),
                   ("Dless (short)", "catalog_short_Dless_sigstage"), ("DlessIn (short)", "catalog_short_DlessIn_sigstage"),
                   ("Dpair (short, best checkpoint, diverged)", "catalog_short_Dpair"),
                   ("Dpair seed 1 (short)", "catalog_short_Dpair_seed1_sigstage"),
                   ("DlessIn + virtuality pooling (short)", "catalog_short_DlessInVirt_sigstage"),
                   ("DlessIn, no diagrams (short)", "catalog_short_DlessInNoDiag_sigstage"),
                   ("DlessIn + coupling orders (short)", "catalog_short_DlessInOrd_sigstage")):
        models[lab] = dict(np.load(os.path.join(HERE, "data", f + ".npz")))
    probes = sorted({k.split("__")[0] for k in z58 if k.endswith("__t")})
    res = {}
    for name, z in models.items():
        for rel in (False, True):
            ps = [p for p in probes if (p in twins) == rel and f"{p}__q" in z]
            d = [z[f"{p}__q"].astype(float) - z[f"{p}__t"].astype(float) for p in ps]
            m = np.array([np.mean(x ** 2) for x in d]); lv = np.array([x.mean() ** 2 for x in d])
            sh = np.array([x.var() for x in d])
            aff = np.array([np.mean((z[f"{p}__q_aff"].astype(float) - z[f"{p}__t"].astype(float)) ** 2) for p in ps])
            r = dict(n=len(ps), median=float(np.median(m)), p10=float(np.quantile(m, .1)), p90=float(np.quantile(m, .9)),
                     level=float(np.median(lv)), shape=float(np.median(sh)), level_fraction=float(np.median(lv / (lv + sh))),
                     affine=float(np.median(aff)))
            pulls = [(z[f"{p}__q"].astype(float) - z[f"{p}__t"].astype(float)) / z[f"{p}__s"].astype(float)
                     for p in ps if f"{p}__s" in z and z[f"{p}__s"].size]
            if pulls:
                r.update(pull_std=float(np.median([x.std() for x in pulls])),
                         cover68=float(np.median([np.mean(np.abs(x) < 1) for x in pulls])))
            key = f"{name}|{'96 relabels' if rel else '43 new'}"
            res[key] = r
            print(f"{key:34s} MSE {r['median']:.3g} (10-90% {r['p10']:.3g}-{r['p90']:.3g}) level {r['level']:.3g} "
                  f"shape {r['shape']:.3g} level frac {r['level_fraction']:.2f} affine {r['affine']:.3g}"
                  + (f" pull std {r['pull_std']:.2f} cover68 {r['cover68']:.2f}" if "pull_std" in r else ""))
    json.dump(res, open(os.path.join(HERE, "data", "catalog_short_heldout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
