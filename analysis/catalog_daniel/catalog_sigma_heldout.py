"""The sigma head on the 139 held-out processes (docs/results.tex, "The sigma head on the held-out processes"): pulls
(q - t) / sigma in log|M|^2 per process, the median over processes of each one's pull standard deviation and of its
fraction of events within one sigma (calibrated: 1 and 0.68), and the pull std with each process's mean error removed.
  hp58 after its sigma stage: data/heldout_preds_sigstage.npz (scripts/job_catalog_heldout_preds.sh
      runs/dancat_het_pre/trial_0058/sigma_stage), read with each process's own statistics: q = mu_p + sd_p h,
      sigma = sd_p sig (mu_sd, h, sig in the npz; the zero-shot statistics would need the twin or g);
  A after its sigma stage: data/catalog_norm_A_sig.npz ($SCRATCH/zz_catalog_norm_A_sigstage);
  B at its best checkpoint (it blew up before a sigma stage): data/catalog_norm_Bsig.npz ($SCRATCH/zz_catalog_norm_B,
      rescored with sigma saved); for A and B q, t and s (= the shared sd times sigma) are in the npz.
Output: the table printed, data/catalog_sigma_heldout.json.
    python analysis/catalog_daniel/catalog_sigma_heldout.py
"""
import json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def pulls(z, p, own):
    t = z[f"{p}__t"].astype(float)
    if own:
        mu, sd = z[f"{p}__mu_sd"]
        q, s = mu + sd * z[f"{p}__h"].astype(float), sd * z[f"{p}__sig"].astype(float)
    else:
        q, s = z[f"{p}__q"].astype(float), z[f"{p}__s"].astype(float)
    return (q - t) / s, (q - t - (q - t).mean()) / s


def main():
    twins = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    res = {}
    for lab, f, own in (("hp58, sigma stage, own statistics", "heldout_preds_sigstage", True),
                        ("A, sigma stage", "catalog_norm_A_sig", False),
                        ("B, best checkpoint (no sigma stage)", "catalog_norm_Bsig", False)):
        z = dict(np.load(os.path.join(HERE, "data", f + ".npz")))
        probes = sorted({k.split("__")[0] for k in z if k.endswith("__t")})
        for rel in (False, True):
            ps = [p for p in probes if (p in twins) == rel]
            pp = [pulls(z, p, own) for p in ps]
            r = dict(n=len(ps), pull_std=float(np.median([a.std() for a, _ in pp])),
                     cover68=float(np.median([np.mean(np.abs(a) < 1) for a, _ in pp])),
                     pull_std_level_removed=float(np.median([b.std() for _, b in pp])))
            key = f"{lab}|{'96 relabels' if rel else '43 new'}"
            res[key] = r
            print(f"{key:52s} pull std {r['pull_std']:.2f}, within one sigma {r['cover68']:.2f}, "
                  f"pull std after the level {r['pull_std_level_removed']:.2f}")
    json.dump(res, open(os.path.join(HERE, "data", "catalog_sigma_heldout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
