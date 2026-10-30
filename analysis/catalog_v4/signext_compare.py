"""The one-loop sign extension on catalog_v4 against its control, as the passed design of v3_prep_explore reads it.

Arms (scripts/job_catalog_v3_short.sh, both at trial_1860's HPs with Muon, seeds 42 and 43):
  control    v4_dimensionless_muon, v4_dimensionless_muon_seed2
  extension  v4_dimensionless_muon_more_signed, v4_dimensionless_muon_more_signed_seed2 (more sign-changing one-loop
             training pools, recipes/catalog_v4_train_signext.yaml)
Decision measure: the zero-shot wrong-sign share on the held-out one-loop pools the scorer reads as sign-changing
(H5_signed.txt, 2000 validation events each, best checkpoint); per pool the mean over the two seeds of each arm; the
number of pools (ties excluded) where the extension is lower, with the one-sided sign test p; the seed-mean medians;
each seed alone. Secondary: zero-shot MSE of log|M|^2 per probe (mean (q - t)^2 of the saved .npz) on every held-out
axis, the per-probe ratio extension over control (seed means), its median, and the control's own seed-43-over-42
median ratio as the seed spread.
    python3 analysis/catalog_v4/signext_compare.py   -> data/signext_compare.json
"""
import glob
import json
import os
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
READS = os.path.join(os.path.dirname(HERE), "catalog_v3", "data", "short_reads")
CTRL = ("v4_dimensionless_muon", "v4_dimensionless_muon_seed2")
EXT = ("v4_dimensionless_muon_more_signed", "v4_dimensionless_muon_more_signed_seed2")
AXES = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c")


def signed(arm):
    out = {}
    for line in open(os.path.join(READS, arm, "H5_signed.txt")):
        if line.startswith("SIGNED_AB "):
            r = json.loads(line[len("SIGNED_AB "):])
            if "wrong_sign" in r:
                out[r["pool"]] = r["wrong_sign"]
    return out


def mse(arm, axis):
    out = {}
    for f in glob.glob(os.path.join(READS, arm, axis, "*.npz")):
        z = np.load(f)
        out[os.path.basename(f)[:-4]] = float(np.mean((z["q"] - z["t"]) ** 2))
    return out


def sign_p(k, n):  # one-sided: P(at least k lower of n) under p = 1/2
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main():
    s = {a: signed(a) for a in CTRL + EXT}
    pools = sorted(set.intersection(*(set(v) for v in s.values())))
    c = np.array([[s[a][p] for p in pools] for a in CTRL])
    e = np.array([[s[a][p] for p in pools] for a in EXT])
    cm, em = c.mean(0), e.mean(0)
    d = em - cm
    lower, higher = int((d < 0).sum()), int((d > 0).sum())
    res = {"pools": len(pools),
           "sign_changing_per_arm": {a: len(v) for a, v in s.items()},
           "seed_mean": {"ext_lower": lower, "ext_higher": higher, "ties": len(pools) - lower - higher,
                         "p_one_sided": sign_p(lower, lower + higher),
                         "median_ctrl_pct": round(100 * float(np.median(cm)), 2),
                         "median_ext_pct": round(100 * float(np.median(em)), 2)},
           "per_seed": {str(sd): {"ext_lower": int((e[i] < c[i]).sum()), "ext_higher": int((e[i] > c[i]).sum()),
                                  "median_ctrl_pct": round(100 * float(np.median(c[i])), 2),
                                  "median_ext_pct": round(100 * float(np.median(e[i])), 2)}
                        for i, sd in enumerate((42, 43))},
           "per_pool_pct": {p: {"ctrl": [round(100 * x, 2) for x in c[:, j]], "ext": [round(100 * x, 2) for x in e[:, j]]}
                            for j, p in enumerate(pools)},
           "mse_ratio": {}}
    for ax in AXES:
        m = {a: mse(a, ax) for a in CTRL + EXT}
        probes = sorted(set.intersection(*(set(v) for v in m.values())))
        if not probes:
            continue
        cmse = np.array([[m[a][p] for p in probes] for a in CTRL]).mean(0)
        emse = np.array([[m[a][p] for p in probes] for a in EXT]).mean(0)
        seed = np.array([m[CTRL[1]][p] / m[CTRL[0]][p] for p in probes])
        res["mse_ratio"][ax] = {"probes": len(probes),
                                "median_ext_over_ctrl": round(float(np.median(emse / cmse)), 3),
                                "per_seed": [round(float(np.median(np.array([m[EXT[i]][p] / m[CTRL[i]][p] for p in probes]))), 3)
                                             for i in range(2)],
                                "ctrl_seed43_over_42": round(float(np.median(seed)), 3),
                                "median_mse_ctrl": round(float(np.median(cmse)), 4),
                                "median_mse_ext": round(float(np.median(emse)), 4)}
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "data", "signext_compare.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_pool_pct"}))


if __name__ == "__main__":
    main()
