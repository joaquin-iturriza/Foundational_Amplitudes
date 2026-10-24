"""D99, the signed pools' short A/B: catalog_v3's sign-changing one-loop pools trained with DlessInOrdQ's short setup on
the MSE (the sign head refuses the het loss; both arms alike), two targets for them: SgnR0, the signed log with r0 three
nats below each pool's 0.5% quantile of |x|, and SgnHead, log|x| floored at its 1% quantile with a sign head. Scored
per pool on its validation events (tools/signed_ab_score.py, run FA-ccin2p3-1f951ace): the MSE of log max(|x|, floor)
with one floor for both arms (the magnitude, on a common footing), the offset and the MSE after it, the wrong-sign
fraction, and the same within a factor 10 of zero. Pools: the 48 trained on (in distribution) and the 14 held out (H5,
zero-shot), recipes/catalog_signed_ab_{indist,heldout}.yaml. Medians over pools, and the per-pool ratio SgnR0 / SgnHead.
    python3 analysis/catalog_daniel/signed_ab.py   -> data/signed_ab.json, the table printed
"""
import json, os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KEYS = ("mse_mag", "bias", "mse_after_offset", "wrong_sign", "near_share", "mse_mag_near", "wrong_sign_near")


def main():
    rows = json.load(open(os.path.join(HERE, "data", "signed_ab_runs.json")))
    by = {(r["arm"], r["pool"]): r for r in rows}
    out = {}
    for setname in ("indist", "heldout"):
        names = [p["name"] for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_signed_ab_{setname}.yaml")))["processes"]]
        pools = sorted({pool for (_, pool) in by if any(pool.startswith(n + "_") for n in names)})
        res = {"pools": len(pools)}
        for arm in ("r0", "head"):
            res[arm] = {k: float(np.median([by[(arm, p)][k] for p in pools if by[(arm, p)].get(k) is not None])) for k in KEYS}
        ratio = np.array([by[("r0", p)]["mse_mag"] / by[("head", p)]["mse_mag"] for p in pools])
        res["ratio_r0_over_head"] = [float(x) for x in np.quantile(ratio, [0.1, 0.5, 0.9])]
        res["r0_better"] = int((ratio < 1).sum())
        out[setname] = res
        print(setname, res["pools"], "pools;", " | ".join(f"{a}: " + ", ".join(f"{k} {res[a][k]:.3g}" for k in KEYS) for a in ("r0", "head")),
              "| r0/head 10/50/90%", np.round(res["ratio_r0_over_head"], 2).tolist(), f"r0 better on {res['r0_better']}")
    json.dump(out, open(os.path.join(HERE, "data", "signed_ab.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
