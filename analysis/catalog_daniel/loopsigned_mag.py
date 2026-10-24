"""The sign-changing one-loop pools (recipes/catalog_loop_signed.yaml) scored zero-shot on the magnitude, log max(|M|^2,
s) (zero_shot_parent.py --signed-magnitude; scripts/job_catalog_norm.sh <arm> eval_short loopsigned), from the two short
catalog models DlessInOrd (run FA-ccin2p3-e1e5d61f) and DlessInOrdQ (FA-ccin2p3-f20eb068), 2026-10-23. Reads the runs'
logs (site logs -n 5000 <run>), keeps each ZERO_SHOT line: per probe the MSE of log|M|^2, the offset (bias), the MSE
after the offset, MSE / variance, the correlation; the probes that failed and why. Medians over the probes both arms
scored.
    python3 analysis/catalog_daniel/loopsigned_mag.py <DlessInOrd log> <DlessInOrdQ log>   -> data/loopsigned_mag.json
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("mse_logm2_real", "bias", "mse_after_offset", "mse_over_var", "corr", "frac_at_floor")


def read(path):
    ok, bad = {}, {}
    for line in open(path):
        if not line.startswith("ZERO_SHOT {"):
            continue
        d = json.loads(line[len("ZERO_SHOT "):])
        name = os.path.basename(d["recipe"]).rsplit(".", 1)[0]
        if "error" in d:
            bad[name] = d["error"][:160]
        else:
            ok[name] = {k: d.get(k, 0.0) for k in KEYS}
    return ok, bad


def main():
    arms = dict(zip(("DlessInOrd", "DlessInOrdQ"), sys.argv[1:3]))
    out = {}
    res = {a: read(p) for a, p in arms.items()}
    both = sorted(set.intersection(*(set(r[0]) for r in res.values())))
    for a, (ok, bad) in res.items():
        out[a] = {"scored": len(ok), "failed": len(bad), "failures": bad, "probes": ok,
                  "median": {k: float(np.median([ok[p][k] for p in both])) for k in KEYS}}
        print(a, len(ok), "scored,", len(bad), "failed;", json.dumps({k: round(v, 3) for k, v in out[a]["median"].items()}))
    out["both"] = both
    json.dump(out, open(os.path.join(HERE, "data", "loopsigned_mag.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
