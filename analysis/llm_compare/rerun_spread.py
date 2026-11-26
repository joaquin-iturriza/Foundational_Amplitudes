"""llm_compare: the same-setup rerun spread, the single-seed margin of the width and level-file reads (steps 13/14/16).
Two runs of one config and seed (his setup, his optimizer settings: first run FA-lxplus-cb9273f1 -> data/control_*,
retrain FA-ccin2p3-0165aa58 -> data/control_retrain_*), per set (seen 1352, relabels 96, new 43 of suprio.split()):
the ratio of their median per-process MSE of log|M|^2, of their pooled RMSE, and the median and 10-90% range of the
per-process |log ratio| as a factor. The margin for a single-seed median is the larger of the two runs' median ratios.
    python analysis/llm_compare/rerun_spread.py [--a control --b control_retrain]
Output: data/rerun_spread.json, and its JSON on the last line.
"""
import argparse, json, os

import numpy as np

import suprio

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag, role):
    return json.load(open(os.path.join(HERE, "data", f"{tag}_{role}.json")))["processes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="control")
    ap.add_argument("--b", default="control_retrain")
    a = ap.parse_args()
    sp = suprio.split()
    out = {"a": a.a, "b": a.b}
    for name, role, keep in (("seen", "seen", None), ("relabels", "heldout", sp["interp"]), ("new", "heldout", sp["extrap"])):
        pa, pb = load(a.a, role), load(a.b, role)
        ns = sorted(keep if keep is not None else set(pa) & set(pb))
        assert all(n in pa and n in pb for n in ns), name
        ma = np.array([pa[n]["mse_logm2"] for n in ns])
        mb = np.array([pb[n]["mse_logm2"] for n in ns])
        r = np.abs(np.log(mb / ma))
        out[name] = {"n": len(ns), "median_mse": [float(np.median(ma)), float(np.median(mb))],
                     "median_ratio": float(max(np.median(ma), np.median(mb)) / min(np.median(ma), np.median(mb))),
                     "pooled_rmse": [float(np.sqrt(ma.mean())), float(np.sqrt(mb.mean()))],
                     "pooled_ratio": float(max(ma.mean(), mb.mean()) / min(ma.mean(), mb.mean())) ** 0.5,
                     "per_process_factor_median": float(np.exp(np.median(r))),
                     "per_process_factor_p10_p90": [float(np.exp(np.quantile(r, 0.1))), float(np.exp(np.quantile(r, 0.9)))]}
    json.dump(out, open(os.path.join(HERE, "data", "rerun_spread.json"), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
