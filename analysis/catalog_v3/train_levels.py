#!/usr/bin/env python3
"""The reference level of the pretraining comparisons' MSEs: each run's own per-process validation MSE of
log|M|^2 on its training pools, at its best checkpoint, in natural-log units (the units the pp and zero-shot reads
are quoted in), against which a read of 0.4 to 3 is placed.

result.json's proc_val_losses_no_reg is in standardized target units; with one shared standardization (asserted)
it is multiplied by that std squared, as pp_compare.collect does for window_train.
    on the site:  python analysis/catalog_v3/train_levels.py collect > analysis/catalog_v3/data/train_levels.json
    at home:      python3 analysis/catalog_v3/train_levels.py        (the medians, last line JSON)"""
import json, math, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "analysis", "catalog_v3", "data", "train_levels.json")
RUNS = {"dimensionless": "runs/catalog_v3_short/dimensionless",                   # AdamW, catalog_v3 windows
        "dimensionless_pp": "runs/catalog_v3_short/dimensionless_pp",             # AdamW, hadronic classes on pp
        "dimensionless_muon_best": "runs/catalog_v3_short/dimensionless_muon_best",
        "dimensionless_muon_best_seed2": "runs/catalog_v3_short/dimensionless_muon_best_seed2",
        "v4_dimensionless_muon": "runs/catalog_v4_short/v4_dimensionless_muon",
        "v4_dimensionless_muon_seed2": "runs/catalog_v4_short/v4_dimensionless_muon_seed2"}
RMS = 0.6        # an rms error in ln|M|^2 read as a factor per event, exp(RMS)


def collect():
    out = {}
    for r, d in RUNS.items():
        res = json.load(open(os.path.join(ROOT, d, "result.json")))
        st = json.load(open(os.path.join(ROOT, d, "data_stats.json")))
        sd = st["prepd_std"]
        assert st.get("shared_standardization") and max(sd) - min(sd) < 1e-9 * max(sd), f"{r}: not one shared std"
        out[r] = {"best_step": res.get("best_step"), "shared_std": sd[0],
                  "mse": {k: float(v) * sd[0] ** 2 for k, v in res["proc_val_losses_no_reg"].items()}}
    json.dump(out, sys.stdout)


def main():
    d = json.load(open(DATA))
    out = {r: {"median": round(statistics.median(v["mse"].values()), 4), "n": len(v["mse"]),
               "best_step": v["best_step"]} for r, v in d.items()}
    out["rms_factor"] = round(math.exp(RMS), 3)
    print(json.dumps(out))


if __name__ == "__main__":
    collect() if sys.argv[1:] == ["collect"] else main()
