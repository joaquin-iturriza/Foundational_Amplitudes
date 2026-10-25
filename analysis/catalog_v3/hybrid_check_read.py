"""Reads the CPU check of data.target_physics_hybrid (hybrid_check.py, run on CC's login node; its HYBRID_CHECK lines
copied home into data/hybrid_check.json) and prints one flat JSON line, the numbers the hybrid paragraph in
docs/results.tex cites. The 2 -> 2 pool is left out of the ranges: its c_p is zero on any events.
  n_pools                         pools with five particles or more
  dcp_min, dcp_max                |c_p(first 10000 train events) - c_p(first 2000 validation events)|
  serr_min, serr_max              the validation estimate's sampling error, c_p's event sd / sqrt(2000)
  ratio_max                       the largest |difference| / sampling error
  worst_{cp_val_minus_prep, shape_sd, level_val}   the identities hybrid_check.py tests, zero up to rounding
    python3 analysis/catalog_v3/hybrid_check_read.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cl = json.load(open(os.path.join(HERE, "data", "hybrid_check.json")))["classes"]
    rs = [r for r in cl.values() if r["N"] >= 5]
    d = [abs(r["cp_train_minus_val"]) for r in rs]
    e = [r["sampling_err"] for r in rs]
    out = {"n_pools": len(rs), "dcp_min": round(min(d), 4), "dcp_max": round(max(d), 4),
           "serr_min": round(min(e), 4), "serr_max": round(max(e), 4),
           "ratio_max": round(max(a / b for a, b in zip(d, e)), 3)}
    for k in ("cp_val_minus_prep", "shape_sd", "level_val"):
        out[f"worst_{k}"] = max(abs(r[k]) for r in cl.values())
    print(json.dumps(out))


if __name__ == "__main__":
    main()
