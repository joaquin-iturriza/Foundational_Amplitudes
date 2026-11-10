#!/usr/bin/env python
"""Rank HETEROSC foundation configs by mu-quality on the foundation's OWN (in-distribution)
val set — the 22-process leave-uug-out data. mu-MSE (preprocessed) is beta-invariant (only
mu, sigma split off by _collect_predictions), so it ranks 'did mu fit' across (lr, beta) and
vs the MSE base22 reference. NOT evaluated on held-out uug (that would confound fit with
transfer). Answers the HP-hypothesis: is the earlier mu-underfit an HP/loss-scale issue? GPU.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root, so siteconf imports from anywhere
import siteconf  # noqa: E402
import argparse
import json
import os
import sys


REPO = siteconf.PROJECT_DIR
WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WT)
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
from diag_ckpt_mu import eval_ckpt  # noqa


def mu_mse_for_run(run_dir, ckpt="model_run0_best.pt"):
    """mu-MSE/MAE (preprocessed, beta-invariant) on the run's own val set: diag_ckpt_mu.eval_ckpt."""
    return eval_ckpt(run_dir, ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default=os.path.join(WT, "runs/heterosc_foundation"))
    ap.add_argument("--tags", default="lr0.004_b0.0,lr0.004_b0.5,lr0.004_b1.0,"
                                      "lr0.016_b0.0,lr0.016_b0.5,lr0.016_b1.0")
    ap.add_argument("--prefix", default="het_")
    ap.add_argument("--mse_ref", default=os.path.join(REPO, "runs/pretrain22_heldout_uug/base"),
                    help="MSE base22 reference (loss=MSE) on the SAME val data; '' to skip")
    ap.add_argument("--out", default=os.path.join(WT, "analysis/divergences/heterosc_rank.json"))
    args = ap.parse_args()

    results = []
    entries = [(t.strip(), os.path.join(args.runs_root, f"{args.prefix}{t.strip()}"))
               for t in args.tags.split(",")]
    if args.mse_ref:
        entries.append(("MSE_base22_ref", args.mse_ref))
    for tag, rd in entries:
        if not os.path.exists(os.path.join(rd, "config.yaml")):
            print(f"  missing {rd}; skip", flush=True); continue
        try:
            r = mu_mse_for_run(rd); r["tag"] = tag
            results.append(r)
            print(f"  {tag:20s} loss={r['loss']:8s} beta={r['beta']:.1f}  "
                  f"val mu-MSE={r['mse']:.4g}  mu-MAE={r['mae']:.4g}  (N={r['n']})", flush=True)
        except Exception as ex:
            print(f"  {tag}: FAILED {ex}", flush=True)

    results.sort(key=lambda r: r["mse"])
    print("\n=== ranked by in-distribution val mu-MSE (best first) ===")
    for r in results:
        print(f"  {r['tag']:20s} mu-MSE={r['mse']:.4g}")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
