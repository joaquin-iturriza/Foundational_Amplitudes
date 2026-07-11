#!/usr/bin/env python
"""Rank HETEROSC foundation configs by mu-quality on the foundation's OWN (in-distribution)
val set — the 22-process leave-uug-out data. mu-MSE (preprocessed) is beta-invariant (only
mu, sigma split off by _collect_predictions), so it ranks 'did mu fit' across (lr, beta) and
vs the MSE base22 reference. NOT evaluated on held-out uug (that would confound fit with
transfer). Answers the HP-hypothesis: is the earlier mu-underfit an HP/loss-scale issue? GPU.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WT)
from experiment import AmplitudeExperiment  # noqa
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
from extract_preds import load_finetuned_state  # noqa


def mu_mse_for_run(run_dir, ckpt="model_run0_best.pt"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.save_source = False; cfg.warm_start_idx = None
        cfg.ema = False; cfg.count_flops = False
        cfg.run_dir = os.path.join(REPO, "runs", "_rank_tmp")
        if cfg.get("fine_tune") is not None:
            cfg.fine_tune.pretrained_path = None
        if "loss" in cfg.model.net:
            cfg.model.net.loss = cfg.training.loss   # force correct head width on rebuild
    beta = float(cfg.training.get("heterosc_beta", 0.0) or 0.0)
    sys.path.insert(0, WT)  # re-assert: hydra imports models.lloca lazily (see extract_sigma)
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data(); exp._init_dataloader(); exp.init_model()
    exp.model.load_state_dict(load_finetuned_state(os.path.join(run_dir, "models", ckpt))["model"])
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    with torch.no_grad():
        pred, truth, _ = exp._collect_predictions(exp.val_loader)   # sigma split off; preprocessed mu
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    d = pred - truth
    return dict(mse=float(np.mean(d ** 2)), mae=float(np.mean(np.abs(d))),
                n=len(d), loss=str(cfg.training.loss), beta=beta)


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
