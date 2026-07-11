#!/usr/bin/env python
"""Diagnose the HETEROSC uug finetune underfit (hand-off priority 1).

Evaluate mu-MSE (preprocessed, beta-invariant) + sigma stats on the run's OWN val set for
a list of (run_dir, ckpt) pairs. Question: is the underfit due to checkpoint selection?
Checkpoint 'best' is picked by val_loss_no_reg = beta-NLL (base_experiment.py:866), NOT by
mu-MSE. If 'last' has much better mu-MSE than 'best', selection-on-NLL is the culprit.

Matched MSE antenna finetune (loss=MSE, identical HPs) is the reference: its uug val mu-MSE
is ~8.5e-5 (preprocessed). GPU (xformers). Run via diag_ckpt_mu.sh.
"""
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


def eval_ckpt(run_dir, ckpt):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.save_source = False; cfg.warm_start_idx = None
        cfg.ema = False; cfg.count_flops = False
        cfg.run_dir = os.path.join(REPO, "runs", "_diag_tmp")
        if cfg.get("fine_tune") is not None:
            cfg.fine_tune.pretrained_path = None
        if "loss" in cfg.model.net:
            cfg.model.net.loss = cfg.training.loss
    beta = float(cfg.training.get("heterosc_beta", 0.0) or 0.0)
    sys.path.insert(0, WT)
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data(); exp._init_dataloader(); exp.init_model()
    exp.model.load_state_dict(load_finetuned_state(os.path.join(run_dir, "models", ckpt))["model"])
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    with torch.no_grad():
        pred, truth, sigmas = exp._collect_predictions(exp.val_loader)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    d = pred - truth
    out = dict(mse=float(np.mean(d ** 2)), mae=float(np.mean(np.abs(d))),
               n=int(len(d)), loss=str(cfg.training.loss), beta=beta, ckpt=ckpt)
    if sigmas is not None:
        s = np.asarray(sigmas, dtype=np.float64).reshape(-1)
        pull = d / np.clip(s, 1e-15, None)
        out.update(sigma_med=float(np.median(s)), sigma_mean=float(np.mean(s)),
                   pull_std=float(np.std(pull)), calib=float(np.std(pull)))
    return out


def main():
    HET = os.path.join(WT, "runs/heterosc_foundation/ft_uug_het")
    MSE = os.path.join(REPO, "runs/pretrain22_heldout_uug/ft_deep_antenna")
    jobs = [
        ("HET_uug_best", HET, "model_run0_best.pt"),
        ("HET_uug_last", HET, "model_run0.pt"),
        ("MSE_uug_best", MSE, "model_run0_best.pt"),
        ("MSE_uug_last", MSE, "model_run0.pt"),
    ]
    results = []
    for label, rd, ckpt in jobs:
        if not os.path.exists(os.path.join(rd, "config.yaml")):
            print(f"  missing {rd}; skip", flush=True); continue
        try:
            r = eval_ckpt(rd, ckpt); r["label"] = label
            results.append(r)
            extra = ""
            if "sigma_med" in r:
                extra = f"  sigma_med={r['sigma_med']:.3g} calib(pull_std)={r['calib']:.3g}"
            print(f"  {label:14s} loss={r['loss']:8s} b={r['beta']:.1f}  "
                  f"val mu-MSE={r['mse']:.4g}  mu-MAE={r['mae']:.4g}{extra}", flush=True)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"  {label}: FAILED {ex}", flush=True)
    out = os.path.join(WT, "analysis/divergences/diag_ckpt_mu.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
