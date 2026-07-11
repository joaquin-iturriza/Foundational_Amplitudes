#!/usr/bin/env python
"""Rank HETEROSC foundation configs by mu-quality: forward the uug deep-IR test through
each run and report mu-MSE (Delta ln|M|^2) + median relative error. beta-invariant (only
mu, not sigma), so it ranks 'did mu fit' across (lr, beta) and vs the MSE base22 reference.

Answers the HP-hypothesis test: does a properly-tuned HETEROSC foundation fit mu as well
as MSE, or does mu underfit at every (lr, beta)? GPU.
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
from extract_ir import ir_observables  # noqa
from extract_preds import load_finetuned_state  # noqa
from eval_heldout import com_normalize  # noqa


def mu_mse_for_run(run_dir, ho, ckpt="model_run0_best.pt", batch=8192):
    rows = np.asarray(ho["rows"], dtype=np.float64)
    pdg = np.asarray(ho["pdg"], dtype=int)
    P = (rows.shape[1] - 1) // 5
    raw_mom = rows[:, : P * 4].reshape(-1, P, 4)
    true_logamp = np.log(rows[:, -1].astype(np.float64))
    y = ir_observables(raw_mom, pdg)["y_min"]

    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.save_source = False; cfg.warm_start_idx = None
        cfg.ema = False; cfg.count_flops = False
        cfg.run_dir = os.path.join(REPO, "runs", "_rank_tmp")
        cfg.data.subsample = 200000
        if cfg.get("fine_tune") is not None:
            cfg.fine_tune.pretrained_path = None
        if "loss" in cfg.model.net:
            cfg.model.net.loss = cfg.training.loss   # force correct head width on rebuild
    is_het = cfg.training.loss == "HETEROSC"
    sys.path.insert(0, WT)  # re-assert: hydra imports models.lloca lazily (see extract_sigma)
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data(); exp._init_dataloader(); exp.init_model()
    exp.model.load_state_dict(load_finetuned_state(os.path.join(run_dir, "models", ckpt))["model"])
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    amp_mean = float(np.atleast_1d(exp.prepd_mean)[0]); amp_std = float(np.atleast_1d(exp.prepd_std)[0])
    out_shape = cfg.model.net.get("out_shape") or cfg.model.net.get("out_channels")

    parts = com_normalize(raw_mom, float(exp.mom_div))
    from particle_ids import global_encode
    toks = global_encode(np.tile(pdg, (len(parts), 1)))
    order_row = np.array(exp._order_row(0, exp._resolve_amp_orders(list(cfg.data.dataset))), dtype=np.float32)
    mus = []
    with torch.no_grad():
        for s in range(0, len(parts), batch):
            e = min(s + batch, len(parts)); nb = e - s
            pt = torch.tensor(parts[s:e], dtype=exp.dtype, device=exp.device).reshape(-1, 4)
            tk = torch.tensor(toks[s:e], dtype=torch.long, device=exp.device).reshape(nb * P)
            ptr = torch.arange(0, nb * P + 1, P, dtype=torch.long, device=exp.device)
            ol = torch.tensor(np.tile(order_row, (nb, 1)), dtype=exp.dtype, device=exp.device)
            pid_t = torch.zeros(nb, dtype=torch.long, device=exp.device)
            yp = exp.model(pt, tk, mean=exp.mom_mean[0], std=exp.mom_std[0],
                           ptr=ptr, order_labels=ol, process_ids=pid_t)
            mu = yp[..., :out_shape] if is_het else yp
            mus.append(mu.detach().cpu().float().numpy().reshape(-1))
    pred_logamp = np.concatenate(mus) * amp_std + amp_mean
    d = pred_logamp - true_logamp
    return dict(mse=float(np.mean(d ** 2)), medrel=float(np.median(np.abs(np.exp(np.abs(d)) - 1.0))),
                n=len(d), loss=cfg.training.loss, beta=float(cfg.training.get("heterosc_beta", 0.0) or 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default=os.path.join(WT, "runs/heterosc_foundation"))
    ap.add_argument("--tags", default="lr0.004_b0.0,lr0.004_b0.5,lr0.004_b1.0,"
                                      "lr0.016_b0.0,lr0.016_b0.5,lr0.016_b1.0")
    ap.add_argument("--prefix", default="het_")
    ap.add_argument("--mse_ref", default=os.path.join(REPO, "runs/pretrain22_heldout_uug/base"),
                    help="MSE base22 reference run dir (loss=MSE) for comparison; '' to skip")
    ap.add_argument("--heldout", default=os.path.join(REPO, "analysis/divergences/uug_deep_test.npz"))
    ap.add_argument("--out", default=os.path.join(WT, "analysis/divergences/heterosc_rank.json"))
    args = ap.parse_args()
    ho = np.load(args.heldout)

    results = []
    for tag in [t.strip() for t in args.tags.split(",")]:
        rd = os.path.join(args.runs_root, f"{args.prefix}{tag}")
        if not os.path.exists(os.path.join(rd, "config.yaml")):
            print(f"  missing {rd}; skip", flush=True); continue
        try:
            r = mu_mse_for_run(rd, ho); r["tag"] = tag
            results.append(r)
            print(f"  {tag:20s} loss={r['loss']:8s} beta={r['beta']:.1f}  mu-MSE={r['mse']:.4g}  "
                  f"med-rel={r['medrel']*100:.2f}%", flush=True)
        except Exception as ex:
            print(f"  {tag}: FAILED {ex}", flush=True)
    if args.mse_ref and os.path.exists(os.path.join(args.mse_ref, "config.yaml")):
        r = mu_mse_for_run(args.mse_ref, ho); r["tag"] = "MSE_base22_ref"
        results.append(r)
        print(f"  {'MSE_base22_ref':20s} loss={r['loss']:8s}          mu-MSE={r['mse']:.4g}  "
              f"med-rel={r['medrel']*100:.2f}%", flush=True)

    results.sort(key=lambda r: r["mse"])
    print("\n=== ranked by mu-MSE (best first) ===")
    for r in results:
        print(f"  {r['tag']:20s} mu-MSE={r['mse']:.4g}  med-rel={r['medrel']*100:.2f}%")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
