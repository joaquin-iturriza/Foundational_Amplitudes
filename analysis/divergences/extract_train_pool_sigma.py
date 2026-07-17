#!/usr/bin/env python
"""Q2 confirmation, step 1: per-event (error, sigma, y_min) over the TRAIN pool.

Forward the (mu, sigma) deep-IR model over the full antenna training pool .npy so we
can resample it by a sigma-driven proposal pi ∝ Q * score^alpha (build_reweight_pools.py).
Saves arrays aligned to the pool's row order:
  pred_logamp, true_logamp, sigma_ln (predicted uncertainty, Delta ln units), y_min.
GPU (one forward pass, a few minutes).
"""
import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
WT = os.path.join(REPO, "worktrees/wt-heterosc")
sys.path.insert(0, WT)
from experiment import AmplitudeExperiment  # noqa
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
from extract_ir import ir_observables  # noqa
from extract_preds import load_finetuned_state  # noqa
from eval_heldout import com_normalize  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(REPO, "data_deep_antenna/ee_uug_91-1000GeV_amplitudes.npy"))
    ap.add_argument("--run_dir", default=os.path.join(WT, "runs/heterosc_twostage2/ft_antenna"))
    ap.add_argument("--ckpt", default="model_run0_best.pt")
    ap.add_argument("--out", default=os.path.join(WT, "analysis/divergences/train_pool_sigma_antenna.npz"))
    ap.add_argument("--batch_events", type=int, default=8192)
    args = ap.parse_args()

    rows = np.load(args.pool).astype(np.float64)
    P = (rows.shape[1] - 1) // 5
    raw_mom = rows[:, : P * 4].reshape(-1, P, 4)
    pdg = rows[:, P * 4 : P * 5].astype(int)
    raw_amp = rows[:, -1].astype(np.float64)
    obs = ir_observables(raw_mom, pdg[0])   # pdg identical per row (single process)
    y_min = obs["y_min"]
    true_logamp = np.log(np.abs(raw_amp))

    cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.save_source = False; cfg.warm_start_idx = None
        cfg.ema = False; cfg.count_flops = False
        cfg.run_dir = os.path.join(REPO, "runs", "_q2_extract_tmp")
        cfg.data.subsample = None
        cfg.fine_tune.pretrained_path = None
        cfg.model.net.loss = cfg.training.loss
    sys.path.insert(0, WT)
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data(); exp._init_dataloader(); exp.init_model()
    exp.model.load_state_dict(load_finetuned_state(
        os.path.join(args.run_dir, "models", args.ckpt))["model"])
    exp.model.to(exp.device, dtype=exp.dtype).eval()

    mom_div = float(exp.mom_div)
    amp_mean = float(np.atleast_1d(exp.prepd_mean)[0])
    amp_std = float(np.atleast_1d(exp.prepd_std)[0])
    out_shape = cfg.model.net.get("out_shape") or cfg.model.net.get("out_channels")

    parts = com_normalize(raw_mom, mom_div)
    from particle_ids import global_encode
    toks = global_encode(pdg)
    order_row = np.array(exp._order_row(0, exp._resolve_amp_orders(list(cfg.data.dataset))),
                         dtype=np.float32)

    mus, sigmas = [], []
    with torch.no_grad():
        for s in range(0, len(parts), args.batch_events):
            e = min(s + args.batch_events, len(parts)); nb = e - s
            pt = torch.tensor(parts[s:e], dtype=exp.dtype, device=exp.device).reshape(-1, 4)
            tk = torch.tensor(toks[s:e], dtype=torch.long, device=exp.device).reshape(nb * P)
            ptr = torch.arange(0, nb * P + 1, P, dtype=torch.long, device=exp.device)
            ol = torch.tensor(np.tile(order_row, (nb, 1)), dtype=exp.dtype, device=exp.device)
            pid_t = torch.zeros(nb, dtype=torch.long, device=exp.device)
            yp = exp.model(pt, tk, mean=exp.mom_mean[0], std=exp.mom_std[0],
                           ptr=ptr, order_labels=ol, process_ids=pid_t)
            mus.append(yp[..., :out_shape].detach().cpu().float().numpy().reshape(-1))
            sigmas.append(yp[..., -out_shape:].detach().cpu().float().numpy().reshape(-1))
    mu_prepd = np.concatenate(mus); sigma_prepd = np.concatenate(sigmas)
    pred_logamp = mu_prepd * amp_std + amp_mean
    sigma_ln = sigma_prepd * amp_std                  # predicted uncertainty, Delta ln units

    np.savez(args.out,
             pred_logamp=pred_logamp.astype(np.float64),
             true_logamp=true_logamp.astype(np.float64),
             sigma_ln=sigma_ln.astype(np.float64),
             y_min=y_min.astype(np.float64))
    err = np.abs(pred_logamp - true_logamp)
    # quick self-check: sigma-vs-error rank quality on the pool
    def spear(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        ra = ra - ra.mean(); rb = rb - rb.mean()
        return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))
    print(f"saved {args.out}  N={len(err)}")
    print(f"pool sigma-vs-|err| Spearman = {spear(sigma_ln, err):.3f}  "
          f"(median err {np.median(err):.4f}, median sigma {np.median(sigma_ln):.4f})")


if __name__ == "__main__":
    main()
