#!/usr/bin/env python
"""L1 step 2 for ee->uu: forward the two-stage (mu, sigma) model over a large candidate pool,
saving per-event predicted uncertainty sigma_ln (Delta-ln units), pred/true log|M|^2, and sqrt(s),
row-aligned to the pool. Feeds the sigma-reweight subsampler (build_reweight_pools_eeuu.py):
pi ∝ sigma^alpha selects WHERE the training budget goes, emphasizing high-uncertainty events on
top of the L0 coverage base. sqrt(s) is carried only for diagnostics/eval, never used to reweight.
GPU (one forward pass). Adapted from extract_train_pool_sigma.py (uug)."""
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
from eval_heldout import com_normalize  # noqa
from extract_preds import load_finetuned_state  # noqa


def _spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(REPO, "data_l1pool_eeuu/ee_uu_91-1000GeV_amplitudes.npy"))
    ap.add_argument("--run_dir", default=os.path.join(REPO, "runs/eeuu_sigfit/mix025_sigma"))
    ap.add_argument("--ckpt", default="model_run0_best.pt")
    ap.add_argument("--out", default=os.path.join(REPO, "analysis/divergences/l1pool_sigma_eeuu.npz"))
    ap.add_argument("--batch_events", type=int, default=8192)
    args = ap.parse_args()

    rows = np.load(args.pool).astype(np.float64)
    P = (rows.shape[1] - 1) // 5
    raw_mom = rows[:, : P * 4].reshape(-1, P, 4)
    pdg = rows[:, P * 4: P * 5].astype(int)
    true_logamp = np.log(rows[:, -1])
    sqrt_s = 2.0 * rows[:, 0]

    cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.save_source = False; cfg.warm_start_idx = None
        cfg.ema = False; cfg.count_flops = False
        cfg.run_dir = os.path.join(REPO, "runs", "_l1_extract_tmp")
        cfg.data.subsample = None
        cfg.fine_tune.pretrained_path = None
        cfg.model.net.loss = cfg.training.loss                 # HETEROSC -> 2ch net
    # the analysis helpers above prepend the MAIN repo to sys.path; re-assert the worktree so
    # hydra instantiates the HETEROSC models.lloca (with sigma_after_pool), not MAIN's 1ch net.
    sys.path.insert(0, WT)
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data(); exp._init_dataloader(); exp.init_model()
    exp.model.load_state_dict(load_finetuned_state(os.path.join(args.run_dir, "models", args.ckpt))["model"])
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    mom_div = float(exp.mom_div)
    amp_mean = float(np.atleast_1d(exp.prepd_mean)[0]); amp_std = float(np.atleast_1d(exp.prepd_std)[0])
    out_shape = cfg.model.net.get("out_shape") or cfg.model.net.get("out_channels") or 1

    parts = com_normalize(raw_mom, mom_div)
    from particle_ids import global_encode
    toks = global_encode(pdg)
    order_row = np.array(exp._order_row(0, exp._resolve_amp_orders(list(cfg.data.dataset))), dtype=np.float32)

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
    pred_logamp = np.concatenate(mus) * amp_std + amp_mean
    sigma_ln = np.concatenate(sigmas) * amp_std               # predicted uncertainty, Delta-ln units

    np.savez(args.out, pred_logamp=pred_logamp.astype(np.float64),
             true_logamp=true_logamp.astype(np.float64),
             sigma_ln=sigma_ln.astype(np.float64), sqrt_s=sqrt_s.astype(np.float64))
    err = np.abs(pred_logamp - true_logamp)
    print(f"saved {args.out}  N={len(err)}")
    print(f"pool sigma-vs-|err| Spearman = {_spear(sigma_ln, err):.3f}  "
          f"(median err {np.median(err):.4f}, median sigma {np.median(sigma_ln):.4f})")


if __name__ == "__main__":
    main()
