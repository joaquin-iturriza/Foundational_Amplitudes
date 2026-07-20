#!/usr/bin/env python
"""Score a trained run on the FIXED held-out deep-IR uugg set, on raw log|M|^2, binned by y_min decade.

Loads a run's config + best checkpoint, re-inits data on the run's round-0 dir (the random-Lorentz aug
is seeded by cfg.seed, so this reproduces the run's frozen mom/amp stats EXACTLY), forwards the held-out
momenta through the (mu[,sigma]) model, inverts the amp preprocessing to predicted log|M|^2, and reports
MSE(pred,true) overall and per y_min decade. Works for any run (sigma arm, base arm, or an existing
MSE finetune like ft_uugg_f100) -> a preprocessing-independent, like-for-like comparison. GPU.
"""
import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, WT)
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
import l2_online_uugg as L                       # preprocess_increment, PDG        # noqa: E402

DECADES = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]
DEFAULT_HELDOUT = os.path.join(REPO, "analysis/divergences/heldout_uugg_deepIR.npz")


def score_and_save(exp, label, heldout_path=DEFAULT_HELDOUT):
    """Score a LIVE exp (frozen stats, model already loaded + in eval mode) on the FIXED held-out deep-IR
    uugg set, on raw log|M|^2, binned by y_min decade. Reports overall + per-decade MSE and saves
    heldout_eval_{label}.npz (row-aligned pred/true/y_min/sigma for downstream plots). This is the SHARED
    core used by both the standalone eval (which rebuilds exp from config.yaml + a checkpoint) and the
    driver's fold-in tail (which passes its already-trained, best-checkpoint-reloaded exp) -- so no
    separate eval job / queue wait is needed for a fresh run."""
    from dataset import AmplitudeDataset, build_flat_arrays, collate_variable_length
    d = np.load(heldout_path)
    rows = d["rows"]; y_min = d["y_min"]
    P_held = rows[:, :L.NP * 4].reshape(-1, L.NP, 4)
    true_logamp = np.log(rows[:, -1]); me2 = rows[:, -1]

    parts, toks, amp_prepd, orders, pids = L.preprocess_increment(exp, P_held, me2)
    pf, tf, off = build_flat_arrays(parts, toks)
    ds = AmplitudeDataset(particles_flat=pf, offsets=off, amplitudes=np.asarray(amp_prepd).reshape(-1, 1),
                          tokens_flat=tf, order_labels=np.asarray(orders),
                          process_ids=pids.astype(np.int64), dtype=exp.dtype)
    loader = torch.utils.data.DataLoader(ds, batch_size=int(exp.cfg.evaluation.batchsize), shuffle=False,
                                         drop_last=False, collate_fn=collate_variable_length, num_workers=0)
    was_training = exp.model.training
    exp.model.eval()
    with torch.no_grad():
        pred_prepd, _truth_prepd, sigma = exp._collect_predictions(loader)
    if was_training:
        exp.model.train()

    # amp target is [log, standardization]; UNDO ONLY the standardization to stay in log space.
    amp_mean = float(np.atleast_1d(exp.prepd_mean)[0]); amp_std = float(np.atleast_1d(exp.prepd_std)[0])
    assert list(exp.cfg.data.amp_trafos) == ["log", "standardization"], exp.cfg.data.amp_trafos
    pred_logamp = pred_prepd.reshape(-1) * amp_std + amp_mean

    err2 = (pred_logamp - true_logamp) ** 2
    print(f"\n=== {label} : MSE(Δlog|M|^2) on held-out deep-IR uugg (N={len(err2)}) ===", flush=True)
    print(f"  OVERALL          MSE = {err2.mean():.4e}   RMSE = {np.sqrt(err2.mean()):.4f}", flush=True)
    print(f"  {'y_min decade':16s} {'N':>7s} {'MSE':>12s} {'RMSE':>8s}", flush=True)
    for lo, hi in DECADES:
        m = (y_min >= lo) & (y_min < hi)
        if m.sum() == 0:
            continue
        print(f"  [{lo:.0e},{hi:.0e})   {int(m.sum()):>7d} {err2[m].mean():>12.4e} {np.sqrt(err2[m].mean()):>8.4f}",
              flush=True)
    out = os.path.join(REPO, "analysis/divergences", f"heldout_eval_{label}.npz")
    np.savez(out, pred_logamp=pred_logamp, true_logamp=true_logamp, y_min=y_min,
             sigma=(sigma.reshape(-1) if sigma is not None else np.array([])))
    print(f"  saved {out}", flush=True)
    return out


def _load_ckpt_state(run_dir, ckpt):
    import gzip, io
    p = os.path.join(run_dir, "models", ckpt)
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        p = p + ".gz"
    if p.endswith(".gz"):
        with gzip.open(p, "rb") as f:
            return torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)["model"]
    return torch.load(p, map_location="cpu", weights_only=False)["model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="runs/<exp>/<run_name>")
    ap.add_argument("--round0_dir", default=None,
                    help="data dir whose npy re-derives the frozen stats (default: from config data_path)")
    ap.add_argument("--ckpt", default="model_run0_best.pt")
    ap.add_argument("--heldout", default=DEFAULT_HELDOUT)
    ap.add_argument("--label", default=None, help="label for the printout (default: run basename)")
    args = ap.parse_args()
    label = args.label or os.path.basename(args.run_dir.rstrip("/"))

    cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.count_flops = False
        cfg.warm_start_idx = None; cfg.fine_tune.pretrained_path = None
        cfg.run_dir = os.path.join(REPO, "runs", "_heldout_eval_tmp")
        if args.round0_dir:
            cfg.data.data_path = args.round0_dir.rstrip("/") + "/"
    torch.set_default_dtype(torch.float32)

    from experiment import AmplitudeExperiment
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data(); exp._init_dataloader(); exp.init_model()
    exp.model.load_state_dict(_load_ckpt_state(args.run_dir, args.ckpt))
    exp.model.to(exp.device, dtype=exp.dtype).eval()

    score_and_save(exp, label, args.heldout)


if __name__ == "__main__":
    main()
