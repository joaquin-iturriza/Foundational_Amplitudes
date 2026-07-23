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


def score_and_save(exp, label, heldout_path=DEFAULT_HELDOUT, mc_samples=1):
    """Score a LIVE exp (frozen stats, model already loaded + in eval mode) on the FIXED held-out deep-IR
    uugg set, on raw log|M|^2, binned by y_min decade. Reports overall + per-decade MSE and saves
    heldout_eval_{label}.npz (row-aligned pred/true/y_min/sigma for downstream plots). This is the SHARED
    core used by both the standalone eval (which rebuilds exp from config.yaml + a checkpoint) and the
    driver's fold-in tail (which passes its already-trained, best-checkpoint-reloaded exp) -- so no
    separate eval job / queue wait is needed for a fresh run."""
    from dataset import AmplitudeDataset, build_flat_arrays, collate_variable_length
    d = np.load(heldout_path)
    rows = d["rows"]; y_min = d["y_min"]
    sqrt_s = d["sqrt_s"] if "sqrt_s" in d.files else None
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
    # A Bayesian (BBB) model must be evaluated THROUGH its posterior, not collapsed to a single
    # mean-weights forward: the point estimate is the PREDICTIVE MEAN E_q[f] ~ (1/K) sum_k f(x,w_k)
    # (marginalised over the weight posterior; != f(x,mu) for a nonlinear net), and the predictive
    # SPREAD is a first-class output we report (calibration), not something to hide. mc_samples>1 with
    # a variational net does exactly this; for a deterministic net it is a no-op (one forward).
    import bbb as BBB
    is_bayes = mc_samples > 1 and len(BBB.collect_variational(exp.model)) > 0
    pred_std_prepd = None
    with torch.no_grad():
        if is_bayes:
            BBB.set_sample_in_eval(exp.model, True)          # sample the weight posterior each pass
            samples = []
            for _ in range(mc_samples):
                p, _t, _s = exp._collect_predictions(loader)
                samples.append(np.asarray(p, dtype=np.float64).reshape(-1))
            BBB.set_sample_in_eval(exp.model, False)
            samples = np.stack(samples, axis=0)              # (K, N) preprocessed
            pred_prepd = samples.mean(0)                     # predictive mean (the point estimate)
            pred_std_prepd = samples.std(0)                  # epistemic predictive std
            sigma = None
        else:
            pred_prepd, _truth_prepd, sigma = exp._collect_predictions(loader)
    if was_training:
        exp.model.train()

    # amp target is [log, standardization]; UNDO ONLY the standardization to stay in log space.
    amp_mean = float(np.atleast_1d(exp.prepd_mean)[0]); amp_std = float(np.atleast_1d(exp.prepd_std)[0])
    assert list(exp.cfg.data.amp_trafos) == ["log", "standardization"], exp.cfg.data.amp_trafos
    pred_logamp = pred_prepd.reshape(-1) * amp_std + amp_mean
    pred_std_logamp = (pred_std_prepd * amp_std) if pred_std_prepd is not None else None

    err2 = (pred_logamp - true_logamp) ** 2
    print(f"\n=== {label} : MSE(Δlog|M|^2) on held-out deep-IR {L.PROCESS} (N={len(err2)}) ===", flush=True)
    print(f"  OVERALL          MSE = {err2.mean():.4e}   RMSE = {np.sqrt(err2.mean()):.4f}", flush=True)
    print(f"  {'y_min decade':16s} {'N':>7s} {'MSE':>12s} {'RMSE':>8s}", flush=True)
    for lo, hi in DECADES:
        m = (y_min >= lo) & (y_min < hi)
        if m.sum() == 0:
            continue
        print(f"  [{lo:.0e},{hi:.0e})   {int(m.sum()):>7d} {err2[m].mean():>12.4e} {np.sqrt(err2[m].mean()):>8.4f}",
              flush=True)
    # --- multi-scale (uug): the s-channel gamma*/Z resonance is a SECOND competing extreme at
    #     sqrt(s) ~ M_Z. Report the sqrt(s) regions (Z-peak / shoulders / continuum) and the deep-IR x
    #     on/off-peak split, so we can see whether one sigma^gamma knob budgets across BOTH extremes or
    #     robs the resonance to pay the IR (the open question this experiment answers). ---
    if sqrt_s is not None:
        MZ = 91.1876
        SREG = [("Z-peak |s-Mz|<3", np.abs(sqrt_s - MZ) < 3.0),
                ("shoulder 3-15",  (np.abs(sqrt_s - MZ) >= 3.0) & (np.abs(sqrt_s - MZ) < 15.0)),
                ("continuum >15",   np.abs(sqrt_s - MZ) >= 15.0)]
        print(f"  {'sqrt(s) region':16s} {'N':>7s} {'MSE':>12s} {'RMSE':>8s}", flush=True)
        for name, m in SREG:
            if m.sum() == 0:
                continue
            print(f"  {name:16s} {int(m.sum()):>7d} {err2[m].mean():>12.4e} {np.sqrt(err2[m].mean()):>8.4f}",
                  flush=True)
        deep = y_min < 1e-3
        for name, m in [("deep-IR & Z-peak", deep & (np.abs(sqrt_s - MZ) < 3.0)),
                        ("deep-IR & contin", deep & (np.abs(sqrt_s - MZ) >= 15.0))]:
            if m.sum() == 0:
                continue
            print(f"  {name:16s} {int(m.sum()):>7d} {err2[m].mean():>12.4e} {np.sqrt(err2[m].mean()):>8.4f}",
                  flush=True)
    # --- Bayesian predictive uncertainty: report calibration of the epistemic predictive std against
    #     the actual error (the whole point of a posterior). z2 = err^2 / sigma_pred^2 should average ~1
    #     if calibrated; a Gaussian NLL scores the full predictive distribution. Epistemic-only sigma
    #     (no aleatoric term) will tend to under-cover (z2>1) where the target has irreducible spread --
    #     itself informative. Also report how well sigma_pred RANKS the error (Spearman), since ranking
    #     is what the L2 keep-rule actually uses. ---
    if pred_std_logamp is not None:
        eps = 1e-12
        s2 = pred_std_logamp.reshape(-1) ** 2 + eps
        z2 = err2 / s2
        nll = 0.5 * (np.log(2 * np.pi * s2) + z2)
        # Spearman rank corr (sigma_pred vs |error|) without scipy: corr of the ranks.
        def _rank(a): return np.argsort(np.argsort(a))
        rp, re = _rank(pred_std_logamp.reshape(-1)), _rank(np.abs(pred_logamp - true_logamp))
        rho = float(np.corrcoef(rp, re)[0, 1])
        print(f"  --- Bayesian predictive uncertainty (K={mc_samples} posterior samples) ---", flush=True)
        print(f"  sigma_pred  mean={pred_std_logamp.mean():.4e}  median={np.median(pred_std_logamp):.4e}"
              f"  (log|M|^2 units)", flush=True)
        print(f"  calibration <z^2>={z2.mean():.3f} (want ~1; >1 = under-confident/under-covers)"
              f"   Gaussian NLL={nll.mean():.4f}", flush=True)
        print(f"  sigma_pred vs |err| Spearman rho={rho:.3f} (ranking power the L2 keep-rule uses)",
              flush=True)
    out = os.path.join(REPO, "analysis/divergences", f"heldout_eval_{label}.npz")
    np.savez(out, pred_logamp=pred_logamp, true_logamp=true_logamp, y_min=y_min,
             sqrt_s=(sqrt_s if sqrt_s is not None else np.array([])),
             sigma=(sigma.reshape(-1) if sigma is not None else np.array([])),
             pred_std=(pred_std_logamp.reshape(-1) if pred_std_logamp is not None else np.array([])))
    print(f"  saved {out}", flush=True)
    # Return the objective(s) so a sweep can read them without re-parsing stdout.
    #   overall_mse : plain event mean. BULK-DOMINATED -- the held-out set is deep-IR-heavy by
    #                 construction in some processes and bulk-heavy in others, so this is not a
    #                 like-for-like objective across arms.
    #   deep_mse    : y_min<1e-3 only. Ignores what concentration COSTS in the bulk, so optimising it
    #                 just drives the keep rule harder and harder -- the wrong objective for any
    #                 question about the concentration trade-off.
    #   logflat_mse : mean of the per-decade MSEs, every decade counting once however many events it
    #                 holds. This is the honest scoring rule for the trade-off (see the Q2 metric
    #                 correction in results.tex) and the right default sweep objective.
    deep_mse = float(err2[y_min < 1e-3].mean()) if (y_min < 1e-3).any() else float(err2.mean())
    per_decade = [err2[(y_min >= lo) & (y_min < hi)].mean()
                  for lo, hi in DECADES if ((y_min >= lo) & (y_min < hi)).sum() > 0]
    logflat_mse = float(np.mean(per_decade)) if per_decade else float(err2.mean())
    print(f"  LOG-FLAT (mean of {len(per_decade)} per-decade MSEs) = {logflat_mse:.4e}", flush=True)
    return {"npz": out, "overall_mse": float(err2.mean()), "deep_mse": deep_mse,
            "logflat_mse": logflat_mse}


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
    ap.add_argument("--process", default="uugg", choices=list(L.PROCESSES),
                    help="must match the run's process (sets NP/PDG for the held-out momenta)")
    ap.add_argument("--run_dir", required=True, help="runs/<exp>/<run_name>")
    ap.add_argument("--round0_dir", default=None,
                    help="data dir whose npy re-derives the frozen stats (default: from config data_path)")
    ap.add_argument("--ckpt", default="model_run0_best.pt")
    ap.add_argument("--heldout", default=DEFAULT_HELDOUT)
    ap.add_argument("--label", default=None, help="label for the printout (default: run basename)")
    ap.add_argument("--mc_samples", type=int, default=1,
                    help="bbb: posterior samples for the predictive mean + calibration (auto-set to 16 "
                         "if a variational checkpoint is detected and this is left at 1)")
    args = ap.parse_args()
    L.set_process(args.process)
    if args.heldout == DEFAULT_HELDOUT:
        args.heldout = os.path.join(REPO, f"analysis/divergences/heldout_{args.process}_deepIR.npz")
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
    state = _load_ckpt_state(args.run_dir, args.ckpt)
    # bbb checkpoints carry '*_rho' posterior-std params: variationalize the freshly-built MSE net so
    # the shapes match, then load. Auto-enable the predictive-mean eval (K>1) so a Bayesian run is not
    # silently scored as a single deterministic forward.
    mc = args.mc_samples
    if any(k.endswith("_rho") for k in state.keys()):
        import bbb as BBB
        BBB.variationalize(exp.model)
        if mc == 1:
            mc = 16
        print(f"[eval] variational checkpoint detected -> predictive-mean eval with K={mc} samples",
              flush=True)
    exp.model.load_state_dict(state)
    exp.model.to(exp.device, dtype=exp.dtype).eval()

    score_and_save(exp, label, args.heldout, mc_samples=mc)


if __name__ == "__main__":
    main()
