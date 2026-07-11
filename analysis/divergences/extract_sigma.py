#!/usr/bin/env python
"""Inspect the heteroscedastic sigma(x) of a HETEROSC LLoCa run across the ee->uug
IR (y_min). Forwards the common deep-IR test set through the (mu, sigma) model and
compares, per y_min decade and in Delta ln|M|^2 units:
  - sigma_ln   : the model's PREDICTED uncertainty (sigma * amp_std)
  - model error: the ACTUAL error |mu - truth|
  - float32 floor: the irreducible label-precision floor (measured separately)

Question: does sigma track the model error (epistemic-usable in this undertrained
regime), and does it sit above the float32 floor (so it is NOT yet noise-limited)?
Calibration ratio median(|err|/sigma) ~ 1 means sigma is a good error predictor. GPU.
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
sys.path.insert(0, WT)  # use the worktree's HETEROSC-wired code
from experiment import AmplitudeExperiment  # noqa
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
from extract_ir import ir_observables  # noqa
from extract_preds import load_finetuned_state  # noqa
from eval_heldout import com_normalize  # noqa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EDGES = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
CEN = np.sqrt(EDGES[:-1] * EDGES[1:])
# float32 momentum-precision floor on |M|^2 (median rel-err per decade), measured in
# analysis (recompute |M|^2 from float32-rounded vs float64 momenta). ~= Delta ln.
FLOAT32_FLOOR = np.array([0.00386, 0.00048, 0.00006, 1e-5, 1e-5, 1e-5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=os.path.join(WT, "runs/pretrain22_heldout_uug/ft_deep_antenna_het"))
    ap.add_argument("--ckpt", default="model_run0_best.pt")
    ap.add_argument("--heldout", default=os.path.join(REPO, "analysis/divergences/uug_deep_test.npz"))
    ap.add_argument("--out_base", default=os.path.join(WT, "analysis/divergences/figs/heterosc_sigma_map"))
    ap.add_argument("--batch_events", type=int, default=8192)
    args = ap.parse_args()

    ho = np.load(args.heldout)
    rows = np.asarray(ho["rows"], dtype=np.float64)
    pdg = np.asarray(ho["pdg"], dtype=int)
    P = (rows.shape[1] - 1) // 5
    raw_mom = rows[:, : P * 4].reshape(-1, P, 4)
    raw_amp = rows[:, -1].astype(np.float64)
    obs = ir_observables(raw_mom, pdg)
    y = obs["y_min"]
    true_logamp = np.log(raw_amp)

    cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.save_source = False; cfg.warm_start_idx = None
        cfg.ema = False; cfg.count_flops = False
        cfg.run_dir = os.path.join(REPO, "runs", "_het_eval_tmp")
        cfg.data.subsample = None
        cfg.fine_tune.pretrained_path = None
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data(); exp._init_dataloader(); exp.init_model()
    exp.model.load_state_dict(load_finetuned_state(os.path.join(args.run_dir, "models", args.ckpt))["model"])
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    mom_div = float(exp.mom_div)
    amp_mean = float(np.atleast_1d(exp.prepd_mean)[0]); amp_std = float(np.atleast_1d(exp.prepd_std)[0])
    out_shape = cfg.model.net.get("out_shape") or cfg.model.net.get("out_channels")

    parts = com_normalize(raw_mom, mom_div)
    from particle_ids import global_encode
    toks = global_encode(np.tile(pdg, (len(parts), 1)))
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
            mu = yp[..., :out_shape]; sig = yp[..., -out_shape:]
            mus.append(mu.detach().cpu().float().numpy().reshape(-1))
            sigmas.append(sig.detach().cpu().float().numpy().reshape(-1))
    mu_prepd = np.concatenate(mus); sigma_prepd = np.concatenate(sigmas)
    pred_logamp = mu_prepd * amp_std + amp_mean
    sigma_ln = sigma_prepd * amp_std                  # predicted uncertainty in Delta ln units
    err_ln = np.abs(pred_logamp - true_logamp)        # actual model error in Delta ln

    med_sig, med_err, pull = [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (y >= lo) & (y < hi)
        if m.sum() < 20:
            med_sig.append(np.nan); med_err.append(np.nan); pull.append(np.nan); continue
        med_sig.append(float(np.median(sigma_ln[m])))
        med_err.append(float(np.median(err_ln[m])))
        pull.append(float(np.median(err_ln[m] / np.clip(sigma_ln[m], 1e-12, None))))
    med_sig, med_err, pull = np.array(med_sig), np.array(med_err), np.array(pull)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(r"HETEROSC $e^+e^-\to u\bar u g$ (antenna pool, 4000 steps): predicted $\sigma$ vs actual "
                 r"error vs float32 floor", fontsize=12)
    axL.plot(CEN, med_sig, "o-", color="#C44E52", lw=2, ms=7, label=r"predicted $\sigma$ (median)")
    axL.plot(CEN, med_err, "s-", color="#4C72B0", lw=2, ms=6, label=r"actual error $|\mu-{\rm truth}|$ (median)")
    axL.plot(CEN, FLOAT32_FLOOR, "d--", color="#888888", lw=1.5, ms=5, label="float32 label floor")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel(r"$y_{\min}$  ($\leftarrow$ deeper IR)")
    axL.set_ylabel(r"$\Delta\ln|\mathcal{M}|^2$ scale")
    axL.grid(True, which="both", alpha=0.25); axL.legend(fontsize=9)
    axR.plot(CEN, pull, "o-", color="#55A868", lw=2, ms=7)
    axR.axhline(1.0, color="grey", ls=":", lw=1)
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel(r"$y_{\min}$  ($\leftarrow$ deeper IR)")
    axR.set_ylabel(r"calibration median$(|{\rm err}|/\sigma)$  (1 = calibrated)")
    axR.grid(True, which="both", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out_base), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_base}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_base}.png/.pdf")

    print(f"\n{'decade':>16} {'pred sigma':>11} {'actual err':>11} {'float32':>9} {'err/sigma':>10}")
    for i in range(len(CEN)):
        print(f"  [{EDGES[i]:.0e},{EDGES[i+1]:.0e}) {med_sig[i]:>11.4g} {med_err[i]:>11.4g} "
              f"{FLOAT32_FLOOR[i]:>9.4g} {pull[i]:>10.3f}")
    out = {"y_center": CEN.tolist(), "median_sigma_ln": med_sig.tolist(),
           "median_err_ln": med_err.tolist(), "float32_floor": FLOAT32_FLOOR.tolist(),
           "pull": pull.tolist()}
    with open(os.path.join(os.path.dirname(args.out_base), "heterosc_sigma_summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
