#!/usr/bin/env python
"""Evaluate each add-back fine-tune over the FIXED held-out NEAR (deep-IR) region of
ee->uug. The held-out set (analysis/divergences/uug_heldtest.npz, y_min<c, the 15th
percentile deep-IR tail) is NEVER in any fine-tune's training data, so f=0 is a pure
extrapolation into the soft/collinear divergence and f=1 the in-support baseline.

Method (why not reuse extract_ir.py directly): the held-out set was not loaded by the
fine-tune runs, so we forward it manually through each run's STORED preprocessing stats
(data_stats.json). init_data recomputes mom_div from a *random* Lorentz aug
(rand_lorentz(generator=None)) and is therefore non-deterministic -> we must use the
stored mom_div / mom_mean / mom_std / prepd_mean / prepd_std, not recomputed ones.
The LLoCa output is Lorentz-invariant, so a COM-boost-only (no random aug) input is the
canonical, deterministic frame. amp_trafos=[log, standardization] with |M|^2>0 => true
log|M|^2 = log(raw_amp); pred log|M|^2 = pred_prepd * prepd_std + prepd_mean.

Guard: recomputed ir_observables(y_min) on the loaded momenta must match the stored
y_min from the npz (validates momenta reshape/alignment). GPU only (xformers).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
from experiment import AmplitudeExperiment  # noqa
from lloca.utils.polar_decomposition import restframe_boost  # noqa
sys.path.insert(0, os.path.join(REPO, "analysis/divergences"))
from extract_ir import ir_observables  # noqa
from extract_preds import load_finetuned_state  # noqa


def com_normalize(raw_mom, mom_div):
    """raw_mom: (N,P,4) physical momenta. On-shell the energies, boost each event to
    the e-e+ COM (restframe_boost), divide by the run's stored mom_div. No random
    Lorentz aug (LLoCa output is invariant) -> deterministic canonical frame.
    Returns (N,P,4) float32 preprocessed momenta."""
    p = torch.tensor(raw_mom, dtype=torch.float64)
    m2 = p[..., 0] ** 2 - (p[..., 1:] ** 2).sum(dim=-1)
    p[..., 0] = torch.sqrt((p[..., 1:] ** 2).sum(dim=-1) + m2.clamp(min=0))
    lab = p[..., :2, :].sum(dim=-2)
    to_com = restframe_boost(lab)
    p = torch.einsum("...ij,...kj->...ki", to_com, p)
    return (p / mom_div).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default=os.path.join(REPO, "runs/pretrain22_heldout_uug"))
    ap.add_argument("--tags", default="000,005,015,050,100")
    ap.add_argument("--ckpt", default="model_run0_best.pt")
    ap.add_argument("--heldout", default=os.path.join(REPO, "analysis/divergences/uug_heldtest.npz"))
    ap.add_argument("--out_dir", default=os.path.join(REPO, "analysis/divergences"))
    ap.add_argument("--batch_events", type=int, default=8192)
    args = ap.parse_args()

    ho = np.load(args.heldout)
    rows = np.asarray(ho["rows"], dtype=np.float64)
    y_stored = np.asarray(ho["y_min"], dtype=np.float64)
    pdg = np.asarray(ho["pdg"], dtype=int)
    cut = float(ho["cut"])
    P = (rows.shape[1] - 1) // 5
    raw_mom = rows[:, : P * 4].reshape(-1, P, 4)
    raw_amp = rows[:, -1].astype(np.float64)
    obs = ir_observables(raw_mom, pdg)
    align_y = float(np.max(np.abs(obs["y_min"] - y_stored)))
    true_logamp = np.log(raw_amp)
    print(f"held-out: N={len(rows)} P={P} pdg={list(pdg)} cut(y_min)={cut:.3e} "
          f"y_min guard max|Δ|={align_y:.2e}{'  !!FAIL' if align_y > 1e-6 else ''}", flush=True)

    summary = []
    for tag in [t.strip() for t in args.tags.split(",")]:
        run_dir = os.path.join(args.runs_root, f"ft_f{tag}")
        cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
        stats = json.load(open(os.path.join(run_dir, "data_stats.json")))
        mom_div = float(stats["mom_div"])
        mom_mean = float(stats["mom_mean"])
        mom_std = float(stats["mom_std"])
        pm = np.atleast_1d(np.asarray(stats["prepd_mean"], dtype=np.float64))
        ps = np.atleast_1d(np.asarray(stats["prepd_std"], dtype=np.float64))
        amp_mean, amp_std = float(pm[0]), float(ps[0])

        # Wire the model on a tiny subsample (fast), then override stats with stored.
        with open_dict(cfg):
            cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
            cfg.use_mlflow = False; cfg.save_source = False; cfg.warm_start_idx = None
            cfg.ema = False; cfg.count_flops = False
            cfg.run_dir = os.path.join(REPO, "runs", "_ho_eval_tmp")
            cfg.data.subsample = 2000
            cfg.fine_tune.pretrained_path = None   # don't reload base; we load ft ckpt below
        exp = AmplitudeExperiment(cfg)
        exp._init(); exp.init_physics(); exp.init_geometric_algebra()
        exp.init_data(); exp._init_dataloader(); exp.init_model()
        exp.model.load_state_dict(load_finetuned_state(os.path.join(run_dir, "models", args.ckpt))["model"])
        exp.model.to(exp.device, dtype=exp.dtype).eval()
        exp.mom_mean = [mom_mean]; exp.mom_std = [mom_std]

        parts = com_normalize(raw_mom, mom_div)                       # (N,P,4)
        from particle_ids import global_encode
        toks = global_encode(np.tile(pdg, (len(parts), 1)))           # (N,P) int property-table indices
        order_row = np.array(exp._order_row(0, exp._resolve_amp_orders(list(cfg.data.dataset))),
                             dtype=np.float32)

        preds = []
        with torch.no_grad():
            for s in range(0, len(parts), args.batch_events):
                e = min(s + args.batch_events, len(parts))
                nb = e - s
                pt = torch.tensor(parts[s:e], dtype=exp.dtype, device=exp.device).reshape(-1, 4)
                tk = torch.tensor(toks[s:e], dtype=torch.long, device=exp.device).reshape(nb * P)
                ptr = torch.arange(0, nb * P + 1, P, dtype=torch.long, device=exp.device)
                ol = torch.tensor(np.tile(order_row, (nb, 1)), dtype=exp.dtype, device=exp.device)
                pid_t = torch.zeros(nb, dtype=torch.long, device=exp.device)
                yp = exp.model(pt, tk, mean=exp.mom_mean[0], std=exp.mom_std[0],
                               ptr=ptr, order_labels=ol, process_ids=pid_t)
                preds.append(yp.detach().cpu().float().numpy().reshape(-1))
        pred_prepd = np.concatenate(preds)
        pred_logamp = pred_prepd * amp_std + amp_mean

        d = pred_logamp - true_logamp
        rms = float(np.sqrt(np.mean(d ** 2)))
        mae = float(np.mean(np.abs(d)))
        # deep-IR bins by y_min
        ybins = [(0, 1e-3), (1e-3, 3e-3), (3e-3, 1e-2), (1e-2, cut)]
        binrms = []
        for lo, hi in ybins:
            msk = (obs["y_min"] >= lo) & (obs["y_min"] < hi)
            binrms.append((lo, hi, int(msk.sum()),
                           float(np.sqrt(np.mean(d[msk] ** 2))) if msk.any() else float("nan")))
        print(f"  ft_f{tag}: RMS Δlog|M|^2={rms:.4f}  MAE={mae:.4f}  (N={len(d)})", flush=True)
        for lo, hi, n, r in binrms:
            print(f"      y_min[{lo:.0e},{hi:.0e}) n={n:6d} RMS={r:.4f}", flush=True)

        out = os.path.join(args.out_dir, f"heldout_eval_ft_f{tag}.npz")
        extra = {k: obs[k] for k in ("x_q", "x_qbar") if k in obs}   # Dalitz vars (uug)
        np.savez_compressed(out, true_logamp=true_logamp, pred_logamp=pred_logamp,
                            y_min=obs["y_min"], x_gmin=obs["x_gmin"], sqrt_s=obs["sqrt_s"],
                            amp_mean=amp_mean, amp_std=amp_std, cut=np.array(cut), **extra)
        summary.append(dict(tag=tag, f=int(tag) / 100.0, rms=rms, mae=mae,
                            binrms=[[lo, hi, n, r] for lo, hi, n, r in binrms]))
        print(f"    saved -> {out}", flush=True)

    with open(os.path.join(args.out_dir, "heldout_eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote heldout_eval_summary.json", flush=True)


if __name__ == "__main__":
    main()
