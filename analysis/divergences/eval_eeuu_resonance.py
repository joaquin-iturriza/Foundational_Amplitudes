#!/usr/bin/env python
"""Evaluate the L0 flat-log|M|^2 A/B on the ee->uu s-channel Z resonance.

Forward the COMMON held-out test pool (data_test_eeuu/, native RAMBO density, never in either
arm's training data) through each fine-tune's OWN preprocessing stats, de-standardize to true
log|M|^2, and bin the residual by sqrt(s) --- the divergence coordinate the resampler NEVER used
(it read only |M|^2). Reports per-sqrt(s)-bin MSE(Δlog|M|^2), plus a LOG-FLAT-PER-DECADE overall
(equal weight per sqrt(s) region --- the Thread-A/Q2 equal-footing metric, so the RAMBO-dominant
high-sqrt(s) bulk doesn't drown the resonance) alongside the naive event-weighted overall.

Reuses the forwarding path proven in eval_heldout.py (com_normalize, run-stats recomputation via
init_data on the run's own pool, load_finetuned_state). GPU only (xformers). ee->uu is 4-body:
rows = [16 mom | 4 pdg | amp], sqrt(s) = 2*E_beam = 2*rows[:,0].
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
sys.path.insert(0, os.path.join(REPO, "analysis/divergences"))
from eval_heldout import com_normalize  # noqa
from extract_preds import load_finetuned_state  # noqa

# sqrt(s) bins: dense across the Z peak (91-95), coarsening into the bulk.
SBINS = [(91, 93), (93, 95), (95, 100), (100, 110), (110, 150),
         (150, 300), (300, 600), (600, 1000)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default=os.path.join(REPO, "runs/eeuu_flatlogm"))
    ap.add_argument("--tags", default="raw,flatlogm")
    ap.add_argument("--run_prefix", default="ft_")
    ap.add_argument("--ckpt", default="model_run0_best.pt")
    ap.add_argument("--test", default=os.path.join(REPO, "data_test_eeuu/ee_uu_91-1000GeV_amplitudes.npy"))
    ap.add_argument("--out_dir", default=os.path.join(REPO, "analysis/divergences"))
    ap.add_argument("--out_prefix", default="eeuu_reson_")
    ap.add_argument("--summary", default="eeuu_reson_summary.json")
    ap.add_argument("--batch_events", type=int, default=8192)
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    # IN-PROCESS STATE LEAK: evaluating >1 tag in one process corrupts every tag after the
    # first (a module/MuP global survives across AmplitudeExperiment instances; a per-tag
    # run_dir does NOT fix it -> confirmed the leak is in-process, not file-based). Isolate by
    # running exactly ONE tag per subprocess, then merge the per-tag summaries. Each child hits
    # the `else` single-tag branch below with a fresh interpreter -> reproducible, uncorrupted.
    if len(tags) > 1:
        import json as _json
        import subprocess as _sp
        merged = []
        for t in tags:
            sub_summary = f".sub_{args.summary}_{t}.json"
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--runs_root", args.runs_root, "--tags", t, "--run_prefix", args.run_prefix,
                   "--ckpt", args.ckpt, "--test", args.test, "--out_dir", args.out_dir,
                   "--out_prefix", args.out_prefix, "--summary", sub_summary,
                   "--batch_events", str(args.batch_events)]
            _sp.run(cmd, check=True)
            with open(os.path.join(args.out_dir, sub_summary)) as f:
                merged.extend(_json.load(f))
            os.remove(os.path.join(args.out_dir, sub_summary))
        with open(os.path.join(args.out_dir, args.summary), "w") as f:
            _json.dump(merged, f, indent=1)
        print(f"wrote {args.summary} (merged {len(merged)} tags, subprocess-isolated)", flush=True)
        return

    rows = np.load(args.test).astype(np.float64)
    P = (rows.shape[1] - 1) // 5
    raw_mom = rows[:, : P * 4].reshape(-1, P, 4)
    pdg = rows[0, P * 4: P * 5].astype(int)          # fixed final state (e-,e+,u,ubar)
    raw_amp = rows[:, -1]
    true_logamp = np.log(raw_amp)
    sqrt_s = 2.0 * rows[:, 0]
    print(f"test: N={len(rows)} P={P} pdg={list(pdg)} sqrt(s) {sqrt_s.min():.1f}..{sqrt_s.max():.1f}",
          flush=True)

    from particle_ids import global_encode
    summary = []
    for tag in [t.strip() for t in args.tags.split(",")]:
        run_dir = os.path.join(args.runs_root, f"{args.run_prefix}{tag}")
        cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
        with open_dict(cfg):
            cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
            cfg.use_mlflow = False; cfg.save_source = False; cfg.warm_start_idx = None
            cfg.ema = False; cfg.count_flops = False
            # per-tag run_dir: a SHARED tmp dir leaks MuP base_shapes.bsh / data_stats.json
            # from tag N into tag N+1 -> only the first tag evaluates correctly.
            cfg.run_dir = os.path.join(REPO, "runs", f"_eeuu_eval_tmp_{tag}")
            cfg.data.subsample = None
            cfg.fine_tune.pretrained_path = None
        exp = AmplitudeExperiment(cfg)
        exp._init(); exp.init_physics(); exp.init_geometric_algebra()
        exp.init_data(); exp._init_dataloader(); exp.init_model()
        exp.model.load_state_dict(load_finetuned_state(os.path.join(run_dir, "models", args.ckpt))["model"])
        exp.model.to(exp.device, dtype=exp.dtype).eval()
        mom_div = float(exp.mom_div)
        amp_mean = float(np.atleast_1d(exp.prepd_mean)[0])
        amp_std = float(np.atleast_1d(exp.prepd_std)[0])

        parts = com_normalize(raw_mom, mom_div)
        toks = global_encode(np.tile(pdg, (len(parts), 1)))
        order_row = np.array(exp._order_row(0, exp._resolve_amp_orders(list(cfg.data.dataset))),
                             dtype=np.float32)
        preds = []
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
                preds.append(yp.detach().cpu().float().numpy().reshape(-1))
        pred_logamp = np.concatenate(preds) * amp_std + amp_mean
        d = pred_logamp - true_logamp

        binmse = []
        for lo, hi in SBINS:
            m = (sqrt_s >= lo) & (sqrt_s < hi)
            binmse.append((lo, hi, int(m.sum()), float(np.mean(d[m] ** 2)) if m.any() else float("nan")))
        overall = float(np.mean(d ** 2))
        logflat = float(np.nanmean([b[3] for b in binmse]))   # equal weight per sqrt(s) bin
        peak = binmse[0][3]
        print(f"  {tag}: overall(event-wtd) MSE={overall:.4g}  logflat(per-bin) MSE={logflat:.4g}  "
              f"Zpeak[91,93) MSE={peak:.4g}", flush=True)
        for lo, hi, n, r in binmse:
            print(f"      sqrt(s)[{lo:>4},{hi:>4}) n={n:6d} MSE={r:.4g}", flush=True)

        out = os.path.join(args.out_dir, f"{args.out_prefix}{tag}.npz")
        np.savez_compressed(out, true_logamp=true_logamp, pred_logamp=pred_logamp,
                            sqrt_s=sqrt_s, amp_mean=amp_mean, amp_std=amp_std)
        summary.append(dict(tag=tag, overall=overall, logflat=logflat, zpeak=peak,
                            binmse=[[lo, hi, n, r] for lo, hi, n, r in binmse]))
        print(f"    saved -> {out}", flush=True)

    with open(os.path.join(args.out_dir, args.summary), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {args.summary}", flush=True)


if __name__ == "__main__":
    main()
