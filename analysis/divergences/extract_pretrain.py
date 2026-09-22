#!/usr/bin/env python
"""Extract true vs predicted log|M|^2 + phase-space coords for selected 2->2
processes as learned JOINTLY by a pretrained model (NOT fine-tuned).

Robust to the two preprocessing regimes:
  * amp preprocessing scope: GLOBAL (source=files -> one pooled mean/std) or
    PER-DATASET (source=recipes -> one mean/std/trafo per process). We detect it
    via len(prepd_mean)>1 and un-preprocess each process with ITS OWN stats.
  * momentum scaling: per-dataset (files) or global mom_div (recipes). This only
    affects the physical sqrt_s scale, which we sidestep by taking kinematics from
    the RAW momenta (Lorentz-exact), aligning via the deterministic default_rng(42)
    event shuffle.

A per-event check that preprocess(raw_amp, per-process stats) == the loader's
true_prepd guards against a scope/stat/alignment mismatch. GPU only.
Handles only 2->2 (4 particles). Multi-body finals are skipped.
"""
import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

REPO = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"
sys.path.insert(0, REPO)
from experiment import AmplitudeExperiment  # noqa
from dataset import AmplitudeDataset, collate_variable_length  # noqa
from preprocessing import preprocess_amplitude  # noqa
sys.path.insert(0, os.path.join(REPO, "analysis/divergences"))
from extract_preds import boost_to_com_and_angle, load_finetuned_state  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="model_run0_best.pt.gz")
    ap.add_argument("--processes", required=True,
                    help="comma-separated dataset names to extract (2->2 only)")
    ap.add_argument("--subsample", type=int, default=300000)
    ap.add_argument("--max_per_proc", type=int, default=300000)
    ap.add_argument("--batch_events", type=int, default=8192)
    ap.add_argument("--out_dir", default="analysis/divergences")
    ap.add_argument("--tag", default="pretrain8")
    ap.add_argument("--no_forward", action="store_true",
                    help="CPU validation: check raw-alignment + per-process stats "
                         "against stored true_prepd; skip model build/forward")
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    ckpt = os.path.join(args.run_dir, "models", args.ckpt)
    src = cfg.data.get("source", "files")
    assert src == "files", (
        f"raw-alignment kinematics assume source=files; this run is source={src}. "
        "For recipe runs, derive kinematics from stored momenta * mom_div instead.")
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.save_source = False; cfg.warm_start_idx = None
        cfg.run_dir = os.path.join(REPO, "runs", "_pretrain_infer_tmp")
        cfg.ema = False; cfg.count_flops = False
        cfg.data.subsample = args.subsample

    import time as _clk
    _t = _clk.time()
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data(); exp._init_dataloader()
    if not args.no_forward:
        exp.init_model()
        exp.model.load_state_dict(load_finetuned_state(ckpt)["model"])
        exp.model.to(exp.device, dtype=exp.dtype).eval()

    ds_names = list(cfg.data.dataset)
    prepd_mean = np.atleast_1d(np.asarray(exp.prepd_mean, dtype=np.float64))
    prepd_std = np.atleast_1d(np.asarray(exp.prepd_std, dtype=np.float64))
    per_dataset = len(prepd_mean) > 1
    trafos_pp = getattr(exp, "_amp_trafos_pp", None)
    print(f"[{_clk.time()-_t:.1f}s] ready. preprocessing="
          f"{'PER-DATASET' if per_dataset else 'GLOBAL'}; "
          f"n_amp_stats={len(prepd_mean)}; trafos={list(cfg.data.amp_trafos)}", flush=True)

    # per-dataset loaded counts (files loads raw[:subsample]) -> pre-shuffle block starts
    counts = []
    for n in ds_names:
        N = int(np.load(os.path.join(cfg.data.data_path, f"{n}.npy"),
                        mmap_mode="r").shape[0])
        counts.append(min(args.subsample, N))
    counts = np.array(counts)
    block_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    N_total = int(counts.sum())
    perm = np.random.default_rng(seed=42).permutation(N_total)   # matches init_data
    assert N_total == int(exp.N_events), (N_total, exp.N_events)

    def stats_for(pid):
        k = pid if per_dataset else 0
        tr = (trafos_pp[pid] if trafos_pp is not None else list(cfg.data.amp_trafos))
        return float(prepd_mean[k]), float(prepd_std[k]), tr

    for name in [x.strip() for x in args.processes.split(",")]:
        if name not in ds_names:
            print(f"!! {name} not in pretrain dataset list; skipping", flush=True)
            continue
        pid = ds_names.index(name)
        idx = np.where(exp.all_process_ids == pid)[0]
        P = int(exp.offsets[idx[0], 1] - exp.offsets[idx[0], 0])
        if P != 4:
            print(f"!! {name}: {P} particles (not 2->2); skipping", flush=True)
            continue
        idx = idx[: args.max_per_proc]
        m, s, tr = stats_for(pid)
        true = np.asarray(exp.all_amplitudes[idx], dtype=np.float64).reshape(-1)

        # raw rows for these (shuffled) events: pre-shuffle index = perm[idx],
        # minus this dataset's block start -> row into raw[:subsample]
        raw_rows = perm[idx] - block_start[pid]
        assert raw_rows.min() >= 0 and raw_rows.max() < counts[pid], "alignment off"
        raw = np.asarray(np.load(os.path.join(cfg.data.data_path, f"{name}.npy"),
                                 mmap_mode="r")[raw_rows], dtype=np.float64)
        raw_mom = raw[:, :16].reshape(-1, 4, 4)
        raw_amp = raw[:, [-1]]
        sqrt_s, cos_t = boost_to_com_and_angle(raw_mom)

        # ALIGNMENT + STATS CHECK (no model needed): raw amp preprocessed with THIS
        # process's stats must equal the stored preprocessed truth for these events.
        chk, _, _ = preprocess_amplitude(raw_amp, trafos=tr, mean=m, std=s)
        align_err = float(np.max(np.abs(chk.reshape(-1) - true)))
        flag = "  !! FAIL" if align_err > 1e-4 else ""
        print(f"  {name}: N={len(true)}  align/stat max|Δ|={align_err:.2e}{flag} "
              f"(mean={m:.3f} std={s:.3f} trafos={tr})", flush=True)

        if args.no_forward:
            continue

        ds = AmplitudeDataset(
            particles_flat=exp.particles_flat, offsets=exp.offsets[idx],
            amplitudes=exp.all_amplitudes[idx], tokens_flat=exp.tokens_flat,
            order_labels=exp.all_order_labels[idx],
            process_ids=exp.all_process_ids[idx], dtype=exp.dtype)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_events, shuffle=False, num_workers=2,
            collate_fn=collate_variable_length)
        preds = []
        with torch.no_grad():
            for data in loader:
                particles, y, tokens, order_labels, ptr, process_ids = data
                yp = exp.model(
                    particles.to(exp.device), tokens.to(exp.device),
                    mean=exp.mom_mean[0], std=exp.mom_std[0],
                    ptr=ptr.to(exp.device),
                    order_labels=order_labels.to(exp.device),
                    process_ids=process_ids.to(exp.device))
                preds.append(yp.detach().cpu().float().numpy().reshape(-1))
        pred = np.concatenate(preds)
        mse = float(np.mean((pred - true) ** 2))
        print(f"    {name}: MSE(prepd)={mse:.4e}", flush=True)

        out = os.path.join(args.out_dir, f"preds_{args.tag}_{name}.npz")
        np.savez_compressed(
            out, sqrt_s=sqrt_s, cos_theta=cos_t,
            true_prepd=true, pred_prepd=pred,
            true_logamp=true * s + m, pred_logamp=pred * s + m,
            split=np.zeros(len(pred), dtype=np.int8),
            prepd_mean=np.array(m), prepd_std=np.array(s))
        print(f"    saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
