#!/usr/bin/env python
"""Extract true vs predicted log|M|^2 + IR observables for the 2->3 / 2->4 gluon
channels of a joint pretrain (ee->uug, ee->uugg), which carry genuine SOFT and
COLLINEAR singularities.

IR observables (all Lorentz invariants, built from raw momenta):
  s      = (p_e- + p_e+)^2
  x_g    = 2 (p_g . Q)/s        gluon energy fraction  (soft divergence at x_g->0)
  x_gmin = min over gluons      (softest gluon)
  y_ij   = (p_i+p_j)^2 / s      for gluon-involving colored pairs
  y_min  = min y_ij             master IR resolution var (->0 soft AND collinear)
  (ee->uug only) x_q, x_qbar = 2(p_q.Q)/s, 2(p_qbar.Q)/s   Dalitz energy fractions

Same faithful machinery as extract_pretrain.py: loads all pretrain datasets together
(global amp stats + dataset-0 momentum norm), forwards the requested processes, and
recovers raw momenta by inverting the deterministic default_rng(42) event shuffle.
A per-event preprocess(raw_amp)==stored-truth check guards alignment/stats. GPU only.
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
from extract_preds import load_finetuned_state  # noqa


def _dot(a, b):
    """Minkowski dot (+,-,-,-) over last axis of (...,4) arrays."""
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(axis=-1)


def ir_observables(raw_mom, pdg):
    """raw_mom: (N,P,4) physical four-momenta; pdg: (P,) integer PDG ids.
    Returns dict of IR observables (see module docstring)."""
    N, P, _ = raw_mom.shape
    Q = raw_mom[:, 0, :] + raw_mom[:, 1, :]           # e- + e+
    s = _dot(Q, Q)
    sqrt_s = np.sqrt(np.clip(s, 0.0, None))
    apdg = np.abs(pdg)
    glu = np.where(apdg == 21)[0]
    quark = np.where((apdg >= 1) & (apdg <= 6))[0]
    colored = np.sort(np.concatenate([glu, quark]))

    # gluon energy fractions
    xg = np.stack([2.0 * _dot(raw_mom[:, g, :], Q) / s for g in glu], axis=1)  # (N,ng)
    x_gmin = xg.min(axis=1)

    # y_ij over colored pairs that involve at least one gluon (the IR-singular ones)
    gset = set(glu.tolist())
    pairs = [(colored[a], colored[b])
             for a in range(len(colored)) for b in range(a + 1, len(colored))
             if (colored[a] in gset or colored[b] in gset)]
    yv = np.stack([_dot(raw_mom[:, i, :] + raw_mom[:, j, :],
                        raw_mom[:, i, :] + raw_mom[:, j, :]) / s
                   for i, j in pairs], axis=1)                                  # (N,npair)
    y_min = np.clip(yv, 1e-12, None).min(axis=1)

    out = {"sqrt_s": sqrt_s, "x_gmin": x_gmin, "y_min": y_min}
    if len(glu) == 1 and len(quark) == 2:   # ee->q qbar g : Dalitz
        q, qb = quark[0], quark[1]
        out["x_q"] = 2.0 * _dot(raw_mom[:, q, :], Q) / s
        out["x_qbar"] = 2.0 * _dot(raw_mom[:, qb, :], Q) / s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="model_run0_best.pt.gz")
    ap.add_argument("--processes", required=True)
    ap.add_argument("--subsample", type=int, default=2000000)
    ap.add_argument("--max_per_proc", type=int, default=300000)
    ap.add_argument("--batch_events", type=int, default=8192)
    ap.add_argument("--out_dir", default="analysis/divergences")
    ap.add_argument("--tag", default="pretrain8")
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    ckpt = os.path.join(args.run_dir, "models", args.ckpt)
    assert cfg.data.get("source", "files") == "files", "raw-alignment assumes source=files"
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.save_source = False; cfg.warm_start_idx = None
        cfg.run_dir = os.path.join(REPO, "runs", "_ir_infer_tmp")
        cfg.ema = False; cfg.count_flops = False
        cfg.data.subsample = args.subsample

    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data(); exp._init_dataloader(); exp.init_model()
    exp.model.load_state_dict(load_finetuned_state(ckpt)["model"])
    exp.model.to(exp.device, dtype=exp.dtype).eval()

    ds_names = list(cfg.data.dataset)
    prepd_mean = np.atleast_1d(np.asarray(exp.prepd_mean, dtype=np.float64))
    prepd_std = np.atleast_1d(np.asarray(exp.prepd_std, dtype=np.float64))
    per_dataset = len(prepd_mean) > 1
    trafos_pp = getattr(exp, "_amp_trafos_pp", None)
    print(f"ready. preprocessing={'PER-DATASET' if per_dataset else 'GLOBAL'}; "
          f"trafos={list(cfg.data.amp_trafos)}", flush=True)

    counts = np.array([min(args.subsample,
                           int(np.load(os.path.join(cfg.data.data_path, f"{n}.npy"),
                                       mmap_mode="r").shape[0])) for n in ds_names])
    block_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    perm = np.random.default_rng(seed=42).permutation(int(counts.sum()))
    assert int(counts.sum()) == int(exp.N_events)

    for name in [x.strip() for x in args.processes.split(",")]:
        if name not in ds_names:
            print(f"!! {name} not in dataset list; skipping", flush=True); continue
        pid = ds_names.index(name)
        k = pid if per_dataset else 0
        m = float(prepd_mean[k]); sd = float(prepd_std[k])
        tr = trafos_pp[pid] if trafos_pp is not None else list(cfg.data.amp_trafos)
        idx = np.where(exp.all_process_ids == pid)[0][: args.max_per_proc]
        true = np.asarray(exp.all_amplitudes[idx], dtype=np.float64).reshape(-1)

        raw_rows = perm[idx] - block_start[pid]
        assert raw_rows.min() >= 0 and raw_rows.max() < counts[pid]
        raw = np.asarray(np.load(os.path.join(cfg.data.data_path, f"{name}.npy"),
                                 mmap_mode="r")[raw_rows], dtype=np.float64)
        P = (raw.shape[1] - 1) // 5
        raw_mom = raw[:, :P * 4].reshape(-1, P, 4)
        pdg = np.load(os.path.join(cfg.data.data_path, f"{name}.npy"),
                      mmap_mode="r")[0, P * 4:P * 5].astype(int)
        raw_amp = raw[:, [-1]]
        obs = ir_observables(raw_mom, pdg)

        chk, _, _ = preprocess_amplitude(raw_amp, trafos=tr, mean=m, std=sd)
        align = float(np.max(np.abs(chk.reshape(-1) - true)))
        print(f"  {name}: P={P} pdg={list(pdg)} N={len(true)} align max|Δ|={align:.2e}"
              f"{'  !!FAIL' if align > 1e-4 else ''}", flush=True)

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
                yp = exp.model(particles.to(exp.device), tokens.to(exp.device),
                               mean=exp.mom_mean[0], std=exp.mom_std[0],
                               ptr=ptr.to(exp.device),
                               order_labels=order_labels.to(exp.device),
                               process_ids=process_ids.to(exp.device))
                preds.append(yp.detach().cpu().float().numpy().reshape(-1))
        pred = np.concatenate(preds)
        print(f"    {name}: MSE(prepd)={np.mean((pred-true)**2):.4e}", flush=True)

        save = dict(true_prepd=true, pred_prepd=pred,
                    true_logamp=true * sd + m, pred_logamp=pred * sd + m,
                    prepd_mean=np.array(m), prepd_std=np.array(sd))
        save.update({kk: vv for kk, vv in obs.items()})
        out = os.path.join(args.out_dir, f"preds_ir_{args.tag}_{name}.npz")
        np.savez_compressed(out, **save)
        print(f"    saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
