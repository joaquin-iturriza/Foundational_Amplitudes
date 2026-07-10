#!/usr/bin/env python
"""Extract true vs predicted log-amplitudes together with per-event phase-space
coordinates (sqrt_s, cos_theta*) for a finetuned run, so we can study how the
model behaves across the phase space and, in particular, near the amplitude
divergences (soft/collinear/resonant/threshold regions).

Runs the *exact* run pipeline (same config, same preprocessing, same μP model),
loads the finetuned checkpoint, and does a forward pass over the train/val/test
eval loaders. sqrt_s and cos_theta* are Lorentz invariants, so they are recovered
directly from the (COM-boosted + randomly-Lorentz-augmented, unit-scaled) momenta
the model actually sees — no dependence on the random augmentation.

GPU only (xformers attention is CUDA-only). Submit via sbatch.
"""
import argparse
import gzip
import io
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

# repo root on path
REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)

from experiment import AmplitudeExperiment  # noqa: E402
from dataset import AmplitudeDataset, collate_variable_length  # noqa: E402


def boost_to_com_and_angle(P):
    """P: (nev, nparticles, 4) physical four-momenta (E, px, py, pz), metric (+,-,-,-).
    Returns sqrt_s (nev,) and cos_theta* (nev,), the scattering angle between the
    incoming e- (slot 0) and the outgoing fermion (slot 2) in the c.o.m. frame.
    Boosting both legs into the c.o.m. frame makes the angle between them
    frame-independent (invariant under the input Lorentz augmentation)."""
    P = np.asarray(P, dtype=np.float64)
    Q = P[:, 0, :] + P[:, 1, :]                    # total four-momentum of initial pair
    s = Q[:, 0] ** 2 - (Q[:, 1:] ** 2).sum(axis=1)
    sqrt_s = np.sqrt(np.clip(s, 0.0, None))

    beta = Q[:, 1:] / Q[:, [0]]                    # (nev,3) velocity of com frame in lab
    b2 = (beta ** 2).sum(axis=1)
    b2 = np.clip(b2, 0.0, 1.0 - 1e-15)
    gamma = 1.0 / np.sqrt(1.0 - b2)

    def boost(p):
        # boost four-vector p (nev,4) INTO the com rest frame (Q at rest)
        E = p[:, 0]
        vec = p[:, 1:]
        bp = (beta * vec).sum(axis=1)              # beta . p_vec
        Ep = gamma * (E - bp)
        # spatial part; guard b2->0
        coef = np.where(b2 > 1e-12, (gamma - 1.0) * bp / np.where(b2 > 0, b2, 1.0), 0.0)
        vecp = vec + (coef - gamma * E)[:, None] * beta
        return Ep, vecp

    _, e_minus = boost(P[:, 0, :])                 # incoming e-
    _, fermion = boost(P[:, 2, :])                 # outgoing fermion (slot 2)
    n1 = e_minus / (np.linalg.norm(e_minus, axis=1, keepdims=True) + 1e-30)
    n2 = fermion / (np.linalg.norm(fermion, axis=1, keepdims=True) + 1e-30)
    cos_theta = np.clip((n1 * n2).sum(axis=1), -1.0, 1.0)
    return sqrt_s, cos_theta


def load_finetuned_state(ckpt):
    if not ckpt.endswith(".gz") and not os.path.exists(ckpt) and os.path.exists(ckpt + ".gz"):
        ckpt = ckpt + ".gz"          # tolerate a checkpoint gzipped by cleanup after training
    if ckpt.endswith(".gz"):
        with gzip.open(ckpt, "rb") as f:
            buf = io.BytesIO(f.read())
        return torch.load(buf, map_location="cpu", weights_only=False)
    return torch.load(ckpt, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="finetune trial dir with config.yaml + models/")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--max_events", type=int, default=200000,
                    help="cap on number of events run through the model")
    ap.add_argument("--splits", default="test,val,train",
                    help="which splits to draw from, in fill order")
    ap.add_argument("--batch_events", type=int, default=8192,
                    help="events per forward batch (GPU; xformers isolates events)")
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    ckpt = os.path.join(args.run_dir, "models", "model_run0.pt.gz")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(args.run_dir, "models", "model_run0.pt")

    # --- reproduce the TRAINING-TIME order_labels (avoid resolver drift) ---
    # These runs (2026-06-17) predate the name-based amp_orders resolver
    # (commit accef58, 2026-06-26). The old code used cfg.data.amp_orders[proc_idx]
    # verbatim; the stale 8-entry [0,0] list (1 dataset) therefore fed order_labels
    # = [0,0]. Today's resolver instead derives [1,0] from "nlo_virt" in the name,
    # which is NOT what the model was trained with and degrades every prediction.
    # Slice the stale list to len(datasets) so the current resolver returns it as-is.
    _orders = cfg.data.get("amp_orders", None)
    if _orders is not None:
        _ds = list(cfg.data.dataset)
        if len(_orders) != len(_ds):
            with open_dict(cfg):
                cfg.data.amp_orders = [list(o) for o in list(_orders)[:len(_ds)]]
            print(f"amp_orders drift fix: using {list(cfg.data.amp_orders)} "
                  f"(training-time value), not the name-derived resolver output",
                  flush=True)

    # Inference-only overrides: no writes, no train, no plot, cold build (we load
    # the finetuned weights ourselves). Keep every data/model knob as the run had it
    # so preprocessing (mom_div, amp trafos, prepd mean/std) is reproduced exactly.
    with open_dict(cfg):
        cfg.train = False
        cfg.evaluate = False
        cfg.plot = False
        cfg.save = False
        cfg.use_mlflow = False
        cfg.save_source = False
        cfg.warm_start_idx = None
        cfg.run_dir = os.path.join(REPO, "runs", "_divergence_infer_tmp")
        cfg.ema = False
        cfg.count_flops = False

    import time as _clk
    _t = _clk.time()
    def _mark(msg):
        print(f"[{_clk.time()-_t:6.1f}s] {msg}", flush=True)

    exp = AmplitudeExperiment(cfg)
    exp._init(); _mark("_init done")
    exp.init_physics(); _mark("init_physics done")
    exp.init_geometric_algebra(); _mark("init_geometric_algebra done")
    exp.init_data(); _mark("init_data done")
    exp._init_dataloader(); _mark("_init_dataloader done")
    exp.init_model(); _mark("init_model done")       # builds μP model (+pretrained)
    state = load_finetuned_state(ckpt)
    exp.model.load_state_dict(state["model"]); _mark("finetuned weights loaded")
    exp.model.to(exp.device, dtype=exp.dtype)
    exp.model.eval()

    mom_div = float(exp.mom_div)
    prepd_mean = float(exp.prepd_mean[0])
    prepd_std = float(exp.prepd_std[0])
    amp_trafos = list(cfg.data.amp_trafos)
    print(f"mom_div={mom_div} prepd_mean={prepd_mean} prepd_std={prepd_std} trafos={amp_trafos}", flush=True)

    # --- reconstruct the positional train/val/test split (files data path) ---
    N = int(exp.N_events)
    ttv = [float(x) for x in cfg.data.train_test_val]
    n_train = int(N * ttv[0])
    if n_train % 2 != 0 and n_train > 1:
        n_train -= 1
    elif n_train == 1:
        n_train = 2
    val_ratio = ttv[2] / ttv[0]
    n_val = max(int(n_train * val_ratio), 2)
    if n_val % 2 != 0:
        n_val -= 1
    idx_by_split = {
        "train": np.arange(0, n_train),
        "val": np.arange(n_train, n_train + n_val),
        "test": np.arange(n_train + n_val, N),
    }
    split_code = {"train": 0, "val": 1, "test": 2}

    # fill order: draw from requested splits until max_events reached (events are
    # already physics-shuffled, so a prefix of a split is a random phase-space sample)
    want = [s for s in args.splits.split(",") if s in idx_by_split]
    chosen_idx, chosen_split = [], []
    budget = args.max_events
    for s in want:
        take = idx_by_split[s][: max(0, budget)]
        chosen_idx.append(take)
        chosen_split.append(np.full(len(take), split_code[s], dtype=np.int8))
        budget -= len(take)
        if budget <= 0:
            break
    indices = np.concatenate(chosen_idx)
    split_arr = np.concatenate(chosen_split)
    print(f"N_events={N}, running {len(indices)} events "
          f"(splits={want}, sizes={[len(x) for x in chosen_idx]})", flush=True)

    # batched loader; on GPU xformers builds a block-diagonal mask so each event
    # only attends within itself -> batching is exactly correct and fast.
    ds = AmplitudeDataset(
        particles_flat=exp.particles_flat,
        offsets=exp.offsets[indices],
        amplitudes=exp.all_amplitudes[indices],
        tokens_flat=exp.tokens_flat,
        order_labels=exp.all_order_labels[indices],
        process_ids=exp.all_process_ids[indices],
        dtype=exp.dtype,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_events, shuffle=False, num_workers=2,
        collate_fn=collate_variable_length)

    out = {k: [] for k in ["sqrt_s", "cos_theta", "true_prepd", "pred_prepd"]}
    import time as _tm
    t0 = _tm.time()
    with torch.no_grad():
        for data in loader:
            particles, y, tokens, order_labels, ptr, process_ids = data
            particles = particles.to(exp.device)
            tokens = tokens.to(exp.device)
            order_labels = order_labels.to(exp.device)
            ptr_d = ptr.to(exp.device)
            process_ids = process_ids.to(exp.device)
            y_pred = exp.model(
                particles, tokens,
                mean=exp.mom_mean[0], std=exp.mom_std[0],
                ptr=ptr_d, order_labels=order_labels,
                process_ids=process_ids,
            )
            out["pred_prepd"].append(y_pred.detach().cpu().float().numpy().reshape(-1))
            out["true_prepd"].append(y.detach().cpu().float().numpy().reshape(-1))

            # per-event kinematics from the (boosted, unit-scaled) momenta the model saw
            p = particles.detach().cpu().float().numpy() * mom_div     # (N_total, 4)
            ptr_np = ptr.numpy()
            counts = np.diff(ptr_np)
            P = int(counts[0])
            assert (counts == P).all(), "unexpected variable particle count"
            B = len(counts)
            sqrt_s, cos_theta = boost_to_com_and_angle(p.reshape(B, P, 4))
            out["sqrt_s"].append(sqrt_s)
            out["cos_theta"].append(cos_theta)
    print(f"forward over {len(indices)} events in {_tm.time()-t0:.2f}s", flush=True)
    out["split"] = [split_arr]

    res = {k: np.concatenate(v) for k, v in out.items() if v}
    res["prepd_mean"] = np.array(prepd_mean)
    res["prepd_std"] = np.array(prepd_std)
    res["amp_trafos"] = np.array(amp_trafos, dtype=object)
    res["mom_div"] = np.array(mom_div)
    # convenience: physical log-amplitude (= log|M|^2 when trafo is 'log')
    res["true_logamp"] = res["true_prepd"] * prepd_std + prepd_mean
    res["pred_logamp"] = res["pred_prepd"] * prepd_std + prepd_mean

    # sanity check: preprocessed-space MSE per split (test should match the run's
    # logged/JSON test_loss, confirming faithful weights + preprocessing)
    d_prepd = res["pred_prepd"] - res["true_prepd"]
    for s, code in (("train", 0), ("val", 1), ("test", 2)):
        m = res["split"] == code
        if m.any():
            print(f"  MSE(prepd) {s}: {np.mean(d_prepd[m]**2):.4e}  (N={int(m.sum())})",
                  flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **res)
    print(f"Saved {res['sqrt_s'].shape[0]} events -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
