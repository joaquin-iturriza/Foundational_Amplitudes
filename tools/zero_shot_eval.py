#!/usr/bin/env python3
"""
zero_shot_eval.py — Evaluate a frozen pretrained amplitude model on a process it
was NEVER trained on (true zero-shot, no fine-tuning).

Protocol (fair across encodings; see CLAUDE.md A/B rules):
  * Load the checkpoint's own config (architecture + encoding flags) so inputs are
    built exactly as at training time. The physics property matrix is a fixed
    global table (dataset-independent); the PID tokenizer is reused from the
    pretrained run so PDG->index mapping is identical.
  * Point data at a single held-out process in `files` mode. Momentum and
    amplitude standardization are recomputed on the held-out events themselves
    (per-process unit scale), so the preprocessed val loss (val_loss_no_reg = MSE
    on standardized log-amplitudes) is comparable across processes and across the
    two encodings, and a predict-the-mean model scores MSE ~= 1.0.
  * PID encodability guard: a one-hot PID model can only represent particles seen
    in training. If the held-out process contains an unseen PDG id (e.g. top for a
    model never trained on top), it is structurally un-encodable -> we report
    {"encodable": false, "unseen_pdgs": [...]} instead of silently growing the
    vocab (which would break the input layer). This is itself a result.

Reported per split (val/test): preprocessed MSE (val_loss_no_reg), raw mean/median
|rel err| on the physical amplitude, and the trivial predict-the-mean MSE baseline.

Usage:
  python tools/zero_shot_eval.py \
      --ckpt-run-dir runs/pretrain25/trial_0009 \
      --dataset ee_uu_91-1000GeV_amplitudes \
      --out analysis/zero_shot/phys_ee_uu.json [--subsample 30000]
"""
import argparse, json, os, sys
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from experiment import AmplitudeExperiment
from misc import get_device


def _pick_weights(run_dir, run_idx=0):
    mdir = os.path.join(run_dir, "models")
    for fname in (f"model_run{run_idx}_best.pt", f"model_run{run_idx}_best.pt.gz",
                  f"model_run{run_idx}.pt", f"model_run{run_idx}.pt.gz"):
        p = os.path.join(mdir, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no checkpoint under {mdir}")


def _held_out_pdgs(npy_path):
    a = np.load(npy_path, mmap_mode="r")
    n = (a.shape[1] - 1) // 5
    pdg = np.asarray(a[:, 4 * n:5 * n]).astype(int)
    return sorted(int(x) for x in np.unique(pdg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-run-dir", required=True,
                    help="run dir with config.yaml, models/, particle_tokenizer.json")
    ap.add_argument("--dataset", required=True, help="held-out .npy basename (no .npy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-idx", type=int, default=0)
    ap.add_argument("--subsample", type=int, default=30000)
    ap.add_argument("--data-path", default=os.path.join(_project_dir, "data"))
    args = ap.parse_args()

    run_dir = os.path.abspath(args.ckpt_run_dir)
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    use_pids = bool(cfg.data.get("use_PIDs", False))
    weights = _pick_weights(run_dir, args.run_idx)

    npy_path = os.path.join(args.data_path, args.dataset + ".npy")
    held_pdgs = _held_out_pdgs(npy_path)

    result = {
        "ckpt_run_dir": run_dir, "weights": weights, "dataset": args.dataset,
        "use_PIDs": use_pids, "held_out_pdgs": held_pdgs, "subsample": args.subsample,
    }

    # --- PID encodability guard --------------------------------------------
    if use_pids:
        from particle_ids import ParticleTokenizer
        tok_path = os.path.join(run_dir, "particle_tokenizer.json")
        tok = ParticleTokenizer.load(tok_path)
        known = set(tok._pdg_to_idx.keys())
        unseen = [p for p in held_pdgs if p not in known]
        result["train_vocab_pdgs"] = sorted(int(x) for x in known)
        if unseen:
            result["encodable"] = False
            result["unseen_pdgs"] = unseen
            result["note"] = ("PID one-hot model cannot encode this process: "
                              f"PDG ids {unseen} were never seen in training.")
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            json.dump(result, open(args.out, "w"), indent=2)
            print(json.dumps(result, indent=2))
            return
    result["encodable"] = True

    # --- build eval-only experiment ----------------------------------------
    with open_dict(cfg):
        cfg.data.source = "files"
        cfg.data.dataset = [args.dataset]
        cfg.data.data_path = args.data_path + "/"
        cfg.data.amp_orders = None          # auto-derive (LO) from name
        cfg.data.subsample = args.subsample
        cfg.data.processes = None
        cfg.data.processes_file = None
        cfg.data.require_cache = False
        cfg.warm_start_idx = None
        cfg.train = False
        cfg.evaluate = True
        cfg.plot = False
        cfg.save = False
        cfg.run_dir = run_dir               # so PID tokenizer is reloaded from here
        if "fine_tune" not in cfg or cfg.fine_tune is None:
            cfg.fine_tune = {}
        cfg.fine_tune.pretrained_path = weights   # loads weights into the fresh model

    exp = AmplitudeExperiment(cfg)
    exp.warm_start = False
    exp.device = get_device()
    exp.dtype = (
        torch.bfloat16 if cfg.training.float16 and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        else torch.float16 if cfg.training.float16 else torch.float32
    )
    exp.ema = None
    torch.backends.cuda.enable_flash_sdp(cfg.training.enable_flash_sdp)
    torch.backends.cuda.enable_math_sdp(cfg.training.enable_math_sdp)
    torch.backends.cuda.enable_mem_efficient_sdp(cfg.training.enable_mem_efficient_sdp)

    exp.init_physics()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    exp._init_regularization()
    exp.init_model()      # instantiates arch + MuP shapes, then loads pretrained weights
    exp.evaluate()

    key = "combined" if len(cfg.data.dataset) > 1 else cfg.data.dataset[0]
    for split, attr in [("val", "results_val"), ("test", "results_test")]:
        r = getattr(exp, attr).get(key, {})
        pre = r.get("preprocessed", {})
        raw = r.get("raw", {})
        t, p = raw.get("truth"), raw.get("prediction")
        rel_med = (float(np.median(np.abs(t - p) / np.abs(t)))
                   if t is not None and p is not None else None)
        result[split] = {
            "mse_prepd": float(pre["mse"]),     # = val_loss_no_reg (per-process std space)
            "l1_prepd": float(pre["l1"]),
            "rel_err_mean": float(raw["l1_rel"]),   # Mean |rel err| on physical amplitude
            "rel_err_median": rel_med,
            "raw_mse": float(raw["mse"]),
            "n_events": int(t.shape[0]) if t is not None else None,
        }
    # trivial predict-the-mean baseline in standardized space ~ Var(target) ~ 1.0
    result["trivial_mse_prepd"] = 1.0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps({k: result[k] for k in result if k != "train_vocab_pdgs"}, indent=2))


if __name__ == "__main__":
    main()
