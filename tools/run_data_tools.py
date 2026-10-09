"""Tools that rebuild a finished run's own data pipeline (its recipe, pools, per-dataset statistics, input flags) and
apply a model to it, for the continual-pretraining test (the user's call of 2026-10-09, site decision D21).

  fisher  the diagonal Fisher of the run's best checkpoint on the run's own training data, written to
          <run_dir>/models/ewc_fisher.pt: what fine_tune.ewc.fisher_path reads, so EWC protects the parent's processes
          (the built-in estimate uses the child's new data). Same estimator as fine_tune.EWC: squared gradients of
          the batch loss, averaged over --n-batches batches.
  eval    per-process validation loss of any checkpoint (--weights) on the run's data: how much a model fine-tuned
          away from this run forgot of its processes, read in the run's own preprocessed units (val_loss_no_reg).

    python tools/run_data_tools.py fisher --run-dir runs/tp3_finale/trial_0073 [--n-batches 64]
    python tools/run_data_tools.py eval --run-dir runs/tp3_finale/trial_0073 --weights runs/X/trial_Y/models/model_run0_best.pt.gz [...] --out f.json
Runs on a GPU node (a job), never on a login node.
"""
import argparse, json, os, sys

import torch
from omegaconf import OmegaConf, open_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiment import AmplitudeExperiment
from misc import get_device


def best_weights(run_dir):
    for f in ("model_run0_best.pt.gz", "model_run0_best.pt"):
        p = os.path.join(run_dir, "models", f)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no best checkpoint under {run_dir}/models")


def build(run_dir, weights):
    """The run's experiment with its data and loaders, the model loaded from `weights`; nothing trains or saves."""
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    with open_dict(cfg):
        # a run copied from another site names that site's paths: the recipe is re-rooted on this checkout
        pf = cfg.data.get("processes_file")
        if pf and not os.path.exists(pf) and "/recipes/" in pf:
            cfg.data.processes_file = os.path.join(ROOT, "recipes", pf.split("/recipes/", 1)[1])
        cfg.warm_start_idx = None
        cfg.train = False
        cfg.evaluate = True
        cfg.plot = False
        cfg.save = False
        cfg.run_dir = run_dir
        if "fine_tune" not in cfg or cfg.fine_tune is None:
            cfg.fine_tune = {}
        cfg.fine_tune.pretrained_path = weights
        cfg.fine_tune.target_stats = "own"      # the run's own pools give the run's own statistics
        if "ewc" in cfg.fine_tune and cfg.fine_tune.ewc is not None:
            cfg.fine_tune.ewc.enabled = False
    exp = AmplitudeExperiment(cfg)
    exp.warm_start = False
    exp.device = get_device()
    exp.dtype = torch.float32
    exp.ema = None
    exp.init_physics()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    exp._init_regularization()
    exp.init_model()
    return exp


def cmd_fisher(a):
    from fine_tune import EWC
    w = best_weights(a.run_dir)
    exp = build(a.run_dir, w)
    exp.model.to(exp.device)
    fisher = EWC._compute_fisher(None, exp.model, exp.train_loader, a.n_batches, exp.device,
                                 lambda batch: exp._batch_loss(batch)[0])
    out = os.path.join(a.run_dir, "models", "ewc_fisher.pt")
    torch.save({"fisher": {k: v.cpu() for k, v in fisher.items()}, "n_batches": a.n_batches, "weights": w}, out)
    tot = sum(float(v.sum()) for v in fisher.values())
    print("FISHER " + json.dumps({"out": out, "n_params": len(fisher), "sum": tot, "weights": w}))


def cmd_eval(a):
    """One or more checkpoints, each built afresh on the run's data; --out holds one record, or a list for several."""
    outs = []
    for w in a.weights:
        exp = build(a.run_dir, w)
        exp.evaluate()
        # per process (results_per_proc[name]["val"][name]), the pooled value as "combined": forgetting is per process
        res = {}
        for name, splits in getattr(exp, "results_per_proc", {}).items():
            pre = ((splits.get("val") or {}).get(name) or {}).get("preprocessed") or {}
            if "mse" in pre:
                res[name] = float(pre["mse"])
        for name, r in exp.results_val.items():
            pre = (r or {}).get("preprocessed") or {}
            if "mse" in pre:
                res[name] = float(pre["mse"])
        outs.append({"run_dir": a.run_dir, "weights": w, "val_mse_prepd": res})
        print("EVAL " + json.dumps(outs[-1]), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(outs[0] if len(outs) == 1 else outs, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fisher"); s.add_argument("--run-dir", required=True); s.add_argument("--n-batches", type=int, default=64)
    s.set_defaults(fn=cmd_fisher)
    s = sub.add_parser("eval"); s.add_argument("--run-dir", required=True); s.add_argument("--weights", required=True, nargs="+")
    s.add_argument("--out", required=True); s.set_defaults(fn=cmd_eval)
    a = ap.parse_args()
    a.run_dir = os.path.abspath(a.run_dir)
    a.fn(a)


if __name__ == "__main__":
    main()
