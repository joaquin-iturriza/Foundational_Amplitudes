"""Per-event predictions of a run's best checkpoint on its own splits, for the Delta and pull plots (decision D42):
preds_<split>.npz (evaluation.save_predictions: standardized pred/truth, raw |M|^2 pred/truth, pool_row) written to
<out-dir>/<sweep>__<trial>/, never into the run directory. Built as tools/eval_best_val.py builds a run: EMA off and
the checkpoint's "model" weights (the EMA average where EMA was on; a fresh EMA would score random weights), the
fine-tune head kept, the rebuilt amplitude statistics asserted against the run's data_stats.json, and the validation
aggregate recomputed and printed next to the run's recorded one. Needs a GPU.
    python tools/eval_delta.py --out-dir $SCRATCH/zz_delta runs/tp3_scr_ee_ddbar_d8/trial_0010 [...]
Relative run directories are read under the site's sweep_dir.
"""
import argparse, json, os, sys, traceback

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import siteconf  # noqa: E402
from rebuild_run import _build, load_best_state  # noqa: E402


def one(run_dir, out_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.count_flops = False; cfg.ema = False
        cfg.warm_start_idx = None
        cfg.training.validate_frac = 0; cfg.training.validate_every_n_steps = 1
        if cfg.get("fine_tune") is not None:
            cfg.fine_tune.pretrained_path = None
            cfg.fine_tune.reset_output_head = False
        cfg.evaluation.save_predictions = True
        cfg.run_dir = out_dir
        if cfg.data.get("processes_file"):
            cfg.data.processes_file = siteconf._expand_str(str(cfg.data.processes_file))
    saved = json.load(open(os.path.join(run_dir, "data_stats.json")))
    if cfg.data.get("offshell_per_event", False) and saved.get("offshell_stats") is not None:
        with open_dict(cfg):
            cfg.data.offshell_stats = saved["offshell_stats"]
    os.makedirs(out_dir, exist_ok=True)
    exp = _build(cfg, None, None, None)
    for k in ("prepd_mean", "prepd_std"):
        got = [float(x) for x in getattr(exp, k)]
        assert np.allclose(got, saved[k], rtol=1e-6), f"{run_dir}: rebuilt {k} {got} != {saved[k]}"
    exp.model.load_state_dict(load_best_state(run_dir))
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    exp.ema, exp.train_sampler = None, None
    exp.val_loss, exp.val_loss_no_reg, exp.val_mse = [], [], []
    exp.proc_val_losses, exp.proc_val_losses_no_reg = {}, {}
    exp._init_regularization()
    with torch.no_grad():
        exp._validate(0)
        exp.evaluate()
    rec = {"run_dir": run_dir, "out_dir": out_dir, "val_loss_rebuilt": exp.val_loss_no_reg[-1],
           "files": sorted(f for f in os.listdir(out_dir) if f.startswith("preds_"))}
    json.dump(rec, open(os.path.join(out_dir, "eval_delta.json"), "w"), indent=1)
    print("EVAL_DELTA " + json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("run_dirs", nargs="+")
    a = ap.parse_args()
    bad = []
    for rd in a.run_dirs:
        rd = rd if os.path.isabs(rd) else os.path.join(siteconf.SWEEP_DIR, rd)
        rd = os.path.normpath(rd)
        tag = "__".join(rd.split(os.sep)[-2:])
        try:
            one(rd, os.path.join(a.out_dir, tag))
        except Exception:
            traceback.print_exc()
            print("EVAL_DELTA_FAIL " + json.dumps({"run_dir": rd}), flush=True)
            bad.append(rd)
    print(f"EVAL_DELTA_DONE {len(a.run_dirs) - len(bad)}/{len(a.run_dirs)}", flush=True)


if __name__ == "__main__":
    main()
