"""Score a run's best checkpoint on its own validation split, as training did: val_loss_no_reg (the validation
aggregate without the regularizer) and each process's standardized loss. For runs that diverged before writing a
result: their best checkpoint is still their value (CLAUDE.md, reported values), and their logs print the
regularized loss to four decimals, which the L2 term swamps at large lambda. The checkpoint's "model" holds the
weights validation used (the EMA average where EMA was on), so the run is rebuilt with EMA off and those weights.
The rebuilt statistics are asserted against the run's data_stats.json. Prints one EVAL_BEST JSON line per run.
Needs a GPU.
    python tools/eval_best_val.py <run_dir> [<run_dir> ...]
"""
import json, os, sys, tempfile

import numpy as np
import torch
import siteconf
from omegaconf import OmegaConf, open_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from rebuild_run import _build, load_best_state  # noqa: E402

for run_dir in sys.argv[1:]:
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.count_flops = False; cfg.ema = False
        cfg.warm_start_idx = None
        cfg.training.validate_frac = 0; cfg.training.validate_every_n_steps = 1   # _validate records every call
        if cfg.get("fine_tune") is not None:
            cfg.fine_tune.pretrained_path = None
        cfg.run_dir = tempfile.mkdtemp(prefix="evalbest_", dir=os.environ["SCRATCH"])
        # a run copied from another site keeps that site's absolute paths (its recipe): re-root them here
        if cfg.data.get("processes_file"):
            cfg.data.processes_file = siteconf._expand_str(str(cfg.data.processes_file))
    saved = json.load(open(os.path.join(run_dir, "data_stats.json")))
    if cfg.data.get("offshell_per_event", False) and saved.get("offshell_stats") is not None:
        with open_dict(cfg):
            cfg.data.offshell_stats = saved["offshell_stats"]
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
    print("EVAL_BEST " + json.dumps({
        "run_dir": run_dir, "val_loss": exp.val_loss_no_reg[-1],
        "proc_val_losses_no_reg": {n: v[-1] for n, v in exp.proc_val_losses_no_reg.items()},
        "prepd_std": saved["prepd_std"]}), flush=True)
