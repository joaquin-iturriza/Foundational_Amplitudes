"""Zero-shot loss of a pretraining on a transfer-study probe (docs/results.tex sec:ladder hand-off, "Distance from the
pretraining"): the pretrained weights, before any fine-tuning step, scored on the probe's validation split exactly as the
fine-tune's first validation would score them. Each argument is a fine-tune run dir (any trial of a tp3_<parent>fte cell):
its config fixes the probe, its pool and the probe's own target statistics (fine_tune.target_stats: own), and its
fine_tune.pretrained_path the parent. The value is val_loss_no_reg in the probe's standardized units (a constant
prediction of the mean scores ~1) and, times prepd_std^2, MSE of log|M|^2. Prints one ZERO_SHOT JSON line per run dir.
Needs a GPU.
    python tools/zero_shot_ft.py <ft_run_dir> [<ft_run_dir> ...]
"""
import json, os, sys, tempfile

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import siteconf  # noqa: E402
from rebuild_run import _build  # noqa: E402

for run_dir in sys.argv[1:]:
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.count_flops = False; cfg.ema = False
        cfg.warm_start_idx = None
        cfg.training.validate_frac = 0; cfg.training.validate_every_n_steps = 1
        cfg.run_dir = tempfile.mkdtemp(prefix="zeroshot_", dir=os.environ["SCRATCH"])
        if cfg.data.get("processes_file"):
            cfg.data.processes_file = siteconf._expand_str(str(cfg.data.processes_file))
        # the parent checkpoint, re-rooted to this site (a config written on another site keeps its paths)
        cfg.fine_tune.pretrained_path = siteconf._expand_str(str(cfg.fine_tune.pretrained_path))
    saved = json.load(open(os.path.join(run_dir, "data_stats.json")))
    if cfg.data.get("offshell_per_event", False) and saved.get("offshell_stats") is not None:
        with open_dict(cfg):
            cfg.data.offshell_stats = saved["offshell_stats"]
    try:
        exp = _build(cfg, None, None, None)          # init_model loads the parent's weights, as the fine-tune did
    except Exception as e:                            # a missing parent or pool on this site: listed, not dropped silently
        print("ZERO_SHOT " + json.dumps({"run_dir": run_dir, "error": repr(e)[:300]}), flush=True)
        continue
    for k in ("prepd_mean", "prepd_std"):
        got = [float(x) for x in getattr(exp, k)]
        assert np.allclose(got, saved[k], rtol=1e-6), f"{run_dir}: rebuilt {k} {got} != {saved[k]}"
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    exp.ema, exp.train_sampler = None, None
    exp.val_loss, exp.val_loss_no_reg, exp.val_mse = [], [], []
    exp.proc_val_losses, exp.proc_val_losses_no_reg = {}, {}
    exp._init_regularization()
    with torch.no_grad():
        exp._validate(0)
    print("ZERO_SHOT " + json.dumps({
        "run_dir": run_dir, "parent": str(cfg.fine_tune.pretrained_path), "val_loss": exp.val_loss_no_reg[-1],
        "prepd_std": saved["prepd_std"],
        "mse_logm2": exp.val_loss_no_reg[-1] * float(np.asarray(saved["prepd_std"]).ravel()[0]) ** 2}), flush=True)
