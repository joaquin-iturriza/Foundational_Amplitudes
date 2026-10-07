"""Zero-shot loss of a pretraining on a transfer-study probe (docs/results.tex sec:ladder hand-off, "Distance from the
pretraining"): the pretrained weights, before any fine-tuning step, scored on the probe's validation split exactly as the
fine-tune's first validation would score them. Each argument is a fine-tune run dir (any trial of a tp3_<parent>fte cell):
its config fixes the probe, its pool and the probe's own target statistics (fine_tune.target_stats: own), and its
fine_tune.pretrained_path the parent. The value is val_loss_no_reg in the probe's standardized units (a constant
prediction of the mean scores ~1) and, times prepd_std^2, MSE of log|M|^2. The parent's outputs are in its own
standardized units, not the probe's, so that value mixes a units mismatch with what the parent knows; val_loss_affine
is the loss after the best affine map of the parent's output onto the probe's target, min_{a,b} E[(a + b h - z)^2]
= 1 - corr(h, z)^2, the share of the probe's variance the parent cannot explain up to an offset and a scale (a and b
fitted on the same validation split: two parameters on its 10^4-scale pool). Prints one ZERO_SHOT JSON line per run
dir. Needs a GPU.
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

# --parent <pretraining run dir>: score that pretraining instead of the run dir's own parent, on the run dir's probe
# (its config and pool); with a parent trained under data.shared_standardization the prediction is a real
# log|M|^2, mu + sigma h, scored against the probe's true log|M|^2 with no fit (mse_logm2_real)
args, parent = sys.argv[1:], None
if "--parent" in args:
    i = args.index("--parent"); parent = args[i + 1]; args = args[:i] + args[i + 2:]
for run_dir in args:
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
        if parent:
            cfg.fine_tune.pretrained_path = os.path.join(parent, "models", "model_run0_best.pt")
    saved = json.load(open(os.path.join(run_dir, "data_stats.json")))
    # the run's own off-shellness scale is pinned only for its own parent: with --parent the stats it recorded were merged
    # with a different parent's, so the fine-tune's merge (fine_tune.offshell_stats=parent) is redone with the new one
    if not parent and cfg.data.get("offshell_per_event", False) and saved.get("offshell_stats") is not None:
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
        hs, zs = [], []
        for data in exp.val_loader:
            y_pred, y, _, _, _ = exp._forward_lloca(data)
            hs.append(y_pred.reshape(-1).double().cpu()); zs.append(y.reshape(-1).double().cpu())
    h, z = torch.cat(hs).numpy(), torch.cat(zs).numpy()
    A = np.stack([np.ones_like(h), h], 1)
    coef = np.linalg.lstsq(A, z, rcond=None)[0]
    affine = float(np.mean((A @ coef - z) ** 2))
    real = None
    pstats = json.load(open(os.path.join(os.path.dirname(os.path.dirname(str(cfg.fine_tune.pretrained_path))),
                                         "data_stats.json")))
    if pstats.get("shared_standardization"):
        tr = (exp._amp_trafos_pp or [exp.cfg.data.amp_trafos])[0]
        assert str(tr[0]).startswith("log"), f"{run_dir}: probe transform {tr} (a signed probe has no log|M|^2 target)"
        mu_c, sd_c = float(pstats["prepd_mean"][0]), float(pstats["prepd_std"][0])
        mu_p, sd_p = float(saved["prepd_mean"][0]), float(saved["prepd_std"][0])
        real = float(np.mean(((mu_c + sd_c * h) - (mu_p + sd_p * z)) ** 2))
    print("ZERO_SHOT " + json.dumps({
        "mse_logm2_real": real, "shared_parent": bool(pstats.get("shared_standardization")),
        "run_dir": run_dir, "parent": str(cfg.fine_tune.pretrained_path), "val_loss": exp.val_loss_no_reg[-1],
        "prepd_std": saved["prepd_std"], "val_loss_affine": affine, "affine_ab": [float(c) for c in coef],
        "n_val": int(len(z)), "val_loss_check": float(np.mean((h - z) ** 2)),
        "mse_logm2": exp.val_loss_no_reg[-1] * float(np.asarray(saved["prepd_std"]).ravel()[0]) ** 2}), flush=True)
