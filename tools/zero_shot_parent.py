"""Real zero-shot of a pretraining on a process it has no data of, built from the PRETRAINING's own config (model,
data flags, particle order, causal mask), with only the recipe swapped for the probe's. tools/zero_shot_ft.py takes
the probe's config from a ladder fine-tune, whose model block is the plain one, so it cannot score a pretraining
with model.net.causal_mask (and needs a fine-tune cell per probe; the multiplicity probes W+4g, Z+4g, Z+5g have none
from rung 9).

The parent's prediction in log|M|^2: mu_c + sigma_c h, with h its standardized output and (mu_c, sigma_c) from its
data_stats.json: data.affine_multiplicity_std, mu_N = a + b N at the probe's particle count N (the paper draft's
mu_n = a + n b, Sec. "Extrapolating to more legs"); data.shared_standardization, the pooled (mu, sigma); a
per-process parent has no prediction for a new process (only val_loss_affine). The truth: the probe's log|M|^2,
mu_p + sigma_p z, with the probe's own statistics fitted on its train split (fine_tune.target_stats=own; a
representation only, inverted exactly). Scored on the probe's validation split. Prints one ZERO_SHOT JSON line
and one ZERO_SHOT_HIST line per probe. Needs a GPU.
    python tools/zero_shot_parent.py --parent <pretraining run dir> <probe recipe.yaml> [<probe recipe.yaml> ...]
"""
import argparse, json, os, sys, tempfile

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import siteconf  # noqa: E402
from rebuild_run import _build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--parent", required=True, help="pretraining run dir (config.yaml, data_stats.json, models/)")
ap.add_argument("--train-subsample", type=int, default=10000,
                help="probe train events read for its own target statistics (a representation, inverted exactly)")
ap.add_argument("recipes", nargs="+")
a = ap.parse_args()

parent = os.path.abspath(a.parent)
pstats = json.load(open(os.path.join(parent, "data_stats.json")))
ckpt = os.path.join(parent, "models", "model_run0_best.pt")
for recipe in a.recipes:
    recipe = os.path.abspath(siteconf._expand_str(recipe))
    cfg = OmegaConf.load(os.path.join(parent, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.count_flops = False; cfg.ema = False
        cfg.warm_start_idx = None
        cfg.training.validate_frac = 0; cfg.training.validate_every_n_steps = 1
        cfg.run_dir = tempfile.mkdtemp(prefix="zeroshot_", dir=os.environ["SCRATCH"])
        cfg.data.source = "recipes"
        cfg.data.processes_file = recipe
        cfg.data.train_subsample = a.train_subsample
        # the probe's own target statistics (one process: no pooled or affine fit)
        cfg.data.shared_standardization = False
        cfg.data.affine_multiplicity_std = False
        cfg.fine_tune.pretrained_path = ckpt
        cfg.fine_tune.target_stats = "own"
        cfg.fine_tune.offshell_stats = "parent"
    try:
        exp = _build(cfg, None, None, None)          # init_model loads the parent's weights, as a fine-tune does
    except Exception as e:                            # a missing pool or sidecar on this site: listed, not dropped
        print("ZERO_SHOT " + json.dumps({"recipe": recipe, "parent": parent, "error": repr(e)[:300]}), flush=True)
        continue
    assert len(exp.cfg.data.dataset) == 1, f"{recipe}: one probe process per recipe"
    probe = str(exp.cfg.data.dataset[0])
    N = int(exp._proc_npart[0])
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    exp.ema, exp.train_sampler = None, None
    exp.val_loss, exp.val_loss_no_reg, exp.val_mse = [], [], []
    exp.proc_val_losses, exp.proc_val_losses_no_reg = {}, {}
    exp._init_regularization()
    with torch.no_grad():
        hs, zs = [], []
        for data in exp.val_loader:
            y_pred, y, _, _, _ = exp._forward_lloca(data)
            hs.append(y_pred.reshape(-1).double().cpu()); zs.append(y.reshape(-1).double().cpu())
    h, z = torch.cat(hs).numpy(), torch.cat(zs).numpy()
    A = np.stack([np.ones_like(h), h], 1)
    coef = np.linalg.lstsq(A, z, rcond=None)[0]
    affine = float(np.mean((A @ coef - z) ** 2))
    tr = (exp._amp_trafos_pp or [exp.cfg.data.amp_trafos])[0]
    assert str(tr[0]).startswith("log"), f"{recipe}: probe transform {tr} (a signed probe has no log|M|^2 target)"
    mu_p, sd_p = float(exp.prepd_mean[0]), float(exp.prepd_std[0])
    t = mu_p + sd_p * z                                                   # the probe's log|M|^2
    if pstats.get("affine_multiplicity"):
        am = pstats["affine_multiplicity"]
        mu_c, sd_c, kind = float(am["a"]) + float(am["b"]) * N, float(am["sigma"]), "affine_multiplicity"
    elif pstats.get("shared_standardization"):
        mu_c, sd_c, kind = float(pstats["prepd_mean"][0]), float(pstats["prepd_std"][0]), "shared"
    else:
        mu_c = sd_c = None; kind = "per_process"
    rec = {"probe": probe, "n_particles": N, "recipe": recipe, "parent": parent, "parent_stats": kind,
           "n_val": int(len(z)), "var_logm2": float(np.var(t)), "val_loss_affine": affine,
           "affine_ab": [float(c) for c in coef]}
    if mu_c is not None:
        q = mu_c + sd_c * h                                               # the zero-shot log|M|^2
        mse = float(np.mean((q - t) ** 2))
        rec.update({"mu_c": mu_c, "sigma_c": sd_c, "mse_logm2_real": mse, "mse_over_var": mse / float(np.var(t)),
                    "bias": float(np.mean(q - t)), "mse_after_offset": float(np.var(q - t)),
                    "corr": float(np.corrcoef(q, t)[0, 1]), "spread_ratio": float(np.std(q) / np.std(t))})
        lo, hi = float(min(t.min(), q.min())), float(max(t.max(), q.max()))
        e = np.linspace(lo, hi, 61)
        print("ZERO_SHOT_HIST " + json.dumps({
            "probe": probe, "parent": parent, "edges": e.tolist(), "truth": np.histogram(t, e)[0].tolist(),
            "pred": np.histogram(q, e)[0].tolist(), "h2_range": [lo, hi],
            "h2": np.histogram2d(t, q, bins=40, range=[[lo, hi], [lo, hi]])[0].astype(int).tolist()}), flush=True)
    print("ZERO_SHOT " + json.dumps(rec), flush=True)
