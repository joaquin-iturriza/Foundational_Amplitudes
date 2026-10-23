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
--twins: a JSON {probe: training process} for a per-process parent: a probe that relabels a training process is scored
in log|M|^2 with that process's statistics (prepd_mean/std at its index in the parent's recipe), parent_stats "twin".
--save-dir: per probe <probe>.npz with the truth t, the affine-mapped prediction q_aff, q_own (the output in the probe's
own standardized units) and, when there is one, the prediction q, all in log|M|^2 on the validation split (the Delta plots and per-process MSEs are built from these).
    python tools/zero_shot_parent.py --parent <pretraining run dir> [--twins J] [--save-dir D] <probe recipe.yaml> [...]
"""
import argparse, json, os, sys, tempfile

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import siteconf  # noqa: E402
from rebuild_run import _build, load_best_weights  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--parent", required=True, help="pretraining run dir (config.yaml, data_stats.json, models/)")
ap.add_argument("--train-subsample", type=int, default=10000,
                help="probe train events read for its own target statistics (a representation, inverted exactly)")
ap.add_argument("--twins", default=None, help="JSON {probe: training process it relabels} (per-process parent)")
ap.add_argument("--save-dir", default=None, help="write <probe>.npz (t, q_aff, q) here")
ap.add_argument("--eval-subsample", type=int, default=2000, help="held-out val events per probe")
ap.add_argument("recipes", nargs="+")
a = ap.parse_args()

parent = os.path.abspath(a.parent)
pstats = json.load(open(os.path.join(parent, "data_stats.json")))
ckpt = os.path.join(parent, "models", "model_run0_best.pt")
twins = json.load(open(a.twins)) if a.twins else {}
if twins:
    import yaml
    _pc = OmegaConf.load(os.path.join(parent, "config.yaml"))
    _pr = yaml.safe_load(open(siteconf._expand_str(str(_pc.data.processes_file))))
    _pr = _pr.get("processes", _pr) if isinstance(_pr, dict) else _pr
    parent_index = {p["name"]: i for i, p in enumerate(_pr)}   # the order prepd_mean/std follow (experiment recipe load)
    assert len(parent_index) == len(pstats["prepd_mean"]), "parent recipe and its data_stats disagree in length"
if a.save_dir:
    os.makedirs(a.save_dir, exist_ok=True)
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
        cfg.data.memmap = False                      # one probe fits in memory; a memmap would build a cache per probe
        # the held-out val events per probe: 2000 unless --eval-subsample says otherwise, whatever the parent's own cap (hp58's
        # was 2000, the scale every held-out number so far was read at; a short design run validates on 200)
        cfg.data.eval_subsample = a.eval_subsample
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
    # the parent's recorded loss is its EMA weights' (docs/results.tex sec:ema-ckpt), so that is what is scored here,
    # as fine-tunes start from it since D63 (base_experiment._ckpt_scored_state)
    ema_applied = load_best_weights(exp.model, parent)
    probe = str(exp.cfg.data.dataset[0])
    N = int(exp._proc_npart[0])
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    exp.ema, exp.train_sampler = None, None
    exp.val_loss, exp.val_loss_no_reg, exp.val_mse = [], [], []
    exp.proc_val_losses, exp.proc_val_losses_no_reg = {}, {}
    exp._init_regularization()
    with torch.no_grad():
        hs, zs, ss = [], [], []
        for data in exp.val_loader:
            y_pred, y, _, sig, _ = exp._forward_lloca(data)
            hs.append(y_pred.reshape(-1).double().cpu()); zs.append(y.reshape(-1).double().cpu())
            if sig is not None:                       # HETEROSC: the per-event sigma, in the parent's standardized units
                ss.append(sig.reshape(-1).double().cpu())
    h, z = torch.cat(hs).numpy(), torch.cat(zs).numpy()
    sig = torch.cat(ss).numpy() if ss else np.zeros(0)
    A = np.stack([np.ones_like(h), h], 1)
    coef = np.linalg.lstsq(A, z, rcond=None)[0]
    affine = float(np.mean((A @ coef - z) ** 2))
    tr = (exp._amp_trafos_pp or [exp.cfg.data.amp_trafos])[0]
    if not str(tr[0]).startswith("log"):
        # a signed probe (a one-loop finite part that changes sign) has no log|M|^2 target: listed, not dropped, and
        # the other probes of the call still scored
        print("ZERO_SHOT " + json.dumps({"recipe": recipe, "parent": parent, "probe": probe,
                                         "error": f"signed probe: transform {list(map(str, tr))}, no log|M|^2 target"}), flush=True)
        continue
    mu_p, sd_p = float(exp.prepd_mean[0]), float(exp.prepd_std[0])
    t = mu_p + sd_p * z                                                   # the probe's log|M|^2
    if pstats.get("affine_multiplicity"):
        am = pstats["affine_multiplicity"]
        mu_c, sd_c, kind = float(am["a"]) + float(am["b"]) * N, float(am["sigma"]), "affine_multiplicity"
    elif pstats.get("shared_standardization"):
        mu_c, sd_c, kind = float(pstats["prepd_mean"][0]), float(pstats["prepd_std"][0]), "shared"
    elif probe in twins:
        i = parent_index[twins[probe]]
        mu_c, sd_c, kind = float(pstats["prepd_mean"][i]), float(pstats["prepd_std"][i]), "twin"
    else:
        mu_c = sd_c = None; kind = "per_process"
    rec = {"probe": probe, "n_particles": N, "recipe": recipe, "parent": parent, "ema_applied": ema_applied, "parent_stats": kind,
           "n_val": int(len(z)), "var_logm2": float(np.var(t)), "val_loss_affine": affine,
           "affine_ab": [float(c) for c in coef], "twin": twins.get(probe)}
    if mu_c is not None:
        q = mu_c + sd_c * h                                               # the zero-shot log|M|^2
        mse = float(np.mean((q - t) ** 2))
        rec.update({"mu_c": mu_c, "sigma_c": sd_c, "mse_logm2_real": mse, "mse_over_var": mse / float(np.var(t)),
                    "bias": float(np.mean(q - t)), "mse_after_offset": float(np.var(q - t)),
                    "corr": float(np.corrcoef(q, t)[0, 1]), "spread_ratio": float(np.std(q) / np.std(t))})
        if len(sig) == len(q):                        # the sigma head's calibration on this probe: pulls in log|M|^2
            pull = (q - t) / (sd_c * sig)
            rec.update({"pull_std": float(np.std(pull)), "pull_mean": float(np.mean(pull)),
                        "cover68": float(np.mean(np.abs(pull) < 1)), "cover95": float(np.mean(np.abs(pull) < 1.96))})
        lo, hi = float(min(t.min(), q.min())), float(max(t.max(), q.max()))
        e = np.linspace(lo, hi, 61)
        print("ZERO_SHOT_HIST " + json.dumps({
            "probe": probe, "parent": parent, "edges": e.tolist(), "truth": np.histogram(t, e)[0].tolist(),
            "pred": np.histogram(q, e)[0].tolist(), "h2_range": [lo, hi],
            "h2": np.histogram2d(t, q, bins=40, range=[[lo, hi], [lo, hi]])[0].astype(int).tolist()}), flush=True)
    # the per-event target factor (data.target_propagators / target_physics_factor): t and q are log(|M|^2 F), so
    # t - logF is log|M|^2; their difference, hence every MSE and Delta here, does not depend on it
    F = getattr(exp, "all_amp_factor", None)
    logF = np.zeros(0, np.float32)
    if F is not None:
        Fv = np.asarray(F)[exp._split_indices["val"]].reshape(-1)
        if len(Fv) == len(z):
            logF = np.log(Fv).astype(np.float32)
    if a.save_dir:
        # q_own: the raw output read in the probe's own standardized units (its train split's mean and std of
        # log|M|^2, two numbers from its data, no fit to the predictions)
        np.savez_compressed(os.path.join(a.save_dir, probe + ".npz"), t=t.astype(np.float32),
                            q_aff=(mu_p + sd_p * (A @ coef)).astype(np.float32),
                            q_own=(mu_p + sd_p * h).astype(np.float32), h=h.astype(np.float32),
                            mu_sd=np.array([mu_p, sd_p]), logF=logF,
                            sig=sig.astype(np.float32),     # raw, the parent's standardized units
                            s=(sd_c * sig if (mu_c is not None and len(sig) == len(h)) else np.zeros(0)).astype(np.float32),
                            q=(q if mu_c is not None else np.zeros(0)).astype(np.float32))
    print("ZERO_SHOT " + json.dumps(rec), flush=True)
