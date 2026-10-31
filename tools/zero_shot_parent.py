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
representation only, inverted exactly). Scored on the probe's validation split (--split test: its test split, a pool no
run selected a checkpoint on; the first --eval-subsample rows of either). Prints one ZERO_SHOT JSON line
and one ZERO_SHOT_HIST line per probe. Needs a GPU.
--twins: a JSON {probe: training process} for a per-process parent: a probe that relabels a training process is scored
in log|M|^2 with that process's statistics (prepd_mean/std at its index in the parent's recipe), parent_stats "twin".
--save-dir: per probe <probe>.npz with the truth t, the affine-mapped prediction q_aff, q_own (the output in the probe's
own standardized units) and, when there is one, the prediction q, all in log|M|^2 on the scored split (--split; the Delta plots and per-process MSEs are built from these).
Physics check (tools/pool_physics.py): before a probe is scored, its pool's physics block (model with restriction, MB,
ymb, MT, ymt, WT, WZ, WW, WH as its backend read them, coupling-order clause, order vector, alpha_s rule) and the b's
particle-table row are compared with the parent's training catalog (data.processes_file): its entry for the same
process, else the catalog-wide values. A difference refuses the probe (a ZERO_SHOT line with error "physics mismatch:
...", one PHYSICS_CHECK line with the details) unless --allow-physics-mismatch "<reason>" (or the environment variable
ZS_ALLOW_PHYSICS_MISMATCH) is given; the reason goes into the ZERO_SHOT record and the probe's .npz (physics_check).
    python tools/zero_shot_parent.py --parent <pretraining run dir> [--twins J] [--save-dir D]
        [--allow-physics-mismatch "<reason>"] <probe recipe.yaml> [...]
"""
import argparse, json, os, sys, tempfile

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import siteconf  # noqa: E402
import pool_physics  # noqa: E402
from rebuild_run import _build, load_best_weights, scored_split  # noqa: E402
from multiplicity_extrap import parent_target_standardization  # noqa: E402
from preprocessing import unresolved_amp_trafos  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--parent", required=True, help="pretraining run dir (config.yaml, data_stats.json, models/)")
ap.add_argument("--train-subsample", type=int, default=10000,
                help="probe train events read for its own target statistics (a representation, inverted exactly)")
ap.add_argument("--twins", default=None, help="JSON {probe: training process it relabels} (per-process parent)")
ap.add_argument("--save-dir", default=None, help="write <probe>.npz (t, q_aff, q) here")
ap.add_argument("--eval-subsample", type=int, default=2000, help="held-out events per probe (the first rows of --split)")
ap.add_argument("--split", choices=("val", "test"), default="val",
                help="the probe pool's role scored: val (default, every read so far) or test (for a process the parent "
                     "trained on, its val pool was the parent's checkpoint-selection pool)")
ap.add_argument("--window-names", action="store_true",
                help="save <probe>_<lo>-<hi>GeV.npz (the recipe's sqrt(s) window), for a probe set with several windows of one process")
ap.add_argument("--signed-magnitude", action="store_true",
                help="a probe whose target changes sign is scored on log|M| (data.signed_abslog: the 'abslog' transform), "
                     "the magnitude a log|M|^2 model predicts; its sign, which such a model has no output for, is not scored")
ap.add_argument("--allow-physics-mismatch", default=os.environ.get(pool_physics.ALLOW_ENV) or None, metavar="REASON",
                help="score a probe whose pool's physics differs from the parent's training catalog; the reason is saved")
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
# the parent's training catalog, the reference of every probe's physics check (built once, from the entries' backends)
_pcfg = OmegaConf.load(os.path.join(parent, "config.yaml"))
_train_recipe = _pcfg.data.get("processes_file", None)
physics_ref = (pool_physics.TrainingReference(siteconf._expand_str(str(_train_recipe)))
               if _train_recipe and str(_train_recipe) != "None" else None)

# a role a probe has no pool of is read from one it has (rebuild_run.stand_in_split_set; EE4: the test pool only); the
# role its own target statistics came from is recorded (own_stats_role): q_own and val_loss_affine are in its units,
# t, q and mse_logm2_real do not depend on it
import datagen  # noqa: E402
from rebuild_run import recipe_role_counts, stand_in_role, stand_in_split_set  # noqa: E402
_ensure_split_set = datagen.ensure_split_set

for recipe in a.recipes:
    recipe = os.path.abspath(siteconf._expand_str(recipe))
    _counts = recipe_role_counts(recipe)
    datagen.ensure_split_set = stand_in_split_set(_ensure_split_set, _counts, a.split)
    _roles = sorted({stand_in_role(_counts, n, "train", a.split)[0] for n in _counts})
    own_stats_role = _roles[0] if len(_roles) == 1 else _roles
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
        if a.signed_magnitude:
            cfg.data.signed_abslog = True
        # a het run's sigma_stage/config.yaml written before run.py kept the unresolved base carries the main run's
        # resolved representative ('signedlog:<s>'), which resolve_amp_trafos would pass through to every probe
        cfg.data.amp_trafos = unresolved_amp_trafos(cfg.data.amp_trafos)
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
    # the physics check, before anything is scored
    _spec = exp._recipe_specs[0]
    loader, _pool, split_idx = scored_split(exp, a.split, _spec["name"])
    import mg5_pipeline_final as _mg
    physics_chk = pool_physics.check_probe(_spec["name"], _spec.get("physics"), _pool, physics_ref,
                                           cfg=_mg.PROCESSES.get(_spec["name"]) or _mg.PROCESSES[_spec["base"]])
    _ok = pool_physics.apply_allowance(physics_chk, a.allow_physics_mismatch)
    print("PHYSICS_CHECK " + json.dumps(physics_chk, default=str), flush=True)
    if not _ok:
        print("ZERO_SHOT " + json.dumps({"recipe": recipe, "parent": parent, "probe": _spec["name"],
                                         "error": "physics mismatch: " + pool_physics.summary(physics_chk)}), flush=True)
        continue
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
        for data in loader:
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
    if not (str(tr[0]).startswith("log") or (a.signed_magnitude and str(tr[0]).startswith("abslog"))):
        # a signed probe (a one-loop finite part that changes sign) has no log|M|^2 target: listed, not dropped, and
        # the other probes of the call still scored
        print("ZERO_SHOT " + json.dumps({"recipe": recipe, "parent": parent, "probe": probe,
                                         "error": f"signed probe: transform {list(map(str, tr))}, no log|M|^2 target"}), flush=True)
        continue
    mu_p, sd_p = float(exp.prepd_mean[0]), float(exp.prepd_std[0])
    t = mu_p + sd_p * z                                                   # the probe's log|M|^2 (abslog: log max(|M|^2, s))
    if pstats.get("affine_multiplicity") or pstats.get("shared_standardization"):
        # the statistics a fine-tune with fine_tune.target_stats=parent_shared standardizes the probe with
        (mu_c,), (sd_c,), kind = parent_target_standardization(pstats, [N])
    elif probe in twins:
        i = parent_index[twins[probe]]
        mu_c, sd_c, kind = float(pstats["prepd_mean"][i]), float(pstats["prepd_std"][i]), "twin"
    else:
        mu_c = sd_c = None; kind = "per_process"
    rec = {"probe": probe, "split": a.split, "n_particles": N, "recipe": recipe, "parent": parent, "ema_applied": ema_applied, "parent_stats": kind,
           "own_stats_role": own_stats_role,
           "n_val": int(len(z)),   # the events scored, of --split (key kept for the readers)
           "var_logm2": float(np.var(t)), "val_loss_affine": affine,
           "affine_ab": [float(c) for c in coef], "twin": twins.get(probe), "transform": str(tr[0]),
           "physics": pool_physics.record(physics_chk)}
    # abslog: the truth is clipped at log s (s the probe's 1% quantile of |x| on its train split, data.signedlog_quantile),
    # the prediction is not; the share of val events at that floor is part of what the MSE means
    floor = np.log(float(str(tr[0]).split(":", 1)[1])) if str(tr[0]).startswith("abslog:") else None
    at_floor = (t <= floor + 1e-4 * max(1.0, abs(floor))) if floor is not None else np.zeros(len(t), bool)
    if floor is not None:
        rec.update({"abslog_floor": float(floor), "frac_at_floor": float(np.mean(at_floor))})
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
    # t - logF is log|M|^2 (abslog: log|M|^2 only off the floor, at_floor below; the floor is on |M|^2 F); their
    # difference, hence every MSE and Delta here, does not depend on it
    F = getattr(exp, "all_amp_factor", None)
    logF = np.zeros(0, np.float32)
    if F is not None:
        Fv = np.asarray(F)[split_idx].reshape(-1)
        if len(Fv) == len(z):
            logF = np.log(Fv).astype(np.float32)
    if a.save_dir:
        # q_own: the raw output read in the probe's own standardized units (its train split's mean and std of
        # log|M|^2, two numbers from its data, no fit to the predictions)
        stem = probe
        if a.window_names:
            import yaml
            lo, hi = yaml.safe_load(open(recipe))["processes"][0]["sqrts"]
            stem = f"{probe}_{lo}-{hi}GeV"
        np.savez_compressed(os.path.join(a.save_dir, stem + ".npz"), t=t.astype(np.float32),
                            q_aff=(mu_p + sd_p * (A @ coef)).astype(np.float32),
                            q_own=(mu_p + sd_p * h).astype(np.float32), h=h.astype(np.float32),
                            mu_sd=np.array([mu_p, sd_p]), logF=logF, at_floor=at_floor, split=np.array(a.split),
                            sig=sig.astype(np.float32),     # raw, the parent's standardized units
                            s=(sd_c * sig if (mu_c is not None and len(sig) == len(h)) else np.zeros(0)).astype(np.float32),
                            q=(q if mu_c is not None else np.zeros(0)).astype(np.float32),
                            # the physics check of this probe (tools/pool_physics.py); a mismatch read on purpose
                            # carries its reason here and in physics_reason
                            physics_check=np.array(json.dumps(physics_chk, default=str)),
                            physics_reason=np.array(physics_chk.get("reason") or ""),
                            # the scored events' positions in the experiment's split (idx) and their rows in the split's pool (pool_row), in scoring order, and that pool's path:
                            # a per-event read from the pool's momenta (v3_prep_explore, the dipole residual) aligns on
                            # these, never by matching log|M|^2 (the sampling-sensitivity dead ends)
                            idx=(np.asarray(split_idx).reshape(-1) if len(np.asarray(split_idx).reshape(-1)) == len(t)
                                 else np.zeros(0, np.int64)),
                            # idx are positions in the experiment's role-contiguous array, not pool rows: the pool
                            # row of each scored event is the role's shuffle (_role_perm) at its position in the role
                            pool_row=(np.asarray((getattr(exp, "_role_perm", None) or {}).get(a.split, []))[
                                np.asarray(split_idx).reshape(-1) - int(np.min(split_idx))]
                                if len(np.asarray(split_idx).reshape(-1)) == len(t)
                                and len((getattr(exp, "_role_perm", None) or {}).get(a.split, [])) == len(t)
                                else np.zeros(0, np.int64)),
                            pool=np.array(str(_pool or "")))
    print("ZERO_SHOT " + json.dumps(rec), flush=True)
