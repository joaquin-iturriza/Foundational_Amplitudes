"""Rebuild a finished run for inference on another split: the run's own config (so its own train pool,
momentum scale, target factors and amplitude stats) with one role's pool file replaced, and its best
checkpoint loaded. Used by tools/steer_pool.py (the reference model scores candidates) and
analysis/transfer/cross_eval.py (a run scored on another recipe's test split).

The rebuilt statistics are checked against the run's frozen data_stats.json, which is the evidence
that the inputs go through exactly the run's preprocessing. The off-shellness feature's column stats are
fitted on every role together, so they are pinned to the run's own (data.offshell_stats; from
data_stats.json, or for older runs a build on the run's own pools): a swapped split otherwise shifts
every event's input (the steering cross-pool offset, docs/results.tex sec:ladder).
"""
import gzip, io, json, os, sys, tempfile

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, open_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import datagen  # noqa: E402


def _load_best_ckpt(run_dir):
    p = os.path.join(run_dir, "models", "model_run0_best.pt")
    if os.path.exists(p):
        return torch.load(p, map_location="cpu", weights_only=False)
    with gzip.open(p + ".gz", "rb") as f:
        return torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)


def load_best_state(run_dir):
    """The best checkpoint's "model" entry, plain or gzipped. With EMA on it is the EMA average that validation
    scored only in a checkpoint written since D63 (marker "model_is_ema"); in an older one it is the LIVE weights at
    the best step (base_experiment._save_model took state_dict() inside ema.average_parameters() but saved after the
    context restored the live weights into the same storage; docs/results.tex sec:ema-ckpt). To score a run as it
    was validated, use load_best_weights."""
    return _load_best_ckpt(run_dir)["model"]


def load_best_weights(model, run_dir):
    """Load the best checkpoint (es_load_best_model: the validation minimum) into `model` as the run
    validated it. A checkpoint written since D63 with EMA on ("model_is_ema") holds the EMA weights in "model"; an
    older one holds the live weights there, and the EMA shadow is applied over model.parameters() (the order
    ExponentialMovingAverage was built with). Returns True if the weights loaded are the EMA weights."""
    ck = _load_best_ckpt(run_dir)
    model.load_state_dict(ck["model"])
    if ck.get("model_is_ema"):
        return True
    ema = ck.get("ema")
    if not ema:
        return False
    params = list(model.parameters())
    shadow = ema["shadow_params"]
    assert len(shadow) == len(params), f"{run_dir}: EMA shadow has {len(shadow)} tensors, model {len(params)}"
    with torch.no_grad():
        for p, s in zip(params, shadow):
            assert p.shape == s.shape, f"{run_dir}: EMA shadow shape {tuple(s.shape)} != {tuple(p.shape)}"
            p.copy_(s.to(p.device, p.dtype))
    return True


def scored_split(exp, split, name):
    """What a probe read scores on one held-out role of a built experiment: its loader, the pool file the physics
    check reads and the event indices of that role (for the per-event target factor). split is "val" (the
    zero-shot reads' default since they began) or "test" (a pool no run selects a checkpoint on: for a process the
    parent trained on, its val pool was the parent's own checkpoint-selection pool)."""
    assert split in ("val", "test"), f"split {split!r}: val or test"
    loader = exp.val_loader if split == "val" else exp.test_loader
    pool = (getattr(exp, "_pool_paths", None) or {}).get(split, {}).get(name)
    return loader, pool, exp._split_indices[split]


def recipe_role_counts(recipe):
    """{process: {role: events}} of a recipe file."""
    r = yaml.safe_load(open(recipe))
    return {e["name"]: {"train": int(e.get("n_train", 0) or 0), "val": int(e.get("n_val", 0) or 0),
                        "test": int(e.get("n_test", 0) or 0)} for e in (r.get("processes", r) if isinstance(r, dict) else r)}


def stand_in_role(counts, name, role, scored):
    """(role, events) a probe's `role` is read from: its own when it has events, else the first role it has of the
    scored split, test, val, train; (role, 0) when it has none."""
    c = counts.get(name) or {}
    if c.get(role):
        return role, c[role]
    return next(((r, c[r]) for r in [scored] + [x for x in ("test", "val", "train") if x != scored] if c.get(r)),
                (role, 0))


def stand_in_split_set(real, counts, scored):
    """datagen.ensure_split_set for a probe read: a role a probe has no pool of (an EE4 probe of
    recipes/catalog_v4_ee_test.yaml: n_train 0 and n_val 0, its test pool only) is read from a role it has (stand_in_role).
    Its train role gives only the probe's own target statistics (fine_tune.target_stats=own), a representation
    inverted exactly: the truth t = mu_p + sd_p z, the prediction q and mse_logm2_real do not depend on it."""
    def split(specs, role, seed, dest_dir=None, require_cache=False):
        out = {}
        for s in specs:
            r, n = (role, s["n_events"]) if int(s["n_events"]) > 0 else stand_in_role(counts, s["process"], role, scored)
            out.update(real([dict(s, n_events=n)], role=r, seed=seed, dest_dir=dest_dir, require_cache=require_cache))
        return out
    return split


def _build(cfg, swap, path, name):
    """The experiment built through init_model, with the `swap` role's pool read from `path` (None: none)."""
    real = datagen.ensure_split_set

    def split(specs, role, seed, dest_dir=None, require_cache=False):     # the loader's own signature
        return ({name: path} if role == swap else
                real(specs, role=role, seed=seed, dest_dir=dest_dir, require_cache=require_cache))
    datagen.ensure_split_set = split
    try:
        torch.set_default_dtype(torch.float32)
        from experiment import AmplitudeExperiment
        exp = AmplitudeExperiment(cfg)
        exp._init(); exp.init_physics(); exp.init_geometric_algebra()
        exp.init_data(); exp._init_dataloader(); exp.init_model()
    finally:
        datagen.ensure_split_set = real
    return exp


def own_offshell_stats(run_dir):
    """The off-shellness column stats [mean, std] the run trained with, from a build on its own pools (for runs
    older than their record in data_stats.json); checked against the run's frozen amplitude stats."""
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    assert cfg.data.get("offshell_per_event", False), f"{run_dir} has no off-shellness input"
    name = yaml.safe_load(open(os.path.expandvars(str(cfg.data.processes_file))))["processes"][0]["name"]
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.count_flops = False
        cfg.warm_start_idx = None; cfg.fine_tune.pretrained_path = None
        cfg.run_dir = tempfile.mkdtemp(prefix="rebuild_", dir=os.environ["SCRATCH"])
    exp = _build(cfg, None, None, name)
    saved = json.load(open(os.path.join(run_dir, "data_stats.json")))
    assert np.allclose([float(x) for x in exp.prepd_std], saved["prepd_std"], rtol=1e-6), "not the run's own build"
    return exp._offshell_stats


def rebuild(run_dir, role, path, n_events):
    """(exp, loader): the run rebuilt with `role` ('val' or 'test') read from `path` (n_events rows),
    in eval mode with its best weights; loader iterates that role in the order of exp._role_perm[role]."""
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    own_recipe = os.path.expandvars(str(cfg.data.processes_file))
    rec = yaml.safe_load(open(own_recipe))
    (spec,) = rec["processes"]
    name = spec["name"]
    spec[f"n_{role}"] = int(n_events)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(rec, tmp); tmp.close()
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.count_flops = False
        cfg.warm_start_idx = None; cfg.fine_tune.pretrained_path = None
        cfg.data.processes_file = tmp.name
        cfg.data.eval_subsample = None            # the swapped split is read whole (the recipe count caps it)
        cfg.run_dir = tempfile.mkdtemp(prefix="rebuild_", dir=os.environ["SCRATCH"])
    saved = json.load(open(os.path.join(run_dir, "data_stats.json")))
    if cfg.data.get("offshell_per_event", False) and cfg.data.get("offshell_stats", None) is None:
        # the off-shellness column stats are fitted on every role together (experiment.py), so the swapped
        # split would move them and shift every event's input; pin the run's own. Runs older than the
        # record in data_stats.json get them from a build on the run's own pools
        own = saved.get("offshell_stats")
        if own is None:
            own = own_offshell_stats(run_dir)
        with open_dict(cfg):
            cfg.data.offshell_stats = own
    exp = _build(cfg, role, path, name)
    got = dict(mom_div=float(exp.mom_div), prepd_mean=[float(x) for x in exp.prepd_mean],
               prepd_std=[float(x) for x in exp.prepd_std])
    for k, v in got.items():
        assert np.allclose(v, saved[k], rtol=1e-6), f"rebuilt {k} {v} != the run's {saved[k]}"
    if saved.get("tchannel_norm") is not None:
        assert np.allclose(exp._tch_norm, saved["tchannel_norm"], rtol=1e-6), (exp._tch_norm, saved["tchannel_norm"])
    load_best_weights(exp.model, run_dir)
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    loader = exp.val_loader if role == "val" else exp.test_loader
    n = len(loader.dataset) if hasattr(loader.dataset, "__len__") else None
    assert n in (None, int(n_events)), f"{role} loader holds {n} events, expected {n_events}"
    return exp, loader
