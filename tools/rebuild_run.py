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


def load_best_state(run_dir):
    """The best checkpoint (es_load_best_model: the validation minimum), plain or gzipped."""
    p = os.path.join(run_dir, "models", "model_run0_best.pt")
    if os.path.exists(p):
        return torch.load(p, map_location="cpu", weights_only=False)["model"]
    with gzip.open(p + ".gz", "rb") as f:
        return torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)["model"]


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
    exp.model.load_state_dict(load_best_state(run_dir))
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    loader = exp.val_loader if role == "val" else exp.test_loader
    n = len(loader.dataset) if hasattr(loader.dataset, "__len__") else None
    assert n in (None, int(n_events)), f"{role} loader holds {n} events, expected {n_events}"
    return exp, loader
