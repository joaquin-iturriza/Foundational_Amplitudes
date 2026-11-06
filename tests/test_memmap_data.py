"""data.memmap (data_mmap.py, format 3) against the in-memory recipe path: the same data, batches and RNG per seed.

CPU only. Run as a script (pytest is shadowed by the stray root-level py.py):
    python tests/test_memmap_data.py                       # sections 0-1: reductions, dataset, sampler
    python tests/test_memmap_data.py --sweep-config sweep/sweep_config_dancat_het_pre.yaml --n-procs 4 \\
        --out $SCRATCH/zz_memmap_test                       # + section 2: end to end on a recipe's pools
    python tests/test_memmap_data.py --run-config runs/catalog_v4_short/v4_het_default25 --n-procs 12 \
        --set ++data.train_subsample=2000 --out $SCRATCH/zz_memmap_test   # section 2 on a run's own config
        (dimensionless_inputs, target_physics_factor with target_physics_orders, off-shellness)
Section 0's streamed reductions reproduce numpy's own summation order (8192-element pairwise blocks for a
contiguous float32 sum, sequential rows for a C-order axis-0 sum): checked here on the numpy the sites pin.
Section 2 checks that a memmap run without a cache refuses, prebuilds the cache, then builds the experiment with the
flag off and on for seeds 42 and 43 (both opening the one cache) and as a fine-tune (frozen statistics) both ways,
and compares every event's momenta, tokens and order features, every statistic and permutation, the torch RNG state
after init_data, and the first batches of the train loader (2 workers), the val loader and the per-process loaders
tensor by tensor.
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import data_mmap as dm  # noqa: E402
from dataset import AmplitudeDataset, build_flat_arrays, collate_variable_length  # noqa: E402

PINNED_NUMPY = "1.26.4"
RUN_CONFIG, SET = None, []     # --run-config / --set (section 2 on a run's own config)
SYNTHETIC = False              # --synthetic: section 2 on the synthetic pilot recipe (CPU, the laptop's own pools)
FACTOR_RTOL = 1e-10            # data.target_physics_factor's float64 factor and the statistics fitted after it


def section0_reductions():
    assert np.__version__ == PINNED_NUMPY, (
        f"numpy {np.__version__}: the bitwise claim was checked on {PINNED_NUMPY} (the sites' pin); rerun section 0")
    rng = np.random.default_rng(0)
    for M in (3, 2047, 2049, 100000, 1048577, 3000001):
        a = (rng.standard_normal((M, 4)) * rng.uniform(0.1, 1e3) + rng.uniform(-5, 5)).astype(np.float32)
        for cs in (M, 1000, 777777):
            ch = lambda: (a[i:i + cs] for i in range(0, M, cs))
            m, s = dm.f32_mean_std(ch)
            assert m == a.mean() and s == a.std(), (M, cs)
            m2, s2 = dm.f32_mean_std(ch, scale=float(s))
            b = a / float(s)
            assert float(m2) == float(b.mean()) and float(s2) == float(b.std()), (M, cs)
    for N in (1, 5, 8193, 100003, 3000001):
        a = rng.standard_normal((N, 3)) * np.array([1e3, 1, 1e-3]) + np.array([5, 0, 1])
        for cs in (N, 1000, 1 << 20):
            m, s = dm.rows_mean_std(lambda: (a[i:i + cs] for i in range(0, N, cs)))
            assert np.array_equal(m, a.mean(0)) and np.array_equal(s, a.std(0)), (N, cs)
    print("section 0: streamed float32 mean/std and axis-0 float64 mean/std are bitwise numpy's")


def _eq_batch(x, y, what):
    assert len(x) == len(y), what
    for i, (u, v) in enumerate(zip(x, y)):
        assert u.dtype == v.dtype and u.shape == v.shape and torch.equal(u, v), f"{what}: tensor {i} differs"


def section1_dataset_sampler():
    """RandomBatchSampler: the same batches as DataLoader(shuffle=True, drop_last) over the same global RNG (the
    memmap dataset's batches against the in-memory computation: tests/test_mmap_key.py)"""
    rng = np.random.default_rng(1)
    n = 5000
    counts = rng.integers(4, 6, size=n)
    parts = [rng.normal(size=(c, 4)).astype(np.float32) for c in counts]
    toks = [rng.integers(1, 20, size=c) for c in counts]
    fp, ft, off = build_flat_arrays(parts, toks)
    amp = rng.normal(size=(n, 1)).astype(np.float32)
    order = rng.normal(size=(n, 9)).astype(np.float32)
    pid = rng.integers(0, 7, size=n)
    ref = AmplitudeDataset(fp, off, amp, ft.astype(np.int64), order, torch.float32, process_ids=pid.astype(np.int64))
    for drop in (True, False):
        torch.manual_seed(5)
        a = [b[5] for _, b in zip(range(10 ** 6), torch.utils.data.DataLoader(
            ref, batch_size=512, shuffle=True, drop_last=drop, collate_fn=collate_variable_length))]
        after_a = torch.get_rng_state()
        torch.manual_seed(5)
        bs = dm.RandomBatchSampler(n, 512, drop)
        b = [x[5] for x in torch.utils.data.DataLoader(ref, batch_sampler=bs, collate_fn=collate_variable_length)]
        assert len(a) == len(b) == len(bs), (len(a), len(b), len(bs))
        assert all(torch.equal(u, v) for u, v in zip(a, b)), "RandomBatchSampler order differs"
        assert torch.equal(after_a, torch.get_rng_state()), "RandomBatchSampler drew the global RNG differently"
    print("section 1: RandomBatchSampler order is bitwise DataLoader's")


# ---------------------------------------------------------------------------------------------------------------
def _build(sweep_config, n_procs, out, extra):
    from mem_profile_data import sweep_cfg, run_cfg
    from experiment import AmplitudeExperiment
    if SYNTHETIC:       # recipes/synthetic_pilot.yaml with the ladder's data settings (tests/test_synthetic_amplitudes.py)
        import json as _json
        from hydra import compose, initialize_config_dir
        from test_synthetic_amplitudes import LADDER_DATA, RECIPE
        import mg5_pipeline_final as mg
        import yaml
        doc = yaml.safe_load(open(RECIPE))
        mg.register_recipe_processes(doc["processes"], default_sampling=doc.get("sampling"))
        fmt = lambda v: _json.dumps(v) if isinstance(v, list) else (str(v).lower() if isinstance(v, bool) else v)
        ov = [f"{k}={fmt(v)}" for k, v in LADDER_DATA.items() if k != "data.phase_space_on_pool"]
        ov += [f"data.processes_file={RECIPE}", "data.require_cache=true", "model=lloca", f"run_dir={out}",
               "data.phase_space_on_pool=false", "data.train_subsample=3000", "data.eval_subsample=500",
               "training.batchsize=512", "evaluation.batchsize=512", "save=true"] + list(SET) + list(extra)
        ov = [o for o in ov if not o.startswith("training.num_workers")] + ["training.num_workers=2"]
        with initialize_config_dir(config_dir=os.path.join(ROOT, "config"), version_base=None):
            cfg = compose("amplitudes", overrides=ov)
    elif RUN_CONFIG:    # a run's own config.yaml (--run-config), its overrides (--set) first
        cfg, _ = run_cfg(RUN_CONFIG, n_procs, out, list(SET) + list(extra) + ["training.num_workers=2"])
    else:
        cfg, _ = sweep_cfg(sweep_config, n_procs, out, list(extra) + ["training.num_workers=2"])
    torch.set_default_dtype(torch.float32)
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data()
    rng_after = torch.get_rng_state().clone()
    exp._init_dataloader()
    snap = {"rng": rng_after}
    for k in ("mom_div",):
        snap[k] = exp.mom_div
    for k in ("mom_mean", "mom_std", "prepd_mean", "prepd_std", "_offshell_stats", "_amp_trafos_pp",
              "_role_counts", "n_order_features", "N_events", "_slot_pdgs_by_name", "_proc_npart",
              "_dimless_mass_stats"):
        snap[k] = getattr(exp, k, None)
    snap["amp_trafos"] = list(exp.cfg.data.amp_trafos)
    snap["vocab"] = dict(exp.tokenizer._pdg_to_idx)
    snap["cfg_net"] = (exp.cfg.model.net.in_channels, exp.cfg.model.net.num_scalars)
    snap.update(_events(exp))
    for k in ("all_amplitudes", "all_process_ids"):
        snap[k] = np.asarray(getattr(exp, k))
    snap["all_amp_factor"] = None if exp.all_amp_factor is None else np.asarray(exp.all_amp_factor)
    snap["flags"] = {k: bool(exp.cfg.data.get(k, False)) for k in (
        "dimensionless_inputs", "target_physics_factor", "target_physics_orders", "offshell_per_event")}
    snap["perm"] = {r: np.asarray(v) for r, v in exp._role_perm.items()}
    it = iter(exp.train_loader)
    snap["train"] = [next(it) for _ in range(4)]
    snap["n_train_batches"] = len(exp.train_loader)
    snap["val"] = list(exp.val_loader)
    snap["proc_val"] = {k: list(v) for k, v in exp.proc_val_loaders.items()}
    snap["proc_train_eval"] = {k: next(iter(v)) for k, v in exp.proc_train_eval_loaders.items()}
    snap["train_eval"] = next(iter(exp.train_eval_loader))
    snap["split_train"] = np.asarray(exp._split_indices["train"])
    snap["mmap_dir"] = getattr(exp, "_mmap_dir", None)
    snap["run_dir"] = exp.cfg.run_dir
    del exp
    return snap


def _events(exp):
    """every event's momenta, tokens and order features in the run's (shuffled) event order, as a batch carries
    them: the in-memory arrays gathered by their offsets, or the memmap source gathered in chunks"""
    src = getattr(exp, "_mmap_src", None)
    if src is None:
        off = np.asarray(exp.offsets)
        cnt = off[:, 1] - off[:, 0]
        csum = np.cumsum(cnt)
        g = np.repeat(off[:, 0] - (csum - cnt), cnt) + np.arange(int(csum[-1]))
        return {"particles_flat": np.asarray(exp.particles_flat[g]), "tokens_flat": np.asarray(exp.tokens_flat[g]),
                "all_order_labels": np.asarray(exp.all_order_labels)}
    moms, toks, ords = [], [], []
    pid_all = np.asarray(exp.all_process_ids).astype(np.int64)
    for i in range(0, exp.N_events, 1 << 16):
        v = np.arange(i, min(exp.N_events, i + (1 << 16)))
        m, t, o, _ = src.gather(v, pid_all[v])
        moms.append(m); toks.append(t); ords.append(o)
    return {"particles_flat": np.concatenate(moms), "tokens_flat": np.concatenate(toks),
            "all_order_labels": np.concatenate(ords)}


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_plain(v) for v in (x.tolist() if isinstance(x, np.ndarray) else x)]
    if isinstance(x, (np.floating, float)):
        return float(x).hex()
    if isinstance(x, np.integer):
        return int(x)
    return x


def _diagnose(A, B, label):
    """every array and statistic, equal or not, with its largest relative difference (to locate a mismatch)"""
    for k in A:
        a, b = A[k], B[k]
        if k in ("mmap_dir", "run_dir", "train", "val", "proc_val", "proc_train_eval", "train_eval", "rng", "perm",
                 "flags", "vocab", "_slot_pdgs_by_name", "amp_trafos", "_amp_trafos_pp", "cfg_net"):
            continue
        try:
            x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            if k == "all_amp_factor":
                x, y = x.reshape(-1), y.reshape(-1)
        except (TypeError, ValueError):
            continue
        if x.shape != y.shape:
            print(f"  [diag {label}] {k}: shapes {x.shape} vs {y.shape}")
            continue
        d = np.abs(x - y) / np.maximum(np.abs(x), 1e-300)
        n = int((x != y).sum())
        print(f"  [diag {label}] {k}: {n} of {x.size} differ, max rel {float(d.max()) if d.size else 0.0:.3g}"
              + (f"; first at {np.argwhere(x != y)[0].tolist()}" if n else ""))


def _compare(A, B, label):
    _diagnose(A, B, label)
    for k in A:
        if k in ("mmap_dir", "run_dir", "train", "val", "proc_val", "proc_train_eval", "train_eval"):
            continue
        a, b = A[k], B[k]
        if k == "rng":
            assert torch.equal(a, b), f"{label}: torch RNG state after init_data differs"
        elif k == "all_amp_factor":
            # the float64 target factor and the float64 statistics fitted on the factored target are not bitwise
            # (measured on catalog_v4 12 processes: factor up to 2.6e-12 relative on a third of the events, prepd
            # mean/std 1 ulp; cause not found, the momenta and alpha_s fed to _physics_logF are bitwise equal): held
            # to FACTOR_RTOL; every float32 array the network reads (targets included) is still compared bitwise
            assert (a is None) == (b is None) and (a is None or (a.shape == b.shape and np.allclose(
                a, b, rtol=FACTOR_RTOL, atol=0))), f"{label}: {k} differs beyond rtol {FACTOR_RTOL}"
        elif k in ("prepd_mean", "prepd_std") and A.get("all_amp_factor") is not None:
            assert np.allclose(np.asarray(a, np.float64), np.asarray(b, np.float64), rtol=FACTOR_RTOL, atol=0), \
                f"{label}: {k} differs beyond rtol {FACTOR_RTOL}: {a} vs {b}"
        elif k in ("particles_flat", "tokens_flat", "all_amplitudes", "all_process_ids",
                   "all_order_labels", "split_train"):
            assert a.shape == b.shape and np.array_equal(a.astype(np.float64), b.astype(np.float64)), \
                f"{label}: {k} differs"
        elif k == "perm":
            assert a.keys() == b.keys() and all(np.array_equal(a[r], b[r]) for r in a), f"{label}: perm differs"
        else:
            assert repr(_plain(a)) == repr(_plain(b)), f"{label}: {k} differs: {a} vs {b}"
    for i, (x, y) in enumerate(zip(A["train"], B["train"])):
        _eq_batch(x, y, f"{label}: train batch {i}")
    assert len(A["val"]) == len(B["val"])
    for i, (x, y) in enumerate(zip(A["val"], B["val"])):
        _eq_batch(x, y, f"{label}: val batch {i}")
    assert A["proc_val"].keys() == B["proc_val"].keys()
    for k in A["proc_val"]:
        for x, y in zip(A["proc_val"][k], B["proc_val"][k]):
            _eq_batch(x, y, f"{label}: per-process val {k}")
    assert A["proc_train_eval"].keys() == B["proc_train_eval"].keys()
    for k in A["proc_train_eval"]:
        _eq_batch(A["proc_train_eval"][k], B["proc_train_eval"][k], f"{label}: per-process train-eval {k}")
    _eq_batch(A["train_eval"], B["train_eval"], f"{label}: train-eval batch")
    print(f"  {label}: identical ({A['N_events']} events, flags {A['flags']}, {A['particles_flat'].shape[0]} particles, "
          f"{A['n_train_batches']} train batches; 4 train batches, {len(A['val'])} val batches, "
          f"{len(A['proc_val'])} per-process loaders compared)")


def section2_end_to_end(sweep_config, n_procs, out):
    """data.memmap against the in-memory path for two seeds from one cache (format 3: the cache holds no seed, cap or
    statistic; the augmentation is drawn per event from (seed, pool, index) in both paths), the refusal without a
    cache, the prebuild, and a fine-tune with frozen statistics opening the same cache."""
    shutil.rmtree(out, ignore_errors=True)
    cache = os.path.join(out, "cache")
    on = ["++data.memmap=true", f"++data.memmap_dir={cache}"]
    try:
        _build(sweep_config, n_procs, os.path.join(out, "refused"), on + ["seed=42"])
        raise AssertionError("a run without a cache must refuse, not build one")
    except RuntimeError as e:
        assert "prebuild" in str(e), e
    try:
        _build(sweep_config, n_procs, os.path.join(out, "prebuild"), on + ["seed=42", "++data.memmap_prebuild=true"])
        raise AssertionError("the prebuild must exit after the build")
    except SystemExit as e:
        assert e.code in (0, None), e.code
    snaps = {}
    for seed in (42, 43):
        A = _build(sweep_config, n_procs, os.path.join(out, f"off{seed}"), [f"seed={seed}"])
        B = _build(sweep_config, n_procs, os.path.join(out, f"on{seed}"), on + [f"seed={seed}"])
        _compare(A, B, f"seed {seed}: in memory vs memmap")
        snaps[seed] = (A, B)
    assert snaps[42][1]["mmap_dir"] == snaps[43][1]["mmap_dir"], "two seeds must open one cache"
    assert not np.array_equal(snaps[42][0]["particles_flat"], snaps[43][0]["particles_flat"]), \
        "seeds 42 and 43 drew the same augmentation"
    # smaller caps: a prefix of the same cache (train_subsample / eval_subsample are not in its key)
    small = ["++data.train_subsample=1000", "++data.eval_subsample=200", "seed=42"]
    A1 = _build(sweep_config, n_procs, os.path.join(out, "off_small"), small)
    B1 = _build(sweep_config, n_procs, os.path.join(out, "on_small"), on + small)
    assert B1["mmap_dir"] == snaps[42][1]["mmap_dir"], "smaller caps must read a prefix of the same cache"
    _compare(A1, B1, "smaller caps: in memory vs memmap")
    # frozen statistics: a fine-tune of the seed-42 in-memory run reads its data_stats.json, and opens the same cache
    A = snaps[42][0]
    ft = [f"fine_tune.pretrained_path={os.path.join(A['run_dir'], 'models', 'model_run0_best.pt')}", "seed=42"]
    D = _build(sweep_config, n_procs, os.path.join(out, "ft_off"), ft)
    E = _build(sweep_config, n_procs, os.path.join(out, "ft_on"), ft + on)
    assert E["mmap_dir"] == snaps[42][1]["mmap_dir"], "frozen statistics must not key another cache"
    _compare(D, E, "fine-tune (frozen stats) in memory vs memmap")
    print("section 2: data.memmap and the in-memory path give the same data, batches and RNG state for seeds 42 and "
          "43 from one cache; a run without a cache refuses; smaller caps and a fine-tune open the same cache")


def section3_gpu_training(sweep_config, n_procs, out, steps):
    """A few training steps with the flag off, on and off again (the last is the run-to-run control on the GPU):
    the train and validation losses of every step / validation, compared exactly."""
    from mem_profile_data import sweep_cfg
    from experiment import AmplitudeExperiment
    assert torch.cuda.is_available(), "section 3 needs a GPU"
    shutil.rmtree(out, ignore_errors=True)
    runs = {}
    on = ["data.memmap=true", f"data.memmap_dir={os.path.join(out, 'cache')}"]
    try:                # the cache comes from a prebuild (a run never builds one)
        _build(sweep_config, n_procs, os.path.join(out, "prebuild"), on + ["++data.memmap_prebuild=true"])
    except SystemExit:
        pass
    for tag, extra in (("off", []), ("on", on),
                       ("off2", [])):
        cfg, _ = sweep_cfg(sweep_config, n_procs, os.path.join(out, tag), extra + [
            f"training.iterations={steps}", "training.validate_frac=0.1", "evaluate=false", "plot=false",
            "training.num_workers=2", "ema=false", "training.regularization_lambda=1e-8"])
        torch.set_default_dtype(torch.float32)
        exp = AmplitudeExperiment(cfg)
        exp()
        runs[tag] = (np.asarray(exp.train_loss, np.float64), np.asarray(exp.val_loss_no_reg, np.float64))
        print(f"  {tag}: train loss steps 1..{len(exp.train_loss)}: first {runs[tag][0][:3]}, last "
              f"{runs[tag][0][-3:]}; val_loss_no_reg {runs[tag][1]}")
    for a, b in (("off", "on"), ("off", "off2")):
        tr = np.array_equal(runs[a][0], runs[b][0])
        va = np.array_equal(runs[a][1], runs[b][1])
        rel = float(np.max(np.abs(runs[a][1] - runs[b][1]) / np.abs(runs[a][1])))
        print(f"  {a} vs {b}: train losses identical {tr}, val losses identical {va} (max rel. diff {rel:.3g})")
    print("section 3 done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-config")
    ap.add_argument("--run-config", help="a run's directory holding its config.yaml (section 2 on that config)")
    ap.add_argument("--set", action="append", default=[], help="an override applied to --run-config (repeatable)")
    ap.add_argument("--n-procs", type=int, default=4)
    ap.add_argument("--synthetic", action="store_true", help="section 2 on the synthetic pilot recipe (CPU)")
    ap.add_argument("--out")
    ap.add_argument("--gpu-steps", type=int, default=0, help="section 3 only: train this many steps off/on/off")
    a = ap.parse_args()
    RUN_CONFIG, SET, SYNTHETIC = a.run_config, a.set, a.synthetic
    if a.gpu_steps:
        section3_gpu_training(a.sweep_config, a.n_procs, os.path.expandvars(a.out), a.gpu_steps)
        sys.exit(0)
    section0_reductions()
    section1_dataset_sampler()
    if a.sweep_config or a.run_config or a.synthetic:
        section2_end_to_end(a.sweep_config, a.n_procs, os.path.expandvars(a.out))
    print("OK")
