"""data.memmap (data_mmap.py) against the in-memory recipe path: bitwise the same data, batches and RNG.

CPU only. Run as a script (pytest is shadowed by the stray root-level py.py):
    python tests/test_memmap_data.py                       # sections 0-1: reductions, dataset, sampler
    python tests/test_memmap_data.py --sweep-config sweep/sweep_config_dancat_het_pre.yaml --n-procs 4 \\
        --out $SCRATCH/zz_memmap_test                       # + section 2: end to end on a recipe's pools
Section 0's streamed reductions reproduce numpy's own summation order (8192-element pairwise blocks for a
contiguous float32 sum, sequential rows for a C-order axis-0 sum): checked here on the numpy the sites pin.
Section 2 builds the experiment with the flag off, on (cache built), on again (cache hit), and as a fine-tune
of the first (frozen statistics) both ways, and compares every array, statistic, permutation, the torch RNG
state after init_data, and the first batches of the train loader (2 workers), the val loader and the
per-process loaders tensor by tensor.
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
    d = tempfile.mkdtemp()
    try:
        arrs = {"p": fp, "o": off, "a": amp, "t": ft.astype(np.int16), "l": order, "i": pid.astype(np.int32)}
        mm = {}
        for k, v in arrs.items():
            np.save(os.path.join(d, k + ".npy"), v)
            mm[k] = np.load(os.path.join(d, k + ".npy"), mmap_mode="r")
        for rows in (range(0, n), np.sort(rng.choice(n, 900, replace=False))):
            ds = dm.MmapAmplitudeDataset(mm["p"], mm["o"], mm["a"], mm["t"], mm["l"], torch.float32, mm["i"], rows)
            sub = AmplitudeDataset(fp, off[np.asarray(rows)], amp[np.asarray(rows)], ft.astype(np.int64),
                                   order[np.asarray(rows)], torch.float32, process_ids=pid[np.asarray(rows)])
            idx = rng.choice(len(ds), 300, replace=False)
            _eq_batch(collate_variable_length(ds.__getitems__(idx)), collate_variable_length(sub.__getitems__(idx)),
                      "mmap vs in-memory batch")
            _eq_batch(ds[17], sub[17], "mmap vs in-memory item")
        # the sampler: same batches as DataLoader(shuffle=True, drop_last) over the same global RNG
        for drop in (True, False):
            torch.manual_seed(5)
            a = [b[5] for _, b in zip(range(10 ** 6), torch.utils.data.DataLoader(
                ref, batch_size=512, shuffle=True, drop_last=drop, collate_fn=collate_variable_length))]
            after_a = torch.get_rng_state()
            torch.manual_seed(5)
            dsr = dm.MmapAmplitudeDataset(mm["p"], mm["o"], mm["a"], mm["t"], mm["l"], torch.float32, mm["i"],
                                          range(0, n))
            bs = dm.RandomBatchSampler(n, 512, drop)
            b = [x[5] for x in torch.utils.data.DataLoader(dsr, batch_sampler=bs, collate_fn=collate_variable_length)]
            assert len(a) == len(b) == len(bs), (len(a), len(b), len(bs))
            assert all(torch.equal(u, v) for u, v in zip(a, b)), "RandomBatchSampler order differs"
            assert torch.equal(after_a, torch.get_rng_state()), "RandomBatchSampler drew the global RNG differently"
    finally:
        shutil.rmtree(d)
    print("section 1: MmapAmplitudeDataset batches/items and RandomBatchSampler order are bitwise the in-memory ones")


# ---------------------------------------------------------------------------------------------------------------
def _build(sweep_config, n_procs, out, extra):
    from mem_profile_data import sweep_cfg
    from experiment import AmplitudeExperiment
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
              "_role_counts", "n_order_features", "N_events", "_slot_pdgs_by_name", "_proc_npart"):
        snap[k] = getattr(exp, k, None)
    snap["amp_trafos"] = list(exp.cfg.data.amp_trafos)
    snap["vocab"] = dict(exp.tokenizer._pdg_to_idx)
    snap["cfg_net"] = (exp.cfg.model.net.in_channels, exp.cfg.model.net.num_scalars)
    for k in ("particles_flat", "tokens_flat", "offsets", "all_amplitudes", "all_process_ids", "all_order_labels"):
        snap[k] = np.asarray(getattr(exp, k))
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


def _compare(A, B, label):
    for k in A:
        if k in ("mmap_dir", "run_dir", "train", "val", "proc_val", "proc_train_eval", "train_eval"):
            continue
        a, b = A[k], B[k]
        if k == "rng":
            assert torch.equal(a, b), f"{label}: torch RNG state after init_data differs"
        elif k in ("particles_flat", "tokens_flat", "offsets", "all_amplitudes", "all_process_ids",
                   "all_order_labels", "split_train"):
            assert a.shape == b.shape and np.array_equal(a.astype(np.float64), b.astype(np.float64)), \
                f"{label}: {k} differs"
        elif k == "perm":
            assert a.keys() == b.keys() and all(np.array_equal(a[r], b[r]) for r in a), f"{label}: perm differs"
        else:
            assert a == b, f"{label}: {k} differs: {a} vs {b}"
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
    print(f"  {label}: identical ({A['N_events']} events, {A['particles_flat'].shape[0]} particles, "
          f"{A['n_train_batches']} train batches; 4 train batches, {len(A['val'])} val batches, "
          f"{len(A['proc_val'])} per-process loaders compared)")


def section2_end_to_end(sweep_config, n_procs, out):
    shutil.rmtree(out, ignore_errors=True)
    cache = os.path.join(out, "cache")
    on = ["data.memmap=true", f"data.memmap_dir={cache}"]
    A = _build(sweep_config, n_procs, os.path.join(out, "off"), [])
    B = _build(sweep_config, n_procs, os.path.join(out, "on"), on)
    assert B["mmap_dir"] and os.path.exists(os.path.join(B["mmap_dir"], "meta.json"))
    _compare(A, B, "flag off vs on (cache built)")
    C = _build(sweep_config, n_procs, os.path.join(out, "on_hit"), on)
    assert C["mmap_dir"] == B["mmap_dir"]
    _compare(A, C, "flag off vs on (cache hit)")
    # frozen statistics: a fine-tune of A reads A's data_stats.json (and its off-shellness scale)
    ft = [f"fine_tune.pretrained_path={os.path.join(A['run_dir'], 'models', 'model_run0_best.pt')}"]
    D = _build(sweep_config, n_procs, os.path.join(out, "ft_off"), ft)
    E = _build(sweep_config, n_procs, os.path.join(out, "ft_on"), ft + on)
    assert E["mmap_dir"] != B["mmap_dir"], "the frozen statistics must key a different cache"
    _compare(D, E, "fine-tune (frozen stats) off vs on")
    print("section 2: data.memmap on and off give bitwise the same data, batches and RNG state")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-config")
    ap.add_argument("--n-procs", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    section0_reductions()
    section1_dataset_sampler()
    if a.sweep_config:
        section2_end_to_end(a.sweep_config, a.n_procs, a.out)
    print("OK")
