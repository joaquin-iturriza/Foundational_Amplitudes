#!/usr/bin/env python3
"""
test_worker_latency.py — Benchmark DataLoader step latency with old vs new _n_batches formula.

Replicates the exact DataLoader+sampler config used in training (num_workers=4,
persistent_workers=True, drop_last=True, ProcessBalancedSampler) but with a synthetic
dataset, isolating the overhead from DataLoader epoch restarts.

Root cause under test
---------------------
With the OLD formula:
    _n_batches = max(1, min_proc_size * n_processes // batch_size)
...an imbalanced dataset (e.g. n_uu_train=200, n_aa_train=8000, n_aaa_train=8000)
gives min_proc_size=200 → 200*3//512=1 → _n_batches=1.  Each DataLoader "epoch"
contains exactly one batch.  _cycle() calls iter(loader) every step, leaving the
4 persistent workers idle between steps.  On a busy cluster the OS scheduling
wakeup latency is 150-400 ms and is counted in avg_iter_time via next(iterator).

With the NEW formula:
    _n_batches = max(100, total_proc_size // batch_size)
...workers are kept continuously busy and the restart overhead is amortised over
≥100 steps.

Run from lxplus with the amplitudes env active (no GPU needed):
    python sweep/test_worker_latency.py

Expected output: OLD formula is visibly slower (median >100 ms/step) on an
imbalanced config; NEW formula reduces this to ~1-5 ms/step.
"""

import sys
import os
import time
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

# ── Import ProcessBalancedSampler from parent directory ──────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from dataset import ProcessBalancedSampler

# ── Benchmark parameters ─────────────────────────────────────────────────────
N_STEPS  = 300   # steps to time
N_WARMUP = 30    # steps to discard (DataLoader worker startup / JIT)
NUM_WORKERS = 4  # must match training config

# ── Dataset configs from the sweep ───────────────────────────────────────────
#
# For each config we compute the training-split sizes (40% of n_data per dataset).
# batch_size is min(512, n_train_total // 2) as in experiment.py.
#
# SLOW configs trigger n_batches=1 via the old formula:
#   min_proc_size * n_processes // batch_size  = 200*3//512 = 1
#
# FAST configs have n_batches > 1 even with the old formula.
#
CONFIGS = [
    {
        "label":   "imbalanced  nd500_20000_20000",
        "n_data":  [500, 20000, 20000],        # per dataset
    },
    {
        "label":   "balanced    nd5000_20000_20000",
        "n_data":  [5000, 20000, 20000],
    },
    {
        "label":   "all-small   nd500_300_300",
        "n_data":  [500, 300, 300],
    },
    {
        "label":   "large       nd500000_20000_20000",
        "n_data":  [500000, 20000, 20000],
    },
]

TRAIN_FRAC = 0.4   # matches data.train_test_val[0] = 0.4
BATCH_SIZE_DEFAULT = 512


def _n_batches_old(proc_sizes, batch_size):
    return max(1, min(proc_sizes) * len(proc_sizes) // batch_size)


def _n_batches_new(proc_sizes, batch_size):
    return max(100, sum(proc_sizes) // batch_size)


def _make_sampler(n_per_proc, batch_size, n_batches_override):
    total = sum(n_per_proc)
    proc_ids = np.concatenate([
        np.full(n, i, dtype=np.int64) for i, n in enumerate(n_per_proc)
    ])
    sampler = ProcessBalancedSampler(proc_ids, batch_size=batch_size, seed=0)
    sampler._n_batches = n_batches_override
    return sampler


def _make_loader(total_n, sampler):
    # Synthetic dataset: one float per event (content doesn't matter — we only
    # measure DataLoader iteration overhead, not forward-pass time)
    ds = TensorDataset(torch.zeros(total_n, 1))
    return DataLoader(
        ds,
        batch_size         = sampler.batch_size,
        sampler            = sampler,
        drop_last          = True,
        num_workers        = NUM_WORKERS,
        persistent_workers = NUM_WORKERS > 0,
        pin_memory         = False,
    )


def _cycle(iterable):
    """Identical to base_experiment.BaseExperiment._cycle."""
    while True:
        for x in iterable:
            yield x


def _benchmark(loader, n_steps, n_warmup):
    it = _cycle(loader)
    times = []
    for step in range(n_steps + n_warmup):
        t0 = time.perf_counter()
        _ = next(it)
        elapsed = time.perf_counter() - t0
        if step >= n_warmup:
            times.append(elapsed)
    return np.array(times)


def _summarise(label, times):
    ms = times * 1000
    print(f"    {label:6s}  mean={np.mean(ms):6.1f} ms  "
          f"median={np.median(ms):6.1f} ms  "
          f"p95={np.percentile(ms, 95):6.1f} ms  "
          f"p99={np.percentile(ms, 99):6.1f} ms")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"DataLoader worker-latency benchmark")
    print(f"  num_workers={NUM_WORKERS}  persistent_workers=True  n_steps={N_STEPS}  warmup={N_WARMUP}")
    print()

    for cfg in CONFIGS:
        n_data     = cfg["n_data"]
        n_per_proc = [max(2, int(n * TRAIN_FRAC)) for n in n_data]

        n_train_total = sum(n_per_proc)
        batch_size    = min(BATCH_SIZE_DEFAULT, n_train_total // 2)

        nb_old = _n_batches_old(n_per_proc, batch_size)
        nb_new = _n_batches_new(n_per_proc, batch_size)

        print(f"{'─'*65}")
        print(f"  {cfg['label']}")
        print(f"  n_train per proc : {n_per_proc}  total={n_train_total}")
        print(f"  batch_size       : {batch_size}")
        print(f"  _n_batches  OLD  : {nb_old}")
        print(f"  _n_batches  NEW  : {nb_new}")

        if nb_old == nb_new:
            print("  (formulas agree — only running once)")
            sampler = _make_sampler(n_per_proc, batch_size, nb_old)
            loader  = _make_loader(n_train_total, sampler)
            times   = _benchmark(loader, N_STEPS, N_WARMUP)
            _summarise("BOTH", times)
        else:
            # OLD
            sampler_old = _make_sampler(n_per_proc, batch_size, nb_old)
            loader_old  = _make_loader(n_train_total, sampler_old)
            print(f"  Running OLD...", flush=True)
            times_old   = _benchmark(loader_old, N_STEPS, N_WARMUP)
            _summarise("OLD", times_old)

            # NEW
            sampler_new = _make_sampler(n_per_proc, batch_size, nb_new)
            loader_new  = _make_loader(n_train_total, sampler_new)
            print(f"  Running NEW...", flush=True)
            times_new   = _benchmark(loader_new, N_STEPS, N_WARMUP)
            _summarise("NEW", times_new)

            speedup = np.mean(times_old) / np.mean(times_new)
            print(f"  Speedup: {speedup:.2f}x")

        print()


if __name__ == "__main__":
    main()
