#!/usr/bin/env python3
"""Pre-materialize all (process, role) datasets for a recipe spec, in parallel.

Decouples dataset generation from GPU training: run this once (as a CPU job —
see prebuild_recipes.sh) so every training/sweep trial gets a pure cache hit and
never burns GPU time generating data.

Pipeline:
  1. Skip any (process, role) already cached (matching recipe_id).
  2. Serially compile any missing matrix-element backends (avoids workers racing
     to compile the same process).
  3. Split every remaining dataset into fixed-size chunks and generate all chunks
     — across ALL datasets — in one flat pool over `--workers` CPUs, so cores
     stay saturated regardless of how the work is distributed across processes.
  4. Concatenate each dataset's chunks (in order) into the canonical .npy.

Chunking is reproducible (fixed boundaries + per-chunk seeds), so the result is
bit-identical to serial generation and independent of the worker count.

Frozen val/test land on $WORK/datasets; the train cache on $SCRATCH (override
with AMP_FROZEN_DIR / AMP_TRAIN_CACHE_DIR). Idempotent.

Usage:
    python prebuild_recipes.py recipes/pretrain8_D1e5.yaml --workers 24 [--seed 42]
"""
import argparse, os, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datagen
import mg5_pipeline_final as mg

COUNT_KEY = {"train": "n_train", "val": "n_val", "test": "n_test"}


def _gen_chunk(task):
    return datagen.gen_chunk(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1),
                    help="parallel chunk-generation workers (default: min(24, ncpu))")
    ap.add_argument("--auto-workers", action="store_true",
                    help="raise --workers up to ncpu when the cost estimate shows "
                         "more cores would shorten the prebuild (free on prepost)")
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    procs = spec["processes"] if isinstance(spec, dict) else spec

    print(f"Spec        : {args.spec}")
    print(f"Frozen dir  : {datagen.frozen_dir()}")
    print(f"Train cache : {datagen.train_cache_dir()}")
    print(f"{len(procs)} processes x 3 roles, seed={args.seed}, "
          f"target_chunk_cost={datagen.TARGET_CHUNK_COST}, workers={args.workers}\n")

    t_all = time.time()

    # 1) Plan: skip cached datasets, collect chunk tasks for the rest.
    all_tasks, datasets = [], {}   # datasets[key] = metadata for finalize
    n_cached = 0
    for role in ("val", "test", "train"):
        dest = datagen.dest_for_role(role)
        for p in procs:
            name = p["name"]
            smin, smax = float(p["sqrts"][0]), float(p["sqrts"][1])
            n_ev = int(p[COUNT_KEY[role]])
            recipe = mg.variable_energy_recipe(name, smin, smax, n_ev,
                                               role=role, seed=args.seed)
            out, rid = mg.recipe_output_path(recipe, dest), mg.recipe_id(recipe)
            if datagen._is_cached(out, rid):
                n_cached += 1
                print(f"  [cached] [{role:5s}] {name}")
                continue
            key  = (name, role)
            work = os.path.join(dest, f".chunks_{rid}")
            tasks = datagen.chunk_tasks(name, smin, smax, n_ev, role, args.seed, work)
            datasets[key] = dict(process=name, smin=smin, smax=smax, n_ev=n_ev,
                                 role=role, dest=dest, work=work, n_chunks=len(tasks))
            for t in tasks:
                t["_key"] = key
                all_tasks.append(t)

    print(f"\n{n_cached} datasets already cached; "
          f"{len(datasets)} to build ({len(all_tasks)} chunks total).")
    if not datasets:
        print(f"\nNothing to do — all cached. ({time.time()-t_all:.1f}s)")
        return

    # 1b) Cost report + load-balancing. est_cost is relative wall-time
    # (events × per-event weight, in 2→2-equivalent events); it sizes the
    # schedule, never the data. Per-process cost shows which processes dominate.
    total_cost = sum(t["est_cost"] for t in all_tasks)
    by_proc = defaultdict(lambda: [0.0, 0])   # name -> [cost, n_chunks]
    for t in all_tasks:
        by_proc[t["process"]][0] += t["est_cost"]
        by_proc[t["process"]][1] += 1
    longest = max(t["est_cost"] for t in all_tasks)
    print("\nEstimated generation cost (2→2-equivalent events; load-balancing only):")
    for name in sorted(by_proc, key=lambda n: -by_proc[n][0]):
        cost, nch = by_proc[name]
        print(f"  {name:12s} weight={datagen.process_gen_weight(name):6.1f}  "
              f"chunks={nch:4d}  cost={cost/1e6:7.3f}M  ({100*cost/total_cost:4.1f}%)")

    # Makespan is bounded below by BOTH total_cost/workers (perfect balance) and
    # the single longest chunk (indivisible). Cost-aware chunking shrinks the
    # latter; report the predicted makespan in cost-units so over/under-provisioning
    # of cores is visible. (Wall-clock = cost-units × per-2→2-event time.)
    def makespan(nw):
        return max(total_cost / nw, longest)
    ncpu = os.cpu_count() or args.workers
    if args.auto_workers and args.workers < ncpu and makespan(ncpu) < makespan(args.workers):
        print(f"\n[auto-workers] {args.workers} → {ncpu} cores "
              f"(makespan {makespan(args.workers)/1e6:.2f}M → {makespan(ncpu)/1e6:.2f}M units)")
        args.workers = ncpu
    print(f"\nTotal cost {total_cost/1e6:.2f}M units; longest chunk {longest/1e6:.3f}M; "
          f"predicted makespan @ {args.workers}w ≈ {makespan(args.workers)/1e6:.2f}M units "
          f"({100*makespan(args.workers)*args.workers/total_cost:.0f}% of cores busy avg).")

    # Longest-processing-time-first: run the most expensive chunks first so the
    # cheap ones fill the tail → near-optimal makespan, zero effect on bytes.
    all_tasks.sort(key=lambda t: -t["est_cost"])

    # 2) Compile backends serially (one per unique process to build).
    names = sorted({m["process"] for m in datasets.values()})
    print(f"Ensuring {len(names)} backends compiled (serial)...")
    for name in names:
        t0 = time.time()
        datagen.ensure_backend(name)
        print(f"  backend {name:10s} ready ({time.time()-t0:.1f}s)")

    # 3) Generate all chunks across all datasets in one flat pool.
    print(f"\nGenerating {len(all_tasks)} chunks across {args.workers} workers...")
    results = defaultdict(dict)   # key -> {idx: path}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_gen_chunk, t): t["_key"] for t in all_tasks}
        for fut in as_completed(futs):
            key = futs[fut]
            idx, path = fut.result()
            results[key][idx] = path
            done += 1
            if done % max(1, len(all_tasks) // 20) == 0 or done == len(all_tasks):
                print(f"  chunks {done}/{len(all_tasks)}")

    # 4) Concatenate each dataset's chunks (in index order) -> canonical .npy.
    print("\nFinalizing datasets...")
    for key, meta in datasets.items():
        ordered = [results[key][i] for i in sorted(results[key])]
        t0 = time.time()
        final = datagen.finalize_dataset(
            meta["process"], meta["smin"], meta["smax"], meta["n_ev"],
            meta["role"], args.seed, ordered, meta["dest"], meta["work"])
        print(f"  [{meta['role']:5s}] {meta['process']:10s} "
              f"({meta['n_chunks']} chunks) -> {final}  ({time.time()-t0:.1f}s)")

    print(f"\nBuilt {len(datasets)} datasets in {time.time()-t_all:.1f}s "
          f"({n_cached} were cached).")


if __name__ == "__main__":
    main()
