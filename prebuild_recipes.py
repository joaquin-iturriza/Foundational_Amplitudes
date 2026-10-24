#!/usr/bin/env python3
"""Pre-materialize all (process, role) datasets for a recipe spec, in parallel.

Decouples dataset generation from GPU training: run this once (as a CPU job —
see prebuild_recipes.sh) so every training/sweep trial gets a pure cache hit and
never burns GPU time generating data.

Pipeline:
  1. Skip any (process, role) already cached (matching recipe_id).
  2. Compile any missing matrix-element backends, in parallel over distinct backends (no worker races:
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
import argparse, os, shutil, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datagen
import mg5_pipeline_final as mg

COUNT_KEY = {"train": "n_train", "val": "n_val", "test": "n_test"}



def _cores_available():
    """Cores this process may actually run on: the SLURM allocation, else the affinity
    mask. os.cpu_count() is the whole node (512 logical CPUs on the htc nodes) and once
    scaled a 48-core prebuild to 432 simultaneous MadGraph outputs, which died on fork."""
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1

def _gen_chunk(task):
    """One chunk, retried on a transient OS failure (EAGAIN on fork when the node's
    process limit is momentarily hit) so a single hiccup does not abort a pool that has
    hours of finished chunks in flight."""
    import time
    for attempt in range(4):
        try:
            return datagen.gen_chunk(task)
        except OSError as e:
            if attempt == 3:
                raise
            print(f"[RETRY] chunk {task['process']} {task.get('role')} {task['idx']}: {e} (attempt {attempt+1})", flush=True)
            time.sleep(5.0 * (attempt + 1))


def _run_lhe_jobs(lhe_jobs, args):
    """lhe_pp processes (catalog_daniel): one MadEvent run per process, workers // lhe_cores of them
    at once, costliest first (2->3 before 2->2, more diagrams first). A failed process is reported
    and skipped, never aborts the rest; the run is resumable (built pools are cache hits next time).
    Sidecars + manifest are written here, serially. Returns the failed process names."""
    lhe_cores = max(1, args.lhe_cores)
    nw = max(1, args.workers // lhe_cores)
    order = sorted(lhe_jobs, key=lambda n: (-int(mg.PROCESSES[n]["nfinal"]),
                                            -int(mg.PROCESSES[n].get("n_diagrams", 0)), n))
    print(f"\nMadEvent: {len(order)} lhe_pp processes, {nw} at once x {lhe_cores} core(s); "
          f"work dir {mg.lhe_work_dir()}", flush=True)
    failed, t0 = [], time.time()
    jobs = [{"process": n, "items": lhe_jobs[n], "mg_cores": lhe_cores} for n in order]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = [ex.submit(datagen.build_lhe_pp_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            name, res, err, dt = fut.result()
            if err:
                failed.append(name)
                print(f"  [LHE FAIL] {name} ({dt:.0f}s): {err}", flush=True)
                continue
            for out, full in res:
                mg.write_recipe(out, full)
            print(f"  [LHE {i}/{len(jobs)}] {name}: {len(res)} pools in {dt:.0f}s "
                  f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)
    print(f"MadEvent phase: {len(jobs)-len(failed)} ok, {len(failed)} failed"
          + (f" ({' '.join(failed)})" if failed else "") + f", {(time.time()-t0)/60:.1f} min", flush=True)
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="space to leave free on the destination after the build (a shared filesystem)")
    ap.add_argument("--workers", type=int, default=min(24, _cores_available()),
                    help="parallel chunk-generation workers (default: min(24, ncpu))")
    ap.add_argument("--lhe-cores", type=int, default=int(os.environ.get("AMP_LHE_CORES", "1")),
                    help="MadEvent cores per lhe_pp process (catalog_daniel); the pool then runs "
                         "workers // lhe-cores processes at once (default 1: run_mode 0)")
    ap.add_argument("--auto-workers", action="store_true",
                    help="raise --workers up to ncpu when the cost estimate shows "
                         "more cores would shorten the prebuild (free on prepost)")
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    procs = spec["processes"] if isinstance(spec, dict) else spec

    # Register any per-dataset physics scans / decorated-base entries into
    # mg.PROCESSES so generation can address them by their (decorated) dataset name
    # — same registration the training recipe path does in init_physics.
    mg.register_recipe_processes(procs, default_sampling=(spec.get("sampling") if isinstance(spec, dict) else None))

    print(f"Spec        : {args.spec}")
    print(f"Frozen dir  : {datagen.frozen_dir()}")
    print(f"Train cache : {datagen.train_cache_dir()}")
    print(f"{len(procs)} processes x 3 roles, seed={args.seed}, "
          f"target_chunk_cost={datagen.TARGET_CHUNK_COST}, workers={args.workers}\n")

    t_all = time.time()

    # 1) Plan: skip cached datasets, collect chunk tasks for the rest. NLO α_s-scan
    # datasets are NOT generated by MadLoop per point — they reweight from their
    # stripped base (one MadLoop run feeds all α_s points), so we plan the base once
    # and reweight the children in a cheap post-pass (4).
    all_tasks, datasets = [], {}   # datasets[key] = metadata for finalize
    reweight_jobs = []             # (child_name, child_out, parent_out, role, child_recipe)
    lhe_jobs = {}                  # lhe_pp process -> [(recipe, out)]: one MadEvent run each
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
            if mg.is_lhe_pp(name):
                lhe_jobs.setdefault(name, []).append((recipe, out))
                continue
            if datagen.is_reweightable_virt_scan(name):
                # plan the stripped parent (dedup across this base's α_s points)
                parent = mg.PROCESSES[name]["scan_base"]
                precipe = mg.variable_energy_recipe(parent, smin, smax, n_ev,
                                                    role=role, seed=args.seed)
                pout, prid = mg.recipe_output_path(precipe, dest), mg.recipe_id(precipe)
                pkey = (parent, role)
                if not datagen._is_cached(pout, prid) and pkey not in datasets:
                    pwork  = os.path.join(dest, f".chunks_{prid}")
                    ptasks = datagen.chunk_tasks(parent, smin, smax, n_ev, role, args.seed, pwork)
                    datasets[pkey] = dict(process=parent, smin=smin, smax=smax, n_ev=n_ev,
                                          role=role, dest=dest, work=pwork, n_chunks=len(ptasks))
                    for t in ptasks:
                        t["_key"] = pkey
                        all_tasks.append(t)
                reweight_jobs.append((name, out, pout, role, recipe))
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
          f"{len(datasets)} to build ({len(all_tasks)} chunks total); "
          f"{len(reweight_jobs)} NLO α_s datasets reweighted from "
          f"{len({j[2] for j in reweight_jobs})} stripped parents; "
          f"{len(lhe_jobs)} lhe_pp processes ({sum(map(len, lhe_jobs.values()))} pools) from MadEvent.")
    lhe_failed = _run_lhe_jobs(lhe_jobs, args) if lhe_jobs else []
    if not datasets and not reweight_jobs:
        if lhe_failed:
            sys.exit(1)
        print(f"\nNothing else to do{' (lhe_pp built above)' if lhe_jobs else ' — all cached'}. "
              f"({time.time()-t_all:.1f}s)")
        return

    if not datasets:
        print("\nAll parents cached — only reweighting to do.")

    # Generation (1b cost report → 2 compile → 3 pool → 4 finalize) runs only when
    # there is something to MadLoop/cpp-generate; with all stripped parents cached
    # only the reweight post-pass below executes.
    if all_tasks:
        # 1b) Cost report + load-balancing. est_cost is relative wall-time
        # (events × per-event weight, in 2→2-equivalent events); it sizes the
        # schedule, never the data.
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
        # the single longest chunk (indivisible).
        def makespan(nw):
            return max(total_cost / nw, longest)
        ncpu = _cores_available()   # the cores THIS job may use, not the node's
        if args.auto_workers and args.workers < ncpu and makespan(ncpu) < makespan(args.workers):
            print(f"\n[auto-workers] {args.workers} → {ncpu} cores "
                  f"(makespan {makespan(args.workers)/1e6:.2f}M → {makespan(ncpu)/1e6:.2f}M units)")
            args.workers = ncpu
        print(f"\nTotal cost {total_cost/1e6:.2f}M units; longest chunk {longest/1e6:.3f}M; "
              f"predicted makespan @ {args.workers}w ≈ {makespan(args.workers)/1e6:.2f}M units "
              f"({100*makespan(args.workers)*args.workers/total_cost:.0f}% of cores busy avg).")

        # Longest-processing-time-first so the cheap chunks fill the tail.
        all_tasks.sort(key=lambda t: -t["est_cost"])

        # 1c) Free space: the datasets still to build, float64 rows of 5 N + 1 columns, against the free space of the
        # destination filesystem, with the 1 MB-block overhead of the chunk files (CC's /sps) and one dataset in two
        # copies at assembly. Refused when it does not fit, before anything is written (2026-10-24, /sps/lpnhe full).
        # Chunks already on disk (a resumed run) count as written; assembly then needs only the largest dataset twice.
        need, peak = 0, 0
        for meta in datasets.values():
            nf = int(mg.PROCESSES.get(meta["process"], {}).get("nfinal", 2))
            size = meta["n_ev"] * (5 * (nf + 2) + 1) * 8
            have = sum(os.path.getsize(os.path.join(r, fn)) for r, _, fns in os.walk(meta["work"]) for fn in fns
                       if fn == "chunk.npy") if os.path.isdir(meta["work"]) else 0
            n_have = sum(1 for r, _, fns in os.walk(meta["work"]) for fn in fns if fn == "chunk.npy") \
                if os.path.isdir(meta["work"]) else 0
            need += max(0, size - have) + max(0, meta["n_chunks"] - n_have) * (1 << 20)
            peak = max(peak, size)
        need += peak
        dests = {meta["dest"] for meta in datasets.values()}
        for d in dests:
            os.makedirs(d, exist_ok=True)
            free = shutil.disk_usage(d).free
            print(f"  space: {need / 1e9:.1f} GB to write (chunk overhead included), {free / 1e9:.1f} GB free on {d}")
            if free < need * 1.1 + args.min_free_gb * 1e9:
                sys.exit(f"REFUSED: {need / 1e9:.1f} GB to write and {free / 1e9:.1f} GB free on {d} "
                         f"(margin 10% + --min-free-gb {args.min_free_gb}); free space first")

        # 2) Compile backends, in parallel over distinct backends (each in its own dir).
        names = sorted({m["process"] for m in datasets.values()})
        datagen.ensure_backends(names, workers=args.workers)

        # 3) Generate all chunks across all datasets in one flat pool.
        print(f"\nGenerating {len(all_tasks)} chunks across {args.workers} workers...")
        results = defaultdict(dict)   # key -> {idx: path}
        done = 0

        # 4) Each dataset is concatenated into its canonical .npy (chunks in index order) the moment its last chunk
        # lands, and its chunk dir removed then: the disk holds one dataset twice at a time, not the whole recipe
        # (2026-10-24: catalog_v3's stage 1 filled /sps/lpnhe with 47808 chunks before a single pool was assembled).
        def finalize(key):
            meta = datasets[key]
            ordered = [results[key][i] for i in sorted(results[key])]
            t0 = time.time()
            final = datagen.finalize_dataset(
                meta["process"], meta["smin"], meta["smax"], meta["n_ev"],
                meta["role"], args.seed, ordered, meta["dest"], meta["work"])
            print(f"  [{meta['role']:5s}] {meta['process']:10s} "
                  f"({meta['n_chunks']} chunks) -> {final}  ({time.time()-t0:.1f}s)", flush=True)

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_gen_chunk, t): t["_key"] for t in all_tasks}
            for fut in as_completed(futs):
                key = futs[fut]
                idx, path = fut.result()
                results[key][idx] = path
                done += 1
                if done % max(1, len(all_tasks) // 20) == 0 or done == len(all_tasks):
                    print(f"  chunks {done}/{len(all_tasks)}")
                if len(results[key]) == datasets[key]["n_chunks"]:
                    finalize(key)

    # 5) Reweight NLO α_s-scan datasets from their (now materialized) stripped
    # parents — cheap numpy, no MadLoop. Each parent feeds all its α_s points.
    if reweight_jobs:
        print(f"\nReweighting {len(reweight_jobs)} NLO α_s datasets from stripped parents...")
        for (name, out, pout, role, recipe) in reweight_jobs:
            datagen.reweight_virt_alphas(pout, name, out)
            full = dict(recipe)
            full["derived_from"] = os.path.basename(pout)
            mg.write_recipe(out, full)
        print(f"  reweighted {len(reweight_jobs)} datasets.")

    print(f"\nBuilt {len(datasets)} datasets + reweighted {len(reweight_jobs)} "
          f"in {time.time()-t_all:.1f}s ({n_cached} were cached).")
    if lhe_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
