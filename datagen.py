"""
On-demand amplitude dataset materialization for training.

Bridges the experiment to the MG5 generator (`mg5_pipeline_final`): given a
per-process spec + split role, returns a path to a ready `.npy`, generating it
reproducibly from a recipe if it is not already present.

Storage policy (Jean Zay):
  - val / test : FROZEN, persistent on $WORK/datasets. Generated once and reused
                 forever — the benchmark numbers must be stable across every run.
  - train      : ephemeral cache on $SCRATCH. Reused across a sweep's trials via
                 the recipe_id (identical train recipe ⇒ generated at most once);
                 regenerated only when the recipe changes. $SCRATCH is auto-purged.

Override the directories with env vars AMP_FROZEN_DIR / AMP_TRAIN_CACHE_DIR.
"""
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

import mg5_pipeline_final as mg

# NLO-virtual generation lives in tools/nlo_virtual_pipeline.py and goes through
# MadLoop (matrix2py.so), which chdir's and is a process-singleton — so a virt
# chunk is generated in its own subprocess rather than in a shared pool worker.
_NLO_PIPELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "tools", "nlo_virtual_pipeline.py")


def is_virt(process):
    """True if `process` is an NLO-virtual process routed through MadLoop."""
    return mg.PROCESSES.get(process, {}).get("kind") == "virt"

# Cost-aware events-per-chunk policy for parallel generation.
#
# Chunk boundaries and per-chunk seeds derive ONLY from (recipe, process cost) —
# never from the worker count — so a dataset is bit-identical whether generated
# serially or across N cores. The resolved per-process chunk_size is folded into
# the recipe and into recipe_id (see mg5_pipeline_final.variable_energy_recipe /
# recipe_id): a different chunk policy yields a different id, so cost-aware data
# never silently shares an id with the old fixed-100k bytes.
#
# Why cost-aware: per-event generation cost grows steeply with final-state
# multiplicity (a 2→4 point is ~tens of times a 2→2). With a *fixed* events-per-
# chunk size, an expensive process becomes one fat, indivisible work unit: the
# whole prebuild then waits on that single slow chunk while every other core sits
# idle. Sizing each process's chunks for roughly EQUAL WALL-TIME instead spreads
# the expensive process across many small chunks (fine-grained parallelism, short
# tail) while leaving cheap processes coarse (low per-chunk overhead). What you
# ask of each process is "how long it takes", not "how many events".
#
# MAX_CHUNK is the old fixed size (cheap 2→2 keeps ~100k-event chunks → unchanged
# bytes). TARGET_CHUNK_COST is one chunk's worth of work in 2→2-equivalent events;
# an expensive process gets chunk_size = TARGET / gen_weight (clamped to
# [MIN_CHUNK, MAX_CHUNK]). The cost model lives in mg5_pipeline_final (next to
# PROCESSES + the recipe builders, since chunk_size is identity-bearing); these
# names re-export it for the generation call sites here.
MAX_CHUNK          = mg.MAX_CHUNK
MIN_CHUNK          = mg.MIN_CHUNK
TARGET_CHUNK_COST  = mg.TARGET_CHUNK_COST
CHUNK_SIZE         = mg.MAX_CHUNK     # back-compat alias (cheap-process chunk size)
process_gen_weight = mg.process_gen_weight
process_chunk_size = mg.process_chunk_size


def frozen_dir():
    """Persistent home for frozen val/test datasets."""
    d = os.environ.get("AMP_FROZEN_DIR")
    if d:
        return d
    work = os.environ.get("WORK")
    return f"{work}/datasets" if work else mg.OUTPUT_DIR


def train_cache_dir():
    """Purgeable cache for on-demand train datasets (shared across a sweep)."""
    d = os.environ.get("AMP_TRAIN_CACHE_DIR")
    if d:
        return d
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return f"{scratch}/amp_data_cache"
    job = os.environ.get("JOBSCRATCH")
    if job:
        return f"{job}/amp_data_cache"
    return frozen_dir()


def dest_for_role(role):
    return frozen_dir() if role in ("val", "test") else train_cache_dir()


def _virt_standalone_dir(process):
    base = mg.PROCESSES[process]["virt_base"]
    return f"{mg.WORK_DIR}/{base}_virt_standalone"


def ensure_virt_backend(process):
    """Build the [virt=QCD] MadLoop standalone (matrix2py.so) for an NLO-virtual
    process if absent, then pole-certify it. Idempotent; run serially before the
    pool, like the LO backends."""
    base = mg.PROCESSES[process]["virt_base"]
    sa   = _virt_standalone_dir(process)
    import glob
    if glob.glob(f"{sa}/SubProcesses/P0_*/matrix2py*.so"):
        return sa
    env = dict(os.environ, SETUPTOOLS_USE_DISTUTILS="stdlib")
    subprocess.run([sys.executable, _NLO_PIPELINE, base, "--build", "--certify"],
                   check=True, env=env)
    return sa


def ensure_backend(process):
    """Compile the matrix-element backend for `process` if it isn't built yet.

    Run this serially (once per unique process) before parallel generation so
    that concurrent workers never race to compile the same backend."""
    if is_virt(process):
        return ensure_virt_backend(process)
    cfg = dict(mg.PROCESSES[process])
    standalone_dir = f"{mg.WORK_DIR}/{process}_standalone"
    backend, subproc_dirs, driver_bin, _ = mg.detect_compiled_backend(standalone_dir)
    if not subproc_dirs or (backend == "cpp" and driver_bin is None):
        mg.generate_mg5_process(process, cfg)
        mg.compile_backends(standalone_dir, cfg["nfinal"] + 2)
    return standalone_dir


def _is_cached(out_path, wanted_id):
    """True if `out_path` already holds the dataset for recipe_id `wanted_id`."""
    try:
        with open(out_path + ".recipe.json") as f:
            return json.load(f).get("recipe_id") == wanted_id
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def plan_chunks(n_events, chunk_size=None):
    """Split n_events into as few, as-equal chunks as the policy allows.
    Boundaries depend only on (n_events, chunk_size), never the worker count.
    Pass a per-process `chunk_size` (from process_chunk_size) for cost-aware
    granularity; defaults to the cheap-process MAX_CHUNK."""
    if chunk_size is None:
        chunk_size = MAX_CHUNK
    nch = max(1, math.ceil(n_events / chunk_size))
    base, rem = divmod(n_events, nch)
    return [base + (1 if i < rem else 0) for i in range(nch)]


def _child_seed(base_seed, role, idx):
    """Deterministic per-chunk seed from (base_seed, role, chunk index)."""
    off = mg.ROLE_SEED_OFFSET.get(role, 0)
    return int(np.random.SeedSequence([int(base_seed), int(off), int(idx)])
               .generate_state(1, dtype=np.uint32)[0])


def chunk_tasks(process, sqrts_min, sqrts_max, n_events, role, seed, work_dir):
    """Independent chunk work-units for one (process, role) dataset. Each is a
    self-contained dict that gen_chunk() can run in any worker / any order."""
    tasks = []
    weight     = process_gen_weight(process)
    chunk_size = process_chunk_size(process)
    for idx, count in enumerate(plan_chunks(n_events, chunk_size)):
        tasks.append({
            "process":   process,
            "sqrts_min": float(sqrts_min),
            "sqrts_max": float(sqrts_max),
            "count":     int(count),
            "seed":      _child_seed(seed, role, idx),
            "out_dir":   os.path.join(work_dir, f"c{idx:04d}"),
            "idx":       idx,
            # est_cost: relative wall-time of this chunk (events × per-event
            # weight), in 2→2-equivalent events. For LPT scheduling + reporting
            # only — never touches the data.
            "est_cost":  float(count) * weight,
        })
    return tasks


def gen_virt_chunk(task):
    """Worker: generate one NLO-virtual chunk in an isolated subprocess (MadLoop's
    matrix2py.so chdir's and is a process-singleton, so it can't be loaded in a
    shared pool worker). Deterministic via the per-chunk seed. Returns (idx, path)."""
    process = task["process"]
    base    = mg.PROCESSES[process]["virt_base"]
    os.makedirs(task["out_dir"], exist_ok=True)
    out_path = os.path.join(task["out_dir"], "chunk.npy")
    env = dict(os.environ, SETUPTOOLS_USE_DISTUTILS="stdlib")
    cmd = [sys.executable, _NLO_PIPELINE, base, "--generate",
           "--n", str(task["count"]), "--seed", str(task["seed"]),
           "--sqrts-min", repr(task["sqrts_min"]), "--sqrts-max", repr(task["sqrts_max"]),
           "--out", out_path]
    subprocess.run(cmd, check=True, env=env)
    return task["idx"], out_path


def gen_chunk(task):
    """Worker: generate one chunk into its own temp dir. The backend must already
    be compiled (call ensure_backend first). Returns (idx, chunk_npy_path).

    Calls build_dataset_variable_energy directly (NOT generate_from_recipe) so a
    worker writes only its own chunk .npy — no .recipe.json / manifest.json
    update, which would race across the pool. The single recipe/manifest write
    happens once, serially, in finalize_dataset."""
    process        = task["process"]
    if is_virt(process):
        return gen_virt_chunk(task)
    cfg            = dict(mg.PROCESSES[process])
    standalone_dir = f"{mg.WORK_DIR}/{process}_standalone"
    backend, subproc_dirs, driver_bin, eff_dir = mg.detect_compiled_backend(standalone_dir)
    if not subproc_dirs or (backend == "cpp" and driver_bin is None):
        raise RuntimeError(
            f"gen_chunk: backend for {process} not compiled — call ensure_backend first.")
    os.makedirs(task["out_dir"], exist_ok=True)
    out_path = os.path.join(task["out_dir"], "chunk.npy")
    rng = np.random.default_rng(task["seed"])
    mg.build_dataset_variable_energy(
        task["count"], task["sqrts_min"], task["sqrts_max"],
        eff_dir, backend, subproc_dirs, driver_bin, cfg, out_path, rng=rng)
    return task["idx"], out_path


def finalize_dataset(process, sqrts_min, sqrts_max, n_events, role, seed,
                     ordered_chunk_paths, dest_dir, work_dir):
    """Concatenate chunk files (in index order) into the canonical dataset +
    .recipe.json (written with the ORIGINAL recipe, so recipe_id matches the
    non-chunked path and the cache key is unchanged). Removes the temp work dir."""
    rec   = mg.variable_energy_recipe(process, sqrts_min, sqrts_max, n_events,
                                      role=role, seed=seed)
    final = mg.recipe_output_path(rec, dest_dir)
    os.makedirs(os.path.dirname(final) or ".", exist_ok=True)
    data = np.concatenate([np.load(p) for p in ordered_chunk_paths], axis=0)
    tmp  = final + ".tmp.npy"
    np.save(tmp, data)
    os.replace(tmp, final)
    full = dict(rec)
    full["effective_seed"] = seed + mg.ROLE_SEED_OFFSET.get(role, 0)
    full["chunked"]        = True
    # n_chunks (in `rec` when non-legacy) carries the identity; record the nominal
    # chunk_size + gen_weight as excluded traceability metadata.
    full["chunk_size"]     = process_chunk_size(process)
    full["gen_weight"]     = process_gen_weight(process)
    mg.write_recipe(final, full)
    shutil.rmtree(work_dir, ignore_errors=True)
    return final


def ensure_dataset(process, sqrts_min, sqrts_max, n_events, role, seed,
                   dest_dir=None, require_cache=False):
    """Path to the (process, role) dataset, generated reproducibly if absent
    (cache hit otherwise). `role` in {'train','val','test'}.

    Generation goes through the chunked path (serially here, in parallel in the
    prebuild) — same bytes either way. If `require_cache` is set, a missing/stale
    dataset raises instead of being generated, so GPU training jobs never burn GPU
    time materializing data (run the prebuild job first)."""
    if dest_dir is None:
        dest_dir = dest_for_role(role)
    recipe = mg.variable_energy_recipe(
        process, sqrts_min, sqrts_max, n_events, role=role, seed=seed)
    out    = mg.recipe_output_path(recipe, dest_dir)
    wanted = mg.recipe_id(recipe)

    if _is_cached(out, wanted):
        return out
    if require_cache:
        raise RuntimeError(
            f"require_cache: dataset for ({process}, {role}, {n_events} ev) "
            f"not prebuilt at {out} (recipe_id={wanted}). Run the prebuild "
            f"job first: sbatch prebuild_recipes.sh <spec.yaml>.")

    ensure_backend(process)
    work    = os.path.join(dest_dir, f".chunks_{wanted}")
    results = dict(gen_chunk(t) for t in
                   chunk_tasks(process, sqrts_min, sqrts_max, n_events, role, seed, work))
    ordered = [results[i] for i in sorted(results)]
    return finalize_dataset(process, sqrts_min, sqrts_max, n_events, role, seed,
                            ordered, dest_dir, work)


def ensure_split_set(specs, role, seed, dest_dir=None, require_cache=False):
    """Materialize a list of process specs for one role.

    `specs`: iterable of dicts {process, sqrts_min, sqrts_max, n_events}.
    Returns {process: npy_path}.
    """
    return {
        s["process"]: ensure_dataset(
            s["process"], s["sqrts_min"], s["sqrts_max"], s["n_events"],
            role=role, seed=seed, dest_dir=dest_dir, require_cache=require_cache)
        for s in specs
    }
