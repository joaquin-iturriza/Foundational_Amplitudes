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

import numpy as np

import mg5_pipeline_final as mg

# Fixed events-per-chunk policy for parallel generation. Boundaries and per-chunk
# seeds derive ONLY from this constant + the recipe (never from the worker count),
# so a dataset is bit-identical whether generated serially or across N cores, and
# whatever its recipe_id, the cache stays valid. ~100k keeps per-chunk overhead
# (driver startup + concat) under a few percent while allowing wide parallelism
# (a 1M-event set → 10 chunks; the whole prebuild's chunks share one pool).
CHUNK_SIZE = 100_000


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


def ensure_backend(process):
    """Compile the matrix-element backend for `process` if it isn't built yet.

    Run this serially (once per unique process) before parallel generation so
    that concurrent workers never race to compile the same backend."""
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
    """Split n_events into as few, as-equal chunks as the fixed policy allows.
    Boundaries depend only on n_events (+ CHUNK_SIZE), never the worker count."""
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
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
    for idx, count in enumerate(plan_chunks(n_events)):
        tasks.append({
            "process":   process,
            "sqrts_min": float(sqrts_min),
            "sqrts_max": float(sqrts_max),
            "count":     int(count),
            "seed":      _child_seed(seed, role, idx),
            "out_dir":   os.path.join(work_dir, f"c{idx:04d}"),
            "idx":       idx,
        })
    return tasks


def gen_chunk(task):
    """Worker: generate one chunk into its own temp dir. The backend must already
    be compiled (call ensure_backend first). Returns (idx, chunk_npy_path).

    Calls build_dataset_variable_energy directly (NOT generate_from_recipe) so a
    worker writes only its own chunk .npy — no .recipe.json / manifest.json
    update, which would race across the pool. The single recipe/manifest write
    happens once, serially, in finalize_dataset."""
    process        = task["process"]
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
    full["chunk_size"]     = CHUNK_SIZE
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
