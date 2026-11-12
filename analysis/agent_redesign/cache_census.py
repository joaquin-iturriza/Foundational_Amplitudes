#!/usr/bin/env python3
"""What the CC data caches hold (Joaquin 2026-11-12: 'how is it possible that we need more than 100 GB of data
caches'). Read-only, on the site: every file of $SCRATCH/amp_data_cache (the train pools and their sidecars) and of
$SCRATCH/amp_mmap_cache (one directory per training-set key), against $DATA_DIR (the frozen val/test pools).
Each data-cache file is put in one reader class, the first that reads it: a recipe named by an open plan entry
(plan/plan.yaml, status not done), another catalog_v4 recipe, an older recipe (catalog_v3, v2, Daniel's, ...), or
no recipe in recipes/*.yaml. Which files a recipe reads is tools/check_site_data.py --paths (the pool, its
.recipe.json, and a default-share mixture's component pools), run in this process once per recipe so the imports
are paid once; a recipe that does not load is listed under recipes_failed. Duplicates are counted by --quick: pool
files whose .recipe.json carry the same recipe id (same content); one process and role in several energy windows or
samplings are variants, not copies.
    python analysis/agent_redesign/cache_census.py --out <json>
One JSON line (also written to --out): GB by prefix and reader class, mmap keys, datasets."""
import contextlib, copy, glob, io, json, os, re, runpy, sys
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, ROOT)
from lhe_parts import _du as du  # noqa: E402
S, D = os.environ["SCRATCH"], os.environ["DATA_DIR"]
POOL = re.compile(r"^(?P<name>.+?)_(?P<lo>[\d.]+)-(?P<hi>[\d.]+)GeV_(?P<role>train|val|test)_(?P<tag>.*?)amplitudes\.npy$")
_REG = None
CLASSES = ("open plan entry", "other catalog_v4 recipe", "older recipe only", "no recipe")


def open_recipes():
    import yaml
    es = yaml.safe_load(open(os.path.join(ROOT, "plan", "plan.yaml")))["entries"]
    es = es if isinstance(es, list) else list(es.values())
    return {os.path.basename(r) for e in es if e.get("status") != "done"
            for r in re.findall(r"recipes/[\w./-]+?\.yaml", json.dumps(e))}


def recipe_files(spec):
    """The files check_site_data --paths prints for one recipe, run in this process."""
    import mg5_pipeline_final as mg
    global _REG
    if _REG is None:
        _REG = copy.deepcopy(mg.PROCESSES)
    mg.PROCESSES.clear(); mg.PROCESSES.update(copy.deepcopy(_REG))   # no recipe's sampling carries into the next
    buf = io.StringIO()
    sys.argv = ["check_site_data.py", "--paths", spec]
    with contextlib.redirect_stdout(buf):
        try:
            runpy.run_path(os.path.join(ROOT, "tools", "check_site_data.py"), run_name="__main__")
        except SystemExit:
            pass
    return {l.split(" ", 1)[1] for l in buf.getvalue().splitlines() if l.startswith("POOL ")}


def quick():
    """--quick, no recipe pass: the chunk work dirs (.chunks_<recipe id>, left by a build that never finalized: debris
    once a pool of that recipe id is in the cache), set_aside/, the duplicates (pool files sharing one recipe id: GB
    beyond the first file of each id), and the variants (one process and role in several windows or samplings)."""
    cache = os.path.join(S, "amp_data_cache")
    rid = {}             # pool file -> its recipe id
    for f in os.listdir(cache):
        if f.endswith(".npy.recipe.json"):
            try:
                rid[f[:-len(".recipe.json")]] = json.load(open(os.path.join(cache, f))).get("recipe_id")
            except (OSError, ValueError):
                pass
    built = set(rid.values())
    ch = [(d, du(os.path.join(cache, d))) for d in os.listdir(cache) if d.startswith(".chunks_")]
    done = [(d, s) for d, s in ch if d[len(".chunks_"):] in built]
    by_rid, variants, by_tag = defaultdict(list), defaultdict(int), defaultdict(float)
    for f in os.listdir(cache):
        m = POOL.match(f)
        if m:
            sz = os.lstat(os.path.join(cache, f)).st_size
            if rid.get(f):
                by_rid[rid[f]].append(sz)
            variants[(m["name"], m["role"])] += 1
            by_tag[m["tag"].rstrip("_") or "flat"] += sz
    dup = {k: v for k, v in by_rid.items() if len(v) > 1}
    return {"chunk_dirs": len(ch), "chunk_dirs_gb": round(sum(s for _, s in ch) / 1e9, 2),
            "chunk_dirs_pool_built": len(done), "chunk_dirs_pool_built_gb": round(sum(s for _, s in done) / 1e9, 2),
            "set_aside_gb": round(du(os.path.join(cache, "set_aside")) / 1e9, 2),
            "recipe_ids_in_several_files": len(dup), "duplicate_gb": round(sum(sum(v[1:]) for v in dup.values()) / 1e9, 2),
            "pool_files_without_recipe_json": sum(1 for f in os.listdir(cache) if POOL.match(f) and f not in rid),
            "process_roles_with_variants": sum(n > 1 for n in variants.values()),
            "pool_gb_by_sampling_tag": {k: round(v / 1e9, 2) for k, v in sorted(by_tag.items(), key=lambda x: -x[1])}}


def main():
    out = sys.argv[sys.argv.index("--out") + 1]
    if "--quick" in sys.argv:
        res = quick()
        json.dump(res, open(out, "w"), indent=1); print(json.dumps(res)); return
    os.chdir(ROOT)
    opened = open_recipes()
    reader = {}          # file -> best (lowest) class index over the recipes that read it
    failed = []
    for r in sorted(glob.glob(os.path.join(ROOT, "recipes", "*.yaml"))):
        b = os.path.basename(r)
        cls = 0 if b in opened else 1 if b.startswith("catalog_v4") else 2
        try:
            fs = recipe_files(os.path.relpath(r, ROOT))
        except Exception as e:   # a recipe that does not load (old schema, missing process): not a reader
            failed.append(f"{b}: {type(e).__name__}"); continue
        for f in fs:
            reader[f] = min(reader.get(f, 3), cls)
        print(f"CENSUS_RECIPE {b} {len(fs)}", file=sys.stderr, flush=True)
    cache = os.path.join(S, "amp_data_cache")
    g = defaultdict(float)
    by_key = defaultdict(list)
    n_files = 0
    for f in os.listdir(cache):
        full = os.path.join(cache, f)
        sz = du(full) if os.path.isdir(full) else os.lstat(full).st_size
        n_files += 1
        base = f
        for suf in (".recipe.json", ".alphas.npy"):
            if f.endswith(suf):
                base = f[:-len(suf)] + ("" if suf == ".recipe.json" else ".npy")
        m = POOL.match(base)
        if os.path.isdir(full) or not m:
            g["other (chunk dirs, stray files) | no recipe"] += sz; continue
        cls = CLASSES[reader.get(os.path.join(cache, base), 3)]
        g[f"{m['name'].split('_')[0]} | {cls}"] += sz
        if f.endswith("amplitudes.npy"):
            by_key[m["role"]].append(sz)
    by_class = defaultdict(float)
    for k, v in g.items():
        by_class[k.split(" | ")[1]] += v
    mm = os.path.join(S, "amp_mmap_cache")
    res = {"data_cache_gb": round(sum(g.values()) / 1e9, 2), "data_cache_files": n_files,
           "data_cache_by_class_gb": {k: round(v / 1e9, 2) for k, v in sorted(by_class.items(), key=lambda x: -x[1])},
           "data_cache_by_prefix_class_gb": {k: round(v / 1e9, 2) for k, v in sorted(g.items(), key=lambda x: -x[1])},
           "pool_files_by_role": {r: len(by_key[r]) for r in ("train", "val", "test")},
           "mmap_cache_gb": {d: round(du(os.path.join(mm, d)) / 1e9, 2) for d in sorted(os.listdir(mm))} if os.path.isdir(mm) else {},
           "datasets_gb": {"val": round(sum(os.lstat(p).st_size for p in glob.glob(D + "/*_val_*amplitudes.npy")) / 1e9, 2),
                           "test": round(sum(os.lstat(p).st_size for p in glob.glob(D + "/*_test_*amplitudes.npy")) / 1e9, 2),
                           "all": round(du(D) / 1e9, 2)},
           "open_entry_recipes": sorted(opened), "recipes_failed": failed, "files_read_by_recipes": len(reader)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
