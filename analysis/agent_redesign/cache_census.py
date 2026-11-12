#!/usr/bin/env python3
"""What the CC data caches hold (Joaquin 2026-11-12: 'how is it possible that we need more than 100 GB of data
caches'). Read-only, on the site: every file of $SCRATCH/amp_data_cache (the train pools and their sidecars) and of
$SCRATCH/amp_mmap_cache (one directory per training-set key), against $DATA_DIR (the frozen val/test pools).
Per data-cache pool file: its process, role, sampling tag; whether its recipe id is the one a recipe in recipes/*.yaml
asks today (read through tools/check_site_data.py --paths, which prints every pool path a recipe resolves to);
several files of one process and role (other energy window, sampling tag or size) counted as extra copies.
    python analysis/agent_redesign/cache_census.py [--recipes 'recipes/catalog_v4*.yaml'] --out <json>
One JSON line (also written to --out): GB by kind."""
import glob, json, os, re, subprocess, sys
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from lhe_parts import _du as du  # noqa: E402
S, D = os.environ["SCRATCH"], os.environ["DATA_DIR"]
POOL = re.compile(r"^(?P<name>.+?)_(?P<lo>[\d.]+)-(?P<hi>[\d.]+)GeV_(?P<role>train|val|test)_(?P<tag>.*?)amplitudes\.npy$")


def main():
    a = sys.argv[1:]
    pat = a[a.index("--recipes") + 1] if "--recipes" in a else "recipes/*.yaml"
    out = a[a.index("--out") + 1]
    wanted = set()
    for r in sorted(glob.glob(os.path.join(ROOT, pat))):   # one call per recipe: a recipe that does not parse is skipped
        p = subprocess.run([sys.executable, "tools/check_site_data.py", "--paths", os.path.relpath(r, ROOT)],
                           cwd=ROOT, capture_output=True, text=True)
        wanted |= {l.split()[1] for l in p.stdout.splitlines() if l.startswith("POOL ") and l.endswith(".npy")}
        print(f"CENSUS_RECIPE {os.path.basename(r)} {len(wanted)}", flush=True)
    cache = os.path.join(S, "amp_data_cache")
    g = defaultdict(float)
    by_key = defaultdict(list)
    for f in os.listdir(cache):
        full = os.path.join(cache, f)
        sz = du(full) if os.path.isdir(full) else os.lstat(full).st_size
        base = f
        for suf in (".recipe.json", ".alphas.npy"):
            if f.endswith(suf):
                base = f[:-len(suf)] + ("" if suf == ".recipe.json" else ".npy")
        m = POOL.match(base) or POOL.match(f)
        if os.path.isdir(full) or not m:
            g["other (chunk dirs, stray files)"] += sz; continue
        pre = m["name"].split("_")[0]
        kind = "read by a current recipe" if os.path.join(cache, base) in wanted else "read by no current recipe"
        g[f"{pre} {kind}"] += sz
        if f.endswith("amplitudes.npy"):
            by_key[(m["name"], m["role"])].append((sz, os.path.join(cache, f) in wanted))
    extra = sum(sz for v in by_key.values() if len(v) > 1 for sz, w in v if not w)
    res = {"data_cache_gb": round(sum(g.values()) / 1e9, 2),
           "data_cache_by_kind_gb": {k: round(v / 1e9, 2) for k, v in sorted(g.items(), key=lambda x: -x[1])},
           "pools": len(by_key), "pools_with_several_files": sum(len(v) > 1 for v in by_key.values()),
           "extra_copies_gb": round(extra / 1e9, 2),
           "mmap_cache_gb": {d: round(du(os.path.join(S, "amp_mmap_cache", d)) / 1e9, 2)
                             for d in sorted(os.listdir(os.path.join(S, "amp_mmap_cache")))},
           "datasets_gb": {"val": round(sum(os.lstat(p).st_size for p in glob.glob(D + "/*_val_*amplitudes.npy")) / 1e9, 2),
                           "test": round(sum(os.lstat(p).st_size for p in glob.glob(D + "/*_test_*amplitudes.npy")) / 1e9, 2),
                           "all": round(du(D) / 1e9, 2)},
           "recipes": pat, "wanted_pool_files": len(wanted)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
