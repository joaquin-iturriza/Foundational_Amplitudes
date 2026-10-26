#!/usr/bin/env python3
"""Census of the training cache $SCRATCH/amp_data_cache (CPU, read-only), as a manifest of what to keep and delete.

One row per file (and one per directory, after its files): path, apparent size, mtime, atime, the recipe file(s)
whose train pools are that path, the entry it belongs to, keep or delete, and why. An entry is one pool and its
sidecars (`<pool>.npy`, `.npy.recipe.json`, `.alphas.npy`, ...), a MadEvent banner (`<process>_lhe_iseed*.banner.txt`,
with that process's lhe pools), a chunk directory `.chunks_<recipe_id>` (with the pool of that recipe id), or a
lone file. An entry is kept when any of its files was modified or read in the last --days days, or when its pool is
a train pool of an in-use recipe (--in-use globs under recipes/); every other entry is deleted.

Which recipe a pool belongs to is computed the way datagen does it: for every recipe yaml under recipes/,
register_recipe_processes, then variable_energy_recipe(name, sqrts, n_train, role=train, seed) and
recipe_output_path in the train cache (the stripped parent of a reweighted alpha_s scan too). Each recipe file is
read in its own forked process, so the registrations of one never leak into another.

    python analysis/storage/cache_census.py --in-use 'catalog_v3*' 'catalog_v4*' --days 3 > manifest.tsv
"""
import argparse, fnmatch, glob, os, re, sys, time
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)


def pools_of(args):
    """[(pool basename, recipe_id)] of the train pools of one recipe file; [] with an error string if it is not one."""
    path, seed = args
    try:
        import yaml
        import datagen
        import mg5_pipeline_final as mg
        spec = yaml.safe_load(open(path))
        procs = spec.get("processes") if isinstance(spec, dict) else spec
        if not isinstance(procs, list) or not procs or not all(isinstance(p, dict) and "sqrts" in p for p in procs):
            return path, [], "not a recipe"
        mg.register_recipe_processes(procs, default_sampling=(spec.get("sampling") if isinstance(spec, dict) else None))
        dest, out, bad = datagen.train_cache_dir(), [], 0
        for p in procs:
            try:
                smin, smax, n = float(p["sqrts"][0]), float(p["sqrts"][1]), int(p["n_train"])
                rec = mg.variable_energy_recipe(p["name"], smin, smax, n, role="train", seed=seed)
                out.append((os.path.basename(mg.recipe_output_path(rec, dest)), mg.recipe_id(rec)))
                if datagen.is_reweightable_virt_scan(p["name"]):
                    par = mg.PROCESSES[p["name"]]["scan_base"]
                    prec = mg.variable_energy_recipe(par, smin, smax, n, role="train", seed=seed)
                    out.append((os.path.basename(mg.recipe_output_path(prec, dest)), mg.recipe_id(prec)))
            except Exception:
                bad += 1
        return path, out, (f"{bad} entries not resolved" if bad else "")
    except Exception as e:
        return path, [], f"{type(e).__name__}: {e}"


def entry_of(name):
    """The entry key of a top-level cache item."""
    if name.startswith(".chunks_"):
        return "chunks:" + name[len(".chunks_"):]
    m = re.match(r"^(.*)_lhe_iseed\d+\.banner\.txt$", name)
    if m:
        return "lhe:" + m.group(1)
    i = name.find("_amplitudes")
    if i >= 0:
        return "pool:" + name[:i + len("_amplitudes")] + ".npy"
    return "file:" + name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-use", nargs="+", required=True, help="recipe basename globs (under recipes/) in use")
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import datagen
    cache = datagen.train_cache_dir()
    now = time.time()
    cut = now - 86400 * a.days

    recipes = sorted(glob.glob(os.path.join(ROOT, "recipes", "**", "*.yaml"), recursive=True))
    with Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(pools_of, [(r, a.seed) for r in recipes], chunksize=1)
    by_pool, by_rid, inuse_pools, inuse_rids = {}, {}, set(), set()
    for path, out, err in res:
        rel = os.path.relpath(path, ROOT)
        used = any(fnmatch.fnmatch(os.path.basename(rel), g) for g in a.in_use)
        print(f"# recipe {rel}\t{len(out)} train pools\t{'IN_USE' if used else ''}\t{err}", file=sys.stderr)
        for base, rid in out:
            by_pool.setdefault(base, set()).add(rel)
            by_rid.setdefault(rid, set()).add(base)
            if used:
                inuse_pools.add(base)
                inuse_rids.add(rid)
    inuse_lhe = {re.sub(r"_[0-9.]+-[0-9.]+GeV_train_lhe_amplitudes\.npy$", "", b) for b in inuse_pools if "_lhe_" in b}
    lhe_recipes = {}
    for b, rs in by_pool.items():
        if "_train_lhe_" in b:
            lhe_recipes.setdefault(re.sub(r"_[0-9.]+-[0-9.]+GeV_train_lhe_amplitudes\.npy$", "", b), set()).update(rs)

    # walk the top level; a directory's files are listed under it
    entries = {}
    for name in sorted(os.listdir(cache)):
        p = os.path.join(cache, name)
        rows = []
        if os.path.isdir(p) and not os.path.islink(p):
            for dp, dns, fns in os.walk(p):
                for f in fns:
                    fp = os.path.join(dp, f)
                    st = os.lstat(fp)
                    rows.append((fp, st.st_size, st.st_mtime, st.st_atime, "f"))
            dirs = [dp for dp, _, _ in os.walk(p)]
            for dp in sorted(dirs, key=len, reverse=True):     # deepest first, so a delete can rmdir in order
                st = os.lstat(dp)
                rows.append((dp, 0, st.st_mtime, st.st_atime, "d"))
        else:
            st = os.lstat(p)
            rows.append((p, st.st_size, st.st_mtime, st.st_atime, "f"))
        entries.setdefault(entry_of(name), []).extend(rows)

    print("path\tsize_bytes\tmtime\tatime\tkind\trecipe\tentry\tdecision\treason")
    for key in sorted(entries):
        rows = entries[key]
        last = max(max(m, t) for _, _, m, t, _ in rows)
        kind, val = key.split(":", 1)
        if kind == "pool":
            rs = by_pool.get(val, set())
            used, rid_note = val in inuse_pools, ""
        elif kind == "chunks":
            bases = by_rid.get(val, set())
            rs = set().union(*(by_pool.get(b, set()) for b in bases)) if bases else set()
            used, rid_note = val in inuse_rids, ""
        elif kind == "lhe":
            rs = lhe_recipes.get(val, set())
            used, rid_note = val in inuse_lhe, ""
        else:
            rs, used, rid_note = set(), False, ""
        recipe = ",".join(sorted(rs)[:4]) + (f",+{len(rs) - 4} more" if len(rs) > 4 else "") if rs else "none"
        if last >= cut:
            dec, why = "keep", f"touched {time.strftime('%Y-%m-%d %H:%M', time.localtime(last))}, within {a.days:g} days"
        elif used:
            dec, why = "keep", "train pool of an in-use recipe (" + ",".join(a.in_use) + ")"
        else:
            dec = "delete"
            why = (f"last touched {time.strftime('%Y-%m-%d', time.localtime(last))}; "
                   + ("its recipes are not in use" if rs else "produced by no recipe under recipes/"))
        why += rid_note
        for fp, sz, m, t, k in rows:
            print("\t".join([fp, str(sz), time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(m)),
                             time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)), k, recipe, key, dec, why]))


if __name__ == "__main__":
    main()
