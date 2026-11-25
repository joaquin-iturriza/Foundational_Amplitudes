"""Who reads CC's caches (cc_disk, D556), read-only, on the site.
  RECIPE lines: each recipe (data.processes_file) named by a run or sweep config written in the last --days days under
                runs/ or $SWEEP_DIR (queued sweep trials included: their configs exist before they start), with a count
  MMAP lines:   each amp_mmap_cache key: apparent GiB, its depths, days since any depth was last used (data_mmap's
                last_used, refreshed every 30 min by an open run), live slurm jobs naming it (cache_guard.site_users),
                the run logs and scheduler logs of the last --days days naming it, and the first process names it holds
                and its training pools: how many, how many are train pools of an in-use recipe (recipe_id match through
                cache_census.pools_of, as datagen names them), and those recipes
  INUSE line:   the recipe basenames for cache_census.py --in-use: the recent configs' plus every recipe under recipes/
                whose name (with or without .yaml) is a whole word in the full plan/plan.yaml record of any entry not done
                or withdrawn (held entries included)
    python analysis/storage/cache_readers.py --days 14 --plan plan/plan.yaml
"""
import argparse, collections, fnmatch, glob, json, os, re, sys, time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
PF = re.compile(r"^\s*processes_file:\s*(\S+)", re.M)


def recent(paths, cut):
    return [p for p in paths if os.path.getmtime(p) > cut]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--plan", help="plan/plan.yaml: the recipes its not-done entries name are in use")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cut = time.time() - a.days * 86400
    scratch, sweeps = os.environ["SCRATCH"], os.environ.get("SWEEP_DIR", "")
    cfgs = recent([p for g in ("runs/*/config.yaml", "runs/*/*/config.yaml", "runs/*/*/*/config.yaml")
                   for p in glob.glob(os.path.join(ROOT, g))], cut)
    if sweeps:
        cfgs += recent([p for g in ("*/*.yaml", "*/*/config.yaml", "*/*/*.yaml") for p in glob.glob(os.path.join(sweeps, g))], cut)
    rec = collections.Counter()
    for c in cfgs:
        for m in PF.findall(open(c, errors="replace").read()):
            if m not in ("null", "None"):
                rec[os.path.basename(m.strip("'\""))] += 1
    for r, n in rec.most_common():
        print("RECIPE\t%d\t%s" % (n, r))
    allrec = sorted(glob.glob(os.path.join(ROOT, "recipes", "**", "*.yaml"), recursive=True))
    inuse = set(rec)
    if a.plan:
        import yaml
        texts = [yaml.safe_dump(e) for e in yaml.safe_load(open(a.plan))["entries"]
                 if e.get("status") not in ("done", "withdrawn")]
        words = set(re.findall(r"[A-Za-z0-9_.*-]+", "\n".join(texts)))
        globs = [w for w in words if "*" in w and len(w.replace("*", "")) >= 6]     # a family named as a glob
        for r in allrec:
            b = os.path.basename(r)
            if b in words or b[:-5] in words or any(fnmatch.fnmatch(b[:-5], g.removesuffix(".yaml")) for g in globs):
                inuse.add(b)
        print("PLAN\t%d entries not done\t%d of %d recipes named" % (len(texts), len(inuse - set(rec)), len(allrec)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cache_census import pools_of
    files = [r for r in allrec if any(fnmatch.fnmatch(os.path.basename(r), g) for g in inuse)]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(pools_of, [(r, 42) for r in files], chunksize=1)
    rid_recipes = collections.defaultdict(set)
    for path, pools, _ in res:
        for _, rid in pools:
            rid_recipes[rid].add(os.path.basename(path))
    logs = recent(glob.glob(os.path.join(ROOT, "*.out")) + glob.glob(os.path.join(ROOT, "runs", "**", "out_*.log"), recursive=True)
                  + (glob.glob(os.path.join(sweeps, "**", "*.log"), recursive=True) if sweeps else []), cut)   # sweep trials too
    texts = {}
    for p in logs:
        try:
            texts[p] = open(p, errors="replace").read()
        except OSError:
            pass
    from cache_guard import site_users
    root = os.path.join(scratch, "amp_mmap_cache")
    for k in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        kd = os.path.join(root, k)
        if not os.path.isdir(kd) or k.startswith("."):
            continue
        size = sum(os.lstat(os.path.join(d, f)).st_size for d, _, fs in os.walk(kd) for f in fs)
        deps = [d for d in os.listdir(kd) if os.path.isdir(os.path.join(kd, d)) and not d.startswith(".")]
        lu = [os.path.getmtime(os.path.join(kd, d, "last_used")) if os.path.exists(os.path.join(kd, d, "last_used"))
              else os.path.getmtime(os.path.join(kd, d)) for d in deps]
        kdp = os.path.join(kd, "key_doc.json")
        kdoc = json.load(open(kdp)) if os.path.exists(kdp) else {}
        names = kdoc.get("names", [])
        train = [i[0] for i in kdoc.get("pools", {}).get("train", []) if isinstance(i, list) and i]
        used_by = sorted(set().union(*[rid_recipes.get(rid, set()) for rid in train])) if train else []
        readers = [os.path.relpath(p, ROOT) for p, t in texts.items() if k in t]
        print("MMAP\t%s\t%.2f GiB\t%d depths\tlast used %.1f d ago\tlive %s\t%d recent logs %s\t%d processes %s"
              "\t%d train pools, %d of in-use recipes %s" % (
            k, size / 2 ** 30, len(deps), (time.time() - max(lu)) / 86400 if lu else -1, site_users(k) or "-",
            len(readers), ",".join(readers[:4]), len(names), ",".join(names[:3]),
            len(train), sum(rid in rid_recipes for rid in train), ",".join(used_by[:6]) or "-"))
    print("INUSE\t" + " ".join(sorted(inuse)))


if __name__ == "__main__":
    main()
