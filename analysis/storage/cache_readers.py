"""Who reads CC's caches (cc_disk, D556), read-only, on the site.
  RECIPE lines: each recipe (data.processes_file) named by a run or sweep config written in the last --days days under
                runs/ or $SWEEP_DIR (queued sweep trials included: their configs exist before they start), with a count
  MMAP lines:   each amp_mmap_cache key: apparent GiB, its depths, days since any depth was last used (data_mmap's
                last_used, refreshed every 30 min by an open run), live slurm jobs naming it (cache_guard.site_users),
                the run logs and scheduler logs of the last --days days naming it, and the first process names it holds
  INUSE line:   the recipe basenames for cache_census.py --in-use
    python analysis/storage/cache_readers.py --days 14
"""
import argparse, collections, glob, json, os, re, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
PF = re.compile(r"^\s*processes_file:\s*(\S+)", re.M)


def recent(paths, cut):
    return [p for p in paths if os.path.getmtime(p) > cut]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=14.0)
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
    logs = recent(glob.glob(os.path.join(ROOT, "*.out")) + glob.glob(os.path.join(ROOT, "runs", "**", "out_*.log"), recursive=True), cut)
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
        names = json.load(open(kdp)).get("names", []) if os.path.exists(kdp) else []
        readers = [os.path.relpath(p, ROOT) for p, t in texts.items() if k in t]
        print("MMAP\t%s\t%.2f GiB\t%d depths\tlast used %.1f d ago\tlive %s\t%d recent logs %s\t%d processes %s" % (
            k, size / 2 ** 30, len(deps), (time.time() - max(lu)) / 86400 if lu else -1, site_users(k) or "-",
            len(readers), ",".join(readers[:4]), len(names), ",".join(names[:3])))
    print("INUSE\t" + " ".join(sorted(rec)))


if __name__ == "__main__":
    main()
