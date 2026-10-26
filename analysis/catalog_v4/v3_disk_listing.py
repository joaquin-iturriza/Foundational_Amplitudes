#!/usr/bin/env python3
"""Read-only listing of catalog_v3's files on a site, for the deletion manifest (catalog_coverage, D163).

Prints one JSON object per line (nothing is changed or removed):
  {"kind": "item", "root": R, "path": P, "dir": bool, "alloc": bytes, "apparent": bytes, "files": n, "mtime": t,
   "atime": t, "first": <first file name inside a directory>}
for every top-level item whose name starts with one of the catalog prefixes (v3_, zp_, thdm_, pp13_) or is a chunk
directory (.chunks_*), under: the frozen val/test pools ($DATA_DIR = $WORK/datasets), the train cache
($SCRATCH/amp_data_cache), the width rebuild's staging and quarantine ($SCRATCH/zz_width_regen, walked to depth 3),
the MadGraph backends ($WORK/mg5amcnlo), the diagram sidecars (data/diagrams); and
  {"kind": "run", "path": run dir, "processes_file": ..., "mtime": t}
for every config.yaml under runs/ (depth 4) whose data.processes_file names a catalog_v3 recipe.

    site run ccin2p3 FA --item catalog_coverage -- python analysis/catalog_v4/v3_disk_listing.py > listing.jsonl
"""
import json, os, re, sys

PREFIX = ("v3_", "zp_", "thdm_", "pp13_", ".chunks_")


def size(p):
    st = os.lstat(p)
    if not os.path.isdir(p) or os.path.islink(p):
        return st.st_blocks * 512, st.st_size, 1, st.st_mtime, st.st_atime, None
    alloc = app = n = 0
    mt, at, first = st.st_mtime, st.st_atime, None
    for dp, dns, fns in os.walk(p):
        for f in fns:
            try:
                s = os.lstat(os.path.join(dp, f))
            except OSError:
                continue
            first = first or f
            alloc += s.st_blocks * 512; app += s.st_size; n += 1
            mt = max(mt, s.st_mtime); at = max(at, s.st_atime)
    return alloc, app, n, mt, at, first


def emit(root, p):
    a, s, n, mt, at, first = size(p)
    print(json.dumps({"kind": "item", "root": root, "path": p, "dir": os.path.isdir(p), "alloc": a, "apparent": s,
                      "files": n, "mtime": int(mt), "atime": int(at), "first": first}), flush=True)


def scan(root, d, depth=1):
    if not os.path.isdir(d):
        print(json.dumps({"kind": "missing", "root": root, "path": d}), flush=True)
        return
    for e in sorted(os.listdir(d)):
        p = os.path.join(d, e)
        if e.startswith(PREFIX):
            emit(root, p)
        elif depth > 1 and os.path.isdir(p) and not os.path.islink(p):
            scan(root, p, depth - 1)


def main():
    env = os.environ
    proj = env.get("PROJECT_DIR", os.getcwd())
    scan("datasets", env["DATA_DIR"])
    scan("amp_data_cache", os.path.join(env["SCRATCH"], "amp_data_cache"))
    scan("width_regen", os.path.join(env["SCRATCH"], "zz_width_regen"), depth=4)
    scan("mg5amcnlo", os.path.join(env["WORK"], "mg5amcnlo"))
    scan("diagrams", os.path.join(proj, "data", "diagrams"))
    pf = re.compile(r"processes_file:\s*(\S+)")
    for dp, dns, fns in os.walk(os.path.join(proj, "runs")):
        if dp.count(os.sep) - proj.count(os.sep) > 5:
            dns[:] = []
            continue
        dns[:] = [d for d in dns if d not in ("models", "plots", "checkpoints", "mlruns")]
        if "config.yaml" in fns:
            try:
                m = pf.search(open(os.path.join(dp, "config.yaml")).read())
            except OSError:
                continue
            if m and "catalog_v3" in m.group(1):
                print(json.dumps({"kind": "run", "path": dp, "processes_file": m.group(1),
                                  "mtime": int(os.stat(os.path.join(dp, "config.yaml")).st_mtime)}), flush=True)
    print(json.dumps({"kind": "done"}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
