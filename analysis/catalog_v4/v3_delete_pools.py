#!/usr/bin/env python3
"""Delete catalog_v3's pools on a site: the 'pools' group of the deletion manifest, nothing else (catalog_coverage,
the user's go of 2026-10-26, "Yes. delete catalog 3 and do the regeneration").

Every manifest item of the group is re-checked right before it is removed and refused (left in place, logged) when:
it is not a regular file (directory, symlink, missing), its name starts with pp13_ (the 13 TeV pp pools are kept),
or its allocated size or mtime differs from the manifest. Only paths listed in the manifest are ever touched.
Prints one tab-separated line per item: DELETED|REFUSED|DRYRUN, path, allocated bytes, mtime, reason; then a SUMMARY.

    site run --item catalog_coverage ccin2p3 FA -- python analysis/catalog_v4/v3_delete_pools.py [--apply] [--shard I/N]
Without --apply nothing is removed.
"""
import argparse, json, os, stat, sys

MANIFEST = "analysis/catalog_v4/data/catalog_v3_delete_manifest.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--group", default="pools")
    a = ap.parse_args()
    if a.group != "pools":
        sys.exit("only the 'pools' group is named by the user's deletion call")
    i, n = (int(x) for x in a.shard.split("/"))
    m = json.load(open(MANIFEST))
    items = sorted((os.path.join(d, x[0]), x[1], x[3]) for d, v in m["groups"][a.group].items() for x in v)
    freed = done = refused = 0
    for k, (p, alloc, mtime) in enumerate(items):
        if k % n != i:
            continue
        why = None
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            st, why = None, "missing"
        if st is not None:
            if not stat.S_ISREG(st.st_mode):
                why = "not a regular file"
            elif os.path.basename(p).startswith("pp13_"):
                why = "pp13 pool (kept)"
            elif st.st_blocks * 512 != alloc or int(st.st_mtime) != mtime:
                why = f"changed (alloc {st.st_blocks * 512}, mtime {int(st.st_mtime)})"
        if why:
            refused += 1
            print(f"REFUSED\t{p}\t{alloc}\t{mtime}\t{why}", flush=True)
            continue
        if a.apply:
            os.unlink(p)
            print(f"DELETED\t{p}\t{alloc}\t{mtime}\t", flush=True)
        else:
            print(f"DRYRUN\t{p}\t{alloc}\t{mtime}\t", flush=True)
        done += 1
        freed += alloc
    print(f"SUMMARY\tshard {i}/{n}\t{'deleted' if a.apply else 'would delete'} {done}\trefused {refused}\t"
          f"bytes {freed}", flush=True)


if __name__ == "__main__":
    main()
