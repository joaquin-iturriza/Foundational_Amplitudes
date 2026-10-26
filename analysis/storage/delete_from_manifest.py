#!/usr/bin/env python3
"""Delete the rows marked `delete` in a cache manifest (analysis/storage/cache_census.py), and nothing else.

Only paths listed in the manifest are removed: files with os.remove, then the listed directories with os.rmdir
(an rmdir of a non-empty directory fails, it never recurses). A row is skipped, and reported, if its path is not
under $SCRATCH/amp_data_cache, is a symlink, no longer exists, or its size, mtime or atime differ from the
manifest's (changed or read since the census). Without --apply it only prints what it would remove.

    python analysis/storage/delete_from_manifest.py analysis/storage/cache_manifest_2026-10-26.tsv [--apply]
"""
import argparse, csv, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
NEVER = ("zz_lowd", "zz_archive_lxera", "/datasets", "/runs/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    import datagen
    cache = os.path.realpath(datagen.train_cache_dir()) + os.sep
    rows = [r for r in csv.DictReader(open(a.manifest), delimiter="\t") if r["decision"] == "delete"]
    done = skipped = freed = 0
    fmt = lambda t: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))
    for r in [r for r in rows if r["kind"] == "f"] + [r for r in rows if r["kind"] == "d"]:
        p = r["path"]
        why = None
        if not os.path.realpath(p).startswith(cache) or any(n in p for n in NEVER):
            why = "outside the cache"
        elif os.path.islink(p):
            why = "symlink"
        elif not os.path.lexists(p):
            why = "gone"
        else:
            st = os.lstat(p)
            if r["kind"] == "f" and (st.st_size != int(r["size_bytes"]) or fmt(st.st_mtime) != r["mtime"]
                                     or fmt(st.st_atime) != r["atime"]):
                why = "changed or read since the census"
        if why:
            skipped += 1
            print(f"SKIP {why}: {p}")
            continue
        if a.apply:
            try:
                os.remove(p) if r["kind"] == "f" else os.rmdir(p)
            except OSError as e:
                skipped += 1
                print(f"SKIP {e}: {p}")
                continue
        done += 1
        freed += int(r["size_bytes"])
    print(f"SUMMARY {'removed' if a.apply else 'would remove'} {done} paths, {freed / 1e9:.2f} GB apparent; "
          f"skipped {skipped}")


if __name__ == "__main__":
    main()
