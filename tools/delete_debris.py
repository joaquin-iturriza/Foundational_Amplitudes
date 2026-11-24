#!/usr/bin/env python3
"""Delete our own cheap-to-rebuild debris on a site (D550, cc_disk, 2026-11-25: "and yes, delete the debris."), as a
CPU job (scripts/job_delete_debris.sh), never on a login node. Classes, each re-checked here at run time:
  smoke    the named old smoke, test and probe outputs under $SCRATCH (NAMED below: none cited in the repo, 2026-11-24)
  mmap     amp_mmap_cache keys in MMAP format 2 (current code reads only data_mmap.MMAP_VERSION), kept if any live
           slurm job of the user names the key in its stdout (cache_guard.site_users)
  scratch  the scratch run dirs the probe tools left in $SCRATCH (tempfile.mkdtemp prefixes below), older than --scratch-days
  joblogs  loose scheduler logs (*.out, *.err) in the checkout root older than --log-days whose job id is not queued and
           that hold no result line (RUN_SUMMARY, RESULT_JSON: analysis/catalog_v4/size_figs.py reads the root's
           catalog_v3_short_*.out RUN_SUMMARY lines)
Writes the manifest (class, apparent bytes, path) to --manifest before deleting anything, then deletes and prints the
apparent GiB per class. --dry-run lists only.
    python tools/delete_debris.py --manifest $SCRATCH/zz_cc_disk/deleted.tsv [--dry-run]
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
NAMED = ["zz_v3_smoke_Q_62382593", "zz_v3_smoke_Q_62382608", "zz_v3_smoke_baseline_more_signed_62412176",
         "zz_v3_smoke_baseline_sign_near_62411994", "zz_v3_smoke_baseline_sign_pairs_62411993",
         "zz_v3_smoke_baseline_sign_pairs_bounded_62441776", "zz_v3_smoke_dimensionless_muon_62411859",
         "zz_v3_smoke_v4_dimensionless_muon_62461424", "zz_v3_smoke_v4_dimensionless_muon_62464686",
         "zz_v3_smoke_v4_dimensionless_muon_pp_62884376", "zz_v3_smoke_v4_dimensionless_muon_pp_62889334",
         "zz_v3_smoke_v4_het_control_62899632", "zz_v3_smoke_v4_het_default25_63533679",
         "zz_v3_smoke_v4_het_default25_63533681", "zz_lowd_smoke", "zz_memmap_test", "zz_memmap_test_dimless",
         "zz_memmap_test_dimless_inmem_prev.npz", "zz_memmap_test_v3.log", "zz_probe_long_w16_62929016",
         "zz_probe_long_w16_62932850", "zz_probe_long_w32_62932851", "zz_probe_long_w8_62929015",
         "zz_probe_long_w8_62932622"]
NAMED += [n + ".recipe.yaml" for n in NAMED if n.startswith("zz_v3_smoke_")]
SCRATCH_PREFIXES = ("zeroshot_", "signed_ab_", "kshot_", "rebuild_", "f24net_run_", "kitprec_", "evalbest_")
JOBID = re.compile(r"_(\d{7,9})\.(out|err)$")
RESULT_LINES = (b"RUN_SUMMARY", b"RESULT_JSON")


def holds_result(p):
    with open(p, "rb") as f:
        return any(m in f.read() for m in RESULT_LINES)


def apparent(p):
    if not os.path.isdir(p) or os.path.islink(p):
        return os.lstat(p).st_size
    t = 0
    for d, dns, fns in os.walk(p):
        for f in fns:
            try:
                t += os.lstat(os.path.join(d, f)).st_size
            except OSError:
                pass
    return t


def queued():
    r = subprocess.run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i"], capture_output=True, text=True,
                       timeout=120, check=True)
    return {j.split("_")[0] for j in r.stdout.split()}


def candidates(scratch, project, scratch_days, log_days):
    from data_mmap import MMAP_VERSION
    from cache_guard import site_users
    now = time.time()
    for n in NAMED:
        p = os.path.join(scratch, n)
        if os.path.lexists(p):
            yield "smoke", p
    for k in glob.glob(os.path.join(scratch, "amp_mmap_cache", "*", "meta.json")):
        key = os.path.basename(os.path.dirname(k))
        d = json.load(open(k))
        fmt = d.get("key_doc", d).get("version")
        if fmt == 2 and MMAP_VERSION > 2 and not site_users(key):
            yield "mmap", os.path.dirname(k)
    with os.scandir(scratch) as it:
        for e in it:
            if e.name.startswith(SCRATCH_PREFIXES) and now - e.stat(follow_symlinks=False).st_mtime > scratch_days * 86400:
                yield "scratch", e.path
    live = queued()
    with os.scandir(project) as it:
        for e in it:
            m = JOBID.search(e.name)
            if (m and e.is_file(follow_symlinks=False) and m.group(1) not in live
                    and now - e.stat().st_mtime > log_days * 86400 and not holds_result(e.path)):
                yield "joblogs", e.path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scratch-days", type=float, default=3.0)
    ap.add_argument("--log-days", type=float, default=14.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    scratch, project = os.environ["SCRATCH"], os.environ["PROJECT_DIR"]
    rows = [(c, apparent(p), p) for c, p in candidates(scratch, project, a.scratch_days, a.log_days)]
    for c, _, p in rows:      # every path is a direct child of SCRATCH, of its mmap cache root, or of the checkout
        assert os.path.dirname(p) in (scratch, project, os.path.join(scratch, "amp_mmap_cache")), p
    with open(a.manifest, "w") as f:
        for r in rows:
            f.write("%s\t%d\t%s\n" % r)
    tot = {}
    for c, b, p in rows:
        if not a.dry_run:
            (shutil.rmtree if os.path.isdir(p) and not os.path.islink(p) else os.remove)(p)
        n, s = tot.get(c, (0, 0))
        tot[c] = (n + 1, s + b)
    for c, (n, s) in sorted(tot.items()):
        print("%s %s: %d entries, %.2f GiB apparent" % ("listed" if a.dry_run else "deleted", c, n, s / 2 ** 30))
    print("DEBRIS_DONE")


if __name__ == "__main__":
    main()
