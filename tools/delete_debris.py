#!/usr/bin/env python3
"""Delete our own cheap-to-rebuild debris on a site (D550, cc_disk, 2026-11-25: "and yes, delete the debris."), as a
CPU job (scripts/job_delete_debris.sh), never on a login node. Classes, each re-checked here at run time:
  smoke    the named old smoke, test and probe outputs under $SCRATCH (NAMED below: none cited in the repo, 2026-11-24)
  mmap     amp_mmap_cache keys in MMAP format 2 (current code reads only data_mmap.MMAP_VERSION), kept if any live
           slurm job of the user names the key in its stdout (cache_guard.site_users)
  scratch  the scratch run dirs the probe tools left in $SCRATCH (tempfile.mkdtemp prefixes below), older than --scratch-days
  smoke_models  the model files (WEIGHTS, anything under models/) of the launch-path smoke runs in runs/zz_smoke/
           (scripts/job_config_smoke.sh, 50 steps) with no file newer than --scratch-days and whose name no committed
           file names; their configs, logs and metrics stay (D550's "old smoke and probe runs")
Scheduler logs (*.out in the checkout root) are never deleted here: committed analyses parse them (ws_objective_check.py,
cited as % src: in docs/results.tex; size_figs.py), so they are evidence, not debris (design review 2026-11-25).
A class whose listed total exceeds 2x its expected size (--expect-gib) is skipped and reported. Writes the manifest (class, apparent bytes, path) to --manifest before deleting anything, then deletes and prints the
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
WEIGHTS = (".pt", ".pt.gz", ".ckpt", ".pth")
SCRATCH_PREFIXES = ("zeroshot_", "signed_ab_", "kshot_", "rebuild_", "f24net_run_", "kitprec_", "evalbest_")


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


def candidates(scratch, scratch_days):
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
    smokes = os.path.join(os.environ["PROJECT_DIR"], "runs", "zz_smoke")
    named = subprocess.run(["git", "-C", ROOT, "grep", "-h", "-o", "zz_smoke_[A-Za-z0-9_.-]*"], capture_output=True,
                           text=True).stdout.split()
    for r in sorted(glob.glob(os.path.join(smokes, "*/"))):
        r = r.rstrip("/")
        files = [os.path.join(d, f) for d, _, fs in os.walk(r) for f in fs]
        if os.path.basename(r) in named or any(now - os.lstat(f).st_mtime <= scratch_days * 86400 for f in files):
            continue
        for f in files:
            if f.endswith(WEIGHTS) or "/models/" in f[len(r):]:
                yield "smoke_models", f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scratch-days", type=float, default=3.0)
    ap.add_argument("--classes", default="smoke,mmap,scratch,smoke_models")
    ap.add_argument("--expect-gib", default="smoke=1.4,mmap=4.8,scratch=2.2,smoke_models=7.0",
                    help="the inventory's apparent GiB per class (FA-ccin2p3-b949c4ee); a class listing over 2x is skipped")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    scratch = os.environ["SCRATCH"]
    keep = set(a.classes.split(","))
    rows = [(c, apparent(p), p) for c, p in candidates(scratch, a.scratch_days) if c in keep]
    for c, _, p in rows:      # a direct child of SCRATCH or of its mmap cache root; a smoke model file under runs/zz_smoke/
        if c == "smoke_models":
            assert p.startswith(os.path.join(os.environ["PROJECT_DIR"], "runs", "zz_smoke") + "/") and os.path.isfile(p), p
        else:
            assert os.path.dirname(p) in (scratch, os.path.join(scratch, "amp_mmap_cache")), p
    expect = {k: float(v) for k, v in (x.split("=") for x in a.expect_gib.split(","))}
    listed = {}
    for c, b, _ in rows:
        listed[c] = listed.get(c, 0) + b
    over = {c for c, b in listed.items() if b / 2 ** 30 > 2 * expect.get(c, 0)}
    for c in sorted(over):       # the 2x margin rule: a class far above what was approved is a finding, not deleted
        print("SKIPPED %s: listed %.2f GiB, over 2x the expected %.2f" % (c, listed[c] / 2 ** 30, expect.get(c, 0)))
    rows = [r for r in rows if r[0] not in over]
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
