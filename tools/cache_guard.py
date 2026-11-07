#!/usr/bin/env python3
"""Who is using an amp_mmap_cache key? A cache key dir (<SCRATCH>/amp_mmap_cache/<key>, data_mmap.py) is deleted only
while no RUNNING or queued FA job on its site names the key in its log or config. 2026-11-07: two keys were deleted
under D310 while the scaling grid read them (FA-ccin2p3-a77e20b8 was running on 0c897ca2...; its log names the
path three times).

Laptop side (submit_guard's `cache` check, a `site run ... rm ... amp_mmap_cache/<key>`):
    python tools/cache_guard.py <site> <key|path> [...]     exit 1 and the users when any active job names a key
Registry runs of project FA on the site in an active state (RUNNING, SUBMITTED, PENDING, QUEUED, CONFIGURING) have their
log (`site logs <run> -n 100000`), their registry row (note, launch, expect) and any fetched config.yaml searched.
A run whose log cannot be read counts as a user (fails closed).

Site side (data_mmap.prune): site_users(key) greps the stdout files of the user's live slurm jobs for the key."""
import glob, os, re, sqlite3, subprocess, sys

ACTIVE = ("RUNNING", "SUBMITTED", "PENDING", "QUEUED", "CONFIGURING", "UNKNOWN")
RUNS_DB = os.environ.get("SUBMIT_RUNS_DB") or os.path.expanduser("~/.local/share/ccorch/runs.db")
ARTIFACTS = os.path.expanduser("~/.local/share/ccorch/artifacts/FA")


def key_of(s):
    """the cache key of a path or a bare key: the component after amp_mmap_cache/, else the argument itself"""
    m = re.search(r"amp_mmap_cache/+([^/\s'\"]+)", s)
    return m.group(1) if m else s.strip().strip("/").split("/")[-1]


def active_runs(site, project="FA", db=None):
    c = sqlite3.connect("file:%s?mode=ro" % (db or RUNS_DB), uri=True, timeout=10)
    q = "SELECT id, state, note, launch, expect FROM runs WHERE project=? AND site=? AND state IN (%s)" % \
        ",".join("?" * len(ACTIVE))
    return c.execute(q, (project, site) + ACTIVE).fetchall()


def site_logs(run):
    r = subprocess.run(["site", "logs", run, "-n", "100000"], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError("site logs %s failed: %s" % (run, (r.stderr or r.stdout).strip()[-200:]))
    return r.stdout


def users(site, keys, fetch=site_logs, db=None):
    """[(run, state, key, where)] of active runs naming a key in their log, registry row or fetched config"""
    out = []
    for run, state, note, launch, expect in active_runs(site, db=db):
        texts = {"registry": " ".join(str(x or "") for x in (note, launch, expect))}
        for cfg in glob.glob(os.path.join(ARTIFACTS, run, "**", "config.yaml"), recursive=True):
            texts["config"] = texts.get("config", "") + open(cfg, errors="replace").read()
        try:
            texts["log"] = fetch(run)
        except Exception as e:
            out.append((run, state, "*", "log unreadable (%s)" % e))
            continue
        for k in keys:
            for where, t in texts.items():
                if k in t:
                    out.append((run, state, k, where))
    return out


def site_users(key):
    """on a site: the live slurm jobs of this user whose stdout file names the key"""
    try:
        ids = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"], capture_output=True,
                             text=True, timeout=60).stdout.split()
    except Exception:
        return ["squeue unreadable"]
    out = []
    for j in ids:
        info = subprocess.run(["scontrol", "show", "job", "-o", j], capture_output=True, text=True).stdout
        m = re.search(r"StdOut=(\S+)", info)
        paths = [m.group(1).replace("%j", j)] if m else []
        for p in paths:
            try:
                if key in open(p, errors="replace").read():
                    out.append(j)
            except OSError:
                pass
    return out


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) < 2:
        sys.exit(__doc__)
    site, keys = a[0], [key_of(x) for x in a[1:]]
    us = users(site, keys)
    for u in us:
        print("%s (%s) names %s in its %s" % u)
    return 1 if us else 0


if __name__ == "__main__":
    sys.exit(main())
