"""Disk watch for the supervisor (projects.toml `maintain`), run on a site through
`site run --item '*' <site> FA -- python scripts/disk_watch.py`.

CC-IN2P3's /sps/lpnhe group space (512 GB) filled to 100% on 2026-10-25 and jobs failed on write; nothing watched it.
Per filesystem holding PROJECT_DIR, SCRATCH, WORK and SUBMIT_DIR (deduplicated by device) this prints used/total and
the six largest directories of the user's own area there (WORK when it is on that filesystem, else the shortest of
those paths on it), measured with `du -s` under a time budget and cached in .disk_watch_cache.json in the checkout
for 6 h (a timed-out measurement is cached as partial, so a slow filesystem is not re-walked every pass).

Exit status: 0 below --warn (90%); otherwise 1, last line "DISK <fs> at N%" (the supervisor turns a failing maintain
command into an ALERT). At --full (97%) or above it also writes .disk_full in the checkout: sweep/feed_capped.py
submits nothing while that flag is under 30 min old. The flag is refreshed every pass while the disk is full and
removed by this script once every filesystem is back below --full.

    python scripts/disk_watch.py [--warn 90] [--full 97] [--du-budget 300] [--cache-hours 6]
"""
import argparse, json, os, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAG = os.path.join(REPO, ".disk_full")
CACHE = os.path.join(REPO, ".disk_watch_cache.json")
TOP = 6


def mount_point(p):
    """The mount point of p: walk up while the device stays the same."""
    p = os.path.realpath(p)
    dev = os.stat(p).st_dev
    while p != "/":
        up = os.path.dirname(p)
        try:
            if os.stat(up).st_dev != dev:
                break
        except OSError:
            break
        p = up
    return p


def paths():
    proj = os.environ.get("PROJECT_DIR", REPO)
    work = os.environ.get("WORK") or os.path.dirname(proj)
    out = {"WORK": work, "PROJECT_DIR": proj}
    for k in ("SCRATCH", "SUBMIT_DIR"):
        if os.environ.get(k):
            out[k] = os.environ[k]
    return {k: v for k, v in out.items() if os.path.isdir(v)}


def filesystems():
    """{dev: {"mount", "names", "area"}}: one entry per device, area = the user's own directory on it."""
    fss = {}
    for name, p in paths().items():
        dev = os.stat(p).st_dev
        e = fss.setdefault(dev, {"mount": mount_point(p), "names": [], "paths": []})
        e["names"].append(name)
        e["paths"].append(os.path.realpath(p))
    for e in fss.values():
        e["area"] = e["paths"][e["names"].index("WORK")] if "WORK" in e["names"] else min(e["paths"], key=len)
    return fss


def usage(p):
    s = os.statvfs(p)
    total = s.f_blocks * s.f_frsize
    used = (s.f_blocks - s.f_bfree) * s.f_frsize
    avail = s.f_bavail * s.f_frsize
    # df's Capacity: used / (used + available to the user), rounded up
    pct = 0 if used + avail == 0 else -(-100 * used // (used + avail))
    return used, total, int(pct)


def du_top(area, budget):
    """[(bytes, name)] of area's subdirectories, largest first, and whether the budget ran out first."""
    try:
        subs = sorted(d for d in os.listdir(area) if os.path.isdir(os.path.join(area, d))
                      and not os.path.islink(os.path.join(area, d)))
    except OSError:
        return [], False
    sizes, partial, t_end = [], False, time.time() + budget
    for d in subs:
        left = t_end - time.time()
        if left <= 1:
            partial = True
            break
        try:
            r = subprocess.run(["du", "-s", "-B1", os.path.join(area, d)], capture_output=True, text=True,
                               timeout=left)
            out = r.stdout
        except subprocess.TimeoutExpired:
            partial = True
            break
        try:
            sizes.append((int(out.split()[0]), d))
        except (IndexError, ValueError):
            continue
    return sorted(sizes, reverse=True), partial


def load_cache():
    try:
        return json.load(open(CACHE))
    except (OSError, ValueError):
        return {}


def save_cache(c):
    """Best effort: on a full disk the cache cannot be written, and that must not hide the report."""
    tmp = CACHE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(c, f)
        os.replace(tmp, CACHE)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass


def gb(b):
    return "%.1f GB" % (b / 1e9) if b < 1e12 else "%.2f TB" % (b / 1e12)


def set_flag(msg):
    try:
        with open(FLAG, "w") as f:
            f.write(msg + "\n")
    except OSError:                       # no block left for its content: an empty file still carries the mtime
        try:
            os.utime(FLAG) if os.path.exists(FLAG) else open(FLAG, "a").close()
        except OSError as e:
            print("disk_watch: could not write %s: %s" % (FLAG, e), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warn", type=float, default=90)
    ap.add_argument("--full", type=float, default=97)
    ap.add_argument("--du-budget", type=float, default=300, help="seconds of du per filesystem")
    ap.add_argument("--cache-hours", type=float, default=6)
    a = ap.parse_args()
    site = os.environ.get("CCORCH_SITE", "?")
    cache, dirty, alerts, full = load_cache(), False, [], []
    for dev, e in sorted(filesystems().items(), key=lambda kv: kv[1]["mount"]):
        used, total, pct = usage(e["paths"][0])
        print("%s %s (%s): %s of %s used, %d%%" % (site, e["mount"], ", ".join(e["names"]), gb(used), gb(total), pct))
        c = cache.get(e["area"])
        if not c or time.time() - c.get("when", 0) > 3600 * a.cache_hours:
            top, partial = du_top(e["area"], a.du_budget)
            c = {"when": time.time(), "top": top, "partial": partial}
            cache[e["area"]], dirty = c, True
        age = (time.time() - c["when"]) / 3600
        print("  largest in %s (du %.1f h ago%s):" % (e["area"], age,
                                                     ", partial: du ran out of time" if c["partial"] else ""))
        for b, d in c["top"][:TOP]:
            print("    %10s  %s" % (gb(b), d))
        if pct >= a.warn:
            alerts.append("DISK %s at %d%%" % (e["mount"], pct))
        if pct >= a.full:
            full.append("DISK %s at %d%%" % (e["mount"], pct))
    if dirty:
        save_cache(cache)
    if full:
        set_flag("; ".join(full) + " (disk_watch %s): the feeder submits nothing" % time.strftime("%Y-%m-%d %H:%M"))
        print("flag %s set: sweep/feed_capped.py submits nothing while it is under 30 min old" % FLAG)
    elif os.path.exists(FLAG):
        try:
            os.remove(FLAG)
            print("flag %s removed: every filesystem below %g%%" % (FLAG, a.full))
        except OSError as e:
            print("disk_watch: could not remove %s: %s" % (FLAG, e), file=sys.stderr)
    if alerts:
        print("; ".join(alerts), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
