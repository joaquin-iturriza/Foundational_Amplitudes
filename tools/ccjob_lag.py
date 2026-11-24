#!/usr/bin/env python3
"""Lag of the supervisor's maintain commands run as CC-IN2P3 CPU jobs (`site run --recurring`, plan entry
cc_login_jobs, D544): per ccj_ job its queue wait (Start - Submit), run time (End - Start) and age when read (the first
supervisor pass after its End, minus End); per pass the commands skipped (CCJ_NO_RESULT: job not ended) and the
.disk_full age the feeder printed.

    python3 tools/ccjob_lag.py <supervise output file> [--since 2026-11-24T18:30] [--until 2026-11-24T23:09]

sacct is read through `site run ccin2p3 FA` (a scheduler query: allowed on the login node). Times are CC's local time,
the same zone as the supervisor's pass headers (Europe/Paris).
"""
import argparse, hashlib, json, re, subprocess, sys
from datetime import datetime

CMDS = {"disk_watch": "python scripts/disk_watch.py", "feed_capped": "python sweep/feed_capped.py tp3_"}


def key(cmd):
    return hashlib.sha1(("FA\0" + cmd).encode()).hexdigest()[:12]


def passes(path, since):
    """[(time, [maintain lines])] of the supervise passes at or after `since`."""
    out, year = [], since.year
    for line in open(path, errors="replace"):
        m = re.match(r"=== (\d\d-\d\d \d\d:\d\d) supervise", line)
        if m:
            t = datetime.strptime("%d-%s" % (year, m.group(1)), "%Y-%m-%d %H:%M")
            out.append((t, []))
        elif out and "maintain]" in line:
            out[-1][1].append(line.strip())
    return [p for p in out if p[0] >= since]


def jobs(since):
    q = ("sacct -n -X -P -u $USER -S %s -o JobName%%30,JobID,Submit,Start,End,State | grep '^ccj_'"
         % since.strftime("%Y-%m-%dT%H:%M"))
    r = subprocess.run(["site", "run", "ccin2p3", "FA", "--", q], capture_output=True, text=True, timeout=200)
    rows = []
    for l in r.stdout.splitlines():
        f = l.split("|")
        if len(f) == 6 and f[0].startswith("ccj_"):
            ts = [datetime.strptime(x, "%Y-%m-%dT%H:%M:%S") if x[:2] == "20" else None for x in f[2:5]]
            rows.append((f[0][4:], f[1], *ts, f[5]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--since", default="2026-11-24T18:30")
    ap.add_argument("--until", help="only jobs submitted before this time (the end of the saved log)")
    a = ap.parse_args()
    since = datetime.strptime(a.since, "%Y-%m-%dT%H:%M")
    ps = passes(a.log, since)
    names = {key(c): n for n, c in CMDS.items()}
    worst, run_skip, max_skip, flags, waits = {}, 0, 0, [], []
    print("passes: %d (%s)" % (len(ps), ", ".join(p[0].strftime("%H:%M") for p in ps)))
    until = datetime.strptime(a.until, "%Y-%m-%dT%H:%M") if a.until else None
    for k, jid, sub, st, end, state in jobs(since):
        if until and sub and sub >= until:
            continue
        # pass headers carry the minute the pass started: a job ending inside that minute is read by the next pass
        read = next((p[0] for p in ps if end and p[0] > end.replace(second=0)), None)
        wait = (st - sub).total_seconds() if st and sub else None
        run = (end - st).total_seconds() if end and st else None
        age = (read - end).total_seconds() / 60 if read and end else None
        if wait is not None:
            waits.append(wait)
        if age is not None:
            worst[names.get(k, k)] = max(worst.get(names.get(k, k), 0), age)
        print("JOB %-12s %-10s %s submit %s wait %s s run %s s state %s read %s age %s min" % (
            names.get(k, k), jid, k, sub and sub.strftime("%H:%M:%S"), wait, run, state,
            read and read.strftime("%H:%M"), age if age is None else "%.1f" % age))
    for t, lines in ps:
        skipped = sum("CCJ_NO_RESULT" in l for l in lines)
        flag = [m.group(1) for l in lines for m in [re.search(r"\.disk_full, (\d+) min old", l)] if m]
        print("PASS %s skipped %d flag_age_min %s" % (t.strftime("%H:%M"), skipped, ",".join(flag) or "-"))
        run_skip = run_skip + 1 if skipped else 0
        max_skip, flags = max(max_skip, run_skip), flags + [int(f) for f in flag]
    gaps = [(b[0] - a[0]).total_seconds() / 60 for a, b in zip(ps, ps[1:])]
    print("SUMMARY passes %d, pass gap max %.0f min; max read age min %s; passes in a row with a skip %d; "
          "max .disk_full age at a feed read %s min" % (len(ps), max(gaps or [0]), json.dumps(
              {k: round(v, 1) for k, v in worst.items()}), max_skip, max(flags or [None])))
    print(json.dumps({"passes": len(ps), "gap_max_min": round(max(gaps or [0])), "skips_in_row_max": max_skip,
                      "flag_age_max_min": max(flags or [0]), "queue_wait_max_s": max(waits or [0]),
                      "age_max_min": {k: round(v, 1) for k, v in worst.items() if k in CMDS}}))


if __name__ == "__main__":
    sys.exit(main())
