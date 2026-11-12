#!/usr/bin/env python3
"""How many times `site events --wait` returns (main-session wakes) over a stretch of the real event ledger, under
three waiter rules (plan entry agent_redesign, step 23):

  none     every event returns at the next 60 s poll (no batch)
  old      a 600 s window from the first event, ended at once by any kind outside RUN_DONE/DONE/READY/HANDBACK
           (the site code to 2026-11-12)
  new      the same window, ended at once only by RUN, GAP, OVERRUN, ALERT (URGENT_KINDS, 2026-11-12)

An event counts when it stayed open more than --auto seconds (a clean run end the supervisor acks at once wakes no
one). Not modelled: the quiet filter (events of an entry whose owner is working do not wake the main session) and the
time the main session takes to re-arm its waiter; so these are the waiter's returns, an upper bound on wakes from it.

    python3 tools/waiter_replay.py --since "2026-11-11 20:00" --until "2026-11-12 00:00" [--project FA] [--window 600]
Prints one JSON line {rule: returns} with the counted events by kind.
"""
import argparse, calendar, json, os, sqlite3, time
from collections import Counter

ROUTINE = ("RUN_DONE", "DONE", "READY", "HANDBACK")
URGENT = ("RUN", "GAP", "OVERRUN", "ALERT")


def utc(s):
    return calendar.timegm(time.strptime(s, "%Y-%m-%d %H:%M"))


def returns(ev, window, ends_window, poll=60):
    """Number of waiter returns over [(ts, kind)] sorted by ts; a waiter is re-armed right after each return."""
    n, i = 0, 0
    while i < len(ev):
        t0, k0 = ev[i]
        if window and not ends_window(k0):
            end = t0 + window
            hit = [t for t, k in ev[i + 1:] if t < end and ends_window(k)]
            t_ret = hit[0] if hit else end
        else:
            t_ret = t0 + poll       # the poll that sees it also sees what came in that minute
        while i < len(ev) and ev[i][0] <= t_ret:
            i += 1
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True, help="UTC, YYYY-MM-DD HH:MM")
    ap.add_argument("--until", required=True, help="UTC, YYYY-MM-DD HH:MM")
    ap.add_argument("--project", default="FA")
    ap.add_argument("--window", type=int, default=600)
    ap.add_argument("--auto", type=int, default=5, help="events acked within this many seconds woke no one")
    ap.add_argument("--db", default=os.path.expanduser("~/.local/share/ccorch/runs.db"))
    a = ap.parse_args()
    c = sqlite3.connect("file:%s?mode=ro" % a.db, uri=True)
    rows = c.execute("SELECT ts, kind, acked FROM events WHERE (project=? OR project IS NULL) AND ts>=? AND ts<? "
                     "ORDER BY ts, id", (a.project, utc(a.since), utc(a.until))).fetchall()
    ev = [(ts, k) for ts, k, ack in rows if ack is None or ack - ts > a.auto]
    out = {"since": a.since, "until": a.until, "events": len(ev), "by_kind": dict(Counter(k for _, k in ev)),
           "none": returns(ev, 0, lambda k: True),
           "old": returns(ev, a.window, lambda k: k not in ROUTINE),
           "new": returns(ev, a.window, lambda k: k in URGENT)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
