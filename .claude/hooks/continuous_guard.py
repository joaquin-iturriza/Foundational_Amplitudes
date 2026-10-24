#!/usr/bin/env python3
"""Stop hook: refuse to end a turn while a continuous task has gone quiet.

Why: the supervisor, the event waiter and the goal check-ins all wake the agent on JOB events. Work that produces no
job (the CPU preprocessing exploration the user approved under D83: "keep working on this ... try many things out")
raises no event, so after each event the turn ended and the exploration silently stopped for hours (2026-10-22; the
user: "everything is set up so you can continuously work on the goals set, and you are not doing it, this should be
fixed").

Rule: .claude/continuous.json lists the continuous tasks, each with the paths its RESULTS are committed to (data,
figures, never the scripts: a committed script with nothing running kept a task "alive" for hours, 2026-10-24), its
work items, and a maximum age. A task is alive while the newest commit touching its result paths is younger than
max_age_min, or while a run of one of its items is queued or running in the site registry (runs.db). Nothing else
keeps it alive. While any listed task is stale, the turn cannot end:
either produce and commit its next result, start a worker that will, or remove the task from the file (a committed,
deliberate act: when the user stops it or its queue is done and written up).

    {"tasks": [{"name": "the level (D83, D99)", "items": ["catalog_norm"],
                "paths": ["analysis/catalog_daniel/data/norm_*", "analysis/catalog_daniel/figs/qperp*"],
                "max_age_min": 90}]}
"""
import json, os, subprocess, sys, time

REPO = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONF = os.environ.get("CONTINUOUS_CONF") or os.path.join(REPO, ".claude", "continuous.json")
RUNS_DB = os.environ.get("CONTINUOUS_RUNS_DB") or os.path.expanduser("~/.local/share/ccorch/runs.db")
LIVE = ("SUBMITTED", "PENDING", "QUEUED", "RUNNING", "IDLE", "HELD_BY_DEP")


def live_runs(items):
    """runs of these work items that are queued or running in the site registry (a worker on the task)"""
    if not items:
        return []
    try:
        import sqlite3
        c = sqlite3.connect(f"file:{RUNS_DB}?mode=ro", uri=True, timeout=10)
        q = "select id from runs where item in (%s) and state in (%s)" % (",".join("?" * len(items)), ",".join("?" * len(LIVE)))
        return [r[0] for r in c.execute(q, list(items) + list(LIVE))]
    except Exception:
        return None


def newest_commit(paths):
    try:
        out = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%ct", "--", *paths],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out) if out else 0
    except Exception:
        return None


def main():
    if not os.path.exists(CONF):
        return 0
    try:
        tasks = json.load(open(CONF)).get("tasks", [])
    except Exception as e:
        print(f"continuous_guard: cannot read {CONF}: {e}", file=sys.stderr)
        return 2
    now, stale = time.time(), []
    for t in tasks:
        if t.get("stopped") or t.get("finished"):             # stopped on the user's recorded decision (continuous_list_guard.py)
            continue
        ct = newest_commit(t["paths"])
        if ct is None:
            continue                     # git unavailable: never block on the guard's own failure
        age = (now - ct) / 60
        if age <= float(t.get("max_age_min", 90)):
            continue
        runs = live_runs(t.get("items") or [])
        if runs:                         # a queued or running job of the task's items is its worker
            continue
        stale.append((t["name"], age, t["paths"]))
    if not stale:
        return 0
    print("CONTINUOUS TASK STALE: the turn cannot end while a continuous task has produced nothing.", file=sys.stderr)
    for name, age, paths in stale:
        print(f"  {name}: newest commit on {', '.join(paths)} is {age:.0f} min old", file=sys.stderr)
    print("Produce and commit its next result now, or start a background worker that will, or (only if the user\n"
          "stopped it) set \"stopped\": \"D<n>\", or (its queue is finished and written up) set \"finished\": \"<the write-up commit>\" and commit; never ask the user to close finished work.",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
