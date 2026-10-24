#!/usr/bin/env python3
"""Stop hook: refuse to end a turn while a continuous task has gone quiet.

Why: the supervisor, the event waiter and the goal check-ins all wake the agent on JOB events. Work that produces no
job (the CPU preprocessing exploration the user approved under D83: "keep working on this ... try many things out")
raises no event, so after each event the turn ended and the exploration silently stopped for hours (2026-10-22; the
user: "everything is set up so you can continuously work on the goals set, and you are not doing it, this should be
fixed").

Rule: .claude/continuous.json lists the continuous tasks, each with the paths its work commits to and a maximum age.
A task is alive while the newest commit touching its paths is younger than max_age_min. A real result is the only
thing that keeps it alive (a touched file would prove nothing). While any listed task is stale, the turn cannot end:
either produce and commit its next result, start a worker that will, or remove the task from the file (a committed,
deliberate act: when the user stops it or its queue is done and written up).

    {"tasks": [{"name": "preprocessing exploration (D83)",
                "paths": ["analysis/catalog_daniel/norm_explore*", "analysis/catalog_daniel/data/norm_explore*"],
                "max_age_min": 90}]}
"""
import json, os, subprocess, sys, time

REPO = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONF = os.path.join(REPO, ".claude", "continuous.json")


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
        if t.get("stopped"):             # stopped on the user's recorded decision (continuous_list_guard.py)
            continue
        ct = newest_commit(t["paths"])
        if ct is None:
            continue                     # git unavailable: never block on the guard's own failure
        age = (now - ct) / 60
        if age > float(t.get("max_age_min", 90)):
            stale.append((t["name"], age, t["paths"]))
    if not stale:
        return 0
    print("CONTINUOUS TASK STALE: the turn cannot end while a continuous task has produced nothing.", file=sys.stderr)
    for name, age, paths in stale:
        print(f"  {name}: newest commit on {', '.join(paths)} is {age:.0f} min old", file=sys.stderr)
    print("Produce and commit its next result now, or start a background worker that will, or (only if the user\n"
          "stopped it or its queue is finished and written up) remove it from .claude/continuous.json and commit.",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
