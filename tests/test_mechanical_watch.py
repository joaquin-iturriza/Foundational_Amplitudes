#!/usr/bin/env python3
"""D213's live check (tools/agent_tasks.py mechanical_report) on fixtures: the first 30 mechanical-worker tasks, an
ALERT at 5 failures among them.

  M1  30 tasks, 4 failed: no ALERT
  M2  30 tasks, 5 failed: one ALERT (kind ALERT, key mechanical_revert) naming the 5 agents
  M3  35 tasks whose 5 failures are all after the 30th: no ALERT (only the first 30 count)
  M4  other agent types (compact-worker, reviewers) are never counted

Run:  python3 tests/test_mechanical_watch.py
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import agent_tasks as T      # noqa: E402


def rows(n, failed, typ="mechanical-worker"):
    return [{"agent": "a%03d" % i, "type": typ, "model": "claude-sonnet-5-5", "start": "2026-11-01T%02d:%02d" % (i // 60, i % 60),
             "end": "", "description": "step %d" % i, "stood": i not in failed, "usd": 0.1, "rework_usd": 0.0,
             "corrections": [] if i not in failed else [{"kind": "sendmessage"}]} for i in range(n)]


def run(rs):
    with tempfile.TemporaryDirectory() as d:
        fake, calls = os.path.join(d, "site"), os.path.join(d, "calls")
        open(fake, "w").write('#!/bin/bash\nprintf "%%s\\n" "$*" >> %s\n' % calls)
        os.chmod(fake, 0o755)
        out = os.path.join(d, "watch.json")
        T.mechanical_report(rs, out, event=fake)
        rep = json.load(open(out))
        return rep, (open(calls).read().splitlines() if os.path.exists(calls) else [])


def test_all():
    rep, calls = run(rows(30, {1, 5, 9, 20}))
    assert rep["failed"] == 4 and not rep["alert"] and not calls, (rep["failed"], calls)
    print("M1 ok: 4 of 30 failed, no ALERT")
    rep, calls = run(rows(30, {1, 5, 9, 20, 29}))
    assert rep["alert"] and len(calls) == 1 and "--kind ALERT" in calls[0] and "--key mechanical_revert" in calls[0] \
        and "a029" in calls[0], calls
    print("M2 ok: 5 of 30 failed, one ALERT:", calls[0][:120])
    rep, calls = run(rows(35, {30, 31, 32, 33, 34}))
    assert rep["tasks"] == 30 and rep["failed"] == 0 and not calls, rep["failed"]
    print("M3 ok: failures after the 30th are not counted")
    rep, calls = run(rows(10, set(range(10)), typ="compact-worker") + rows(3, set(), typ="repo-reviewer"))
    assert rep["tasks"] == 0 and not calls
    print("M4 ok: other agent types are not counted")


if __name__ == "__main__":
    test_all()
    print("all ok")
