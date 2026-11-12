#!/usr/bin/env python3
"""Waiter returns over 2026-11-12 00:19-12:19 UTC (agent_redesign step 24): tools/waiter_replay.py's three rules
(none, old, new) against the batch window, beside the main session's actual waiter wakes (tools/wake_replay.py).
    python analysis/agent_redesign/plot_waiter_live.py <json> <out base>"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import plot_style as ps  # noqa: E402

d = json.load(open(sys.argv[1]))
w = sorted(d["replay_by_window_s"], key=int)
x = [int(k) / 60 for k in w]
fig, ax = ps.figure()
for rule, col, mk in (("none", ps.C.grey, "s"), ("old", ps.C.blue, "o"), ("new", ps.C.vermillion, "^")):
    ax.plot(x, [d["replay_by_window_s"][k][rule] for k in w], marker=mk, color=col, label=f"replay, {rule} rule")
ax.plot([10], [d["actual"]["by_kind"]["waiter"]], marker="*", ls="", color=ps.C.green, label="actual waiter wakes")
ax.set_xscale("log")
ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x]); ax.set_xticks([], minor=True)
ax.set_xlabel("batch window (min)")
ax.set_ylabel("waiter returns in 12 h")
ps.legend(ax, "upper right")
ps.save(fig, sys.argv[2])
