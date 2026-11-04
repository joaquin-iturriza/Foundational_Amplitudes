"""Replay of context_guard.compacted_agent on every recorded SessionStart:compact event since 2026-10-29 (transcripts read in place): which session compacted, the main one or a subagent, against what the hook would say.

    python3 tools/compaction_replay.py <hooks dir, e.g. .claude/hooks> [--until ISO]

--until: only events before it (a closed window reproduces). The last line printed is one JSON object: events, right,
main and agent events, the distinct agents, and the agent events before the fix (4550ed8c, 2026-11-04T19:23:34Z),
each of which got the main session's plan, its re-push and the clearing of its markers.
"""
import argparse, glob, json, os, sys
from datetime import datetime
ap = argparse.ArgumentParser()
ap.add_argument("hooks")
ap.add_argument("--until", default="2100-01-01T00:00:00Z")
A = ap.parse_args()
FIX = "2026-11-04T19:23:34Z"
sys.path.insert(0, A.hooks); import context_guard as cg
P = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")
def ts(s): return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
events = []   # (session, main transcript, time, truth)
for main_t in glob.glob(P + "/*.jsonl"):
    sid = os.path.basename(main_t)[:-6]
    for f in [main_t] + glob.glob(P + "/%s/subagents/agent-*.jsonl" % sid):
        for l in open(f, errors="replace"):
            if '"SessionStart:compact"' in l and '"hook_success"' in l and 'context_guard' not in l[:0]:
                try: r = json.loads(l)
                except ValueError: continue
                if (r.get("attachment") or {}).get("hookName") == "SessionStart:compact":
                    who = None if f == main_t else os.path.basename(f)[6:-6]
                    events.append((sid, main_t, r["timestamp"], who))
events = sorted(set(events), key=lambda e: e[2])
events = [e for e in events if "2026-10-29" <= e[2] and ts(e[2]) < ts(A.until)]
def boundary_before(cut):
    def b(path, tail=None):
        last = ""
        for l in open(path, errors="replace"):
            if '"compact_boundary"' in l:
                try: t = json.loads(l).get("timestamp") or ""
                except ValueError: continue
                if t and ts(t) <= cut: last = t
        return last
    return b
ok = bad = 0
for sid, main_t, t, who in events:
    now = ts(t)
    cg._boundary_at = boundary_before(now)
    real = os.path.getmtime
    got = cg.compacted_agent({"session_id": sid, "transcript_path": main_t, "source": "compact"}, now=now)
    good = got == who
    ok += good; bad += not good
    print("%s %s truth=%s got=%s" % ("OK " if good else "BAD", t, who or "main", got or "main"))
print("%d/%d right" % (ok, ok + bad))
agent_ev = [e for e in events if e[3]]
def compactions(ev):     # the hooks of one compaction fire within seconds of each other
    out, last = 0, {}
    for e in ev:
        if e[3] not in last or ts(e[2]) - last[e[3]] > 30: out += 1
        last[e[3]] = ts(e[2])
    return out
print(json.dumps({"events": ok + bad, "right": ok, "main_events": len(events) - len(agent_ev),
                  "agent_events": len(agent_ev), "agent_compactions": compactions(agent_ev), "agents": len({e[3] for e in agent_ev}),
                  "agent_events_before_fix": sum(1 for e in agent_ev if ts(e[2]) < ts(FIX)),
                  "agents_before_fix": len({e[3] for e in agent_ev if ts(e[2]) < ts(FIX)}), "until": A.until}))
