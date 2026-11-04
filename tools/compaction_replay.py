"""Replay of context_guard.compacted_agent on every recorded SessionStart:compact event since 2026-10-29 (transcripts read in place): which session compacted, the main one or a subagent, against what the hook would say.

    python3 tools/compaction_replay.py <hooks dir, e.g. .claude/hooks>
"""
import glob, json, os, sys
from datetime import datetime
sys.path.insert(0, sys.argv[1]); import context_guard as cg
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
events = [e for e in events if e[2] >= "2026-10-29"]
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
