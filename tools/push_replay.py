"""Replay of the reference pushes (.claude/hooks/context_push.py) over recorded agent transcripts.

    python3 tools/push_replay.py [--dirs DIR[:DIR]] [--since ISO] [--until ISO] [--check] [--out JSON]

For every subagent transcript under ~/.claude/projects/-home-joaquin-work-FA/*/subagents/ whose first record falls in
[--since, --until), each tool call is replayed in order through context_push.matches() and its once-per-agent
delivery, with the hook registration of .claude/settings.json (PreToolUse on acts, PostToolUse on reads and acts),
using the sections found in --dirs (default: the live section directories). Per agent:
  pushed_chars    characters of REFERENCE text delivered
  token_calls     sum over pushes of tokens(push) * (calls after the push): what the push costs in context over the
                  agent's remaining calls (each later call re-reads it)
  per_call        token_calls / calls: the push's share of context per call
  held_calls      PreToolUse calls the hook held (a section the call triggers did not fit in one output), each sent
                  again and counted as one more call
tokens(push) = 1.722 * chars/4 (tools/context_breakdown.py's fit of API context growth on chars/4, f3624f9c).
--check compares the replayed pushed_chars with the REFERENCE text actually recorded in each transcript (only
meaningful with the live directories over a period when the live sections were in force).
Matched before/after: run twice over the same transcripts with two --dirs; the difference is the cut's saving on the
same tool calls.
"""
import argparse, glob, json, os, re, statistics, sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TR = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")
TOK_PER_C4 = 1.722
PRE = {"Bash", "Edit", "Write", "MultiEdit", "NotebookEdit", "Agent", "SendMessage", "SendUserFile"}
HARNESS_CAP, PREVIEW = 10000, 2178
POST = {"Read", "Grep", "Glob", "Bash", "Edit", "Write", "MultiEdit", "NotebookEdit"}


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_hook(dirs, hook=None, state=None):
    """The push hook module: the live file, or `hook` (a path, or git:<rev> for the file at that commit)."""
    import importlib.util, subprocess, tempfile
    os.environ["CLAUDE_PROJECT_DIR"] = ROOT
    if dirs:
        os.environ["CONTEXT_DIRS"] = dirs
    os.environ["CONTEXT_STATE"] = state or tempfile.mkdtemp(prefix="push_replay_")
    live = os.path.join(ROOT, ".claude", "hooks", "context_push.py")
    path = hook or live
    if path.startswith("git:"):
        src = subprocess.run(["git", "-C", ROOT, "show", path[4:] + ":.claude/hooks/context_push.py"],
                             capture_output=True, text=True, check=True).stdout
        path = os.path.join(os.environ["CONTEXT_STATE"], "context_push_%s.py" % path[4:])
        open(path, "w").write(src)
    sys.path.insert(0, os.path.dirname(live))
    spec = importlib.util.spec_from_file_location("context_push_replayed", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def replay(path, cp):
    """(calls, pushes [(call index, chars received)], recorded REFERENCE chars) of one transcript. A hook output over
    HARNESS_CAP chars reaches the agent as a PREVIEW-char preview (the harness's persisted-output), as recorded."""
    calls, seen, pushes, recorded, holds = -1, set(), [], 0, 0
    agent = os.path.basename(path)[:-6]
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") == "attachment":
            a = d.get("attachment") or {}
            if a.get("type") == "hook_additional_context":
                c = a.get("content")
                c = "\n".join(c) if isinstance(c, list) else str(c or "")
                if "REFERENCE " in c:
                    recorded += len(c)
            continue
        if d.get("type") != "assistant":
            continue
        m = d.get("message") or {}
        if m.get("id") not in seen:
            seen.add(m.get("id"))
            calls += 1
        for b in m.get("content") or []:
            if not isinstance(b, dict) or b.get("type") != "tool_use":
                continue
            tool, ti = b.get("name") or "", b.get("input") or {}
            for phase in ("pre", "post"):
                if tool not in (PRE if phase == "pre" else POST):
                    continue
                inp = {"session_id": "replay_" + agent, "agent_id": agent, "tool_name": tool, "tool_input": ti,
                       "hook_event_name": "PreToolUse" if phase == "pre" else "PostToolUse"}
                while True:
                    if hasattr(cp, "respond_full"):
                        _, text, held = cp.respond_full(inp, phase)
                    else:
                        (_, text), held = cp.respond(inp, phase), []
                    if text:
                        pushes.append((calls, len(text) if len(text) <= HARNESS_CAP else PREVIEW))
                    if not held or phase != "pre":
                        break
                    calls += 1                     # the held call is sent again: one more call
                    holds += 1
    return calls + 1, pushes, recorded, holds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", help="':'-separated section directories (default: live)")
    ap.add_argument("--hook", help="the hook file to replay: a path or git:<rev> (default: the live file)")
    ap.add_argument("--since", default="2026-10-30T11:35:10+00:00")
    ap.add_argument("--until", default="2100-01-01T00:00:00+00:00")
    ap.add_argument("--min-calls", type=int, default=5)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    cp = load_hook(a.dirs, a.hook)
    lo, hi = ts(a.since), ts(a.until)
    rows = []
    for f in sorted(glob.glob(os.path.join(TR, "*", "subagents", "**", "agent-*.jsonl"), recursive=True)):
        try:
            t0 = json.loads(open(f).readline()).get("timestamp")
        except (ValueError, OSError):
            continue
        if not t0 or not (lo <= ts(t0) < hi):
            continue
        n, pushes, rec, holds = replay(f, cp)
        if n < a.min_calls:
            continue
        tc = sum(TOK_PER_C4 * c / 4 * max(0, n - 1 - k) for k, c in pushes)
        rows.append({"agent": os.path.basename(f)[:-6], "calls": n, "pushed_chars": sum(c for _, c in pushes),
                     "recorded_chars": rec, "token_calls": round(tc), "per_call": round(tc / n),
                     "pushes": len(pushes), "held_calls": holds})
    per = [r["per_call"] for r in rows]
    summ = {"agents": len(rows), "calls": sum(r["calls"] for r in rows),
            "pushed_chars_median": statistics.median([r["pushed_chars"] for r in rows]) if rows else 0,
            "per_call_tokens_mean": round(sum(r["token_calls"] for r in rows) / max(1, sum(r["calls"] for r in rows))),
            "per_call_tokens_median_agent": statistics.median(per) if per else 0,
            "held_calls": sum(r["held_calls"] for r in rows),
            "dirs": a.dirs or "live", "hook": a.hook or "live", "since": a.since, "until": a.until}
    if a.check:
        bad = [r for r in rows if abs(r["pushed_chars"] - r["recorded_chars"]) > 0.05 * max(1, r["recorded_chars"])]
        summ["check_agents_off_by_5pct"] = len(bad)
        summ["check_replayed_over_recorded"] = round(sum(r["pushed_chars"] for r in rows) /
                                                     max(1, sum(r["recorded_chars"] for r in rows)), 3)
    print(json.dumps(summ, indent=1))
    if a.out:
        json.dump({"summary": summ, "agents": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
