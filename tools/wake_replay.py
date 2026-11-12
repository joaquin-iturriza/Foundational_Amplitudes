#!/usr/bin/env python3
"""The main session's wakes in a time window, before and after the 2026-11-12 absorb fixes (agent_redesign step 23).

A wake is a turn-starting prompt of the main transcript that is not typed by Joaquin, not a Stop-hook refusal (those
continue a turn) and not a tool result, and that is followed by a model call (an assistant record) before the next
prompt: a prompt the hook blocked, or one folded into a running turn, is recorded with no call of its own. Each is
classed:
  waiter      a background command's end whose summary names the event waiter (the batch's own delivery: kept)
  shell       another background command's end
  agent_done  an agent's 'finished' notification; after the fix absorbed when its hand-back sits in the spool
  agent_msg   an agent's message (hand-back or mid-run) that reached the transcript; after the fix spooled
  other       anything else (artifact notices, ...)
Messages that the hook already spooled never reach the transcript and are counted from absorb.log.

    python3 tools/wake_replay.py --since 2026-11-12T00:00 --until 2026-11-12T00:40 [--transcript T]
"""
import argparse, glob, json, os, re

CCORCH = os.path.expanduser("~/.local/share/ccorch")
DEF_T = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA/71602535-0955-42ad-a58f-23dc320798dc.jsonl")


def text_of(r):
    c = (r.get("message") or {}).get("content")
    if isinstance(c, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
            return None
        return " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    return c


def classify(t):
    if "<task-notification>" in t:
        m = re.search(r"<task-id>(a[0-9a-f]{16})</task-id>", t)
        if m:
            return "agent_done", m.group(1)
        s = (re.findall(r"<summary>(.*?)</summary>", t, re.S) or [""])[0]
        if "<task-type>" in t and "Background command" not in s:
            return "other", s[:60]
        return ("waiter" if re.search(r"waiter", s, re.I) else "shell"), s[:60]
    m = re.match(r"\s*(?:Another Claude session sent a message:\s*)?<agent-message from=\"(a[0-9a-f]{16})\"", t)
    if m:
        return "agent_msg", m.group(1)
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--transcript", default=DEF_T)
    a = ap.parse_args()
    spooled = {os.path.basename(p).split("_")[1] for p in glob.glob(os.path.join(CCORCH, "spool", "FA", "*.txt"))}
    rows, pending = [], None
    for line in open(a.transcript, errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        ts = r.get("timestamp", "")
        if r.get("isSidechain"):
            continue
        if r.get("type") == "assistant" and pending:
            rows.append(pending)
            pending = None
        if r.get("type") == "user" and not r.get("isMeta") and text_of(r):
            pending = None          # a new prompt before any call: the pending one made no call of its own
        if r.get("type") != "user" or r.get("isSidechain") or r.get("isMeta") or not (a.since <= ts < a.until):
            continue
        t = text_of(r)
        if not t:
            continue
        kind, what = classify(t)
        if kind:
            after = "absorbed" if (kind == "agent_done" and what in spooled) or kind == "agent_msg" else "wakes"
            pending = (ts[11:19], kind, what, after)
    for r in rows:
        print("  %s %-10s %-45s after: %s" % r)
    log = [l for l in open(os.path.join(CCORCH, "absorb.log")) if a.since <= l[:16] < a.until]
    before, after = len(rows), sum(r[3] == "wakes" for r in rows)
    print(json.dumps({"since": a.since, "until": a.until, "wakes_before": before, "wakes_after": after,
                      "by_kind": {k: sum(r[1] == k for r in rows) for k in sorted({r[1] for r in rows})},
                      "absorb_log": {"spooled": sum("waiter_alive=True" in l for l in log),
                                     "passed_no_waiter": sum("waiter_alive=False" in l for l in log),
                                     "passed_mid_run_msg": sum("not-one-handback" in l for l in log)}}))


if __name__ == "__main__":
    main()
