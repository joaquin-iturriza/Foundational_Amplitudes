#!/usr/bin/env python3
"""Recall over our own stores (agent_redesign step 27): search the summary records we already keep, then expand one
back to the raw transcript lines behind it. No memory file: every record is read from its store at call time, and
the transcript scan of compaction summaries is cached only as a derived, rebuildable index.

    python3 tools/recall.py search TERM [TERM ...] [-n 10]   ranked records (all terms scored, case-insensitive)
    python3 tools/recall.py expand ID [--before 30] [--after 5] [--grep WORD]
                                                             the raw lines behind record ID: Joaquin's own messages and
                                                             the assistant's text (no tool output) of the agent the
                                                             record belongs to (the main session when unknown), from
                                                             --before to --after minutes around its time
    python3 tools/recall.py index                            rebuild the compaction-summary index

Stores and their record ids:
    D<n>                  ledger decisions (site decisions --project FA --all): Joaquin's quote, kind, covers
    <entry>/finding/<i>   plan findings, <entry>/ask/<i> questions to Joaquin, <entry>/handoff hand-offs (tools/plan.py)
    C<k>                  compaction summaries in the transcripts (main session and subagents)
"""
import glob, json, os, re, subprocess, sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")
INDEX = os.path.expanduser("~/.cache/fa_recall/compactions.jsonl")
MAIN = "main"


def _ts(s):
    s = s.replace("Z", "")[:19]
    return datetime.fromisoformat(s if "T" in s else s + "T00:00:00")


def _transcripts():
    """(agent id, path): main sessions as 'main:<session>', subagents by their id."""
    for p in glob.glob(PROJ + "/*.jsonl"):
        yield MAIN + ":" + os.path.basename(p)[:-6], p
    for p in glob.glob(PROJ + "/*/subagents/agent-*.jsonl"):
        yield os.path.basename(p)[6:-6], p


def index():
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    n = 0
    with open(INDEX, "w") as out:
        for agent, p in _transcripts():
            for line in open(p, errors="replace"):
                if '"isCompactSummary":true' not in line:
                    continue
                d = json.loads(line)
                c = d["message"]["content"]
                text = c if isinstance(c, str) else " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                out.write(json.dumps({"id": "C%d" % n, "kind": "compaction", "agent": agent,
                                      "ts": d["timestamp"][:19], "text": text}) + "\n")
                n += 1
    print("indexed %d compaction summaries" % n)


def _decisions():
    out = subprocess.run(["site", "decisions", "--project", "FA", "--all"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = re.match(r"\s*(D\d+)\s+\S+\s+(\w+)\s+(\S+)(?: \(\w+\))?\s+(.*)", line)
        if m:
            yield {"id": m.group(1), "kind": "decision " + m.group(2), "agent": MAIN, "ts": m.group(3),
                   "text": m.group(4)}


def _plan():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import plan
    for e in plan.load()[0]["entries"]:
        eid = e.get("id")
        for i, f in enumerate(e.get("findings") or []):
            t = f.get("text", "") if isinstance(f, dict) else str(f)
            yield {"id": "%s/finding/%d" % (eid, i), "kind": "finding", "agent": None,
                   "ts": ((f.get("at") or t[:10]) if isinstance(f, dict) else t[:10])[:19], "text": t}
        for i, a in enumerate(e.get("asks") or []):
            yield {"id": "%s/ask/%d" % (eid, i), "kind": "ask", "agent": a.get("agent"),
                   "ts": (a.get("at") or a.get("asked") or a.get("ts") or "")[:19], "text": a.get("ask", "") + " | answer: " + str(a.get("answer"))}
        h = e.get("handoff")
        if h:
            yield {"id": "%s/handoff" % eid, "kind": "handoff", "agent": h.get("agent"), "ts": str(h.get("written", ""))[:19],
                   "text": " ".join("%s: %s" % (k, v) for k, v in h.items())}


def records():
    yield from _decisions()
    yield from _plan()
    if not os.path.exists(INDEX):
        index()
    for line in open(INDEX):
        yield json.loads(line)


def search(terms, n=10):
    terms = [t.lower() for t in terms]
    hits = []
    for r in records():
        low = r["text"].lower()
        score = sum(low.count(t) > 0 for t in terms)
        if score:
            hits.append((score, r["ts"], r))
    hits.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for score, _, r in hits[:n]:
        low = r["text"].lower()
        k = min([low.find(t) for t in terms if t in low] or [0])
        print("%s  [%s] %s %s  ...%s..." % (r["id"], r["kind"], r["ts"], r.get("agent") or "",
                                             r["text"][max(0, k - 120):k + 280].replace("\n", " ")))


def expand(rid, before=30, after=5, grep=None):
    r = next((x for x in records() if x["id"] == rid), None)
    if r is None:
        sys.exit("no record %s" % rid)
    if not r["ts"] or len(r["ts"]) < 10:
        sys.exit("record %s has no time" % rid)
    t0 = _ts(r["ts"])
    lo, hi = t0 - timedelta(minutes=before), t0 + timedelta(minutes=after)
    agent = r.get("agent") or MAIN
    paths = [p for a, p in _transcripts() if a == agent or (agent == MAIN and a.startswith(MAIN))]
    days = {(lo + timedelta(days=k)).strftime("%Y-%m-%d") for k in range((hi - lo).days + 2)}
    print("# %s [%s] %s, agent %s, %s .. %s" % (rid, r["kind"], r["ts"], agent, lo, hi))
    for p in paths:
        for line in open(p, errors="replace"):
            if not any(day in line[:4000] or day in line[-400:] for day in days):
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ts = d.get("timestamp")
            if not ts or not (lo <= _ts(ts) <= hi) or d.get("type") not in ("user", "assistant"):
                continue
            c = (d.get("message") or {}).get("content")
            if isinstance(c, list):
                if any(isinstance(x, dict) and x.get("type") == "tool_result" for x in c):
                    continue
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
            if not c or (d["type"] == "user" and d.get("turnOrigin") not in (None, "human")):
                continue
            if d["type"] == "user" and (c.startswith("<") or c.startswith("Stop hook")):
                continue
            if grep and grep.lower() not in c.lower():
                continue
            who = "JOAQUIN" if d["type"] == "user" else "assistant"
            print("--- %s %s\n%s" % (ts[:19], who, c[:3000]))


def main(a):
    if not a or a[0] in ("-h", "--help"):
        print(__doc__)
    elif a[0] == "index":
        index()
    elif a[0] == "search":
        n = 10
        if "-n" in a:
            i = a.index("-n")
            n = int(a[i + 1])
            a = a[:i] + a[i + 2:]
        search(a[1:], n)
    elif a[0] == "expand":
        opt = lambda k, dv: (a[a.index(k) + 1] if k in a else dv)
        expand(a[1], int(opt("--before", 30)), int(opt("--after", 5)), opt("--grep", None))


if __name__ == "__main__":
    main(sys.argv[1:])
