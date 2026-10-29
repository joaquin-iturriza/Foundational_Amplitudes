#!/usr/bin/env python3
"""Which backlog each reviewer of a session reviewed: per reviewer subagent, the pillar and the commit its review
started from (the first `git diff <sha>` it ran on that pillar's paths, or the `since <sha>` a `review_backlog.sh begin`
printed), grouped so that several reviewers of one backlog show up together.

    python3 tools/review_overlap.py SID [--day YYYY-MM-DD]
"""
import argparse, collections, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_calls import PROJ

PILLAR_OF = (("claudemd-keeper", r"CLAUDE\.md"), ("notes-editor", r"docs/\*?\.?tex|results\.tex"),
             ("repo-reviewer", r"\*\.py|tools|sweep|scripts"))
DIFF = re.compile(r"git (?:-C \S+ )?diff ([0-9a-f]{7,40})(?:\.\.\S+)?\s+--\s+(.*)")
BEGIN_OUT = re.compile(r"(?:since|git diff) ([0-9a-f]{7,40})")


def reviewer_starts(path):
    """(pillar, sha, first timestamp) of one subagent transcript if it reviewed a pillar, else None."""
    first_ts, begin, uses = None, None, {}
    for line in open(path):
        r = json.loads(line)
        first_ts = first_ts or r.get("timestamp")
        for x in (r.get("message") or {}).get("content") or []:
            if not isinstance(x, dict):
                continue
            if x.get("type") == "tool_use" and x.get("name") == "Bash":
                cmd = x["input"].get("command", "")
                m = re.search(r"review_backlog\.sh begin ([a-z-]+)", cmd)
                if m and not begin:
                    begin = [m.group(1), None]
                    uses[x["id"]] = "begin"
                for d in DIFF.finditer(cmd):
                    for name, pat in PILLAR_OF:
                        if re.search(pat, d.group(2)):
                            return name, d.group(1)[:8], first_ts
            if x.get("type") == "tool_result" and begin:
                c = x.get("content")
                c = c if isinstance(c, str) else " ".join(b.get("text", "") for b in c or [] if isinstance(b, dict))
                m = BEGIN_OUT.search(c)
                if m:
                    return begin[0], m.group(1)[:8], first_ts
    return (begin[0], "?", first_ts) if begin else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("--day")
    a = ap.parse_args()
    groups = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(PROJ, a.sid, "subagents", "agent-*.jsonl"))):
        s = reviewer_starts(f)
        if s and (not a.day or (s[2] or "").startswith(a.day)):
            groups[(s[0], s[1])].append((s[2] or "")[:16])
    n = sum(len(v) for v in groups.values())
    print("%d reviewers, %d distinct backlogs" % (n, len(groups)))
    for (p, sha), ts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print("%-16s from %-8s  %2d reviewers  %s" % (p, sha, len(ts), " ".join(sorted(ts))))


if __name__ == "__main__":
    main()
