#!/usr/bin/env python3
"""Agent tasks and whether their output stood, per model: the small-model question (Joaquin 2026-10-30).

    python3 tools/agent_tasks.py [--since 2026-09-01] [--out analysis/agent_redesign/agent_tasks.json]
                                 [--sample 5]

Unit: one agent (an Agent launch and its resumes: one subagents/agent-<id>.jsonl under ~/.claude/projects), with its
model (the model of its API calls), type, size and cost (tools/usage_windows.py prices, calls from its cache:
run `python3 tools/usage_windows.py extract` first).
Type: review (repo-reviewer, notes-editor, claudemd-keeper, design-reviewer); owner (brief opens "You own one entry");
tool fix (it edited files, all under tools/, .claude/, scripts/, tests/, sites/ or ~/.local/bin); analysis (the rest,
read-only work included).
Size band: terciles of the brief's length within each type (known before the task runs; calls and cost are not used,
since a cheaper model lands lower on both). Effort: the launch call's effort field, else "session default".
Scored: only tasks ending before --scored-before (48 h of follow-up); the rest are listed, not scored.
Model family in cells: opus, opus-low (launched at effort low, D138's reference for mechanical steps), sonnet, haiku.
A task failed when, within 48 h of its last call, any of:
  sendmessage  a SendMessage to it whose text points at an error (ERR below), not the sender correcting its own brief;
               rework = its own calls after that message
  review_fix   a reviewer it launched, or one whose verdict names one of its commits, returned FIX/BLOCK (a design
               reviewer's last plan design-pass/design-fix call, else the first verdict word); rework = its
               own calls after the verdict
  later_fix    a later commit by someone else touching one of its files with fix/fixes/revert in the subject's first 120 characters;
               rework = the calls of the agent that made that commit in the hour up to it (0 when the main session
               or an unknown author made it: not counted)
  user         a message typed by Joaquin, within 3 h after the task's result reached the main session and before
               any other agent's result, with ERR words and naming an agent, reviewer, owner or step; rework = not
               counted (the main session's work is shared)
  later_fix needs the later commit to remove at least one line (> 25 characters) the task added (its edits or its
  commits), not only to touch the same file.
Otherwise it stood.
"""
import argparse, glob, json, os, re, subprocess, sys
from collections import defaultdict, Counter
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import usage_windows as uw
from usage_report import text_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVIEWERS = {"repo-reviewer", "notes-editor", "claudemd-keeper", "design-reviewer"}
TOOLPATH = re.compile(r"^(tools/|\.claude/|scripts/|tests/|sites/|/home/joaquin/\.local/bin|~/.local/bin)")
ERR = re.compile(r"\b((is|was|are|were|looks|got it|that'?s|it'?s) (wrong|incorrect|broken)|mistake|"
                 r"not what (i|you|he|joaquin) asked|you (missed|broke|forgot|skipped|dropped)|that'?s not (right|correct|what)|"
                 r"is not right|isn'?t right|bug in your|redo (it|this|the)|doesn'?t add up|don'?t add up|"
                 r"you (said|claimed)[^.]{0,60}but|wrongly|misread|your (fix|change|commit|number|result|analysis) "
                 r"(is|was|did)n?'?t?)\b", re.I)
MENTION = re.compile(r"\b(agent|agents|reviewer|review|owner|step)\b", re.I)
VERDICT = re.compile(r"\b(PASS\w*|FIX|BLOCK\w*|FAIL\w*)\b")
SELF_CORR = re.compile(r"correction to your brief|\bI (said|wrote|told you|gave you)\b[^.]{0,80}(wrong|incorrect)", re.I)
COMMIT = re.compile(r"\[(?:[\w./'-]+) ([0-9a-f]{7,12})\]")
HASH = re.compile(r"\b[0-9a-f]{8}\b")
FIXSUBJ = re.compile(r"\b(fix|fixes|revert|reverts|reverted)\b", re.I)    # in the subject's first 120 characters


def ts(s):
    return datetime.fromisoformat(s[:19])


def scan_agent(f):
    """brief, final text, files edited, commits, SendMessages and Agent launches it made, verdict text."""
    a = {"brief": None, "final": "", "edited": set(), "commits": [], "sent": [], "launched": [], "added": set(), "design_verdict": None, "first": None,
         "last": None}
    pending = {}
    for line in open(f, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        t = (d.get("timestamp") or "")[:19]
        if t:
            a["first"] = a["first"] or t
            a["last"] = t
        m = d.get("message") or {}
        if d.get("type") == "user":
            c = m.get("content")
            if a["brief"] is None:
                a["brief"] = text_of(c)
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") in pending:
                        cmd = pending.pop(b["tool_use_id"])
                        for h in COMMIT.findall(text_of(b.get("content"))):
                            a["commits"].append((h, t))
        elif d.get("type") == "assistant":
            for b in m.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and b.get("text", "").strip():
                    a["final"] = b["text"]
                if b.get("type") != "tool_use":
                    continue
                n, inp = b.get("name"), b.get("input") or {}
                if n in ("Edit", "Write", "NotebookEdit") and inp.get("file_path"):
                    a["added"].update(l.strip() for l in (inp.get("new_string") or inp.get("content") or "").splitlines()
                                      if len(l.strip()) > 25)
                    a["edited"].add(os.path.relpath(inp["file_path"], ROOT) if inp["file_path"].startswith(ROOT)
                                    else inp["file_path"])
                elif n == "Bash" and "git commit" in (inp.get("command") or ""):
                    pending[b["id"]] = inp["command"]
                elif n == "Bash" and re.search(r"plan\.py design-(pass|fix)", inp.get("command") or ""):
                    a["design_verdict"] = re.search(r"design-(pass|fix)", inp["command"]).group(1)
                elif n == "SendMessage":
                    a["sent"].append((t, inp.get("to"), json.dumps(inp.get("message") or inp.get("content") or "")))
                elif n == "SubagentHandback":
                    a["final"] = json.dumps(inp.get("message") or "")
                elif n in ("Agent", "Task"):
                    a["launched"].append((t, b.get("id"), inp.get("subagent_type"), inp.get("effort"), inp.get("model")))
    return a


def scan_main(f):
    """A main session's SendMessages, Agent launches, agent results reaching it and Joaquin's typed messages."""
    out = {"sent": [], "launched": [], "results": [], "typed": []}
    for line in open(f, errors="replace"):
        if '"type":"user"' not in line and '"type":"assistant"' not in line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        t = (d.get("timestamp") or "")[:19]
        m = d.get("message") or {}
        if d.get("type") == "assistant":
            for b in m.get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    inp = b.get("input") or {}
                    if b.get("name") == "SendMessage":
                        out["sent"].append((t, inp.get("to"), json.dumps(inp.get("message") or inp.get("content") or "")))
                    elif b.get("name") in ("Agent", "Task"):
                        out["launched"].append((t, b.get("id"), inp.get("subagent_type"), inp.get("effort"),
                                                inp.get("model")))
            continue
        c = m.get("content")
        s = text_of(c) if not isinstance(c, str) else c
        r = re.search(r"<task-id>([0-9a-f]+)</task-id>|agent-message from=\"([0-9a-f]+)\"", s or "")
        if r:
            out["results"].append((t, r.group(1) or r.group(2)))
        elif isinstance(c, str) and d.get("entrypoint") != "sdk-cli" and not d.get("isMeta") and not c.startswith(
                ("Another Claude", "Stop hook", "<", "This session is being", "Caveat:", "[Request interrupted")):
            out["typed"].append((t, c))
    return out


def git_files(h):
    try:
        r = subprocess.run(["git", "-C", ROOT, "show", "--name-only", "--format=%H%x09%cI%x09%s", h],
                           capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    if r.returncode:
        return None
    lines = r.stdout.strip().splitlines()
    full, when, subj = lines[0].split("\t", 2)
    return {"full": full, "when": when[:19], "subj": subj, "files": [l for l in lines[1:] if l.strip()]}


_DIFF = {}


def diff_lines(h, sign):
    """Lines a commit added ('+') or removed ('-'), stripped, longer than 25 characters."""
    if h not in _DIFF:
        r = subprocess.run(["git", "-C", ROOT, "show", "-U0", "--format=", h], capture_output=True, text=True)
        add, rem = set(), set()
        for l in r.stdout.splitlines():
            if l.startswith("+") and not l.startswith("+++") and len(l[1:].strip()) > 25:
                add.add(l[1:].strip())
            elif l.startswith("-") and not l.startswith("---") and len(l[1:].strip()) > 25:
                rem.add(l[1:].strip())
        _DIFF[h] = {"+": add, "-": rem}
    return _DIFF[h][sign]


def git_log(since):
    r = subprocess.run(["git", "-C", ROOT, "log", "--all", "--since=" + since, "--name-only",
                        "--format=@@%h%x09%cI%x09%s"], capture_output=True, text=True)
    out, cur = [], None
    for l in r.stdout.splitlines():
        if l.startswith("@@"):
            h, when, subj = l[2:].split("\t", 2)
            cur = {"h": h[:8], "when": datetime.fromisoformat(when).astimezone().replace(tzinfo=None), "subj": subj,
                   "files": []}
            # git dates are local with offset; transcripts are UTC
            cur["when"] = datetime.fromisoformat(when).astimezone(__import__("datetime").timezone.utc).replace(tzinfo=None)
            out.append(cur)
        elif l.strip() and cur:
            cur["files"].append(l.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-01")
    ap.add_argument("--out", default="analysis/agent_redesign/agent_tasks.json")
    ap.add_argument("--sample", type=int, default=5)
    ap.add_argument("--scored-before", default="2026-10-29T00:00", help="score only tasks ending before (48 h follow-up)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the hand-check sample")
    a = ap.parse_args()
    calls = [c for c in uw.load() if c["agent"]]
    by_agent = defaultdict(list)
    for c in calls:
        by_agent[c["agent"]].append(c)
    metas = {os.path.basename(f)[6:-10]: json.load(open(f))
             for f in glob.glob(uw.PROJ + "/*/*/subagents/agent-*.meta.json")}
    agent_files = {os.path.basename(f)[6:-6]: f for f in glob.glob(uw.PROJ + "/*/*/subagents/agent-*.jsonl")}
    scans = {k: scan_agent(f) for k, f in agent_files.items() if k in by_agent and by_agent[k][0]["ts"] >= a.since}
    mains = [scan_main(f) for f in glob.glob(uw.PROJ + "/-home-joaquin-work-FA*/*.jsonl")]
    # who launched whom: tool_use id of the launch -> launcher (agent id or "main")
    launcher, effort = {}, {}
    for k, s in scans.items():
        for t, tid, typ, eff, mod in s["launched"]:
            launcher[tid] = k; effort[tid] = eff
    for mm in mains:
        for t, tid, typ, eff, mod in mm["launched"]:
            launcher[tid] = "main"; effort[tid] = eff
    parent = {k: launcher.get(metas.get(k, {}).get("toolUseId")) for k in scans}
    sends = [x for s in scans.values() for x in s["sent"]] + [x for mm in mains for x in mm["sent"]]
    results = sorted(x for mm in mains for x in mm["results"])
    typed = sorted(x for mm in mains for x in mm["typed"])
    commit_owner = {h[:8]: k for k, s in scans.items() for h, _ in s["commits"]}
    log = git_log(a.since)
    rows = []
    for k, s in scans.items():
        cs = by_agent[k]
        model = Counter(c["model"] for c in cs).most_common(1)[0][0]
        typ = metas.get(k, {}).get("agentType") or "?"
        files = {f for f in s["edited"]}
        for h, _ in s["commits"]:
            g = next((x for x in log if x["h"] == h[:8]), None)
            if g:
                files |= set(g["files"])
        kind = ("review" if typ in REVIEWERS else "owner" if (s["brief"] or "").lstrip().startswith("You own one entry")
                else "tool fix" if files and all(TOOLPATH.match(f) for f in files) else "analysis")
        n = len(cs)
        usd = sum(uw.cost(c) or 0 for c in cs)
        t_end = ts(cs[-1]["ts"])
        horizon = t_end + timedelta(hours=48)
        corr = []
        # sendmessage
        for t, to, msg in sends:
            if to == k and t and ts(cs[0]["ts"]) < ts(t) <= horizon and ERR.search(msg) and not SELF_CORR.search(msg):
                after = sum(uw.cost(c) or 0 for c in cs if c["ts"] >= t)
                corr.append({"kind": "sendmessage", "t": t, "evidence": msg[:300], "rework_usd": round(after, 3)})
        # review_fix: a reviewer it launched, or one naming its commits
        mine = {h[:8] for h, _ in s["commits"]}
        for r, rs in scans.items():
            if metas.get(r, {}).get("agentType") not in REVIEWERS or r == k:
                continue
            fin = rs["final"] or ""
            if rs["design_verdict"]:
                bad = rs["design_verdict"] == "fix"
            else:
                v = VERDICT.search(fin)
                bad = bool(v) and not v.group(1).startswith("PASS")
            if not bad:
                continue
            named = mine & set(HASH.findall(fin))
            if parent.get(r) == k or named:
                rt = rs["last"] or ""
                if rt and ts(rt) <= horizon + timedelta(hours=48):
                    after = sum(uw.cost(c) or 0 for c in cs if c["ts"] >= rt)
                    corr.append({"kind": "review_fix", "t": rt, "reviewer": r, "named": sorted(named),
                                 "evidence": fin[:300], "rework_usd": round(after, 3)})
        # later_fix
        if files:
            t0 = ts(cs[0]["ts"])
            for g in log:
                if g["h"] in mine or not (t_end < g["when"] <= horizon) or not FIXSUBJ.search(g["subj"][:120]):
                    continue
                shared = files & set(g["files"])
                if not shared:
                    continue
                mylines = set(s["added"])
                for h in mine:
                    mylines |= diff_lines(h, "+")
                touched = mylines & diff_lines(g["h"], "-")
                if not touched:
                    continue
                who = commit_owner.get(g["h"])
                rw = 0.0
                if who and who != k:
                    lo = (g["when"] - timedelta(hours=1)).isoformat()
                    rw = sum(uw.cost(c) or 0 for c in by_agent[who] if lo <= c["ts"] <= g["when"].isoformat())
                corr.append({"kind": "later_fix", "t": g["when"].isoformat(), "commit": g["h"], "by": who or "main/unknown",
                             "files": sorted(shared)[:5], "lines_changed": len(touched), "evidence": g["subj"][:200], "rework_usd": round(rw, 3)})
        # user
        mine_res = [t for t, aid in results if aid == k]
        for rt in mine_res:
            nxt = next((t for t, aid in results if t > rt and aid != k), None)
            for t, msg in typed:
                if rt < t <= (nxt or "9") and ts(t) <= ts(rt) + timedelta(hours=3) and ERR.search(msg) and \
                        MENTION.search(msg):
                    corr.append({"kind": "user", "t": t, "evidence": msg[:300], "rework_usd": 0.0})
        uniq = {}
        for c in corr:
            uniq.setdefault((c["kind"], c["t"]), c)
        corr = sorted(uniq.values(), key=lambda c: c["t"])
        rows.append({"agent": k, "type": typ, "kind": kind, "model": model, "calls": n,
                     "effort": effort.get(metas.get(k, {}).get("toolUseId")) or "session default",
                     "brief_chars": len(s["brief"] or ""), "scored": cs[-1]["ts"] < a.scored_before,
                     "usd": round(usd, 3), "start": cs[0]["ts"], "end": cs[-1]["ts"], "period": uw.period(cs[0]["ts"]),
                     "parent": parent.get(k), "description": metas.get(k, {}).get("description"),
                     "files": sorted(files)[:20], "commits": sorted(mine), "stood": not corr, "corrections": corr,
                     "rework_usd": round(sum(c["rework_usd"] for c in corr), 3)})
    fam = lambda r: ("opus" + ("-low" if r["effort"] == "low" else "") if "opus" in r["model"] else
                     "sonnet" if "sonnet" in r["model"] else "haiku" if "haiku" in r["model"] else r["model"])
    # size band: terciles of brief length within each kind (known before the task runs)
    for kind in {r["kind"] for r in rows}:
        L = sorted(r["brief_chars"] for r in rows if r["kind"] == kind)
        q1, q2 = L[len(L) // 3], L[2 * len(L) // 3]
        for r in rows:
            if r["kind"] == kind:
                r["band"] = "short brief" if r["brief_chars"] <= q1 else "mid brief" if r["brief_chars"] <= q2 else "long brief"
    cells = defaultdict(lambda: {"n": 0, "stood": 0, "usd": 0.0, "rework_usd": 0.0})
    for r in rows:
        if not r["scored"]:
            continue
        for key in ((r["kind"], r["band"], fam(r)), (r["kind"], "all", fam(r))):
            c = cells[" | ".join(key)]
            c["n"] += 1; c["stood"] += r["stood"]; c["usd"] += r["usd"]; c["rework_usd"] += r["rework_usd"]
    for c in cells.values():
        c["stood_frac"] = round(c["stood"] / c["n"], 3)
        c["usd_per_task"] = round(c["usd"] / c["n"], 3)
        c["usd_per_task_with_rework"] = round((c["usd"] + c["rework_usd"]) / c["n"], 3)
        c["usd_per_stood_task_with_rework"] = round((c["usd"] + c["rework_usd"]) / c["stood"], 3) if c["stood"] else None
        c["decidable"] = c["n"] >= 10
        p = c["stood"] / c["n"]
        c["stood_95ci"] = [round(max(0, p - 1.96 * (p * (1 - p) / c["n"]) ** .5), 3),
                           round(min(1, p + 1.96 * (p * (1 - p) / c["n"]) ** .5), 3)]
        c["usd"], c["rework_usd"] = round(c["usd"], 2), round(c["rework_usd"], 2)
    rep = {"definition": __doc__, "since": a.since, "prices": "tools/usage_windows.py PRICES",
           "tasks": len(rows), "scored_before": a.scored_before, "cells": dict(sorted(cells.items())), "rows": sorted(rows, key=lambda r: r["start"])}
    import random
    rnd = random.Random(a.seed)
    ok = [r for r in rows if r["stood"] and r["scored"]]; bad = [r for r in rows if not r["stood"] and r["scored"]]
    rep["hand_check_sample"] = {"stood": [r["agent"] for r in rnd.sample(ok, min(a.sample, len(ok)))],
                                "failed": [r["agent"] for r in rnd.sample(bad, min(a.sample, len(bad)))]}
    json.dump(rep, open(os.path.join(ROOT, a.out), "w"), indent=1, default=list)
    for k, c in sorted(cells.items()):
        if "| all |" in k or True:
            print("%-34s n=%3d stood=%.2f $/task=%.2f with rework=%.2f" % (k, c["n"], c["stood_frac"], c["usd_per_task"],
                                                                           c["usd_per_task_with_rework"]))
    print("sample", rep["hand_check_sample"])


if __name__ == "__main__":
    sys.exit(main())
