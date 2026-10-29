#!/usr/bin/env python3
"""Tokens spent per source, per 5-hour window and per completed task, read from the Claude Code transcripts.

  python3 tools/task_cost.py [SESSION_ID] [--day YYYY-MM-DD] [--daily A..B] [--agents] [--turns N]

SESSION_ID: default the newest main transcript in ~/.claude/projects/-home-joaquin-work-FA/. --day: a local calendar
day (default today); every time printed is local time. --daily A..B: one line per local day A..B (inclusive) instead of
the per-day sections. --agents: list every agent of the day. --turns N: how many user turns to list (default 12).

Counting. A token count is one API response's usage (tools/agent_costs.responses: records of one response joined,
counted once): uncached input, cache read, cache creation (split by cache lifetime, 5 min and 1 h, as recorded) and
output, kept apart. How the subscription limits weigh these kinds is not documented anywhere read for this tool, so
no weighting is applied to the totals; the column "API$" is the documented API list-price ratio (input 1, cache read
0.1, cache write 1.25 for 5 min and 2 for 1 h, output 5), a price, not the limit's weighting.

Sources: the main session (the session's own transcript); its subagents (subagents/agent-*.jsonl, grouped by type);
other sessions under ~/.claude/projects/*FA* (headless replays such as the worktrees/zz-ab* A/B runs, older main
sessions); other projects (every other ~/.claude/projects/ directory), since the limits are per account.

5-hour windows: a window that holds a limit refusal (the API error "You've hit your 5-hour limit . resets 7pm") ends
where the refusal says and opened 5 h before; any other window opens at the first response of any source after the
previous one closed and lasts 5 h (an estimate; on 2026-10-29 it put the two refused windows' starts 1 and 8 min after
the stated ones). For a refused window the tokens spent before the first refusal are printed.

Tasks (sessions with the plan, 2026-10-25 on). A task is one step of one plan entry: the agents the main session
recorded for it (`plan.py step <entry> --agent <id> --next <n>` in its transcript; else the launch prompt's "Entry: <id>"
and "Your step ... number <n>"), plus every agent those agents launched (design and pillar reviewers, nested). Another
agent launched by the main session is a task of its own, keyed by its launch description, on the entry its prompt names
first (any id plan/plan.yaml has held) or "(no entry)". Each agent is active from its launch to its report (the main
transcript's task-notification or hand-back naming it; if none, its last record); each main-session response at a time
when k tasks have an active agent counts 1/k to each (apportioned by count); responses when none has are "main, no task
open". A task is complete when the latest of its agents the main session launched reported (an agent cut off by the
limit and relaunched does not leave it open). Produced: commits ("plan:" commits are the plan CLI's own bookkeeping,
counted apart; a commit is a hash printed by a command that ran git commit, made at most 15 min before it, owned by
the command that printed it soonest), whether a commit touched docs/results.tex (a write-up), and job submissions (a
site submit, sweep_manager submit or launch_cc_only.sh command that did not fail).

User turns (any session, the baseline before the plan): from one message the user typed to the next, the main
session's responses in between plus every agent launched in between, with what they produced.

Main session's calls: each response classed by what it did, first match wins: agent launch or message (Agent,
SendMessage); relay to the user (SendUserFile); routing and bookkeeping (plan.py next/step/run/ask/answer/evidence/
set-next/done, site ack/events/digest); state reads (site poll/runs/status/logs, git status/log, TaskOutput); edits
(Edit, Write); other tool calls (analysis, git commits, site submit, reads); a text-only reply under 400 characters
(short reply); a text-only reply of 400 or more (a message to the user). Wakes: what started each run of main
responses (the user's message, an agent's report, a background command's end, a Stop hook's feedback, a compaction).
"""
import datetime as dt, glob, json, os, re, statistics as st, subprocess, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_costs import responses                     # noqa: E402
from agent_calls import PROJ, launch_records, text_of  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTS = os.path.dirname(PROJ)
KINDS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")
SHORT = {"input_tokens": "in", "cache_read_input_tokens": "read", "cache_creation_input_tokens": "write",
         "output_tokens": "out"}
ROUTING = re.compile(r"plan\.py (next|step|run|ask|answer|evidence|set-next|add-next|done|dead-end)\b|"
                     r"\bsite (ack|events|digest)\b|plan_check|review_backlog\.sh")
STATE = re.compile(r"\bsite (poll|runs|status|logs|where|items|decisions)\b|git (status|log|diff)\b|plan\.py show")
SUBMIT = re.compile(r"\bsite submit\b|sweep_manager\.py submit|launch_cc_only\.sh")


def utc(ts):
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def local(ts):
    return utc(ts).astimezone().strftime("%m-%d %H:%M") if ts else "?"


def day_range(day):
    d = dt.datetime.strptime(day, "%Y-%m-%d").astimezone()     # local midnight, with that day's offset
    e = (d + dt.timedelta(days=1)).replace(hour=0).astimezone()
    z = lambda x: x.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return z(d), z(e)


def add(acc, u, w=1.0):
    for k in KINDS + ("cache_5m", "cache_1h"):
        acc[k] += w * u[k]
    acc["n"] += w


def price(a):
    w5 = a["cache_5m"] + max(0, a["cache_creation_input_tokens"] - a["cache_5m"] - a["cache_1h"])
    return (a["input_tokens"] + 0.1 * a["cache_read_input_tokens"] + 1.25 * w5 + 2 * a["cache_1h"]
            + 5 * a["output_tokens"])


def total(a):
    return sum(a[k] for k in KINDS)


def row(name, a, width=40):
    return ("%-*s %6.0f %9.1f %8.1f %8.1f %7.2f %7.2f %8.1f" % (
        width, name[:width], a["n"], total(a) / 1e6, a["cache_read_input_tokens"] / 1e6,
        a["cache_creation_input_tokens"] / 1e6, a["input_tokens"] / 1e6, a["output_tokens"] / 1e6, price(a) / 1e6))


HDR = "%-40s %6s %9s %8s %8s %7s %7s %8s" % ("source", "calls", "total M", "read M", "write M", "in M", "out M",
                                             "API$ M")


def acc():
    return defaultdict(float)


def files_since(lo):
    """[(label, path)] of the transcripts outside the session that changed after lo (UTC iso)."""
    t0 = utc(lo + "Z").timestamp()
    out = []
    for d in glob.glob(os.path.join(PROJECTS, "*")):
        fa = "-FA" in os.path.basename(d)
        for f in glob.glob(os.path.join(d, "*.jsonl")) + glob.glob(os.path.join(d, "*", "subagents", "agent-*.jsonl")):
            if os.path.getmtime(f) >= t0:
                out.append(("other FA sessions" if fa else "other projects", f))
    return out


def tool_results(path):
    """[(timestamp, tool_use input, result text, is_error)] of one transcript, tool results joined to their calls."""
    calls, out = {}, []
    for line in open(path):
        if '"tool_use"' not in line and '"tool_result"' not in line:
            continue
        r = json.loads(line)
        if r.get("type") not in ("assistant", "user"):
            continue
        for x in (r.get("message") or {}).get("content") or []:
            if not isinstance(x, dict):
                continue
            if x.get("type") == "tool_use":
                calls[x["id"]] = (x.get("name"), x.get("input") or {})
            elif x.get("type") == "tool_result" and x.get("tool_use_id") in calls:
                name, inp = calls[x["tool_use_id"]]
                c = x.get("content")
                out.append((r.get("timestamp") or "", name, inp, c if isinstance(c, str) else text_of(c),
                            bool(x.get("is_error"))))
    return out


HEX = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}\b")
_commit = {}


def commit_info(toks):
    """token -> (short hash, committer unix time, subject) for the tokens that name a commit of this repository."""
    need = sorted(set(toks) - set(_commit))
    if need:
        p = subprocess.run(["git", "-C", ROOT, "cat-file", "--batch-check=%(objectname) %(objecttype)"],
                           input="\n".join(need) + "\n", capture_output=True, text=True)
        full = {}
        for t, ln in zip(need, p.stdout.splitlines()):
            f = ln.split()
            full[t] = f[0] if len(f) == 2 and f[1] == "commit" else None
        meta = {}
        if any(full.values()):
            q = subprocess.run(["git", "-C", ROOT, "show", "-s", "--format=%H %h %ct %s"]
                               + sorted({v for v in full.values() if v}), capture_output=True, text=True)
            for ln in q.stdout.splitlines():
                f = ln.split(" ", 3)
                meta[f[0]] = (f[1], int(f[2]), f[3] if len(f) > 3 else "")
        for t in need:
            _commit[t] = meta.get(full.get(t))
    return {t: _commit[t] for t in set(toks) if _commit.get(t)}


def commits_in(ts, res):
    """[(hash, subject, delay)] of the commits a git-commit command's output names that were made within 15 min
    before the command ended (delay: seconds from the commit to the end of the command)."""
    t1 = utc(ts).timestamp() if ts else 0
    return [(h, subj, t1 - ct) for h, ct, subj in commit_info(HEX.findall(res)).values() if -120 <= t1 - ct <= 900]


def owned(groups):
    """{key: [(hash, subject, delay)]} -> {key: [(hash, subject)]}: a commit printed by several commands (a later
    git log, another agent's output) belongs only to the one that printed it soonest after it was made."""
    best = {}
    for k, cs in groups.items():
        for h, _, d in cs:
            if h not in best or d < best[h][0]:
                best[h] = (d, k)
    out = {}
    for k, cs in groups.items():
        seen, out[k] = set(), []
        for h, subj, _ in cs:
            if best[h][1] == k and h not in seen:
                seen.add(h); out[k].append((h, subj))
    return out


def produced(path):
    """commit candidates [(hash, subject, delay)] (see owned) and job submissions [(ts, command)] of one
    transcript."""
    commits, jobs = [], []
    for ts, name, inp, res, err in tool_results(path):
        if name != "Bash":
            continue
        cmd = inp.get("command", "")
        if "git commit" in cmd and not err:
            commits += commits_in(ts, res)
        if SUBMIT.search(cmd) and "--dry-run" not in cmd and not err:
            jobs.append((ts, cmd))
    return commits, jobs


_touch = {}


def writeups(hashes):
    """hash -> True when the commit touched docs/results.tex."""
    need = [h for h in hashes if h not in _touch]
    if need:
        p = subprocess.run(["git", "-C", ROOT, "show", "--no-walk", "--name-only", "--format=@%h"] + need,
                           capture_output=True, text=True)
        cur = None
        for ln in p.stdout.splitlines():
            if ln.startswith("@"):
                cur = ln[1:]; _touch[cur] = False
            elif cur and ln.strip() == "docs/results.tex":
                _touch[cur] = True
        for h in need:                               # abbreviations may differ in length
            _touch.setdefault(h, any(v for k, v in _touch.items() if k.startswith(h[:7]) or h.startswith(k)))
    return {h: _touch[h] for h in hashes}


def wake_of(r):
    """What a user-side record is, for the main session: None for a tool result (a continuation)."""
    c = (r.get("message") or {}).get("content")
    if isinstance(c, list):
        if any(isinstance(x, dict) and x.get("type") == "tool_result" for x in c):
            return None
        c = text_of(c)
    c = (c or "").lstrip()
    if c.startswith("<task-notification>"):
        return "agent report" if re.search(r"<task-id>a[0-9a-f]{16}</task-id>", c) else "background command ended"
    if c.startswith("Another Claude session sent a message"):
        return "agent report"
    if c.startswith("Stop hook feedback"):
        return "stop hook"
    if c.startswith("This session is being continued") or "/compact" in c[:60]:
        return "compaction"
    if c.startswith(("<local-command", "<command-name>", "[Request interrupted", "[Usage limit", "[Earlier usage")):
        return "harness note"
    if r.get("isMeta"):
        return "harness note"
    return "user message"


def main_records(path, lo, hi):
    """(responses in [lo, hi), [(ts, wake)] of the user-side records in [lo, hi), {agent id: [report ts]})."""
    resp = [x for x in responses(path) if lo <= x[0][:19] < hi]
    wakes, reports = [], defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if '"user"' not in line[:300] and '"type": "user"' not in line[:300] and '"type":"user"' not in line:
                continue
            r = json.loads(line)
            if r.get("type") != "user":
                continue
            ts = r.get("timestamp") or ""
            w = wake_of(r)
            if w == "agent report":
                c = json.dumps((r.get("message") or {}).get("content"))
                for aid in set(re.findall(r"<task-id>(a[0-9a-f]{16})</task-id>|agent-message from=\\\"(a[0-9a-f]{16})",
                                          c)):
                    reports[aid[0] or aid[1]].append(ts)
            if lo <= ts[:19] < hi and w:
                wakes.append((ts, w, r))
    return resp, wakes, reports


def head(x):
    """A tool call's name, and for Bash the command's first program and its first argument (after cd ... &&)."""
    if x.get("name") != "Bash":
        return x.get("name")
    cmd = re.sub(r"^\s*(cd \S+\s*(&&|;)\s*)+", "", (x.get("input") or {}).get("command", ""))
    w = cmd.split()
    w = w[1:] if w[:1] == ["NOSYNC=1"] else w
    return " ".join(os.path.basename(t) for t in w[:2]) or "Bash"


def classify(blocks):
    tools = [x for x in blocks if x.get("type") == "tool_use"]
    text = "".join(x.get("text", "") for x in blocks if x.get("type") == "text")
    names = {x.get("name") for x in tools}
    cmds = " ; ".join((x.get("input") or {}).get("command", "") for x in tools if x.get("name") == "Bash")
    if names & {"Agent", "SendMessage"}:
        return "agent launch or message"
    if "SendUserFile" in names:
        return "relay to the user (SendUserFile)"
    if tools and names <= {"Bash"} and ROUTING.search(cmds):
        return "routing and bookkeeping (plan, site ack/events)"
    if tools and (names <= {"Bash", "TaskOutput", "ToolSearch"}) and (STATE.search(cmds) or not cmds):
        return "state reads"
    if names & {"Edit", "Write", "NotebookEdit"}:
        return "edits"
    if tools:
        return "other tool calls"
    if len(text) < 400:
        return "short text reply (<400 chars)"
    return "message to the user (400+ chars)"


def agent_info(sid):
    """agent id -> dict(type, launch ts, prompt, parent agent id or None, file)."""
    launch = launch_records(sid)
    out = {}
    for meta in glob.glob(os.path.join(PROJ, sid, "subagents", "agent-*.meta.json")):
        m = json.load(open(meta))
        aid = os.path.basename(meta)[6:-10]
        f = meta[:-10] + ".jsonl"
        if not os.path.exists(f):
            continue
        L = launch.get(m.get("toolUseId"), {})
        par = re.search(r"agent-(a[0-9a-f]+)\.jsonl$", L.get("file", ""))
        typ = "fork" if m.get("agentType") == "fork" else (L.get("type") or m.get("agentType") or "?")
        ts = L.get("ts")
        if not ts:
            with open(f) as fh:
                for line in fh:
                    ts = json.loads(line).get("timestamp") or ""
                    if ts:
                        break
        out[aid] = {"type": typ, "effort": L.get("effort", "?"), "ts": ts, "prompt": L.get("prompt", ""),
                    "desc": L.get("description") or m.get("description", ""), "parent": par.group(1) if par else None,
                    "file": f}
    return out


def entry_ids():
    """Every entry id the plan has held (the git history of the plan file; the plan CLI commits every change)."""
    p = subprocess.run(["git", "-C", ROOT, "log", "-p", "--format=", "--", "plan/plan.yaml"], capture_output=True,
                       text=True)
    return set(re.findall(r"^[+ -]  id: (\S+)", p.stdout, re.M))


def step_records(main_f):
    """agent id -> (entry, step number) from the main session's `plan.py step <entry> --agent <id> [--next N]`."""
    out = {}
    for _, name, inp, res, err in tool_results(main_f):
        if name == "Bash" and not err and "plan.py step" in inp.get("command", ""):
            for m in re.finditer(r"plan\.py step (\S+) --agent (a[0-9a-f]+)(?: --next (\d+))?", inp["command"]):
                out[m.group(2)] = (m.group(1), int(m.group(3) or 1))
    return out


STEP = re.compile(r"Your step[: ]+\(?number (\d+)[^:]*?\)?:\s*([^\n]*)")


def own_task(aid, a, steps):
    """(entry, step number, step text) the agent was launched for, from the plan step record or its prompt."""
    p = a["prompt"]
    m, s = re.search(r"^Entry: (\S+)", p, re.M), STEP.search(p)
    text = s.group(2)[:70] if s else (a["desc"] or "")[:70]
    if aid in steps:
        return steps[aid] + (text,)
    if m:
        return (m.group(1), int(s.group(1)) if s else 0, text)
    return None


def task_of(aid, info, ids, steps, memo):
    """(entry, step number, step text) of an agent, by the rules of the module docstring."""
    if aid in memo:
        return memo[aid]
    a = info[aid]
    t = own_task(aid, a, steps)
    if t is None and a["parent"] in info:
        t = task_of(a["parent"], info, ids, steps, memo)
    if t is None:
        hits = sorted((m.start(), m.group(1)) for m in re.finditer(r"\b([a-z][a-z0-9_]{3,})\b", a["prompt"])
                      if m.group(1) in ids)
        t = (hits[0][1], 0, "(no step) %s: %s" % (a["type"], a["desc"])) if hits else \
            ("(no entry)", 0, "%s: %s" % (a["type"], a["desc"]))
    memo[aid] = t
    return t


def agent_usage(f):
    a, calls = acc(), 0
    for _, _, u, blocks in responses(f):
        add(a, u)
        calls += sum(1 for x in blocks if x.get("type") == "tool_use")
    return a, calls


LIMIT = re.compile(r"hit your 5-hour limit \S resets (\d{1,2})(?::(\d\d))?(am|pm)")


def limit_hits(files):
    """[(timestamp of the first refusal, window end as the refusal states it, UTC iso)] of the 5-hour limit, from the
    API error messages ("You've hit your 5-hour limit . resets 7pm") in the given transcripts."""
    out = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                if "hit your 5-hour limit" not in line:
                    continue
                r = json.loads(line)
                if r.get("type") != "assistant" or not r.get("isApiErrorMessage"):
                    continue
                m = LIMIT.search(text_of((r.get("message") or {}).get("content")))
                if not m:
                    continue
                t = utc(r["timestamp"]).astimezone()
                h = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
                end = t.replace(hour=h, minute=int(m.group(2) or 0), second=0, microsecond=0)
                if end < t:
                    end += dt.timedelta(days=1)
                e = end.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
                out[e] = min(out.get(e, r["timestamp"]), r["timestamp"])
    return sorted((v, k) for k, v in out.items())


def windows(points, hits=()):
    """5-hour windows over sorted timestamps: [(start, end)]. A window that holds a limit refusal ends where the
    refusal says it resets; the others open at their first response and last 5 h."""
    ends = [e for _, e in hits]
    out = []
    for ts in points:
        if not out or ts >= out[-1][1]:
            e = next((e for e in ends if utc(e) - dt.timedelta(hours=5) <= utc(ts) < utc(e)), None)
            if e:
                out.append(((utc(e) - dt.timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%S.000Z"), e))
            else:
                out.append((ts, (utc(ts) + dt.timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%S.000Z")))
    return out


def fmt_tok(a):
    return "%6.1f %6.1f %6.2f %5.2f %6.1f" % (total(a) / 1e6, a["cache_read_input_tokens"] / 1e6,
                                             a["cache_creation_input_tokens"] / 1e6, a["output_tokens"] / 1e6,
                                             price(a) / 1e6)


def report_day(sid, day, list_agents, nturns):
    lo, hi = day_range(day)
    main_f = os.path.join(PROJ, sid + ".jsonl")
    resp, wakes, reports = main_records(main_f, lo, hi)
    info = agent_info(sid)
    todays = {k: v for k, v in info.items() if lo <= v["ts"][:19] < hi}
    print("session %s, local day %s (UTC %s .. %s)" % (sid, day, lo, hi))

    # ---- 1. per source and 5-hour window
    pts = []                                             # (ts, source, usage)
    for ts, _, u, _ in resp:
        pts.append((ts, "main session", u))
    ausage = {}
    for aid, a in todays.items():
        for ts, _, u, _ in responses(a["file"]):
            pts.append((ts, "subagent: " + a["type"], u))
    for label, f in files_since(lo):
        if f.startswith(os.path.join(PROJ, sid)) or f == main_f:
            continue
        for ts, _, u, _ in responses(f):
            if lo <= ts[:19] < hi:
                pts.append((ts, label, u))
    pts.sort(key=lambda x: x[0])
    hits = limit_hits([main_f] + [a["file"] for a in todays.values()] + [f for _, f in files_since(lo)])
    for w0, w1 in windows([p[0] for p in pts], hits):
        src = defaultdict(acc)
        for ts, s, u in pts:
            if w0 <= ts < w1:
                add(src[s], u)
        allw = acc()
        for a in src.values():
            for k, v in a.items():
                allw[k] += v
        print("\n5-hour window %s .. %s (local)\n%s" % (local(w0), local(w1), HDR))
        for s, a in sorted(src.items(), key=lambda kv: -total(kv[1])):
            print(row(s, a))
        print(row("all sources", allw))
        print("  shares of the window's total: " + ", ".join(
            "%s %.0f%%" % (SHORT[k], 100 * allw[k] / max(total(allw), 1)) for k in KINDS))
        for hit, end in hits:
            if w0 <= hit < w1:
                before = acc()
                for ts, _, u in pts:
                    if w0 <= ts < hit:
                        add(before, u)
                print("  5-hour limit hit at %s (the refusal says it resets %s, so the window opened %s); before the "
                      "hit: %.0f calls, %.1fM tokens (read %.1fM, write %.2fM, out %.2fM), API$ %.1fM" % (
                          local(hit), local(end)[6:], local(w0)[6:], before["n"], total(before) / 1e6,
                          before["cache_read_input_tokens"] / 1e6, before["cache_creation_input_tokens"] / 1e6,
                          before["output_tokens"] / 1e6, price(before) / 1e6))

    # ---- 2. per task
    ids, memo, steps = entry_ids(), {}, step_records(main_f)
    tasks = defaultdict(lambda: {"agents": [], "u": acc(), "calls": 0, "spans": [], "done": True, "last": "",
                                 "commits": [], "jobs": [], "main": acc()})
    for aid, a in sorted(todays.items(), key=lambda kv: kv[1]["ts"]):
        e, n, text = task_of(aid, info, ids, steps, memo)
        t = tasks[(e, n or text)]
        t.setdefault("text", text)
        u, calls = agent_usage(a["file"])
        ausage[aid] = (u, calls)
        t["agents"].append(aid); t["calls"] += calls
        for k, v in u.items():
            t["u"][k] += v
        rep = [r for r in reports.get(aid, []) if r >= a["ts"]]
        if a["parent"] is None and a["ts"] >= t["last"]:      # the latest launch from the main session decides
            t["last"], t["done"] = a["ts"], bool(rep)
        end = max(rep) if rep else ""
        if not end:
            with open(a["file"]) as fh:
                for line in fh:
                    end = json.loads(line).get("timestamp") or end
        t["spans"].append((a["ts"], end))
        c, j = produced(a["file"])
        t["commits"] += c; t["jobs"] += j
    for k, cs in owned({k: t["commits"] for k, t in tasks.items()}).items():
        tasks[k]["commits"] = cs
    nomain = acc()
    for ts, _, u, _ in resp:
        open_ = [t for t in tasks.values() if any(s0 <= ts <= s1 for s0, s1 in t["spans"])]
        if not open_:
            add(nomain, u)
        for t in open_:
            add(t["main"], u, 1.0 / len(open_))
    allc = [h for t in tasks.values() for h, s in t["commits"]]
    wu = writeups(allc)
    print("\nPer task (agents launched this day; tokens in M: total, cache read, cache write, output, API$; "
          "main = the main session's responses apportioned to the task)")
    print("%-62s %3s %5s | %-33s | %-33s | %s" % ("entry / step", "ag", "calls", "agents: tot read write out API$",
                                                  "main: tot read write out API$", "produced"))
    order = sorted(tasks.items(), key=lambda kv: -(total(kv[1]["u"]) + total(kv[1]["main"])))
    tot_a, tot_m = acc(), acc()
    for (e, n), t in order:
        s = "%s: %s" % (n, t["text"]) if isinstance(n, int) else t["text"]
        for k, v in t["u"].items():
            tot_a[k] += v
        for k, v in t["main"].items():
            tot_m[k] += v
        types = Counter(todays[x]["type"] for x in t["agents"])
        real = [(h, sub) for h, sub in t["commits"] if not sub.startswith("plan:")]
        prod = "%d commits (%d write-up) + %d plan, %d jobs" % (
            len(real), sum(wu.get(h, False) for h, _ in real), len(t["commits"]) - len(real), len(t["jobs"]))
        print("%-62s %3d %5d | %s | %s | %s%s" % (
            ("%s / %s" % (e, s))[:62], len(t["agents"]), t["calls"], fmt_tok(t["u"]), fmt_tok(t["main"]), prod,
            "" if t["done"] else " [not reported]"))
        if list_agents:
            print("      " + ", ".join("%s %d" % kv for kv in types.most_common()))
            for h, sub in real:
                print("      %s %s%s" % (h, sub[:100], " [results.tex]" if wu.get(h) else ""))
    print("%-62s %3d %5d | %s | %s |" % ("all tasks", sum(len(t["agents"]) for t in tasks.values()),
                                         sum(t["calls"] for t in tasks.values()), fmt_tok(tot_a), fmt_tok(tot_m)))
    print("%-62s %3s %5s | %33s | %s |" % ("main, no task open", "", "", "", fmt_tok(nomain)))
    step_tasks = [t for (e, n), t in tasks.items() if t["done"] and isinstance(n, int)]
    if step_tasks:
        per = [total(t["u"]) + total(t["main"]) for t in step_tasks]
        perp = [price(t["u"]) + price(t["main"]) for t in step_tasks]
        print("complete plan-step tasks: %d; tokens per task (agents + apportioned main) median %.1fM, mean %.1fM; "
              "API$ median %.1fM" % (len(step_tasks), st.median(per) / 1e6, st.mean(per) / 1e6, st.median(perp) / 1e6))

    # ---- user turns
    report_turns(main_f, resp, wakes, info, lo, hi, nturns)

    # ---- 3. the main session's share
    ctx = [u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"] for _, _, u, _ in resp]
    if not ctx:
        return
    cls, heads = defaultdict(acc), defaultdict(Counter)
    for ts, _, u, blocks in resp:
        c = classify(blocks)
        add(cls[c], u)
        for x in blocks:
            if x.get("type") == "tool_use":
                heads[c][head(x)] += 1
    print("\nMain session: %d responses this day, median context per response %.0fk tokens (max %.0fk)" % (
        len(resp), st.median(ctx) / 1e3, max(ctx) / 1e3))
    print(HDR.replace("source", "what the response did"))
    for c, a in sorted(cls.items(), key=lambda kv: -kv[1]["n"]):
        print(row(c, a))
        if heads[c]:
            print("      " + ", ".join("%s %d" % kv for kv in heads[c].most_common(8)))
    seq = sorted([(ts, "R", u) for ts, _, u, _ in resp] + [(ts, "W", w) for ts, w, _ in wakes], key=lambda x: x[0])
    wk = defaultdict(lambda: [0, 0, acc()])
    cur = None
    for ts, kind, x in seq:
        if kind == "W":
            cur = x; wk[cur][0] += 1
        elif cur:
            wk[cur][1] += 1; add(wk[cur][2], x)
    print("wakes of the main session (a run of responses is charged to what started it)")
    print("%-28s %6s %9s %9s %8s" % ("started by", "wakes", "responses", "total M", "API$ M"))
    for w, (n, r, a) in sorted(wk.items(), key=lambda kv: -kv[1][1]):
        print("%-28s %6d %9d %9.1f %8.1f" % (w, n, r, total(a) / 1e6, price(a) / 1e6))


def report_turns(main_f, resp, wakes, info, lo, hi, nturns):
    """Per user turn: main responses from one typed user message to the next, the agents launched in between."""
    starts = [ts for ts, w, _ in wakes if w == "user message"]
    if not starts:
        return
    bounds = list(zip(starts, starts[1:] + [hi + "Z"]))
    turns = []
    for s, e in bounds:
        m = acc()
        for ts, _, u, _ in resp:
            if s <= ts < e:
                add(m, u)
        ag = acc(); n_ag = 0; commits = []; jobs = []
        for aid, a in info.items():
            if s <= a["ts"] < e:
                n_ag += 1
                u, _ = agent_usage(a["file"])
                for k, v in u.items():
                    ag[k] += v
                c, j = produced(a["file"])
                commits += c; jobs += j
        turns.append([s, e, m, ag, n_ag, commits, jobs])
    # the main session's own commits and jobs, by time
    for ts, name, inp, res, err in tool_results(main_f):
        if not (lo <= ts[:19] < hi) or name != "Bash":
            continue
        cmd = inp.get("command", "")
        for t in turns:
            if t[0] <= ts < t[1]:
                if "git commit" in cmd and not err:
                    t[5] += commits_in(ts, res)
                if SUBMIT.search(cmd) and "--dry-run" not in cmd and not err:
                    t[6].append((ts, cmd))
    for i, cs in owned({i: t[5] for i, t in enumerate(turns)}).items():
        turns[i][5] = cs
    wu = writeups([h for t in turns for h, _ in t[5]])
    print("\nUser turns (from a message the user typed to the next): %d" % len(turns))
    print("%-23s %5s %3s | %-33s | %-33s | %s" % ("turn (local)", "resp", "ag", "main: tot read write out API$",
                                                  "agents: tot read write out API$", "produced"))
    prod_turns = []
    for s, e, m, ag, n_ag, commits, jobs in turns:
        real = [(h, x) for h, x in commits if not x.startswith("plan:")]
        if real or jobs:
            prod_turns.append(total(m) + total(ag))
    shown = sorted(turns, key=lambda t: -(total(t[2]) + total(t[3])))[:nturns]
    for s, e, m, ag, n_ag, commits, jobs in sorted(shown):
        real = [(h, x) for h, x in commits if not x.startswith("plan:")]
        print("%-23s %5.0f %3d | %s | %s | %d commits (%d write-up), %d jobs%s" % (
            "%s..%s" % (local(s), local(e)[6:]), m["n"], n_ag, fmt_tok(m), fmt_tok(ag), len(real),
            sum(wu.get(h, False) for h, _ in real), len(jobs),
            ("; " + real[0][1][:60]) if real else ""))
    if prod_turns:
        print("turns that produced a commit or a job: %d of %d; tokens per such turn median %.1fM, mean %.1fM" % (
            len(prod_turns), len(turns), st.median(prod_turns) / 1e6, st.mean(prod_turns) / 1e6))


def report_daily(sid, a, b):
    d0, d1 = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    main_f = os.path.join(PROJ, sid + ".jsonl")
    resp = responses(main_f)
    info = agent_info(sid)
    sub = []
    for x in info.values():
        sub += [(ts, u) for ts, _, u, _ in responses(x["file"])]
    print("%-10s %6s %6s %8s %8s %8s %7s %7s %8s %6s %6s %5s" % (
        "day", "main", "agents", "main M", "agent M", "read M", "write M", "out M", "API$ M", "commit", "w-up", "plan"))
    week = defaultdict(lambda: [acc(), 0])
    d = d0
    while d <= d1:
        lo, hi = day_range(d.isoformat())
        m, s = acc(), acc()
        for ts, _, u, _ in resp:
            if lo <= ts[:19] < hi:
                add(m, u)
        for ts, u in sub:
            if lo <= ts[:19] < hi:
                add(s, u)
        n_ag = sum(1 for x in info.values() if lo <= x["ts"][:19] < hi)
        p = subprocess.run(["git", "-C", ROOT, "log", "--all", "--format=%h %s", "--since=" + lo + "Z",
                            "--until=" + hi + "Z"], capture_output=True, text=True).stdout.splitlines()
        plan = [x for x in p if x.split(" ", 1)[-1].startswith("plan:")]
        real = [x.split()[0] for x in p if x not in plan]
        w = sum(writeups(real).values()) if real else 0
        both = acc()
        for k in set(m) | set(s):
            both[k] = m[k] + s[k]
        print("%-10s %6.0f %6d %8.1f %8.1f %8.1f %7.2f %7.2f %8.1f %6d %6d %5d" % (
            d, m["n"], n_ag, total(m) / 1e6, total(s) / 1e6, both["cache_read_input_tokens"] / 1e6,
            both["cache_creation_input_tokens"] / 1e6, both["output_tokens"] / 1e6, price(both) / 1e6, len(real), w,
            len(plan)))
        d += dt.timedelta(days=1)
    print("(main, agents: responses / agents launched; tokens: main session + its subagents, this session only; "
          "commit: commits on any branch that day excluding the plan CLI's 'plan:' commits; w-up: of those, touching "
          "docs/results.tex)")


def main(argv):
    opt = {}
    for flag in ("--day", "--daily", "--turns"):
        if flag in argv:
            i = argv.index(flag); opt[flag] = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    agents = "--agents" in argv
    argv = [x for x in argv if x != "--agents"]
    sid = argv[1] if len(argv) > 1 else os.path.basename(max(glob.glob(PROJ + "/*.jsonl"), key=os.path.getmtime))[:-6]
    if "--daily" in opt:
        report_daily(sid, *opt["--daily"].split(".."))
        return
    report_day(sid, opt.get("--day", dt.date.today().isoformat()), agents, int(opt.get("--turns", 12)))


if __name__ == "__main__":
    main(sys.argv)
