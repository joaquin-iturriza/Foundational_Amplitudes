#!/usr/bin/env python3
"""Tokens spent per source, per 5-hour window and per completed task, read from the Claude Code transcripts.

  python3 tools/task_cost.py [SESSION_ID] [--day YYYY-MM-DD] [--daily A..B] [--agents] [--turns N]
  python3 tools/task_cost.py [SESSION_ID] --compare DAY,DAY,...    (what differs between days; the last is listed
                                                                  per step)

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

--compare, per day (agents launched that day, the main session's responses that day): (1) the start-up floor: per
agent first-call ctx x calls against the growth after it, and the first call split into tools+system (the cached
prefix a first call read, exact), CLAUDE.md, launch prompt and other reminders (the first request's messages as
recorded, api-request-blob, split in proportion to characters: tokens per character = (first ctx - tools+system) /
characters; days without recorded requests use that day-independent median ratio); (2) for the last day, every agent
per task with its kind (step agent: a general-purpose launch from the main session whose prompt has "Entry:";
reviewers by type), why it started (a step agent is a relaunch when an earlier step agent had the same entry and step
text; a resume of an agent after it ended is a round of its own, a reviewer's a re-check), calls and tokens; (3) every
tool call classed by purpose (agent_calls.segments; first match: review_backlog.sh, work, plan CLI/site ack/events/
collect_run, reads, state reads) with the refusal overlay: a hook or permission refusal and the first call after a run
of refusals are recovery (review-gate handling when review_backlog.sh refused); (4) reviewer runs per day and their
triggers from the launch prompt; (5) agent and main totals, files re-read across agents of one task, step prompts and
plan CLI output read by launch hour.

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
import agent_calls                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTS = os.path.dirname(PROJ)
KINDS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")
SHORT = {"input_tokens": "in", "cache_read_input_tokens": "read", "cache_creation_input_tokens": "write",
         "output_tokens": "out"}
ROUTING = re.compile(r"plan\.py (next|step|run|ask|answer|evidence|set-next|add-next|done|dead-end)\b|"
                     r"\bsite (ack|events|digest)\b|plan_check|review_backlog\.sh")
STATE = re.compile(r"\bsite (poll|runs|status|logs|where|items|decisions)\b|git (status|log|diff)\b|plan\.py show")
SUBMIT = re.compile(r"(^|[;&|(]\s*|\n\s*|\bdo\s+)(\w+=\S*\s+)*(timeout \S+\s+)?"   # a command segment that submits,
                    r"(site (--\S+ \S+ )*(submit\b|run .*sweep_manager\.py submit)|(bash )?\S*launch_cc_only\.sh)")
#                                                                                   not a mention inside a grep or script


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


# ---------------------------------------------------------------------------------------------------------------------
# --compare DAY,DAY,...: what differs between days (start-up floor, agents per step, call purpose, reviewer volume)

REVIEWERS = ("design-reviewer", "repo-reviewer", "notes-editor", "claudemd-keeper")
LIMIT_TXT = "hit your 5-hour limit"


def agent_kind(a):
    if a["type"] in REVIEWERS:
        return a["type"]
    if a["type"] == "general-purpose" and a["parent"] is None and re.search(r"^Entry: ", a["prompt"], re.M):
        return "step agent"
    return "other (%s)" % a["type"]


def transcript(path, days=None):
    """One pass over a transcript: (responses [(ts, ctx, out, usage, blocks, round, after_stop)], tool results
    {tool_use id: (text, is_error)}, start facts {claudemd chars, shape hash, tools+system chars}, rounds).
    days: [(lo, hi)] UTC iso ranges; responses outside them are dropped (the main session). A round is a launch or a
    resume: a user message (not a tool result, not a harness note) after a response that called no tool or handed
    back. after_stop: the response belongs to a run of responses that a Stop hook's feedback started."""
    inr = (lambda ts: True) if days is None else (lambda ts: any(lo <= ts[:19] < hi for lo, hi in days))
    resp, idx, results = [], {}, {}
    start = {"claudemd": 0, "shape": None, "toolsys_chars": 0, "round_wake": {0: "launch"}, "reports": defaultdict(list)}
    rnd, ended, stop_run, seen_user, blobs = 0, False, False, False, {}
    with open(path) as fh:
        for line in fh:
            if '"type":"assistant"' not in line and '"type":"user"' not in line and '"attachment"' not in line \
                    and '"api-request' not in line:
                continue
            r = json.loads(line)
            t = r.get("type")
            if t == "api-request-blob" and "msg_chars" not in start:
                blobs[r.get("hash")] = r.get("message") or {}
                continue
            if t == "api-request" and "msg_chars" not in start:
                # the first request's messages, by part: CLAUDE.md (the instructions reminder), the launch prompt (the
                # first plain text that is not a reminder), the rest (other reminders, tool additions)
                parts = {"claudemd": 0, "prompt": 0, "rest": 0}
                for h in r.get("tail") or []:
                    for x in (blobs.get(h) or {}).get("content") or []:
                        if not isinstance(x, dict):
                            continue
                        txt = x.get("text") if x.get("type") == "text" else None
                        n = len(txt) if txt is not None else len(json.dumps(x))
                        if txt and "Contents of " in txt[:400] and "CLAUDE.md" in txt[:400]:
                            parts["claudemd"] += n
                        elif txt and not txt.lstrip().startswith("<") and not parts["prompt"]:
                            parts["prompt"] += n
                        else:
                            parts["rest"] += n
                start["msg_chars"] = parts
                continue
            if t in ("api-request-blob", "api-request"):
                continue
            if t == "attachment" and not start["claudemd"]:
                a = r.get("attachment") or {}
                if a.get("type") == "instructions" and isinstance(a.get("files"), list):
                    start["claudemd"] = sum(len(x.get("content", "")) for x in a["files"])
                continue
            if t == "api-request-shape" and start["shape"] is None:
                sh = r.get("shape") or {}
                start["shape"] = r.get("shapeHash")
                start["toolsys_chars"] = len(json.dumps(sh.get("system"))) + len(json.dumps(sh.get("tools")))
                continue
            ts = r.get("timestamp") or ""
            m = r.get("message") or {}
            c = m.get("content")
            if t == "user":
                if isinstance(c, list) and any(isinstance(x, dict) and x.get("type") == "tool_result" for x in c):
                    for x in c:
                        if isinstance(x, dict) and x.get("type") == "tool_result":
                            cc = x.get("content")
                            results[x.get("tool_use_id")] = (cc if isinstance(cc, str) else text_of(cc),
                                                             bool(x.get("is_error")), x)
                    continue
                w = wake_of(r)
                if w == "agent report":
                    cj = json.dumps(c)
                    for g in re.findall(r"<task-id>(a[0-9a-f]{16})</task-id>|agent-message from=\\\"(a[0-9a-f]{16})", cj):
                        start["reports"][g[0] or g[1]].append(ts)
                if w == "stop hook":
                    stop_run = True
                elif w and w != "harness note":
                    stop_run = False
                    if seen_user and ended:
                        rnd += 1
                        start["round_wake"][rnd] = w
                    seen_user = True
                    ended = False
                continue
            if t != "assistant":
                continue
            u = m.get("usage") or {}
            blocks = [x for x in c or [] if isinstance(x, dict)]
            mid = m.get("id") or id(r)
            if mid in idx:
                if idx[mid] is None:
                    continue
                rec = resp[idx[mid]]
                rec[4].extend(blocks)
                for k in KINDS:
                    rec[3][k] = max(rec[3][k], u.get(k, 0) or 0)
            else:
                if not inr(ts):
                    idx[mid] = None
                    continue
                idx[mid] = len(resp)
                resp.append([ts, 0, 0, {k: u.get(k, 0) or 0 for k in KINDS}, blocks, rnd, stop_run])
            if idx[mid] is None:
                continue
            tools = [x for x in resp[idx[mid]][4] if x.get("type") == "tool_use"]
            ended = not tools or any(x.get("name") == "SubagentHandback" for x in tools)
    for x in resp:
        x[1] = x[3]["input_tokens"] + x[3]["cache_read_input_tokens"] + x[3]["cache_creation_input_tokens"]
        x[2] = x[3]["output_tokens"]
    return [x for x in resp if x[1]], results, start     # a response with no usage is an API error record


def seg_class(argv):
    """The purpose of one shell segment: review, work, book, read, state, or None (neutral)."""
    while argv and re.fullmatch(r"\w+=.*", argv[0]):
        argv = argv[1:]
    while argv and argv[0] in ("timeout", "nice", "time", "env", "command", "do", "then", "else", "if", "!", "{"):
        argv = argv[1:]
        while argv and re.fullmatch(r"-\S+|[\d.]+[smh]?", argv[0]):
            argv = argv[1:]
    if not argv:
        return None
    c, sub = os.path.basename(argv[0]), (argv[1] if len(argv) > 1 else "")
    prog = argv[1] if c in ("python", "python3", "bash", "sh") and len(argv) > 1 else argv[0]
    if prog.endswith("review_backlog.sh"):
        return "review"
    if prog.endswith(("plan.py", "collect_run.py")) or (c == "site" and sub in ("ack", "events", "digest")):
        return "book"
    if c in ("cd", "echo", "printf", "true", "false", "export", "set", "pwd", ":", "unset", "wait", "date", "for",
             "done", "fi", "sleep", "source", ".", "}"):
        return None
    if c in ("cat", "sed", "head", "tail", "grep", "egrep", "rg", "awk", "nl", "less", "wc", "ls", "find", "tree",
             "stat", "diff", "file", "du", "jq", "column", "sort", "uniq", "cut", "tr", "xargs", "basename", "dirname",
             "realpath", "readlink"):
        return "work" if (c == "sed" and any(a.startswith("-i") for a in argv)) else "read"
    if c == "git":
        return "read" if sub in ("log", "show", "diff", "status", "blame", "grep", "rev-parse", "branch",
                                 "ls-files", "describe", "cat-file", "-C") and "commit" not in argv else "work"
    if c == "site":
        return "state" if sub in ("poll", "runs", "status", "logs", "where", "items", "decisions", "list", "pick",
                                  "-h", "--help") else "work"
    return "work"                       # scripts, tests, python, cp/mv/tee/rm, submissions, anything that computes


PURPOSES = ("the work itself", "reading code or docs", "plan CLI and event bookkeeping",
            "recovery after a hook refusal", "review-gate handling", "other")


def call_purpose(x, rev_ids):
    """(purpose, detail) of one tool call, before the refusal overlay. rev_ids: agent ids of reviewers."""
    name, inp = x.get("name"), x.get("input") or {}
    if name in ("Edit", "Write", "NotebookEdit"):
        return PURPOSES[0], "edit"
    if name in ("Read", "Grep", "Glob"):
        return PURPOSES[1], name
    if name == "Agent":
        return (PURPOSES[4], "reviewer launch") if inp.get("subagent_type") in REVIEWERS else \
            (PURPOSES[5], "agent launch")
    if name == "SendMessage":
        return (PURPOSES[4], "message to a reviewer") if inp.get("to") in rev_ids else (PURPOSES[5], "SendMessage")
    if name != "Bash":
        return PURPOSES[5], name
    segs = agent_calls.segments(inp.get("command", ""))
    if segs is None:
        return PURPOSES[0], "unparsed command"
    ks = {seg_class(argv) for _, argv, red in segs} | ({"work"} if any(red for _, _, red in segs) else set())
    for k, p, d in (("review", 4, "review_backlog.sh"), ("work", 0, "command"), ("book", 2, "plan/site ack/events"),
                    ("read", 1, "shell read"), ("state", 5, "state read (site poll/runs/status/logs)")):
        if k in ks:
            return PURPOSES[p], d
    return PURPOSES[5], "neutral command"


def classify_calls(resp, results, rev_ids):
    """[(purpose, detail, ctx share, output share, refusal source or None, after_stop, ts)] per tool call, plus one row
    per text-only response. Overlay: a call refused by a hook or a permission check, and the calls after it up to and
    including the first call of the same tool that is not refused (at most 8), are 'recovery after a hook refusal',
    or 'review-gate handling' when review_backlog.sh refused."""
    out, pending = [], None                  # pending: (tool name, purpose to charge, calls left)
    for ts, ctx, o, u, blocks, rnd, stop in resp:
        tools = [x for x in blocks if x.get("type") == "tool_use"]
        if not tools:
            out.append((PURPOSES[5], "text reply", ctx, o, None, stop, ts))
            continue
        for x in tools:
            p, d = call_purpose(x, rev_ids)
            txt, err, raw = results.get(x.get("id"), ("", False, {}))
            src = None
            if err and agent_calls.refused({"is_error": True, "content": raw.get("content", txt)}, x) == \
                    "hook/permission refusal":
                src = agent_calls.source({"content": raw.get("content", txt)}, x)
            if src:
                charge = PURPOSES[4] if "review_backlog" in src else PURPOSES[3]
                out.append((charge, "refused: " + src, ctx / len(tools), o / len(tools), src, stop, ts))
                pending = charge
                continue
            if pending:
                out.append((pending, "after refusal: " + d, ctx / len(tools), o / len(tools), None, stop, ts))
                pending = None
                continue
            out.append((p, d, ctx / len(tools), o / len(tools), None, stop, ts))
    return out


def report_compare(sid, days):
    main_f = os.path.join(PROJ, sid + ".jsonl")
    info = agent_info(sid)
    rng = {d: day_range(d) for d in days}
    dayof = lambda ts: next((d for d, (lo, hi) in rng.items() if lo <= ts[:19] < hi), None)
    _, _, reports = main_records(main_f, "", "")
    mresp, mres, _ = transcript(main_f, list(rng.values()))
    sel = {aid: a for aid, a in info.items() if dayof(a["ts"])}
    T = {}
    for aid, a in sel.items():
        resp, res, start = transcript(a["file"])
        T[aid] = (resp, res, start)
        for k, v in start["reports"].items():            # an agent's report reaches the agent that launched it
            reports[k] = reports.get(k, []) + v
    kind = {aid: agent_kind(a) for aid, a in info.items()}
    rev_ids = {aid for aid in info if kind[aid] in REVIEWERS}
    limited = {aid for aid in sel if any(LIMIT_TXT in text_of(x.get("text", "")) for r in T[aid][0] for x in r[4]
                                         if x.get("type") == "text")}
    for aid in sel:                      # the limit refusal is an API error record, not always a text block
        if aid not in limited:
            with open(info[aid]["file"]) as fh:
                if any(LIMIT_TXT in ln and "isApiErrorMessage" in ln for ln in fh):
                    limited.add(aid)
    print("session %s; days %s (local). calls = API responses; ctx = a response's input context (input + cache read +"
          " cache write); tokens = ctx + output." % (sid, ", ".join(days)))

    # ---- 1. start-up floor
    print("\n== 1. Start-up floor: per agent, floor = first-call ctx x calls, growth = the rest of its ctx")
    # tools+system tokens, exact where the cache shows it: the largest first-call cache read among agents with the
    # same request shape (today) or of the same type on the same day (older transcripts have no shape record)
    first = {aid: T[aid][0][0] for aid in sel if T[aid][0]}
    grp = lambda aid: T[aid][2]["shape"] or (info[aid]["type"], dayof(info[aid]["ts"]))
    toolsys = defaultdict(int)
    for aid, r in first.items():
        toolsys[grp(aid)] = max(toolsys[grp(aid)], r[3]["cache_read_input_tokens"])
    # tokens per character of the first request's messages, per agent where the request is recorded: (first ctx -
    # tools+system) / the messages' characters (closure); the parts are then split in proportion to their characters
    ratio = {}
    for aid, r in first.items():
        mc = T[aid][2].get("msg_chars")
        if mc and toolsys[grp(aid)]:
            ratio[aid] = (r[1] - toolsys[grp(aid)]) / max(sum(mc.values()), 1)
    rq = st.median(ratio.values()) if ratio else None
    if ratio:
        q = sorted(ratio.values())
        print("tokens per character of the first request's messages: median %.3f (range %.3f .. %.3f over %d agents "
              "whose request is recorded); older days: CLAUDE.md and prompt characters x %.3f" % (
                  rq, q[0], q[-1], len(q), rq))
    rows = defaultdict(list)
    for aid, a in sel.items():
        resp = T[aid][0]
        if not resp:
            continue
        ctx = [r[1] for r in resp]
        f = ctx[0]
        rows[(dayof(a["ts"]), kind[aid])].append(dict(
            first=f, n=len(ctx), last=ctx[-1], sum=sum(ctx), floor=f * len(ctx), out=sum(r[2] for r in resp),
            ts_tok=toolsys[grp(aid)], cm=T[aid][2]["claudemd"], pr=len(a["prompt"]),
            parts={k: v * ratio[aid] for k, v in T[aid][2]["msg_chars"].items()} if aid in ratio else None))
    print("%-10s %-22s %4s %6s %7s %7s %7s %8s %8s %6s %8s %6s" % (
        "day", "kind", "n", "calls", "first k", "calls~", "last k", "ctx M", "floor M", "floor%", "growth M",
        "out M"))
    for (d, k), L in sorted(rows.items()):
        sm, fl = sum(x["sum"] for x in L), sum(x["floor"] for x in L)
        print("%-10s %-22s %4d %6d %7.1f %7.0f %7.1f %8.1f %8.1f %5.0f%% %8.1f %6.2f" % (
            d, k[:22], len(L), sum(x["n"] for x in L), st.median(x["first"] for x in L) / 1e3,
            st.median(x["n"] for x in L), st.median(x["last"] for x in L) / 1e3, sm / 1e6, fl / 1e6,
            100 * fl / max(sm, 1), (sm - fl) / 1e6, sum(x["out"] for x in L) / 1e6))
    print("(first k, last k: median first- and last-call ctx, k tokens; calls~: median calls per agent)")
    for d in days:
        L = [x for (dd, k), v in rows.items() if dd == d for x in v]
        cmt = sum((x["parts"]["claudemd"] if x["parts"] else x["cm"] * (rq or 0)) * x["n"] for x in L)
        print("%s: agents' ctx %.1fM, floor %.1fM; CLAUDE.md tokens x calls (the CLAUDE.md part of the floor) %.1fM" % (
            d, sum(x["sum"] for x in L) / 1e6, sum(x["floor"] for x in L) / 1e6, cmt / 1e6))
    print("\nfirst-call ctx split, median over the kind's agents, k tokens. tools+system: the cached prefix a first call "
          "read (exact); CLAUDE.md, prompt, rest (other reminders: deferred-tool, agent and skill lists, context, tool "
          "additions): the messages' tokens split by characters (recorded requests); on days without a recorded "
          "request, CLAUDE.md and prompt are characters x the median ratio and 'tools+sys+rest' is the remainder")
    print("%-10s %-22s %7s %9s %9s %7s %7s %14s %9s %9s" % ("day", "kind", "first", "tools+sys", "CLAUDE.md", "prompt",
                                                          "rest", "tools+sys+rest", "CLAUDE ch", "prompt ch"))
    for (d, k), L in sorted(rows.items()):
        if rq is None:
            break
        f = st.median(x["first"] for x in L)
        P = [x for x in L if x["parts"]]
        if P:
            med = lambda key: st.median(x["parts"][key] for x in P)
            tsy = st.median(x["ts_tok"] for x in P)
            print("%-10s %-22s %7.1f %9.1f %9.1f %7.1f %7.1f %14.1f %9.0f %9.0f" % (
                d, k[:22], f / 1e3, tsy / 1e3, med("claudemd") / 1e3, med("prompt") / 1e3, med("rest") / 1e3,
                (tsy + med("rest")) / 1e3, st.median(x["cm"] for x in L), st.median(x["pr"] for x in L)))
        else:
            cm, pr = st.median(x["cm"] for x in L) * rq, st.median(x["pr"] for x in L) * rq
            print("%-10s %-22s %7.1f %9s %9.1f %7.1f %7s %14.1f %9.0f %9.0f" % (
                d, k[:22], f / 1e3, "-", cm / 1e3, pr / 1e3, "-", (f - cm - pr) / 1e3,
                st.median(x["cm"] for x in L), st.median(x["pr"] for x in L)))
    print("\nmain session per day: first call of the day, its calls, median and last ctx, floor (first ctx x calls),"
          " compactions")
    print("%-10s %7s %6s %7s %7s %8s %8s %6s %5s" % ("day", "first k", "calls", "med k", "last k", "ctx M", "floor M",
                                                     "floor%", "comp"))
    for d in days:
        R = [r for r in mresp if dayof(r[0]) == d]
        if not R:
            continue
        ctx = [r[1] for r in R]
        print("%-10s %7.1f %6d %7.1f %7.1f %8.1f %8.1f %5.0f%% %5d" % (
            d, ctx[0] / 1e3, len(ctx), st.median(ctx) / 1e3, ctx[-1] / 1e3, sum(ctx) / 1e6, ctx[0] * len(ctx) / 1e6,
            100 * ctx[0] * len(ctx) / sum(ctx), sum(1 for a, b in zip(ctx, ctx[1:]) if b < 0.5 * a)))
    print("(comp: calls whose ctx fell below half the previous call's, i.e. compactions)")

    # ---- 2. agents per step (the last day)
    day = days[-1]
    ids, memo, steps = entry_ids(), {}, step_records(main_f)
    todays = sorted([aid for aid in sel if dayof(info[aid]["ts"]) == day], key=lambda k: info[k]["ts"])
    prev_by_task = defaultdict(list)
    for aid in sorted(info, key=lambda k: info[k]["ts"]):       # earlier agents of the same step (entry and step
        if kind[aid] == "step agent":                           # text; step numbers are reused), any day
            t = own_task(aid, info[aid], steps)
            if t:
                prev_by_task[(t[0], t[2])].append(aid)

    def outcome(aid):
        if aid in limited:
            return "cut off by the limit"
        return "reported" if [r for r in reports.get(aid, []) if r >= info[aid]["ts"]] else "no report"

    def reason(aid):
        a, k = info[aid], kind[aid]
        if k == "step agent":
            t = own_task(aid, a, steps)
            before = [b for b in prev_by_task[(t[0], t[2])] if info[b]["ts"] < a["ts"]] if t else []
            if not before:
                return "fresh launch"
            return "relaunch after: " + outcome(before[-1])
        if "NOT the pillar backlog" in a["prompt"]:
            return "review experiment (blind review)" + (" rerun" if "rerun" in a["desc"] else "")
        if k == "design-reviewer":
            ent = lambda b: re.search(r"plan entry (\S+)|entry (\S+)", info[b]["prompt"])
            e = ent(aid)
            same = [b for b in info if kind[b] == "design-reviewer" and b != aid and info[b]["ts"] < a["ts"] and
                    dayof(info[b]["ts"]) == dayof(a["ts"]) and ent(b) and e and ent(b).group(0) == e.group(0)]
            return "re-review after a FIX (new reviewer)" if re.search(
                r"revised|amended|redesigned|re-review|after (your|the) FIX", a["prompt"] + " " + a["desc"]) else \
                "first review" + (" (entry reviewed earlier today)" if same else "")
        if k in REVIEWERS:
            same = [b for b in info if kind[b] == k and b != aid and info[b]["parent"] == a["parent"] and
                    info[b]["ts"] < a["ts"] and a["parent"] is not None]
            return "re-check (new reviewer)" if same else "fresh review"
        return "fresh launch"
    print("\n== 2. Agents per step, %s (round 2+: a resume of the same agent by SendMessage, a re-check)" % day)
    tasks = defaultdict(list)
    for aid in todays:
        tasks[task_of(aid, info, ids, steps, memo)].append(aid)
    by_kind, by_reason = defaultdict(lambda: [0, 0, 0.0]), defaultdict(lambda: [0, 0, 0.0])
    for t, L in sorted(tasks.items(), key=lambda kv: info[kv[1][0]]["ts"]):
        tok = sum(r[1] + r[2] for aid in L for r in T[aid][0])
        print("\n%s / %s: %s  [%d agents, %.1fM tokens]" % (t[0], t[1], t[2][:70], len(L), tok / 1e6))
        for aid in L:
            resp = T[aid][0]
            for rnd in sorted({r[5] for r in resp}) or [0]:
                R = [r for r in resp if r[5] == rnd]
                rw = T[aid][2]["round_wake"].get(rnd, "?")
                why = reason(aid) if rnd == 0 else (
                    "re-check (resumed by a message)" if kind[aid] in REVIEWERS else "resumed by: " + rw)
                tk = sum(r[1] + r[2] for r in R)
                print("   %s %s %-18s %-6s %-36s %4d calls %6.2fM  %s" % (
                    aid[:8], info[aid]["ts"][11:16], kind[aid][:18], info[aid]["effort"][:6], why[:36], len(R),
                    tk / 1e6, outcome(aid) if rnd == 0 else ""))
                for acc_, key in ((by_kind, kind[aid]), (by_reason, (kind[aid], why))):
                    acc_[key][0] += 1 if rnd == 0 else 0
                    acc_[key][1] += len(R); acc_[key][2] += tk
    pe = defaultdict(lambda: [0, 0.0])
    for t, L in tasks.items():
        g = "plan entry, step agent's task" if any(kind[a] == "step agent" for a in L) else \
            "plan entry, other launch" if t[0] != "(no entry)" else "no plan entry"
        pe[g][0] += len(L); pe[g][1] += sum(r[1] + r[2] for a in L for r in T[a][0])
    print("\nby task kind (%s): agents, tokens: %s" % (day, "; ".join("%s %d, %.1fM" % (k, v[0], v[1] / 1e6)
                                                              for k, v in sorted(pe.items()))))
    print("\nby kind (%s): agents, calls, tokens" % day)
    for k, (n, c, tk) in sorted(by_kind.items(), key=lambda kv: -kv[1][2]):
        print("   %-26s %4d %6d %8.1fM" % (k, n, c, tk / 1e6))
    eff = defaultdict(lambda: [0, 0, 0.0])
    for aid in todays:
        if kind[aid] == "step agent":
            e = eff[(info[aid]["effort"], outcome(aid))]
            e[0] += 1; e[1] += len(T[aid][0]); e[2] += sum(r[1] + r[2] for r in T[aid][0])
    print("step agents by launch effort and outcome: agents, calls, tokens")
    for k, (n, c, tk) in sorted(eff.items(), key=lambda kv: -kv[1][2]):
        print("   %-8s %-22s %4d %6d %8.1fM" % (k[0], k[1], n, c, tk / 1e6))
    print("by kind and reason: agents (0 = resumed rounds only), calls, tokens")
    for k, (n, c, tk) in sorted(by_reason.items(), key=lambda kv: -kv[1][2]):
        print("   %-26s %-40s %4d %6d %8.1fM" % (k[0], k[1][:40], n, c, tk / 1e6))

    # ---- 3. call purpose
    print("\n== 3. Call purpose (each tool call charged its response's ctx / number of tool calls in it; a text-only "
          "response is one 'other: text reply' row)")
    per = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    det = defaultdict(lambda: defaultdict(Counter))
    hooks = defaultdict(Counter)
    stop_runs = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    srcs = [("main session", mresp, mres)] + [(aid, T[aid][0], T[aid][1]) for aid in sel]
    for who, resp, res in srcs:
        for p, d, cx, o, src, stop, ts in classify_calls(resp, res, rev_ids):
            dd = dayof(ts if who == "main session" else info[who]["ts"])
            if not dd:
                continue
            g = "main session" if who == "main session" else ("step agents" if kind[who] == "step agent" else
                                                              "reviewers" if kind[who] in REVIEWERS else "other agents")
            per[(dd, g)][p][0] += 1; per[(dd, g)][p][1] += cx + o
            det[(dd, g)][p][d.split(":")[0] if d.startswith(("refused", "after")) else d] += 1
            if src:
                hooks[dd][src] += 1
            if stop and who == "main session":
                stop_runs[dd][p][0] += 1; stop_runs[dd][p][1] += cx + o
    print("%-10s %-13s %-31s %6s %5s %8s %5s" % ("day", "who", "purpose", "calls", "%", "tok M", "%"))
    for (dd, g), P in sorted(per.items()):
        nc, nt = sum(v[0] for v in P.values()), sum(v[1] for v in P.values())
        for p in PURPOSES:
            if p in P:
                print("%-10s %-13s %-31s %6d %4.0f%% %8.1f %4.0f%%   %s" % (
                    dd, g, p, P[p][0], 100 * P[p][0] / nc, P[p][1] / 1e6, 100 * P[p][1] / nt,
                    ", ".join("%s %d" % kv for kv in det[(dd, g)][p].most_common(4))))
        print("%-10s %-13s %-31s %6d %5s %8.1f" % (dd, g, "all", nc, "", nt / 1e6))
    print("\nmain session: calls in runs a Stop hook's feedback started (the turn's end was refused), by purpose")
    for dd, P in sorted(stop_runs.items()):
        print("   %s: " % dd + "; ".join("%s %d calls %.1fM" % (p, P[p][0], P[p][1] / 1e6) for p in PURPOSES if p in P))
    print("\nrefusals by source (hooks and permission checks; tool errors excluded), all calls of the day")
    for dd in days:
        print("   %s: %s" % (dd, ", ".join("%s %d" % kv for kv in hooks[dd].most_common(10)) or "none"))

    # ---- 4. reviewer volume
    print("\n== 4. Reviewer runs per day (a run: a launch or a resume by SendMessage); trigger from the launch")

    def trigger(aid):
        a = info[aid]
        par = a["parent"]
        who = "main session" if par is None else ("step agent" if kind.get(par) == "step agent" else
                                                  "reviewer" if kind.get(par) in REVIEWERS else "other agent")
        p = a["prompt"] + " " + a["desc"]
        if "NOT the pillar backlog" in p:
            why = "an agent's review experiment"
        elif kind[aid] == "design-reviewer":
            why = "design guard (before a job)"
        elif re.search(r"flagged|\bdue\b|DUE|overdue|past its (review )?threshold|threshold|backlog", p):
            why = "review_backlog threshold"
        elif "CLAUDE.md" in p and "uncommitted" in p:
            why = "a CLAUDE.md change (any change is due)"
        else:
            why = "other"
        return who, why
    vol = defaultdict(lambda: [0, 0, 0.0, []])
    trig = defaultdict(Counter)
    for aid in sel:
        if kind[aid] not in REVIEWERS:
            continue
        dd = dayof(info[aid]["ts"])
        resp = T[aid][0]
        nr = len({r[5] for r in resp}) or 1
        tk = sum(r[1] + r[2] for r in resp)
        v = vol[(dd, kind[aid])]
        v[0] += 1; v[1] += nr; v[2] += tk; v[3].append(len(resp))
        trig[dd]["%s / %s" % trigger(aid)] += 1
    print("%-10s %-16s %6s %5s %8s %9s %8s" % ("day", "reviewer", "agents", "runs", "tok M", "M per run", "calls~"))
    for (dd, k), v in sorted(vol.items()):
        print("%-10s %-16s %6d %5d %8.1f %9.2f %8.0f" % (dd, k, v[0], v[1], v[2] / 1e6, v[2] / 1e6 / v[1],
                                                          st.median(v[3])))
    for dd in days:
        print("   triggers %s (launcher / cause): %s" % (dd, ", ".join("%s %d" % kv for kv in trig[dd].most_common())))

    # ---- 5. other differences
    print("\n== 5. Other differences")
    for dd in days:
        L = [aid for aid in sel if dayof(info[aid]["ts"]) == dd]
        tk = defaultdict(float)
        for aid in L:
            tk[kind[aid]] += sum(r[1] + r[2] for r in T[aid][0])
        R = [r for r in mresp if dayof(r[0]) == dd]
        mt = sum(r[1] + r[2] for r in R)
        print("%s: main session %d calls %.1fM tokens; agents %d, %.1fM (%s)" % (
            dd, len(R), mt / 1e6, len(L), sum(tk.values()) / 1e6,
            ", ".join("%s %.1fM" % (k, v / 1e6) for k, v in sorted(tk.items(), key=lambda kv: -kv[1]))))
    # files read by more than one agent of the same task (the last day)
    dup_chars, dup_files, dup_tasks = 0, Counter(), 0
    for t, L in tasks.items():
        seen = {}
        hit = False
        for aid in sorted(L, key=lambda k: info[k]["ts"]):
            resp, res, _ = T[aid]
            mine = set()
            for r in resp:
                for x in r[4]:
                    if x.get("type") != "tool_use":
                        continue
                    paths = []
                    inp = x.get("input") or {}
                    if x.get("name") == "Read":
                        paths = [inp.get("file_path", "")]
                    elif x.get("name") == "Bash":
                        for _, argv, _ in agent_calls.segments(inp.get("command", "")) or []:
                            if argv and os.path.basename(argv[0]) in ("cat", "sed", "head", "nl"):
                                tg = agent_calls.read_targets(argv, ROOT)
                                if isinstance(tg, list):
                                    paths += [p for p, *_ in tg]
                    n = len(res.get(x.get("id"), ("",))[0])
                    for pth in paths:
                        pth = os.path.normpath(pth)
                        if pth in seen and seen[pth] != aid:
                            dup_chars += n / max(len(paths), 1); dup_files[os.path.relpath(pth, ROOT)] += 1
                            hit = True
                        mine.add(pth)
            for pth in mine:
                seen.setdefault(pth, aid)
        dup_tasks += hit
    print("\n%s: reads (Read, cat, sed -n, head) of a file another agent of the same task read before: %d reads in %d "
          "of %d tasks, %.0fk result characters; most re-read: %s" % (
              day, sum(dup_files.values()), dup_tasks, len(tasks), dup_chars / 1e3,
              ", ".join("%s %d" % kv for kv in dup_files.most_common(8))))
    # step prompts and the plan CLI output step agents read, by hour of launch
    hours = defaultdict(lambda: [[], [], []])
    for aid in todays:
        if kind[aid] != "step agent":
            continue
        h = utc(info[aid]["ts"]).astimezone().hour // 3 * 3
        resp, res, _ = T[aid]
        plan_out = sum(len(res.get(x.get("id"), ("",))[0]) for r in resp for x in r[4]
                       if x.get("type") == "tool_use" and x.get("name") == "Bash"
                       and "plan.py" in (x.get("input") or {}).get("command", "")
                       and re.search(r"plan\.py (show|next)", x["input"]["command"]))
        hours[h][0].append(len(info[aid]["prompt"])); hours[h][1].append(plan_out)
        hours[h][2].append(T[aid][0][0][1] if T[aid][0] else 0)
    print("\nstep agents by local launch hour: n, median launch-prompt chars, median chars of plan show/next output "
          "read, median first-call ctx k")
    for h in sorted(hours):
        a, b, c = hours[h]
        print("   %02d-%02d h: %3d  prompt %6.0f  plan output %7.0f  first ctx %5.1fk" % (
            h, h + 3, len(a), st.median(a), st.median(b), st.median(c) / 1e3))


def main(argv):
    opt = {}
    for flag in ("--day", "--daily", "--turns", "--compare"):
        if flag in argv:
            i = argv.index(flag); opt[flag] = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    agents = "--agents" in argv
    argv = [x for x in argv if x != "--agents"]
    sid = argv[1] if len(argv) > 1 else os.path.basename(max(glob.glob(PROJ + "/*.jsonl"), key=os.path.getmtime))[:-6]
    if "--compare" in opt:
        report_compare(sid, opt["--compare"].split(","))
        return
    if "--daily" in opt:
        report_daily(sid, *opt["--daily"].split(".."))
        return
    report_day(sid, opt.get("--day", dt.date.today().isoformat()), agents, int(opt.get("--turns", 12)))


if __name__ == "__main__":
    main(sys.argv)
