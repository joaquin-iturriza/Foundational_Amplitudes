#!/usr/bin/env python3
"""How much of our token spend the three kinds of work routed to the lab LLM (D541, D545) represent, and what routing
them is expected to save (Joaquin 2026-11-24: "how many tokens should I expect to save by off loading these tasks").
Read-only over the FA transcripts (~/.claude/projects/-home-joaquin-work-FA: main sessions and their subagents; the
A/B replay folders -home-joaquin-work-FA-worktrees-zz-* are not ours in production and are left out); CPU, ~1 min.

    python3 tools/token_share.py [--since 2026-11-09T00:00] [--until 2026-11-23T00:00] [--out <json>]
                                 [--redo 0.1] [--lookup-check 10000]

Unit: tokens = input + output + cache read + cache write of each API message, deduplicated on message id over all
files (the unit of tools/lab_llm_report.transcript, so the trial's per-task numbers compare); list-price $
(tools/usage_windows.cost) beside it. A message counts in the window by its timestamp.

Roles (per transcript): main (top-level session file); owner (brief opens "You own one entry", or "owner" in the
launch description); event worker (mechanical-worker whose description opens "E<n>" or whose brief names a RUN event
and its log); fix worker (mechanical-worker that edited files, all under tools/, .claude/, scripts/, tests/, sites/,
~/.local/bin: tools/agent_tasks.py's tool-fix rule); other mechanical; Explore (production) and Explore (trial:
descriptions "lookup trial|lsp|graph", "recall trial", the lookup and recall trials' own agents); reviewers; other.

Kind 1, reading a failed run's log: event workers whole, plus every `site logs` tool result in any other transcript
whose turn (until the next user text) also runs `site ack`. For those, two bounds: residency (the result's tokens,
chars/3.5, times the API messages that carry it, the next one up to the context's end: next compaction or file end)
and read turn (residency plus every token of the messages from the `site logs` call to the `site ack` call).
Kind 2, tool fixes: fix workers whole. Kind 3, code lookups: production Explore agents whole; owners' and the main
session's own searches (Grep, Glob, Bash grep/rg/git grep) as residency, listed apart (they feed the agent's own
edit, so they are not routable as such).
"""
import argparse, glob, json, os, re, statistics, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usage_windows import cost  # noqa: E402

PROJ = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")
ROOT = os.path.expanduser("~/work/FA/")
FIXDIRS = ("tools/", ".claude/", "scripts/", "tests/", "sites/", os.path.expanduser("~/.local/bin"))
TRIAL = re.compile(r"^(lookup (trial|lsp\d?|graph)|recall trial)\b", re.I)
REVIEW = {"repo-reviewer", "notes-editor", "claudemd-keeper", "design-reviewer"}
READONLY = re.compile(r"^\s*(cd [^;&]+&&\s*)?(cat|head|tail|sed -n|grep|rg|ls|wc|site (logs|runs|status|poll|events|list))\b")
SEARCH = re.compile(r"(^|[;&|(]\s*)(grep|rg|git grep)\s")
CHARS_PER_TOKEN = 3.5
# an Explore brief naming any of these is not a lookup in trunk's code (results notebook, plan, transcripts, web,
# another repo): the lab route (CLAUDE.md, D541) covers code lookups only
NONCODE = re.compile(r"results\.tex|plan\.yaml|transcript|WebSearch|github\.com|research result", re.I)


def text_of(c):
    if isinstance(c, str):
        return c
    out = []
    for b in c or []:
        if isinstance(b, dict):
            out.append(b.get("text") or (text_of(b.get("content")) if b.get("type") == "tool_result" else ""))
    return "\n".join(x for x in out if x)


def read(f):
    """(brief, events) of a transcript: events are ('msg', id, usage, model, ts, [tool_use (name, cmd, id)]),
    ('result', tool_use_id, chars), ('user', text) and ('boundary',)."""
    ev, brief, at = [], None, {}
    for line in open(f, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") == "system" and d.get("subtype") == "compact_boundary" or d.get("isCompactSummary"):
            ev.append(("boundary",))
            continue
        m = d.get("message") if isinstance(d.get("message"), dict) else {}
        if d.get("type") == "assistant" and m.get("usage") and m.get("id"):
            uses = [(b.get("name"), json.dumps(b.get("input") or {}), b.get("id"))
                    for b in m.get("content") or [] if isinstance(b, dict) and b.get("type") == "tool_use"]
            if m["id"] in at:  # one API message is logged as one record per content block: merge its tool calls
                ev[at[m["id"]]][5].extend(uses)
                continue
            at[m["id"]] = len(ev)
            ev.append(("msg", m["id"], m["usage"], m.get("model"), (d.get("timestamp") or "")[:19], uses))
        elif d.get("type") == "user":
            c = m.get("content")
            if brief is None:
                brief = text_of(c)
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        ev.append(("result", b.get("tool_use_id"), len(text_of(b.get("content")))))
            if isinstance(c, str) or any(isinstance(b, dict) and b.get("type") == "text" for b in c or []):
                ev.append(("user",))
    return brief or "", ev


def tok(u):
    return sum(u.get(k) or 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                        "cache_creation_input_tokens"))


def usd(u, model):
    cc = u.get("cache_creation") or {}
    cw = u.get("cache_creation_input_tokens") or 0
    cw5, cw1h = cc.get("ephemeral_5m_input_tokens") or 0, cc.get("ephemeral_1h_input_tokens") or 0
    return cost({"model": model, "in": u.get("input_tokens") or 0, "out": u.get("output_tokens") or 0, "cw5": cw5,
                 "cw1h": cw1h, "cwu": max(cw - cw5 - cw1h, 0), "cr": u.get("cache_read_input_tokens") or 0}) or 0.0


def role(f, brief, meta, ev):
    if "/subagents/" not in f:
        return "main"
    t, desc = meta.get("agentType", "?"), meta.get("description", "")
    if brief.startswith("You own one entry") or re.search(r"\bowner\b", desc, re.I):
        return "owner"
    if t == "mechanical-worker":
        # workers edit through Bash as often as through Edit: the paths they commit (`git commit ... -- <paths>`)
        edited = {json.loads(c).get("file_path", "") for e in ev if e[0] == "msg" for n, c, _ in e[5]
                  if n in ("Edit", "Write", "NotebookEdit")}
        for e in ev:
            for n, c, _ in e[5] if e[0] == "msg" else ():
                m = n == "Bash" and re.search(r"git commit .*? -- ([^;&|\n]+)", json.loads(c).get("command", ""))
                if m:
                    edited.update(m.group(1).split())
        edited = {p[len(ROOT):] if p.startswith(ROOT) else p for p in edited if p}
        if edited and all(p.startswith(FIXDIRS) for p in edited):
            return "fix worker"
        if any(n == "Bash" and re.search(r"\bsite logs\b", c) for e in ev if e[0] == "msg" for n, c, _ in e[5]):
            return "event worker"
        return "other mechanical"
    if t == "Explore":
        return "Explore (trial)" if TRIAL.match(desc) else "Explore"
    if t in REVIEW or t == "Plan" and desc.lower().startswith("review"):
        return "reviewer"
    return "other agent"


KINDS = [  # (kind, test on (tool name, Bash command or file path)); a message goes to the first kind any call matches
    ("launch or brief agents", lambda n, c: n in ("Agent", "Task", "SendMessage", "TaskStop")),
    ("final report", lambda n, c: n == "SubagentHandback"),
    ("write-up (.tex)", lambda n, c: n in ("Edit", "Write") and c.endswith(".tex")
     or n == "Bash" and re.search(r"\.tex\b", c) and re.search(r"sed -i|python3? - <<|> ?\S+\.tex", c)),
    ("figures", lambda n, c: n in ("Edit", "Write") and re.search(r"fig|plot", c) or n == "Read" and c.endswith(".png")
     or n == "Bash" and re.search(r"check_caption|send_check|figs?/", c)),
    ("code edits", lambda n, c: n in ("Edit", "Write", "NotebookEdit")),
    ("plan and ledger bookkeeping", lambda n, c: n == "Bash" and re.search(
        r"tools/plan\.py|\bsite (ack|events|decide|undecide|items?|digest|decisions|hold|release)\b|detach\.py|"
        r"hook_incident|review_backlog", c)),
    ("jobs: submit, sync, pick", lambda n, c: n == "Bash" and re.search(
        r"\bsite (submit|sync|copy|pick|run|cancel|env)\b|sweep_manager|drive\.py", c)),
    ("git", lambda n, c: n == "Bash" and re.search(r"(^|[;&|]\s*)git ", c)),
    ("run state and logs", lambda n, c: n == "Bash" and re.search(r"\bsite (logs|runs|status|poll|where|fetch|list)\b|"
                                                                  r"labllm\.py", c)),
    ("analyses and tests", lambda n, c: n == "Bash" and re.search(r"python|pytest|\.sh\b", c)),
    ("read files and search", lambda n, c: n in ("Read", "Grep", "Glob") or n == "Bash"),
    ("other tools", lambda n, c: True),
]


def kind_of(uses):
    calls = []
    for n, c, _ in uses:
        try:
            i = json.loads(c)
        except ValueError:
            i = {}
        calls.append((n, i.get("command") or i.get("file_path") or ""))
    if not calls:
        return "text only (reply, reasoning)"
    for k, test in KINDS:
        if any(test(n, c) for n, c in calls):
            return k


def scan(since, until):
    seen = set()
    roles = defaultdict(lambda: {"tokens": 0, "usd": 0.0, "agents": 0, "per_agent": []})
    k1 = {"residency": 0, "read_phase": 0, "read_turn": 0, "usd_residency": 0.0, "usd_read_phase": 0.0,
          "usd_read_turn": 0.0, "reads": 0}
    k3own = {"residency": 0, "usd": 0.0, "searches": 0}
    calib, explore_desc, routed, worker_desc = [], [], defaultdict(list), defaultdict(list)
    files = sorted(glob.glob(PROJ + "/*.jsonl")) + sorted(glob.glob(PROJ + "/*/subagents/agent-*.jsonl"))
    # subagent files first: a subagent's message also appears in its parent's progress records (not read here)
    for f in files:
        mp = f[:-6] + ".meta.json"
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        brief, ev = read(f)
        r = role(f, brief, meta, ev)
        msgs = [e for e in ev if e[0] == "msg" and e[1] not in seen and since <= e[4] < until]
        all_cmds = " ".join(c for e in ev if e[0] == "msg" for _, c, _ in e[5])
        if r in ("event worker", "fix worker") and "labllm.py" in all_cmds:
            routed[r].append((meta.get("description", ""), sum(tok(e[2]) for e in ev if e[0] == "msg")))
        if not msgs:
            continue
        for e in msgs:
            seen.add(e[1])
        t = sum(tok(e[2]) for e in msgs)
        c = sum(usd(e[2], e[3]) for e in msgs)
        R = roles[r]
        R["tokens"] += t
        R["usd"] += c
        R["agents"] += r != "main"
        R["per_agent"].append(t)
        for e in msgs:
            K = R.setdefault("kinds", {}).setdefault(kind_of(e[5]), [0, 0.0, 0])
            K[0] += tok(e[2])
            K[1] += usd(e[2], e[3])
            K[2] += 1
        if r in ("event worker", "fix worker"):
            worker_desc[r].append((meta.get("description", ""), t))
        if r == "Explore":
            explore_desc.append((meta.get("description", ""), t, not NONCODE.search(brief)))
        if r == "event worker":
            continue  # counted whole
        # walk the context: tool uses by id, results with their size, messages after them
        uses, pos = {}, []
        for i, e in enumerate(ev):
            if e[0] == "msg":
                for n, cmd, uid in e[5]:
                    uses[uid] = (n, cmd, i)
        for i, e in enumerate(ev):
            if e[0] != "result" or e[1] not in uses:
                continue
            n, cmd, at = uses[e[1]]
            is_log = n == "Bash" and re.search(r"\bsite logs\b", cmd)
            is_search = r in ("owner", "main") and (n in ("Grep", "Glob") or n == "Bash" and SEARCH.search(
                json.loads(cmd).get("command", "")) and "site " not in json.loads(cmd).get("command", "")[:30])
            if not (is_log or is_search):
                continue
            T = e[2] / CHARS_PER_TOKEN
            later, ack_at, turn_end = [], None, False
            for j in range(i + 1, len(ev)):
                x = ev[j]
                if x[0] == "boundary":
                    break
                if x[0] == "user":
                    turn_end = True
                if x[0] == "msg":
                    later.append(x)
                    if not turn_end and ack_at is None and any(re.search(r"\bsite ack\b", cc) for _, cc, _ in x[5]):
                        ack_at = len(later)
            phase = 0
            for x in later:
                if not x[5] or not all(n in ("Read", "Grep", "Glob") or n == "Bash" and READONLY.search(
                        json.loads(cc).get("command", "")) for n, cc, _ in x[5]):
                    break
                phase += 1
            inwin = [x for x in later if since <= x[4] < until]
            if not inwin:
                continue
            res_t = T * len(inwin)
            # written once into the cache, then read by every later message
            res_c = sum(usd({"cache_read_input_tokens" if k else "cache_creation_input_tokens": T}, x[3])
                        for k, x in enumerate(inwin))
            if is_search:
                k3own["residency"] += res_t
                k3own["usd"] += res_c
                k3own["searches"] += 1
                continue
            if ack_at is None:
                continue  # a log read outside a RUN event's turn (state checks): not routed work
            if len(ev[at][5]) == 1:  # the result is the only new content: next write minus the call's own output
                new_in = sum(later[0][2].get(k) or 0 for k in ("cache_creation_input_tokens", "input_tokens"))
                calib.append((new_in - (ev[at][2].get("output_tokens") or 0)) / max(T, 1))
            turn = [x for x in [ev[at]] + later[:ack_at] if since <= x[4] < until]
            ph = [x for x in [ev[at]] + later[:phase + 1] if since <= x[4] < until]
            k1["read_phase"] += res_t + sum(tok(x[2]) for x in ph)
            k1["usd_read_phase"] += res_c + sum(usd(x[2], x[3]) for x in ph)
            k1["reads"] += 1
            k1["residency"] += res_t
            k1["usd_residency"] += res_c
            k1["read_turn"] += res_t + sum(tok(x[2]) for x in turn)
            k1["usd_read_turn"] += res_c + sum(usd(x[2], x[3]) for x in turn)
    return roles, k1, k3own, calib, explore_desc, routed, worker_desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-11-09T00:00")
    ap.add_argument("--until", default="2026-11-23T00:00")
    ap.add_argument("--out", default="analysis/lab_llm_trial/token_share.json")
    ap.add_argument("--fig", default="analysis/lab_llm_trial/figs/token_share")
    ap.add_argument("--fix-check", type=float, default=41.5e3, help="our tokens a routed fix costs (cheap check, "
                    "par:labllm2)")
    ap.add_argument("--redo", type=float, default=0.1, help="assumed share of routed fixes and lookups redone by us")
    ap.add_argument("--triage-check", type=float, default=38.2e3, help="our tokens a routed log reading costs when no "
                    "routed worker is on record yet: the cheap fix check's median (par:labllm2), the same Sonnet worker "
                    "reading only printed lines")
    ap.add_argument("--lookup-check", type=float, default=10e3, help="assumed our tokens to check one routed lookup")
    a = ap.parse_args()
    roles, k1, k3own, calib, explore_desc, routed, worker_desc = scan(a.since, a.until)
    total = sum(R["tokens"] for R in roles.values())
    total_usd = sum(R["usd"] for R in roles.values())
    ew, fw, ex = roles["event worker"], roles["fix worker"], roles["Explore"]
    mean = lambda R: R["tokens"] / max(R["agents"], 1)
    # routed workers measured since the route went live (2026-11-23..): whole-task tokens
    rt = {k: {"n": len(v), "median": statistics.median([t for _, t in v]) if v else None, "cases": v}
          for k, v in routed.items()}
    tri_cost = rt.get("event worker", {}).get("median") or a.triage_check
    save = {
        "log_owner_kept": {"now": k1["read_turn"] + ew["tokens"], "after": k1["read_turn"] + k1["reads"] * tri_cost,
                           "note": "current rule (CLAUDE.md: the owner still reads the log): a checked DeepSeek "
                                   "reading is added per event, the owner's read stays"},
        "log_owner_trusts": {"now_low": k1["residency"] + ew["tokens"], "now_mid": k1["read_phase"] + ew["tokens"],
                             "now_high": k1["read_turn"] + ew["tokens"],
                             "after": k1["reads"] * tri_cost,
                             "note": "if owners act on the checked reading and open the log only when the cause is "
                                     "left open (Joaquin's open question)"},
        "fixes": {"now": fw["tokens"], "after": fw["agents"] * (a.fix_check + a.redo * mean(fw))},
        "lookups": {"now": sum(t for _, t, code in explore_desc if code),
                    "after": sum(a.lookup_check + a.redo * t for _, t, code in explore_desc if code),
                    "note": "production Explore agents whose brief is a code lookup (NONCODE); %d of %d" % (
                        sum(c for *_, c in explore_desc), len(explore_desc))},
    }
    for v in save.values():
        for k in [k for k in v if k.startswith("now")]:
            v["saved" + k[3:]] = v[k] - v["after"]
            v["share" + k[3:]] = v["saved" + k[3:]] / total
    rep = {"window": [a.since, a.until], "unit": "input+output+cache read+cache write, deduplicated message ids",
           "total_tokens": total, "total_usd": round(total_usd, 2),
           "roles": {k: {"tokens": v["tokens"], "usd": round(v["usd"], 2), "agents": v["agents"],
                         "share": v["tokens"] / total, "share_usd": v["usd"] / total_usd,
                         "mean_per_agent": mean(v) if k != "main" else None,
                         "kinds": {kk: {"tokens": vv[0], "usd": round(vv[1], 2), "messages": vv[2],
                                        "share_of_total": vv[0] / total}
                                   for kk, vv in sorted(v.get("kinds", {}).items(), key=lambda kv: -kv[1][0])}}
                     for k, v in roles.items()},
           "kind1_logs": dict(k1, event_workers=ew["agents"], event_worker_tokens=ew["tokens"],
                              chars_per_token=CHARS_PER_TOKEN,
                              calib_next_write_over_T_median=statistics.median(calib) if calib else None),
           "kind3_own_searches": k3own, "explore_production": explore_desc, "workers": worker_desc, "routed_measured": rt,
           "assumptions": {"triage_check": a.triage_check, "fix_check": a.fix_check, "redo": a.redo, "lookup_check": a.lookup_check},
           "saving": save}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print("window %s .. %s  total %.1fM tokens  $%.0f" % (a.since, a.until, total / 1e6, total_usd))
    for k, v in sorted(rep["roles"].items(), key=lambda kv: -kv[1]["tokens"]):
        print("  %-17s %8.1fM  %5.1f%%  $%7.0f (%4.1f%%)  agents %4d" % (k, v["tokens"] / 1e6, 100 * v["share"], v["usd"],
                                                                        100 * v["share_usd"], v["agents"]))
    print("kind1: %d owner/other log reads in RUN-event turns: residency %.1fM, read phase %.1fM, read turn %.1fM; calib %.2f" % (
        k1["reads"], k1["residency"] / 1e6, k1["read_phase"] / 1e6, k1["read_turn"] / 1e6, rep["kind1_logs"]["calib_next_write_over_T_median"] or 0))
    print("kind3 own searches: %d, residency %.1fM" % (k3own["searches"], k3own["residency"] / 1e6))
    print("routed measured:", {k: (v["n"], v["median"]) for k, v in rt.items()})
    for k, v in save.items():
        print("saving %-17s" % k, {kk: (round(vv / 1e6, 2) if "share" not in kk else round(100 * vv, 2))
                                   for kk, vv in v.items() if kk != "note"}, v.get("note", "")[:60])
    print(json.dumps({"total_M": round(total / 1e6, 1), **{k: round(100 * v.get("share", v.get("share_mid", 0)), 2)
                                                           for k, v in save.items()}}))
    for r in ("owner", "main", "other agent", "reviewer"):
        print("kinds of %s:" % r)
        for k, v in rep["roles"][r]["kinds"].items():
            print("    %-30s %7.1fM  %5.1f%% of all  $%6.0f  msgs %6d  %.0fk/msg" % (
                k, v["tokens"] / 1e6, 100 * v["share_of_total"], v["usd"], v["messages"], v["tokens"] / v["messages"] / 1e3))
    figure(rep, a.fig)


def figure(rep, base):
    """Tokens of each role over the window and, within them, the work routed to the lab LLM."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import plot_style as ps
    R = rep["roles"]
    bars = [("entry owners", R["owner"]["tokens"], 0), ("main session", R["main"]["tokens"], 0),
            ("other agents", R["other agent"]["tokens"], 0), ("reviewers", R["reviewer"]["tokens"], 0),
            ("owners reading failed logs", rep["kind1_logs"]["read_phase"], 1),
            ("other mechanical workers", R["other mechanical"]["tokens"], 0),
            ("tool-fix workers", R["fix worker"]["tokens"], 1),
            ("Explore, not code lookups", R["Explore"]["tokens"], 0),
            ("log-reading workers", R["event worker"]["tokens"], 1)]
    fig, ax = ps.figure(1, 1)
    for i, (name, t, routed) in enumerate(bars):
        ax.barh(i, t / 1e6, color=ps.C.vermillion if routed else ps.C.blue,
                label=None if i not in (0, 4) else ("all work of the role" if not routed else "routed to DeepSeek"))
    ax.set_yticks(range(len(bars)), [b[0] for b in bars])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.8, 1e5)  # clear lower-right corner for the legend (the bottom bars end below 10^2)
    ax.set_xlabel("our tokens over 14 days ($10^6$)")
    ps.legend(ax, "lower right")
    ps.save(fig, base)


if __name__ == "__main__":
    main()
