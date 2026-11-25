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
ENTRIES = []
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
                e = ev[at[m["id"]]]
                e[5].extend(uses)
                ev[at[m["id"]]] = (e[0], e[1], m["usage"], e[3], e[4], e[5])  # the last record's usage is the final one
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
    # a subagent's message also appears in its parent's progress records, which read() never takes
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
    ap.add_argument("--refusals", action="store_true", help="the refusal-cost pass (D558) instead of the routing shares")
    a = ap.parse_args()
    if a.refusals:
        return refusal_report(a.since, a.until, "analysis/agent_redesign/refusal_cost.json",
                              "analysis/agent_redesign/figs/refusal_cost")
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
    slug = lambda k: re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")
    R = rep["roles"]
    # D555: the owner reads the log only where the checked reading leaves the cause open; the read-phase saving is
    # its value when every checked reading settles the cause
    d555 = save["log_owner_trusts"]["share_mid"] + save["fixes"]["share"] + save["lookups"]["share"]
    last = json.dumps({
        "total_M": round(total / 1e6), "usd": round(total_usd),
        "roles_pct": {k.replace(" ", "_"): round(100 * R[k]["share"], 1) for k in ("owner", "main", "other agent", "reviewer")},
        "reviewer_usd_pct": round(100 * R["reviewer"]["share_usd"], 1),
        "fix_workers": R["fix worker"]["agents"], "fix_worker_M": round(R["fix worker"]["tokens"] / 1e6, 1),
        "log_reads": k1["reads"], "log_read_phase_M": round(k1["read_phase"] / 1e6),
        "saving_pct": {"fixes": round(100 * save["fixes"]["share"], 2), "lookups": round(100 * save["lookups"]["share"], 2),
                       "logs_reread": round(100 * save["log_owner_kept"]["share"], 2),
                       "logs_d555": round(100 * save["log_owner_trusts"]["share_mid"], 2), "total_d555": round(100 * d555, 2)},
        "kinds_pct": {r: {slug(k): round(100 * v["share_of_total"], 1) for k, v in R[r]["kinds"].items()}
                      for r in ("owner", "main")}})
    for r in ("owner", "main", "other agent", "reviewer"):
        print("kinds of %s:" % r)
        for k, v in rep["roles"][r]["kinds"].items():
            print("    %-30s %7.1fM  %5.1f%% of all  $%6.0f  msgs %6d  %.0fk/msg" % (
                k, v["tokens"] / 1e6, 100 * v["share_of_total"], v["usd"], v["messages"], v["tokens"] / v["messages"] / 1e3))
    figure(rep, a.fig)
    figure_kinds(rep, a.fig.replace("token_share", "token_activity"))
    print(last)  # the last {...} line: tools/verify_claims.py reads it


def figure_kinds(rep, base):
    """Tokens of the owners and of the main session per kind of activity (each message to the kind of its calls)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import plot_style as ps
    ko, km = rep["roles"]["owner"]["kinds"], rep["roles"]["main"]["kinds"]
    kinds = sorted(set(ko) | set(km), key=lambda k: -(ko.get(k, {}).get("tokens", 0) + km.get(k, {}).get("tokens", 0)))
    fig, ax = ps.figure(1, 1)
    for j, (kk, name, col) in enumerate(((ko, "entry owners", ps.C.blue), (km, "main session", ps.C.vermillion))):
        ax.barh([i + 0.2 * (2 * j - 1) for i in range(len(kinds))], [kk.get(k, {}).get("tokens", 0) / 1e6 for k in kinds],
                height=0.4, color=col, label=name)
    ax.set_yticks(range(len(kinds)), [k.replace(" (reply, reasoning)", ", no tool call") for k in kinds])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1, 1e5)  # clear lower-right corner for the legend (the bottom bars end below 10^2)
    ax.set_xlabel("our tokens over 14 days ($10^6$)")
    ps.legend(ax, "lower right")
    ps.save(fig, base)


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


# ---- refusal cost (D558): tokens each role spends after a hook or tool refusal until the refused action succeeds ----
HOOK_ERR = re.compile(r"^(PreToolUse|PostToolUse):\w+ hook error: \[[^\]]*\]: ?", re.S)
CLASSIFIER = re.compile(r"denied by the Claude Code auto mode classifier|denied by a built-in Claude Code safety")
TOOL_REFUSAL = re.compile(r"^(plan|site gate|site)(:| \S+ refused)", re.M)
ROLE5 = {"main": "main session", "owner": "owners", "reviewer": "reviewers", "event worker": "mechanical workers",
         "fix worker": "mechanical workers", "other mechanical": "mechanical workers"}


def check_of(text):
    """(hook, check) named from a refusal's own text: the hook, then its first reason with names and numbers cut."""
    t = HOOK_ERR.sub("", text).strip()
    if t.startswith("Stop hook feedback:"):
        t = re.sub(r"^Stop hook feedback:\s*(\[[^\]]*\]:\s*)?", "", t)
    if CLASSIFIER.search(t):
        return "auto-mode classifier (not ours)", "permission denied"
    m = (re.match(r"BLOCKED by (?:\.claude/hooks/)?([\w.-]+?)(?:\.sh|\.py)?:\s*(.*)", t, re.S)
         or re.match(r"(git verify guard|plan files guard|site gate|brief_guard|send_check|wait_guard|plan leave-check"
                     r"|md_guard|context_guard|design_guard|figure_guard)\W*(.*)", t, re.S)
         or re.match(r"PLAN CHECK: the turn cannot end yet\.\s*(?:PLAN CHECK: )?()(.*)", t, re.S)
         or re.match(r"(plan|site):\s*(.*)", t, re.S) or re.match(r"(\S+)\s+(.*)", t, re.S))
    if not m:
        return "?", t[:60]
    hook = m.group(1) or "plan_check"
    hook = {"review-backlog": "review_backlog", "git verify guard": "git_verify_guard", "plan files guard":
            "plan_files_guard", "plan": "plan.py", "site": "site", "plan leave-check": "plan_check"}.get(hook, hook)
    r = m.group(2).strip().split("\n")[0]
    r = re.sub(r"^refused \((.*?)\)?\.?$", r"\1", r)
    r = re.sub(r"^(placement|design|budget):.*", r"\1", r) if hook == "submit_guard" else r
    r = re.sub(r"\b(%s)\b" % "|".join(sorted(ENTRIES, key=len, reverse=True)), "<entry>", r) if ENTRIES else r
    r = re.sub(r"\b(repo-reviewer|notes-editor|claudemd-keeper|design-reviewer)\b", "<pillar>", r)
    r = re.sub(r"'[^']*'|`[^`]*`|\"[^\"]*\"|\S*/\S+|\bE\d+|\b[0-9a-f]{12,}\b|\d+(\.\d+)?", "_", r)
    r = re.sub(r"_(\s*,?\s*(and\s+)?_)+", "_", r)
    r = re.split(r"(?<=[\w)])[.;]\s|\s\(| \u2014 ", r)[0]
    return hook, re.sub(r"\s+", " ", r)[:70]


def action_of(name, inp):
    """What a call does, so a later call can be told the same action: the Bash program and its verb, an edited file."""
    if name == "Bash":
        c = re.sub(r"^\s*(cd [^;&]+&&\s*|\w+=\S+\s+)*", "", inp.get("command", ""))
        w = c.split()
        n = 3 if w[:1] in (["python3"], ["python"]) and len(w) > 1 and w[1].endswith(".py") else 2
        return "Bash " + " ".join(w[:n])
    if name in ("Edit", "Write", "NotebookEdit"):
        return name + " " + inp.get("file_path", "")
    return name


def read_refusals(f):
    """The ordered events of a transcript for the refusal pass: ('msg', id, usage, model, ts, [(uid, action)]),
    ('res', uid, refusal (hook, check) or None, error), ('stop', (hook, check), ts) for a Stop bounce,
    ('post', uid, (hook, check)) for a PostToolUse block, ('user',) for a real user turn, ('boundary',)."""
    ev, at = [], {}
    for line in open(f, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        ts = (d.get("timestamp") or "")[:19]
        if d.get("type") == "system" and d.get("subtype") == "compact_boundary" or d.get("isCompactSummary"):
            ev.append(("boundary",))
            continue
        a = d.get("attachment") if isinstance(d.get("attachment"), dict) else {}
        if a.get("type") == "hook_blocking_error" and str(a.get("hookName", "")).startswith("PostToolUse"):
            b = a.get("blockingError")
            ev.append(("post", a.get("toolUseID"), check_of(b.get("blockingError", "") if isinstance(b, dict) else str(b))))
            continue
        m = d.get("message") if isinstance(d.get("message"), dict) else {}
        if d.get("type") == "assistant" and m.get("usage") and m.get("id"):
            uses = [(b.get("id"), action_of(b.get("name"), b.get("input") or {})) for b in m.get("content") or []
                    if isinstance(b, dict) and b.get("type") == "tool_use"]
            if m["id"] in at:
                e = ev[at[m["id"]]]
                ev[at[m["id"]]] = (e[0], e[1], m["usage"], e[3], e[4], e[5] + uses)
                continue
            at[m["id"]] = len(ev)
            ev.append(("msg", m["id"], m["usage"], m.get("model"), ts, uses))
        elif d.get("type") == "user":
            c = m.get("content")
            if isinstance(c, str) and re.match(r"(Subagent)?Stop hook feedback", c):
                ev.append(("stop", check_of(c), ts))
                continue
            for b in c if isinstance(c, list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    t = text_of(b.get("content"))
                    ref = None
                    if b.get("is_error") and (HOOK_ERR.match(t) or CLASSIFIER.search(t)):
                        ref = check_of(t)
                    elif b.get("is_error") and TOOL_REFUSAL.search(t) and "Traceback" not in t and "usage:" not in t:
                        ref = check_of(t[TOOL_REFUSAL.search(t).start():])
                    ev.append(("res", b.get("tool_use_id"), ref, bool(b.get("is_error"))))
            if isinstance(c, str) or any(isinstance(b, dict) and b.get("type") == "text" for b in c or []):
                ev.append(("user",))
    return ev


def refusal_scan(since, until):
    """Per (hook, check, role): refusals, and the tokens of the messages after each one, two bounds. Upper: until the
    refused action succeeds (a later call of the same action not refused; a Stop bounce: the next attempt to end) or
    is dropped (the turn ends, the context is compacted). Lower: until the first message whose calls all go through.
    A message goes to the newest refusal still open, so no token is counted twice; the refused call itself is not
    counted (it was spent before the refusal). A refused call that is never retried is dropped: it costs only until the
    agent moves on (its upper bound is its lower one); a Stop bounce costs until the next attempt to end.
    shortcut: usd_hi and msgs_hi keep a dropped refusal's later messages; tokens are the reported unit.
    shortcut: messages taken back from a dropped refusal go to no older open refusal (a slight undercount)."""
    seen, out, examples = set(), defaultdict(lambda: {"n": 0, "lo": 0, "hi": 0, "usd_hi": 0.0, "msgs_hi": 0,
                                                      "dropped": 0, "repeat": 0}), defaultdict(list)
    files = sorted(glob.glob(PROJ + "/*.jsonl")) + sorted(glob.glob(PROJ + "/*/subagents/agent-*.jsonl"))
    for f in files:
        mp = f[:-6] + ".meta.json"
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        brief, ev0 = read(f)
        r = ROLE5.get(role(f, brief, meta, ev0), "other agents")
        ev = read_refusals(f)
        res = {e[1]: e for e in ev if e[0] == "res"}
        action = {uid: a for e in ev if e[0] == "msg" for uid, a in e[5]}
        opened = []  # open refusals: dict(key, action, lo_open)
        last_stop = None  # the check of the last Stop bounce, until a message with no call passes

        def close(o, dropped=False):
            if o in opened:
                opened.remove(o)
                K = out[o["key"] + (r,)]
                K["dropped"] += dropped
                if dropped and o["action"] not in ("stop", None):
                    K["hi"] -= o.get("after_lo", 0)  # dropped: it cost only until the agent moved on

        for i, e in enumerate(ev):
            if e[0] in ("user", "boundary"):
                for o in list(opened):
                    close(o, dropped=True)
                continue
            new = []
            if e[0] == "res" and e[2] and e[1] in action:
                new.append({"key": e[2], "action": action[e[1]], "lo": True})
            elif e[0] == "stop":
                new.append({"key": e[1], "action": "stop", "lo": True})
                if last_stop == e[1] and since <= e[2] < until:
                    out[e[1] + (r,)]["repeat"] += 1  # the same check bounced the very next attempt to end
                last_stop = e[1]
            elif e[0] == "post":
                new.append({"key": e[2], "action": None, "lo": True})
            for o in new:
                ts = next((x[4] for x in ev[i::-1] if x[0] == "msg"), "")
                if not since <= ts < until:
                    continue
                out[o["key"] + (r,)]["n"] += 1
                if len(examples[o["key"]]) < 6:
                    examples[o["key"]].append((os.path.basename(f), ts))
                opened.append(o)
            if e[0] != "msg" or not opened:
                continue
            if e[1] in seen or not since <= e[4] < until:
                continue
            seen.add(e[1])
            o = opened[-1]
            K = out[o["key"] + (r,)]
            K["hi"] += tok(e[2])
            K["usd_hi"] += usd(e[2], e[3])
            K["msgs_hi"] += 1
            if o["lo"]:
                K["lo"] += tok(e[2])
            else:
                o["after_lo"] = o.get("after_lo", 0) + tok(e[2])  # taken back if the action is dropped
            ok = [uid for uid, _ in e[5] if not (res.get(uid) and (res[uid][2] or res[uid][3]))]
            for o2 in list(opened):
                if all(not (res.get(uid) and (res[uid][2] or res[uid][3])) for uid, _ in e[5]):
                    o2["lo"] = False  # every call of this message went through: the agent has moved on
                if o2["action"] == "stop" and not e[5]:
                    last_stop = None if not (i + 1 < len(ev) and ev[i + 1][0] == "stop") else last_stop
                    close(o2)  # the next attempt to end (its own bounce, if any, opens a new refusal)
                elif o2["action"] is None and not o2["lo"]:
                    close(o2)
                elif o2["action"] and any(a == o2["action"] for uid, a in e[5] if uid in ok):
                    close(o2)
        for o in list(opened):
            close(o, dropped=True)
    return out, examples


def refusal_report(since, until, out_path, fig):
    global ENTRIES
    try:
        import yaml
        ENTRIES = [x.get("id") for x in yaml.safe_load(open(ROOT + "plan/plan.yaml"))["entries"] if x.get("id")]
    except Exception:
        ENTRIES = []
    out, examples = refusal_scan(since, until)
    roles, *_ = scan(since, until)
    total = sum(R["tokens"] for R in roles.values())
    by_role_total = defaultdict(int)
    for k, R in roles.items():
        by_role_total[ROLE5.get(k, "other agents")] += R["tokens"]
    checks = defaultdict(lambda: {"n": 0, "lo": 0, "hi": 0, "usd_hi": 0.0, "dropped": 0, "repeat": 0, "roles": {}})
    for (hook, chk, r), K in out.items():
        C = checks[(hook, chk)]
        for k in ("n", "lo", "hi", "usd_hi", "dropped", "repeat"):
            C[k] += K[k]
        C["roles"][r] = {k: K[k] for k in ("n", "lo", "hi", "msgs_hi", "dropped")}
    rows = sorted(checks.items(), key=lambda kv: -kv[1]["hi"])
    role_sum = defaultdict(lambda: {"n": 0, "lo": 0, "hi": 0})
    for (_, _, r), K in out.items():
        for k in ("n", "lo", "hi"):
            role_sum[r][k] += K[k]
    rep = {"window": [since, until], "unit": "input+output+cache read+cache write, deduplicated message ids",
           "total_tokens": total, "role_tokens": by_role_total,
           "refusal_tokens_by_role": {r: dict(v, share_of_role_hi=v["hi"] / max(by_role_total[r], 1),
                                              share_of_role_lo=v["lo"] / max(by_role_total[r], 1))
                                      for r, v in role_sum.items()},
           "checks": [{"hook": h, "check": c, **{k: v for k, v in C.items()}, "examples": examples[(h, c)]}
                      for (h, c), C in rows]}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(rep, open(out_path, "w"), indent=1, default=float)
    print("window %s .. %s  all tokens %.1fM" % (since, until, total / 1e6))
    for r, v in sorted(rep["refusal_tokens_by_role"].items(), key=lambda kv: -kv[1]["hi"]):
        print("  %-19s refusals %5d  tokens after them %6.1f-%6.1fM  = %4.1f-%4.1f%% of the role's %6.1fM" % (
            r, v["n"], v["lo"] / 1e6, v["hi"] / 1e6, 100 * v["share_of_role_lo"], 100 * v["share_of_role_hi"],
            by_role_total[r] / 1e6))
    for (h, c), C in rows[:25]:
        print("  %6.1fM (lo %6.1fM) n %4d dropped %3d repeat %3d  %-16s %s  | %s" % (
            C["hi"] / 1e6, C["lo"] / 1e6, C["n"], C["dropped"], C["repeat"], h, c,
            " ".join("%s %d/%.1fM" % (r[:5], v["n"], v["hi"] / 1e6) for r, v in sorted(C["roles"].items()))))
    hi = sum(v["hi"] for v in role_sum.values())
    lo = sum(v["lo"] for v in role_sum.values())
    ver = refusal_verdicts(since, until)
    rep["verdicts"] = ver
    json.dump(rep, open(out_path, "w"), indent=1, default=float)
    slug = lambda c: re.sub(r"[^a-z0-9]+", "_", next((w for h, k, w in PLAIN if c["hook"] == h and c["check"].startswith(k)),
                                                     c["hook"] + " " + c["check"][:20]).lower()).strip("_")
    print(json.dumps({"refusals": sum(v["n"] for v in role_sum.values()), "hi_M": round(hi / 1e6, 1),
                      "verdicts": ver,
                      "role_pct_hi": {r.replace(" ", "_"): round(100 * v["share_of_role_hi"], 1)
                                      for r, v in rep["refusal_tokens_by_role"].items()},
                      "role_pct_lo": {r.replace(" ", "_"): round(100 * v["share_of_role_lo"], 1)
                                      for r, v in rep["refusal_tokens_by_role"].items()},
                      "role_n": {r.replace(" ", "_"): v["n"] for r, v in rep["refusal_tokens_by_role"].items()},
                      "top": {slug(c): {"hi_M": round(c["hi"] / 1e6, 1), "lo_M": round(c["lo"] / 1e6, 1), "n": c["n"],
                                        "repeat": c["repeat"], "dropped": c["dropped"]}
                              for c in rep["checks"][:13]},
                      "lo_M": round(lo / 1e6, 1), "share_hi_pct": round(100 * hi / total, 1),
                      "share_lo_pct": round(100 * lo / total, 1),
                      "by_role_hi_M": {r.replace(" ", "_"): round(v["hi"] / 1e6, 1) for r, v in role_sum.items()},
                      }))
    figure_refusals(rep, fig)


CTX_FIX = "2026-11-19T11:12"  # ee7afdfb7 (UTC): context_guard refuses the Stop at most once per compaction period
PILLAR_FILE = {  # review_backlog.sh PILLARS, as path tests (shortcut: prefix tests, not git pathspecs; enough here)
    "claudemd-keeper": lambda p: p == "CLAUDE.md" or p.startswith((".claude/context", ".claude/agents",
                                                                   "plan/step_prompt", "plan/mechanical_prompt")),
    "notes-editor": lambda p: p.startswith("docs/") and p.endswith(".tex"),
    "repo-reviewer": lambda p: (p.endswith((".py", ".sh")) or p.split("/")[0] in (
        "config", "recipes", "models", "sweep", "tools", "tests", "scripts") or p == ".claude/settings.json")
    and not p.endswith((".json", ".log", ".out", ".err")) and not p.startswith(("runs/", "sweeps/", "plan/"))
    and "/data/" not in p}


def refusal_verdicts(since, until):
    """The counts behind the right/misfire reading of four checks (D558): the refused Bash calls of sleep_guard (a loop,
    a sleep over 5 s, one short sleep, no sleep outside quotes) and of review_backlog's commit block (the named paths
    against the blocking pillar); the main session's context-guard bounces and compactions before and after CTX_FIX;
    every 'question open' bounce replayed through today's plan.ask_matches_set (the recorded ask, the texts sent in
    the 3 h before)."""
    import datetime as dt, yaml
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import plan
    sleep, commit = defaultdict(int), defaultdict(int)
    files = sorted(glob.glob(PROJ + "/*.jsonl")) + sorted(glob.glob(PROJ + "/*/subagents/agent-*.jsonl"))
    for f in files:
        uses = {}
        for line in open(f, errors="replace"):
            if "tool_use" not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ts = (d.get("timestamp") or "")[:19]
            m = d.get("message") if isinstance(d.get("message"), dict) else {}
            for b in m.get("content") if isinstance(m.get("content"), list) else []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use":
                    uses[b.get("id")] = (b.get("input") or {}).get("command", "")
                if b.get("type") != "tool_result" or not b.get("is_error") or not since <= ts < until:
                    continue
                t, cmd = text_of(b.get("content"))[:400], uses.get(b.get("tool_use_id"), "")
                if "sleep_guard" in t:
                    bare = re.sub(r"'[^']*'|\"(?:\\.|[^\"\\])*\"", "''", cmd)
                    secs = [float(x) for x in re.findall(r"\bsleep\s+(\d+(?:\.\d+)?)", bare)]
                    sleep["loop" if re.search(r"\b(until|while|for)\b.*\bdo\b|\bwatch\s|tail\s+-\w*[fF]", bare)
                          else "over_5s" if secs and max(secs) > 5 else "short" if secs else "quoted_only"] += 1
                g = re.search(r"a commit while the (\S+) pillar", t)
                if g:
                    named = re.search(r"git commit\b.*?\s--\s+([^;&|\n]+)", cmd)
                    ps = [x[len(ROOT):] if x.startswith(ROOT) else x for x in (named.group(1).split() if named else [])
                          if not x.startswith((">", "2>"))]
                    commit["unknown" if not ps or any("$" in x for x in ps) else
                           "names_pillar_file" if any(PILLAR_FILE[g.group(1)](x) for x in ps) else "names_none"] += 1
    # the main session: context-guard bounces and compactions around the fix; question bounces replayed
    d = yaml.safe_load(open(ROOT + "plan/plan.yaml"))
    asks = [(e, x.get("ask") or "") for e in d["entries"] for x in e.get("asks") or []] + \
           [(e, e.get("ask") or "") for e in d["entries"] if e.get("ask")]
    ctx, texts, ask = defaultdict(int), [], defaultdict(int)
    main = [f for f in files if "/subagents/" not in f and os.path.getsize(f) > 1e8]  # the one long main session
    for line in open(main[0], errors="replace") if main else []:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        ts = (m.get("timestamp") or "")[:19]
        c = (m.get("message") or {}).get("content") if isinstance(m.get("message"), dict) else None
        if m.get("type") == "assistant" and isinstance(c, list):
            t = " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
            if t.strip():
                texts.append((ts, t))
        if not since <= ts < until:
            continue
        half = "before" if ts < CTX_FIX else "after"
        if m.get("type") == "system" and m.get("subtype") == "compact_boundary":
            ctx["compactions_" + half] += 1
        if m.get("type") != "user" or not isinstance(c, str) or not c.startswith("Stop hook feedback"):
            continue
        if "CONTEXT GUARD" in c:
            ctx["bounces_" + half] += 1
        for g in re.finditer(r"  (\S+) has a question open for the user, but no message sent to them asks: (.*?)(?=\n  |\n?$)",
                             c, re.S):
            a = g.group(2).strip()
            full = next((x for e, x in asks if e["id"] == g.group(1) and x[:100] == a[:100]), None)
            if not full:
                ask["ask_not_found"] += 1
                continue
            t0 = (dt.datetime.fromisoformat(ts) - dt.timedelta(hours=3)).isoformat()
            e = dict(next(e for e in d["entries"] if e["id"] == g.group(1)), ask=full)
            ask["passes_today" if plan.ask_matches_set(e, [t for tt, t in texts if t0 <= tt <= ts])
                else "refused_today"] += 1
    return {"sleep_guard": dict(sleep), "commit_block": dict(commit), "context_guard": dict(ctx),
            "question_replay": dict(ask)}


PLAIN = [  # (hook, start of check) -> what the check refuses, in plain words, for the figure
    ("plan_check", "CONTEXT GUARD", "stop: record before compaction"),
    ("plan_check", "<entry>: resume its owner", "stop: resume the entry's owner"),
    ("plan_check", "<entry>: steps you leave", "agent stop: steps left uncovered"),
    ("review_backlog", "a commit while", "commit: a review is overdue"),
    ("plan_check", "<entry> has a question open", "stop: open question not sent"),
    ("brief_guard", "this brief lacks", "launch: brief lacks the rules"),
    ("plan_check", "<entry>: no live owner: open events", "stop: events with no owner"),
    ("plan_check", "no waiter is armed", "stop: no event waiter running"),
    ("plan_check", "<entry>: no live owner: ready step", "stop: ready step with no owner"),
    ("sleep_guard", "foreground waiting", "shell: foreground sleep"),
    ("submit_guard", "placement", "submit: site not picked"),
    ("plan_check", "figures handed back", "stop: figures not sent"),
]


def figure_refusals(rep, base, top=10):
    """Tokens spent after the refusals of each of the `top` costliest checks of ours (the auto-mode classifier is
    Claude Code's, left out), one bar per role, upper bound."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import plot_style as ps
    rows = [c for c in rep["checks"] if not c["hook"].startswith("auto-mode")][:top]
    name = lambda c: next((w for h, k, w in PLAIN if c["hook"] == h and c["check"].startswith(k)),
                          c["hook"] + ": " + c["check"][:20])
    roles = [("main session", ps.C.vermillion), ("owners", ps.C.blue), ("other agents", ps.C.green),
             ("mechanical workers", ps.C.orange), ("reviewers", ps.C.purple)]
    roles = [(r, col) for r, col in roles if any(c["roles"].get(r, {}).get("hi") for c in rows)]
    fig, ax = ps.figure(1, 1)
    h = 0.8 / len(roles)
    for j, (r, col) in enumerate(roles):
        ys = [i - 0.4 + h * (j + 0.5) for i in range(len(rows))]
        ax.barh(ys, [c["roles"].get(r, {}).get("hi", 0) / 1e6 for c in rows], height=h, color=col, label=r)
    ax.set_yticks(range(len(rows)), [name(c) for c in rows])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.1, 1e5)  # clear right side for the legend (the longest bar ends near 10^2)
    ax.set_xlabel("tokens after the refusals, 14 days ($10^6$)")
    ps.legend(ax, "lower right")
    ps.save(fig, base)


if __name__ == "__main__":
    main()
