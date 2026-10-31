"""How the agent set-ups compare: tokens, output, wasted calls, stalls and context, per period, main session and agents
apart. Read-only over the session transcripts and the run registry; CPU, a few minutes.

    python3 tools/usage_report.py [--out analysis/agent_redesign/usage_report.json] [--fig analysis/agent_redesign/figs/usage]
                                  [--samples hook|nothing_new|denial|wait|repeat|reread]   (prints 5 cases of a classifier, for a hand check)

Periods (UTC; boundaries from git, see PERIODS):
  main_alone   the main session doing the work itself, reviewers as the only subagents, until 653aed00
               (2026-10-25 17:02 UTC, "setup redesign (D121): the plan file and its CLI, ... the switch-over": plan
               entries and the step prompt, one step agent per entry)
  step_agents  the previous owner set-up: step agents per plan entry, until 84a15232 (2026-10-30 11:35:10 UTC,
               "settings: register the hooks of the agent redesign's switch-overs A, B and D")
  redesign     switch-overs A to D live (owners per entry, single Stop gate, guards, supervisor run-end sorting)

What is counted (a "call" is one tool_use block; a "turn" starts at a user record carrying text, not a tool result):
  tokens       per API message (deduplicated on message id over all transcripts read), weighted
               input 1, output 5, cache write 1.25, cache read 0.1
  usd          the same messages at list price per model (tools/usage_windows.py PRICES)
  context      input + cache write + cache read of each API message
  runs         registry runs whose id first appears in a tool result as `run FA-<site>-<hex> (` (site submit's line)
  results      `tools/plan.py evidence|finding` calls and `git commit` calls naming docs/results.tex, not errored
  commits      `git commit` calls whose result is no error and not "nothing to commit"
  hook         tool results that say a hook blocked the call ("hook error" / "BLOCKED by")
  denial       permission denials (user rejection, the auto-mode classifier, a built-in safety check)
  repeat       a call identical (tool and input) to an earlier one in the same turn
  wait         sleep, until/while loops, site poll, Monitor, TaskOutput/BashOutput (the waiter `site events --wait`
               is not counted: it is the sanctioned watch)
  reread       a Read of the same file, offset and limit already read in the same context (a transcript; a compaction
               starts a new context)
  nothing_new  main-session turns started by a notification (task-notification, agent message, system notification)
               whose calls are none or only the waiter's re-arm; a share of the main session's turns
  stall        a registry run's end to the first later call (any transcript) that names the run id; median and p90,
               runs with none within 72 h counted apart
"""
import argparse, glob, json, os, re, sqlite3, statistics, sys
from collections import defaultdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPTS = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")
RUNS_DB = os.path.expanduser("~/.local/share/ccorch/runs.db")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import usage_windows as uw     # list prices per model (cost per call)
SEEN_ALL = set()               # message ids over all transcripts: a subagent's call also logged in its parent counts once
W = {"input_tokens": 1.0, "output_tokens": 5.0, "cache_creation_input_tokens": 1.25, "cache_read_input_tokens": 0.1}


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


PERIODS = [("main_alone", None, ts("2026-10-25T17:02:21Z")),
           ("step_agents", ts("2026-10-25T17:02:21Z"), ts("2026-10-30T11:35:10Z")),
           ("redesign", ts("2026-10-30T11:35:10Z"), None)]
PERIOD_EVIDENCE = {"main_alone|step_agents": "653aed00 2026-10-25T18:02:21+01:00 setup redesign (D121): the plan file "
                   "and its CLI, ... the switch-over",
                   "step_agents|redesign": "84a15232 2026-10-30T12:35:10+01:00 settings: register the hooks of the agent "
                   "redesign's switch-overs A, B and D"}


def period_of(t):
    for name, lo, hi in PERIODS:
        if (lo is None or t >= lo) and (hi is None or t < hi):
            return name
    return None


HOOK = re.compile(r"hook error|BLOCKED by|PreToolUse:\S+ hook (?:error|blocked)", re.I)
DENIAL = re.compile(r"doesn't want to proceed with this tool use|Permission for this (?:action|tool use|command) was "
                    r"denied|denied by the Claude Code auto mode classifier|denied by a built-in Claude Code safety")
WAIT = re.compile(r"(^|[;&|(\s])sleep\s+\d|\buntil\b.*;\s*do\b|\bwhile\b.*;\s*do\b|\bsite poll\b")
WAITER = re.compile(r"site events\b.*--wait")
WAKE = re.compile(r"<task-notification>|\[SYSTEM NOTIFICATION|<agent-message|<cross-session-message|"
                  r"background command .* completed|sent a message while you were working", re.I)
RUN_LINE = re.compile(r"\brun (FA-[a-z0-9]+-[0-9a-f]{8})\s+\(")
RUN_ID = re.compile(r"\bFA-[a-z0-9]+-[0-9a-f]{8}\b")


def text_of(content):
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict):
            if b.get("type") == "text":
                out.append(b.get("text", ""))
            elif b.get("type") == "tool_result":
                out.append(text_of(b.get("content")))
    return "\n".join(out)


def new_stats():
    return {"tokens": 0.0, "usd": 0.0, "usd_by_model": defaultdict(float), "tokens_raw": defaultdict(int), "messages": 0, "calls": 0, "runs": set(), "results": 0,
            "commits": 0, "waste": defaultdict(int), "turns": 0, "wake_turns": 0, "contexts": [], "transcripts": set()}


def scan(path, role, S, samples, actions):
    """One transcript. S[period][role] accumulates; actions[run id] collects (time, role) of calls naming it."""
    seen_msg = set()
    calls = {}                       # tool_use id -> (period, name, input, cmd)
    read_seen = set()
    turn_calls, turn = set(), None   # turn: dict(period, wake, calls[])

    def close_turn():
        if turn and role == "main" and turn["wake"]:
            cs = turn["calls"]
            if all(WAITER.search(c) for c in cs):
                S[turn["period"]][role]["waste"]["nothing_new"] += 1
                samples["nothing_new"].append((path, turn["t"], turn["text"][:200], cs[:3]))

    with open(path, errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = d.get("timestamp")
            if not t:
                continue
            t = ts(t)
            per = period_of(t)
            st = S[per][role]
            st["transcripts"].add(path)
            if d.get("type") == "system" and d.get("subtype") == "compact_boundary":
                read_seen = set()
                continue
            msg = d.get("message") or {}
            if d.get("type") == "user":
                content = msg.get("content")
                is_result = isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result"
                                                              for b in content)
                if not is_result and not d.get("isMeta"):
                    txt = text_of(content)
                    if txt.strip():
                        close_turn()
                        wake = bool(WAKE.search(txt))
                        turn = {"period": per, "wake": wake, "calls": [], "t": d.get("timestamp"), "text": txt}
                        turn_calls = set()
                        st["turns"] += 1
                        st["wake_turns"] += wake
                if is_result:
                    for b in content:
                        if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                            continue
                        c = calls.get(b.get("tool_use_id"))
                        if not c:
                            continue
                        cper, name, cmd = c
                        cs = S[cper][role]
                        out = text_of(b.get("content"))
                        err = bool(b.get("is_error"))
                        if err and HOOK.search(out[:3000]):
                            cs["waste"]["hook"] += 1
                            samples["hook"].append((path, d.get("timestamp"), cmd[:160], out[:240]))
                        elif DENIAL.search(out[:3000]):
                            cs["waste"]["denial"] += 1
                            samples["denial"].append((path, d.get("timestamp"), cmd[:160], out[:200]))
                        for rid in RUN_LINE.findall(out):
                            cs["runs"].add(rid)
                        if name == "Bash" and not err:
                            if re.search(r"\bgit commit\b", cmd) and "nothing to commit" not in out \
                                    and "nothing added" not in out:
                                cs["commits"] += 1
                                if "docs/results.tex" in cmd:
                                    cs["results"] += 1
                            if re.search(r"tools/plan\.py (evidence|finding)\b", cmd):
                                cs["results"] += 1
                continue
            if d.get("type") != "assistant":
                continue
            mid = msg.get("id")
            u = msg.get("usage") or {}
            if mid and mid not in seen_msg and mid not in SEEN_ALL and u:
                seen_msg.add(mid)
                SEEN_ALL.add(mid)
                cc = u.get("cache_creation") or {}
                c5, c1 = cc.get("ephemeral_5m_input_tokens") or 0, cc.get("ephemeral_1h_input_tokens") or 0
                usd = uw.cost({"model": msg.get("model"), "in": u.get("input_tokens") or 0,
                               "out": u.get("output_tokens") or 0, "cw5": c5, "cw1h": c1,
                               "cwu": max((u.get("cache_creation_input_tokens") or 0) - c5 - c1, 0),
                               "cr": u.get("cache_read_input_tokens") or 0}) or 0.0
                st["usd"] += usd
                st["usd_by_model"][msg.get("model")] += usd
                st["messages"] += 1
                for k, w in W.items():
                    st["tokens"] += w * (u.get(k) or 0)
                    st["tokens_raw"][k] += u.get(k) or 0
                st["contexts"].append((u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
                                      + (u.get("cache_read_input_tokens") or 0))
            for b in msg.get("content") or []:
                if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                    continue
                name, inp = b.get("name"), b.get("input") or {}
                cmd = inp.get("command") or inp.get("file_path") or inp.get("prompt") or json.dumps(inp)[:400]
                cmd = str(cmd)
                calls[b.get("id")] = (per, name, cmd)
                st["calls"] += 1
                key = name + json.dumps(inp, sort_keys=True)
                if turn is not None:
                    turn["calls"].append(cmd if name == "Bash" else name)
                    if key in turn_calls:
                        st["waste"]["repeat"] += 1
                        samples["repeat"].append((path, d.get("timestamp"), name, cmd[:160]))
                    turn_calls.add(key)
                if name in ("Monitor", "TaskOutput", "BashOutput") or (name == "Bash" and WAIT.search(cmd)
                                                                       and not WAITER.search(cmd)):
                    st["waste"]["wait"] += 1
                    samples["wait"].append((path, d.get("timestamp"), name, cmd[:160]))
                if name == "Read":
                    rk = (inp.get("file_path"), inp.get("offset"), inp.get("limit"), inp.get("pages"))
                    if rk in read_seen:
                        st["waste"]["reread"] += 1
                        samples["reread"].append((path, d.get("timestamp"), name, str(rk)[:160]))
                    read_seen.add(rk)
                for rid in set(RUN_ID.findall(json.dumps(inp))):
                    actions[rid].append((t, role))
    close_turn()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/agent_redesign/usage_report.json")
    ap.add_argument("--fig", default="analysis/agent_redesign/figs/usage")
    ap.add_argument("--all-fa-folders", action="store_true",
                    help="also read the FA worktree and replay folders (-home-joaquin-work-FA-*), not only -home-joaquin-work-FA")
    ap.add_argument("--samples", help="print 5 cases, spread over time, of each listed classifier (comma list: (hook, nothing_new, denial, wait, repeat, reread)")
    a = ap.parse_args()
    S = defaultdict(lambda: defaultdict(new_stats))
    samples, actions = defaultdict(list), defaultdict(list)
    roots = [TRANSCRIPTS] + (sorted(glob.glob(TRANSCRIPTS + "-*")) if a.all_fa_folders else [])
    mains = sorted(p for r in roots for p in glob.glob(os.path.join(r, "*.jsonl")))
    agents = sorted(p for r in roots for p in glob.glob(os.path.join(r, "*", "subagents", "**", "*.jsonl"), recursive=True))
    for p in mains:
        scan(p, "main", S, samples, actions)
    for p in agents:
        scan(p, "agents", S, samples, actions)
    for k in (a.samples or "").split(","):
        if k:
            xs = samples[k]
            step = max(1, len(xs) // 5)
            for x in xs[::step][:5]:
                print(k, json.dumps(x, default=str)[:600])
            print(k, "total", len(xs))
    # stalls: run end -> first later call naming the run
    c = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
    ends = c.execute("SELECT id, finished, submitted FROM runs WHERE project='FA' AND finished IS NOT NULL").fetchall()
    sub = c.execute("SELECT submitted FROM runs WHERE project='FA' AND submitted IS NOT NULL").fetchall()
    stall, unseen = defaultdict(lambda: defaultdict(list)), defaultdict(int)
    for rid, fin, _ in ends:
        nxt = sorted(x for x in actions.get(rid, []) if fin <= x[0] <= fin + 72 * 3600)
        per = period_of(fin)
        if nxt:
            stall[per][nxt[0][1]].append((nxt[0][0] - fin) / 60.0)
        else:
            unseen[per] += 1
    reg_sub = defaultdict(int)
    for (s,) in sub:
        reg_sub[period_of(s)] += 1
    report = {"periods": {n: {"from": lo and datetime.fromtimestamp(lo, timezone.utc).isoformat(),
                              "to": hi and datetime.fromtimestamp(hi, timezone.utc).isoformat()} for n, lo, hi in PERIODS},
              "boundary_evidence": PERIOD_EVIDENCE, "weights": W, "by_period": {}}
    for n, _, _ in PERIODS:
        P = {}
        for role in ("main", "agents"):
            st = S[n][role]
            calls = st["calls"] or 1
            waste = {k: {"count": st["waste"][k], "share": round(st["waste"][k] / calls, 4)}
                     for k in ("hook", "denial", "repeat", "wait", "reread")}
            if role == "main":
                waste["nothing_new"] = {"count": st["waste"]["nothing_new"], "turns": st["turns"],
                                        "wake_turns": st["wake_turns"],
                                        "share_of_turns": round(st["waste"]["nothing_new"] / max(1, st["turns"]), 4)}
            wasted = sum(st["waste"][k] for k in ("hook", "denial", "repeat", "wait", "reread"))
            runs, res, com = len(st["runs"]), st["results"], st["commits"]
            P[role] = {"transcripts": len(st["transcripts"]), "api_messages": st["messages"], "calls": st["calls"],
                       "weighted_tokens_M": round(st["tokens"] / 1e6, 2),
                       "usd_list_price": round(st["usd"], 2),
                       "usd_by_model": {m: round(v, 2) for m, v in sorted(st["usd_by_model"].items(), key=lambda x: -x[1])},
                       "usd_per_result": res and round(st["usd"] / res, 2),
                       "usd_per_commit": com and round(st["usd"] / com, 2),
                       "raw_tokens_M": {k: round(v / 1e6, 2) for k, v in st["tokens_raw"].items()},
                       "runs_submitted": runs, "results": res, "commits": com,
                       "weighted_tokens_M_per_run": runs and round(st["tokens"] / 1e6 / runs, 3),
                       "weighted_tokens_M_per_result": res and round(st["tokens"] / 1e6 / res, 3),
                       "weighted_tokens_M_per_commit": com and round(st["tokens"] / 1e6 / com, 3),
                       "wasted_calls": waste, "wasted_share": round(wasted / calls, 4),
                       "context_median_k": round(statistics.median(st["contexts"]) / 1e3, 1) if st["contexts"] else None,
                       "context_max_k": round(max(st["contexts"]) / 1e3, 1) if st["contexts"] else None,
                       "stall_min": {"n": len(stall[n][role]), "median": stall[n][role] and round(pct(stall[n][role], .5), 1),
                                     "p90": stall[n][role] and round(pct(stall[n][role], .9), 1)}}
        tot_tok = (S[n]["main"]["tokens"] + S[n]["agents"]["tokens"]) / 1e6
        tot_res = S[n]["main"]["results"] + S[n]["agents"]["results"]
        allst = stall[n]["main"] + stall[n]["agents"]
        tot_usd = S[n]["main"]["usd"] + S[n]["agents"]["usd"]
        P["total"] = {"weighted_tokens_M": round(tot_tok, 2), "usd_list_price": round(tot_usd, 2),
                      "usd_per_result": tot_res and round(tot_usd / tot_res, 2),
                      "weighted_tokens_M_per_result": tot_res and round(tot_tok / tot_res, 3),
                      "wasted_share": round(sum(S[n][r]["waste"][k] for r in ("main", "agents")
                                                for k in ("hook", "denial", "repeat", "wait", "reread"))
                                            / max(1, S[n]["main"]["calls"] + S[n]["agents"]["calls"]), 4),
                      "registry_runs_submitted": reg_sub[n],
                      "stall_min": {"n": len(allst), "median": allst and round(pct(allst, .5), 1),
                                    "p90": allst and round(pct(allst, .9), 1), "runs_with_no_action_72h": unseen[n]}}
        report["by_period"][n] = P
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps(report["by_period"], indent=1))
    figure(report, os.path.join(ROOT, a.fig))


def figure(report, base):
    sys.path.insert(0, ROOT)
    import plot_style as ps
    import numpy as np
    names = [n for n, _, _ in PERIODS]
    x = np.arange(len(names))
    fig, (axL, axR) = ps.figure(ncols=2)
    for i, (role, col) in enumerate((("main", ps.C.blue), ("agents", ps.C.vermillion))):
        axL.bar(x + (i - .5) * .38, [100 * report["by_period"][n][role]["wasted_share"] for n in names], .38,
                color=col, label=role)
        axR.bar(x + (i - .5) * .38, [report["by_period"][n][role]["weighted_tokens_M_per_result"] or 0 for n in names],
                .38, color=col, label=role)
    for ax in (axL, axR):
        ax.set_xticks(x)
        ax.set_xticklabels(["main\nalone", "step\nagents", "redesign"])
    axL.set_ylabel("wasted calls [% of calls]")
    axR.set_ylabel("weighted tokens per result [M]")
    ps.legend(axL)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    ps.save(fig, base)


if __name__ == "__main__":
    main()
