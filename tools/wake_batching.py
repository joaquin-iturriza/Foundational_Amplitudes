"""What batching the wakes of one Claude Code session would change: every main-session wake and every owner (subagent)
resume, with its time, cause and tokens, and a replay that merges the wakes not started by the user within a window.

    python3 tools/wake_batching.py [--session 71602535-...] [--windows 0 5 10 20 30 60] [--since ISO] [--until ISO]
                                   [--out analysis/agent_redesign/wake_batching.json]
                                   [--fig analysis/agent_redesign/figs/wake_batching]

Read-only over ~/.claude/projects/-home-joaquin-work-FA/<session>.jsonl and <session>/subagents/; CPU, about a minute.

Tokens of an API call: input + cache read + cache write + output (raw, each once per message id); the weighted sum of
tools/usage_report.py (W: 1, 0.1, 1.25, 5) is reported beside it.

Main session. A wake starts at a user record that is not a tool result and carries text (tools/main_turns.py
trigger_of); its tokens are those of every API call until the next wake. Stop-hook refusals, compaction summaries,
harness nudges, interrupts and local-command output continue the turn they follow and are folded into it. Causes: user (Joaquin's message), site_event
(the event waiter `site events --wait`, the supervisor, a detach or other background command ending), handback (an
agent's SubagentHandback or peer message), task_notification (the "agent finished" notice of a background subagent),
other (goal check-in).
Owners. Every subagent transcript of the session; a resume is a user record (not a tool result, not a system reminder,
not skill text) that reaches the agent after its run ended (its last assistant message ended the turn or called
SubagentHandback). The launch prompt is not a resume. Causes: user (a main-session message carrying Joaquin's words:
"Joaquin", "The user ..."), site_event (a main-session message naming an event E<n> or ended jobs or runs),
main_message (any other main-session message), handback (another agent's message, e.g. a reviewer's verdict),
task_notification (a background command of the agent's own ended). Each agent is its own stream.

Replay of a window w (minutes), per stream: the first non-user wake opens a batch that closes w minutes later; every
non-user wake up to the close joins it; a user wake inside the window takes the batch into its own turn (the batch
closes then). A batch is one wake. Its tokens, two bounds:
  first_call_saved  the members' tokens minus the first API call of every member after the first (the call that
                    reads the context to take in the notice): the work of each member is still done (fewer savings);
  max_member        the tokens of the largest member: the other members' work is absorbed (routine acks, re-arms;
                    more savings).
Delay: from each member's arrival to its batch's close; reported as mean and max (the window is the worst case).
The cache of these sessions is written with a 1 h lifetime (ephemeral_1h), so for windows up to 60 min the cache
stays warm in raw tokens; weighted tokens would pay a cache write instead of a read when a batch closes more than
1 h after the stream's previous call (counted: n_cold).
"""
import argparse, glob, json, os, re, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_turns import trigger_of
from usage_report import ts, W, text_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")
KEYS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")
CONT = {"stop_hook", "compaction", "nudge"}
MAIN_CAUSE = {"user": "user", "waiter": "site_event", "supervisor": "site_event", "bg_other": "site_event",
              "handback": "handback", "agent_done": "task_notification", "goal_checkin": "other"}
MAIN_CONT = ("[Request interrupted", "<local-command-stdout>", "<command-name>")   # part of the user's own turn
AGENT_SKIP = ("<system-reminder>", "Base directory for this skill", "[handback-send-enforce]",
              "[Your previous response had no visible output", "This session is being continued",
              "[Request interrupted")                                             # continuations, not resumes
USER_WORDS = re.compile(r"\bJoaquin\b|\b[Tt]he user (approved|said|asked|answered|decided|wrote)\b|\buser.s words\b")
EVENT_WORDS = re.compile(r"\bE\d{2,}\b|\bjobs? (have |has )?ended\b|\bRUN_DONE\b|\bruns? (have |has )?ended\b|\bevent\b")


def call_of(u):
    raw = sum(u.get(k) or 0 for k in KEYS)
    wt = sum(w * (u.get(k) or 0) for k, w in W.items())
    return {"raw": raw, "wt": wt, "cr": u.get("cache_read_input_tokens") or 0}


def is_result(c):
    return isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c)


def new_wake(t, cause, text):
    return {"t": ts(t), "iso": t, "cause": cause, "text": text[:120], "raw": 0, "wt": 0.0, "calls": 0,
            "first": None, "last_t": ts(t)}


def add_call(w, d, seen):
    m = d.get("message") or {}
    u, mid = m.get("usage"), m.get("id")
    if not u or not mid or mid in seen:
        return
    seen.add(mid)
    c = call_of(u)
    w["raw"] += c["raw"]
    w["wt"] += c["wt"]
    w["calls"] += 1
    w["last_t"] = ts(d["timestamp"])
    if w["first"] is None:
        w["first"] = c


def scan_main(path):
    wakes, seen, bg, cur = [], set(), {}, None
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        typ, t = d.get("type"), d.get("timestamp")
        if typ == "user" and t:
            c = (d.get("message") or {}).get("content")
            if is_result(c):
                continue
            s = text_of(c).lstrip()
            if not s or (d.get("isMeta") and (s.startswith("Base directory for this skill") or s.startswith("[Image")
                                              or s.startswith("<local-command"))):
                continue
            trig = trigger_of(s, d.get("origin"), bg)
            if (trig in CONT or s.startswith(MAIN_CONT)) and cur is not None:
                continue
            cur = new_wake(t, MAIN_CAUSE.get(trig, "other"), s)
            wakes.append(cur)
        elif typ == "assistant" and cur is not None:
            add_call(cur, d, seen)
            for b in (d.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Bash" and \
                        (b.get("input") or {}).get("run_in_background"):
                    bg[b.get("id")] = str(b["input"].get("command") or "")
    return wakes


def owner_cause(d, s):
    o = (d.get("origin") or {}).get("kind")
    if o == "coordinator":
        body = s.split(":", 1)[-1]
        return "user" if USER_WORDS.search(body) else "site_event" if EVENT_WORDS.search(body) else "main_message"
    if o == "peer" or "<agent-message" in s[:300]:
        return "handback"
    if o == "task-notification" or "<task-notification>" in s[:400] or s.startswith("[SYSTEM NOTIFICATION"):
        return "task_notification"
    return "other"


def scan_agent(path):
    wakes, seen, cur, ended, first = [], set(), None, False, True
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        typ, t = d.get("type"), d.get("timestamp")
        if typ == "assistant":
            m = d.get("message") or {}
            ended = m.get("stop_reason") == "end_turn" or any(
                isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "SubagentHandback"
                for b in m.get("content") or [])
            if cur is not None:
                add_call(cur, d, seen)
        elif typ == "user" and t:
            c = (d.get("message") or {}).get("content")
            if is_result(c):
                ended = False
                continue
            s = text_of(c).lstrip()
            if first:
                first = False
                cur = None            # the launch: its calls are not a wake
                continue
            if not s or s.startswith(AGENT_SKIP):
                continue
            if ended:
                cur = new_wake(t, owner_cause(d, s), s)
                wakes.append(cur)
                ended = False
    return wakes


def replay(stream, win_min):
    """Batches of one stream at a window; returns (n_wakes, raw_lo, raw_hi, wt_lo, wt_hi, delays, n_cold)."""
    win = win_min * 60.0
    out = {"wakes": 0, "raw_first_call_saved": 0, "raw_max_member": 0, "wt_first_call_saved": 0.0,
           "wt_max_member": 0.0, "delays": [], "n_cold": 0}
    prev_last = None

    def close(batch, at, host=None):
        nonlocal prev_last
        members = batch + ([host] if host else [])
        saved = [m["first"] for m in members[1:] if m["first"]]
        raw = sum(m["raw"] for m in members)
        wt = sum(m["wt"] for m in members)
        out["raw_first_call_saved"] += raw - sum(c["raw"] for c in saved)
        out["wt_first_call_saved"] += wt - sum(c["wt"] for c in saved)
        out["raw_max_member"] += max(m["raw"] for m in members)
        out["wt_max_member"] += max(m["wt"] for m in members)
        out["wakes"] += 1
        out["delays"] += [at - m["t"] for m in batch]
        if host is None and batch[0]["first"] and prev_last is not None and at - prev_last > 3600 >= batch[0]["t"] - prev_last:
            out["n_cold"] += 1
        prev_last = max(m["last_t"] for m in members)

    batch = []
    for w in stream:
        if batch and w["t"] > batch[0]["t"] + win:
            close(batch, batch[0]["t"] + win)
            batch = []
        if w["cause"] == "user":
            if batch:
                close(batch, w["t"], host=w)
                batch = []
            else:
                _single(out, w)
                prev_last = w["last_t"]
            continue
        batch.append(w)
    if batch:
        close(batch, batch[0]["t"] + win)
    return out


def _single(out, w):
    out["wakes"] += 1
    for k in ("first_call_saved", "max_member"):
        out["raw_" + k] += w["raw"]
        out["wt_" + k] += w["wt"]


def summary(streams, wins):
    allw = [w for s in streams for w in s]
    rec = {"wakes": len(allw), "raw": sum(w["raw"] for w in allw), "wt": sum(w["wt"] for w in allw),
           "by_cause": {}}
    for w in allw:
        c = rec["by_cause"].setdefault(w["cause"], {"wakes": 0, "raw": 0, "wt": 0.0})
        c["wakes"] += 1
        c["raw"] += w["raw"]
        c["wt"] += w["wt"]
    rec["windows"] = {}
    for win in wins:
        agg = {"wakes": 0, "raw_first_call_saved": 0, "raw_max_member": 0, "wt_first_call_saved": 0.0,
               "wt_max_member": 0.0, "delays": [], "n_cold": 0}
        for s in streams:
            r = replay(s, win)
            for k in agg:
                agg[k] += r[k]
        dl = agg.pop("delays")
        agg["delay_mean_min"] = round(statistics.mean(dl) / 60, 2) if dl else 0.0
        agg["delay_max_min"] = round(max(dl) / 60, 2) if dl else 0.0
        agg["batched_members"] = len(dl)
        rec["windows"][str(win)] = agg
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="71602535-0955-42ad-a58f-23dc320798dc")
    ap.add_argument("--windows", type=float, nargs="+", default=[0, 5, 10, 20, 30, 60])
    ap.add_argument("--since", help="ISO time: wakes before it are left out (default: the whole session)")
    ap.add_argument("--until", help="ISO time: wakes at or after it are left out")
    ap.add_argument("--out", default="analysis/agent_redesign/wake_batching.json")
    ap.add_argument("--fig", default="analysis/agent_redesign/figs/wake_batching")
    ap.add_argument("--list", help="also write every wake (time, stream, cause, tokens) to this JSON file")
    a = ap.parse_args()
    lo = ts(a.since) if a.since else float("-inf")
    hi = ts(a.until) if a.until else float("inf")
    keep = lambda ws: [w for w in ws if lo <= w["t"] < hi]
    main_w = keep(scan_main(os.path.join(FOLDER, a.session + ".jsonl")))
    agents = {}
    for p in sorted(glob.glob(os.path.join(FOLDER, a.session, "subagents", "agent-*.jsonl"))):
        ws = keep(scan_agent(p))
        if ws:
            agents[os.path.basename(p)[6:-6]] = ws
    wins = [int(w) if w == int(w) else w for w in a.windows]
    tspan = [w["t"] for w in main_w] + [w["t"] for ws in agents.values() for w in ws]
    days = (max(tspan) - min(tspan)) / 86400 if tspan else 0
    rep = {"session": a.session, "since": a.since, "until": a.until,
           "first": min(tspan) if tspan else None, "last": max(tspan) if tspan else None, "days": round(days, 2),
           "tokens": "raw = input + cache read + cache write + output; wt = usage_report W",
           "main": summary([main_w], wins), "owners": summary(list(agents.values()), wins),
           "owners_resumed": len(agents)}
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1)
    if a.list:
        rows = [dict(stream="main", **{k: w[k] for k in ("iso", "cause", "raw", "wt", "calls", "text")}) for w in main_w]
        rows += [dict(stream=ag, **{k: w[k] for k in ("iso", "cause", "raw", "wt", "calls", "text")})
                 for ag, ws in agents.items() for w in ws]
        json.dump(sorted(rows, key=lambda r: r["iso"]), open(a.list, "w"), indent=0)
    figure(rep, wins, os.path.join(ROOT, a.fig))
    for who in ("main", "owners"):
        r = rep[who]
        print(f"{who}: {r['wakes']} wakes, {r['raw'] / 1e6:.1f}M tokens, by cause " +
              ", ".join(f"{c} {v['wakes']} / {v['raw'] / 1e6:.1f}M" for c, v in sorted(r["by_cause"].items())))
        for win, v in r["windows"].items():
            print(f"  {win:>4} min: {v['wakes']} wakes, {v['raw_first_call_saved'] / 1e6:.1f}M (first call saved) "
                  f"{v['raw_max_member'] / 1e6:.1f}M (max member), delay mean {v['delay_mean_min']} max "
                  f"{v['delay_max_min']} min, cold {v['n_cold']}")
    print(json.dumps({k: rep[k] for k in ("days", "owners_resumed")} |
                     {who: {w: [rep[who]["windows"][w]["wakes"], round(rep[who]["windows"][w]["raw_first_call_saved"] / 1e6, 2),
                                round(rep[who]["windows"][w]["raw_max_member"] / 1e6, 2)] for w in rep[who]["windows"]}
                      for who in ("main", "owners")}))


def figure(rep, wins, base):
    """Tokens (raw, M) against window, main session and owners, each as a band between the two bounds, with the
    recorded total as the point at window 'none'. Second panel: wakes against window."""
    sys.path.insert(0, ROOT)
    import plot_style as ps
    ps.use()
    fig, (axT, axN) = ps.figure(ncols=2)
    x = [float(w) for w in wins]
    for who, col, lab in (("main", ps.C.blue, "main session"), ("owners", ps.C.vermillion, "owners")):
        r = rep[who]
        lo = [r["windows"][str(w)]["raw_max_member"] / 1e6 for w in wins]
        hi = [r["windows"][str(w)]["raw_first_call_saved"] / 1e6 for w in wins]
        axT.fill_between(x, lo, hi, color=col, alpha=.25, lw=0)
        axT.plot(x, hi, "o-", color=col, label=lab + ", work kept")
        axT.plot(x, lo, "s--", color=col, label=lab + ", largest only")
        axN.plot(x, [r["windows"][str(w)]["wakes"] for w in wins], "o-", color=col, label=lab)
        axT.axhline(r["raw"] / 1e6, color=col, lw=.8, ls=":")
        axN.axhline(r["wakes"], color=col, lw=.8, ls=":")
    axT.set_xlabel("batch window [min]")
    axN.set_xlabel("batch window [min]")
    axT.set_ylabel("tokens [millions]")
    axN.set_ylabel("wakes")
    axT.set_ylim(bottom=0)
    axN.set_ylim(bottom=0)
    ps.legend(axT, "lower left")
    ps.legend(axN, "lower left")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    ps.save(fig, base)


if __name__ == "__main__":
    main()
