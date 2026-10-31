"""What fills an agent's context: per agent transcript, the context at each API call split into sources.

    python3 tools/context_breakdown.py [--out analysis/agent_redesign/context_breakdown.json]
                                       [--fig analysis/agent_redesign/figs/context_breakdown] [--min-calls 5]

Reads ~/.claude/projects/-home-joaquin-work-FA/*/subagents/**/*.jsonl. Periods as in tools/usage_report.py, by the
time of the agent's first call (step_agents until 2026-10-30 11:35:10 UTC, redesign after; main_alone is not an agent
period).  A call is one API message (first record of a message id); its context is the API usage input + cache write +
cache read.  Sources, tokens estimated as chars/4 of what was added before the call:
  fixed       first call's API context minus the brief minus the hook pushes before it: system prompt, tool
              definitions, CLAUDE.md files, agent definition, environment (same for every later call of the agent)
  brief       the first user message (+ the harness reminder that follows it)
  hook        hook additionalContext pushes (REFERENCE sections, notes)
  tool results by tool / command family; own text and tool inputs; SendMessage inputs; hand-back inputs
  user_msgs   later user text (messages from other agents, notifications); reminders = other harness attachments
Per agent: the call whose context is the agent's median, and its last call. Reported: mean over agents.
Check of chars/4: over each agent, sum of API context growth between consecutive calls against chars/4 added.
"""
import argparse, glob, json, os, re, statistics, sys
from collections import defaultdict
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TR = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")
BOUND = datetime.fromisoformat("2026-10-30T11:35:10+00:00").timestamp()
PERIODS = ["step_agents", "redesign"]
TOOLNAMES_OWN = {"SendMessage": "sendmessage", "SubagentHandback": "handback"}


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def txt(c):  # dup-ok: counts thinking blocks too (context), usage_report.text_of does not
    if isinstance(c, str):
        return c
    out = []
    for b in c or []:
        if isinstance(b, dict):
            t = b.get("type")
            if t == "text":
                out.append(b.get("text", ""))
            elif t == "tool_result":
                out.append(txt(b.get("content")))
            elif t == "thinking":
                out.append(b.get("thinking", ""))
    return "\n".join(out)


def family(name, inp):
    if name == "Bash":
        c = (inp.get("command") or "").strip()
        c = re.sub(r"^(cd [^;&]+(&&|;)\s*)+", "", c)
        if re.search(r"tools/plan\.py\s+(show|brief)", c): return "res: plan show/brief"
        if re.search(r"tools/plan\.py", c): return "res: plan other"
        if re.search(r"\bsite\s+(events|runs|status|poll|digest|decisions|items)", c): return "res: site events/runs/status"
        if re.search(r"\bsite\s+(logs|fetch)", c): return "res: site logs"
        if re.search(r"\bsite\b", c): return "res: site other"
        if re.search(r"\bgit\b", c): return "res: git"
        if re.search(r"\b(grep|rg|ugrep)\b", c): return "res: grep (bash)"
        if re.search(r"^(cat|sed|head|tail)\b", c): return "res: file read (bash)"
        if re.search(r"\bsed -n '/\^=== SECTION", c): return "res: REFERENCE section read"
        if re.search(r"pytest|python3? ", c): return "res: python/pytest"
        return "res: bash other"
    if name == "Read": return "res: file read"
    if name in ("Grep", "Glob"): return "res: grep"
    if name in ("Edit", "Write", "NotebookEdit"): return "res: edit/write"
    if name in ("Agent", "SendMessage", "SubagentHandback"): return "res: agent/message"
    if name in ("ToolSearch",): return "res: ToolSearch"
    return "res: other (%s)" % name


def scan(path):
    """Return (t0, calls) ; calls = list of dict(ctx, src={source: chars})."""
    src = defaultdict(int)
    tools = {}
    calls, seen = [], set()
    first_user = True
    global LASTPARTS
    LASTPARTS = defaultdict(int)
    with open(path, errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ty = d.get("type")
            t = d.get("timestamp")
            if ty == "attachment":
                a = d["attachment"]; at = a.get("type")
                if not calls and at not in ("hook_additional_context", "hook_success"):
                    LASTPARTS[at] += len(json.dumps({k: v for k, v in a.items() if k != "type"}))
                if at == "hook_additional_context":
                    src["hook"] += sum(len(x) for x in a.get("content", []))
                elif at in ("total_tokens_reminder", "date", "auto_mode"):
                    src["reminders"] += len(txt(a.get("text", "")))
                continue
            m = d.get("message") or {}
            if ty == "user":
                c = m.get("content")
                if isinstance(c, str) or any(b.get("type") == "text" for b in c if isinstance(b, dict)):
                    n = len(txt(c))
                    src["brief" if (first_user or d.get("isMeta")) and len(calls) == 0 else "user_msgs"] += n
                    first_user = False
                else:
                    for b in c:
                        if b.get("type") == "tool_result":
                            name, inp = tools.get(b.get("tool_use_id"), ("?", {}))
                            src[family(name, inp)] += len(txt(b.get("content")))
            elif ty == "assistant":
                mid = m.get("id")
                u = m.get("usage") or {}
                if mid not in seen:
                    seen.add(mid)
                    ctx = (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                    calls.append({"t": ts(t), "ctx": ctx, "src": dict(src)})
                for b in m.get("content", []):
                    bt = b.get("type")
                    if bt == "text":
                        src["own text/tool inputs"] += len(b.get("text", ""))
                    elif bt == "thinking":
                        src["own text/tool inputs"] += len(b.get("thinking", ""))
                    elif bt == "tool_use":
                        tools[b["id"]] = (b["name"], b.get("input") or {})
                        n = len(json.dumps(b.get("input") or {}))
                        src[TOOLNAMES_OWN.get(b["name"], "own text/tool inputs")] += n
    return calls


SCALE, PERCALL = 1.0, 0.0     # tokens per (chars/4), tokens per call; fitted in main() on API context growth


def split(calls, i):
    """Source breakdown in tokens at call i. fixed = first call's API context minus its (calibrated) content;
    each source = SCALE * chars/4; hidden = PERCALL * calls so far (hidden thinking, harness framing)."""
    c0 = calls[0]
    fixed = max(0.0, c0["ctx"] - SCALE * sum(c0["src"].values()) / 4 - PERCALL)
    out = {"fixed (system, tools, CLAUDE.md, agent def)": fixed, "per-call overhead (hidden thinking)": PERCALL * (i + 1)}
    for k, v in calls[i]["src"].items():
        out[k] = out.get(k, 0) + SCALE * v / 4
    return out, calls[i]["ctx"]


def fit(allcalls):
    import numpy as np
    from scipy.optimize import nnls
    X, y = [], []
    for calls in allcalls:
        for i in range(1, len(calls)):
            dA = calls[i]["ctx"] - calls[i - 1]["ctx"]
            if dA < 0:
                continue
            X.append([1.0, (sum(calls[i]["src"].values()) - sum(calls[i - 1]["src"].values())) / 4]); y.append(dA)
    co, _ = nnls(np.array(X), np.array(y))
    return float(co[1]), float(co[0]), len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/agent_redesign/context_breakdown.json")
    ap.add_argument("--fig", default="analysis/agent_redesign/figs/context_breakdown")
    ap.add_argument("--min-calls", type=int, default=5)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(TR, "*", "subagents", "**", "*.jsonl"), recursive=True))
    global SCALE, PERCALL
    agents = {p: [] for p in PERIODS}
    check = []
    scanned, parts = [], defaultdict(lambda: defaultdict(list))
    for fp in files:
        c = scan(fp)
        if c:
            scanned.append((fp, c))
            per0 = "step_agents" if c[0]["t"] < BOUND else "redesign"
            for k, v in LASTPARTS.items():
                parts[per0][k].append(v / 4)
    SCALE, PERCALL, nfit = fit([c for _, c in scanned])
    for fp, calls in scanned:
        per = "step_agents" if calls[0]["t"] < BOUND else "redesign"
        # chars/4 check: growth of API context vs chars/4 added between consecutive calls
        dA = sum(max(0, calls[i]["ctx"] - calls[i - 1]["ctx"]) for i in range(1, len(calls)))
        dC = sum((sum(calls[i]["src"].values()) - sum(calls[i - 1]["src"].values())) / 4 for i in range(1, len(calls)))
        if len(calls) >= 5:
            check.append({"agent": os.path.basename(fp)[:-6], "period": per, "calls": len(calls),
                          "api_growth_tokens": dA, "chars4_tokens": round(dC), "ratio_chars4_over_api": round(dC / dA, 3) if dA else None})
        if len(calls) < a.min_calls:
            continue
        ctxs = [c["ctx"] for c in calls]
        med = sorted(range(len(calls)), key=lambda i: ctxs[i])[len(calls) // 2]
        m, mc = split(calls, med)
        l, lc = split(calls, len(calls) - 1)
        f0, f0c = split(calls, 0)
        agents[per].append({"id": os.path.basename(fp)[:-6], "calls": len(calls), "median": m, "median_ctx": mc,
                            "last": l, "last_ctx": lc, "first": f0, "first_ctx": f0c})
    rep = {"definition": __doc__, "min_calls": a.min_calls, "periods": {}, "chars4_check": check,
           "calibration": {"tokens_per_chars4": round(SCALE, 3), "tokens_per_call": round(PERCALL), "fit_pairs": nfit,
                           "note": "non-negative least squares of API context growth between consecutive calls on chars/4 added"}}
    # check on three agents: predicted final context from fitted constants vs API
    ok = []
    for fp, calls in scanned:
        if len(calls) >= 30 and len(ok) < 3 and (len(ok) == 0 or calls[0]["t"] > BOUND or len(ok) == 1):
            sp, ctx = split(calls, len(calls) - 1)
            ok.append({"agent": os.path.basename(fp)[:-6], "calls": len(calls), "api_last_context": ctx,
                       "predicted": round(sum(sp.values())), "ratio": round(sum(sp.values()) / ctx, 3),
                       "raw_chars4_sum_without_fixed": round(sum(calls[-1]["src"].values()) / 4)})
    rep["three_agent_check"] = ok
    rep["fixed_parts_median_chars4_tokens"] = {per: {k: round(statistics.median(v)) for k, v in parts[per].items()}
                                               for per in PERIODS}
    # call-weighted: every call of every agent with >= min_calls calls (what the usage report's per-call median is over)
    POOLED = {}
    for per in PERIODS:
        acc, n, ctxs = defaultdict(float), 0, []
        for fp, calls in scanned:
            if len(calls) < a.min_calls or (("step_agents" if calls[0]["t"] < BOUND else "redesign") != per):
                continue
            for i in range(len(calls)):
                sp, ctx = split(calls, i)
                for k, v in sp.items():
                    acc[k] += v
                n += 1; ctxs.append(ctx)
        POOLED[per] = {"calls": n, "median_api_context": round(statistics.median(ctxs)),
                                               "mean_tokens_by_source": {k: round(v / n) for k, v in acc.items()}}
    for per in PERIODS:
        A = agents[per]
        P = {"agents": len(A)}
        for which in ("median", "last", "first"):
            keys = sorted({k for x in A for k in x[which]})
            P[which] = {"mean_tokens_by_source": {k: round(sum(x[which].get(k, 0) for x in A) / max(1, len(A))) for k in keys},
                        "mean_api_context": round(sum(x[which + "_ctx"] for x in A) / max(1, len(A))),
                        "median_api_context": round(statistics.median([x[which + "_ctx"] for x in A])) if A else None}
        rep["periods"][per] = P
        P["pooled_calls"] = POOLED[per]
    out = os.path.join(ROOT, a.out)
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep["fixed_parts_median_chars4_tokens"]))
    for per in PERIODS:
        print(per, "pooled", rep["periods"][per]["pooled_calls"]["calls"], rep["periods"][per]["pooled_calls"]["median_api_context"])
        for k, v in sorted(rep["periods"][per]["pooled_calls"]["mean_tokens_by_source"].items(), key=lambda kv: -kv[1])[:14]:
            print("   %-48s %7d" % (k, v))
        for w in ("first", "median", "last"):
            p = rep["periods"][per]
            print(per, w, "agents", p["agents"], "api ctx mean", p[w]["mean_api_context"], "median", p[w]["median_api_context"])
            for k, v in sorted(p[w]["mean_tokens_by_source"].items(), key=lambda kv: -kv[1]):
                print("   %-48s %7d" % (k, v))
    print("check (chars/4 added vs API growth):")
    for c in check[:6] + check[-3:]:
        print("  ", c)
    rs = [c["ratio_chars4_over_api"] for c in check if c["ratio_chars4_over_api"]]
    print("ratio median", statistics.median(rs), "n", len(rs), "calib", rep["calibration"], ok)
    figure(rep, os.path.join(ROOT, a.fig))


def figure(rep, base):
    sys.path.insert(0, ROOT)
    import plot_style as ps
    import numpy as np
    cols = [(p, w) for p in PERIODS for w in ("first", "median", "last")]
    tot = defaultdict(float)
    for p, w in cols:
        for k, v in rep["periods"][p][w]["mean_tokens_by_source"].items():
            tot[k] += v
    keys = [k for k, _ in sorted(tot.items(), key=lambda kv: -kv[1])]
    top = keys[:9]
    fig, ax = ps.figure()
    bottom = np.zeros(len(cols))
    cm = ps.sequence(len(top) + 1, "tab20") if False else None
    import matplotlib
    colors = matplotlib.colormaps["tab10"].colors
    for i, k in enumerate(top + ["other"]):
        vals = np.array([(sum(v for kk, v in rep["periods"][p][w]["mean_tokens_by_source"].items() if kk not in top)
                          if k == "other" else rep["periods"][p][w]["mean_tokens_by_source"].get(k, 0)) / 1e3 for p, w in cols])
        ax.bar(range(len(cols)), vals, bottom=bottom, color=colors[i % 10] if k != "other" else "0.7", label=k, width=.7)
        bottom += vals
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["step\nfirst", "step\nmedian", "step\nlast", "redes.\nfirst", "redes.\nmedian", "redes.\nlast"])
    ax.set_ylabel("context [k tokens, chars/4 + fixed]")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    ps.save(fig, base)


if __name__ == "__main__":
    main()
