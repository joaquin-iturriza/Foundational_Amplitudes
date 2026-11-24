#!/usr/bin/env python3
"""The lab LLM trial's readout (lab_llm_trial): per family and arm, tasks passed, wall time per task and our tokens
per task; the figure.

    python3 tools/lab_llm_trial.py report        (writes analysis/lab_llm_trial/summary.json and figs/)

Our tokens: a Claude agent's transcript (input + output + cache read + cache write, one count per API message):
the timed Sonnet mechanical-worker for its arm, the Sonnet planner for the +plan arms (whose wall time also adds the
planner's), 0 for a lab model alone. Lab tokens (opencode's step usage) are free to us and reported apart.
Fix runs: the last record of each (task, arm) stands; triage runs: every run is an attempt (two per case and arm).
The earlier trial's Sonnet / Opus-low rows (model_trial.json) carry $ only, no wall time.
"""
import glob, json, os, sys
from collections import defaultdict
from datetime import datetime

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
sys.path.insert(0, ROOT)
import model_trial as mt

OUT = os.path.join(ROOT, "analysis", "lab_llm_trial")
HOME = os.path.expanduser("~")
NAME = {"sonnet_timed": "Sonnet", "deepseek+plan": "DeepSeek+plan", "qwen+plan": "Qwen+plan",
        "deepseek": "DeepSeek", "qwen": "Qwen"}
ORDER = ["sonnet_timed", "deepseek+plan", "qwen+plan", "deepseek", "qwen"]


def transcript(agent):
    """(seconds first to last entry, our tokens) of a subagent transcript."""
    fs = glob.glob(os.path.join(HOME, ".claude/projects/-home-joaquin-work-FA/*/subagents/agent-%s.jsonl" % agent))
    ts, seen, tok = [], set(), 0
    for line in open(fs[0]):
        e = json.loads(line)
        if e.get("timestamp"):
            ts.append(datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")))
        m = e.get("message") or {}
        u = m.get("usage") if isinstance(m, dict) else None
        if u and m.get("id") not in seen:
            seen.add(m.get("id"))
            tok += sum(u.get(k, 0) or 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                   "cache_creation_input_tokens"))
    return (max(ts) - min(ts)).total_seconds(), tok


def summary():
    runs = [json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"))]
    last = {(r["family"], r["task"], r["arm"]): r for r in runs if r["family"] in ("fix", "hardfix")}
    rows = [r for r in runs if r["family"] == "triage"] + list(last.values())
    planner = {t: transcript(a) for t, a in json.load(open(os.path.join(OUT, "planners.json"))).items()}
    checks = {(r["task"], r["arm"]): r for r in runs if r["family"] == "hardcheck"}
    s = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        c = s[r["family"]][r["arm"]]
        u, u2 = r.get("usage") or {}, r.get("usage_rework") or {}
        sec = r["seconds"] + r.get("seconds_rework", 0)
        ours = r.get("our_tokens", 0)
        if r["arm"].endswith("+plan"):
            if r["task"] not in planner:
                sys.exit("lab_llm_report: no planner recorded for %s (planners.json)" % r["task"])
            sec += planner[r["task"]][0]; ours = planner[r["task"]][1]
        if r["family"] == "hardfix" and r["arm"].startswith("deepseek"):   # the Sonnet check of each DeepSeek fix (D541)
            ck = checks.get((r["task"], r["arm"]))
            if ck is None:
                sys.exit("lab_llm_report: no Sonnet check recorded for %s %s (lab_llm_trial.py grade_check)"
                         % (r["task"], r["arm"]))
            ours += ck["our_tokens"]; sec += ck["seconds"]
            c["agree"].append(ck["agree"]); c["agree_matches"].append(ck["matches_grader"])
        c["first"].append(bool(r["first_pass"])); c["final"].append(bool(r.get("final_pass")))
        c["per_task"].append({"task": r["task"], "first": bool(r["first_pass"]), "min": round(sec / 60, 2),
                              "ours": ours, "lab": sum(u.get(k, 0) + u2.get(k, 0)
                                                       for k in ("in", "out", "reasoning", "cache_read"))})
        c["min"].append(sec / 60); c["ours"].append(ours)
        c["lab"].append(sum(u.get(k, 0) + u2.get(k, 0) for k in ("in", "out", "reasoning", "cache_read")))
        c["by_class"].append((r.get("key") or {}).get("class"))
    out = {}
    for f, arms in s.items():
        out[f] = {}
        for a, c in arms.items():
            n = len(c["first"])
            out[f][a] = {"n": n, "passed_first": sum(c["first"]), "passed_final": sum(c["final"]),
                         "minutes_per_task": round(sum(c["min"]) / n, 2),
                         "minutes_median": round(sorted(c["min"])[n // 2], 2),
                         "our_tokens_per_task": round(sum(c["ours"]) / n),
                         "lab_tokens_per_task": round(sum(c["lab"]) / n)}
            if c["agree"]:
                out[f][a]["check_agree"] = sum(c["agree"])
                out[f][a]["check_matches_grader"] = sum(c["agree_matches"])
            if f == "hardfix":
                out[f][a]["per_task"] = c["per_task"]
            if f == "triage":
                out[f][a]["passed_by_class"] = {k: [sum(p for p, q in zip(c["first"], c["by_class"]) if q == k),
                                                    c["by_class"].count(k)] for k in set(c["by_class"])}
    out["lookup"] = lookups(runs)
    hr = os.path.join(OUT, "hand_read.json")
    if os.path.exists(hr):                         # the owner's read of why and resubmit_as_is against the ack note
        for o in json.load(open(hr))["reads"]:
            t = out["triage"][o["arm"]]
            t["hand_ok"] = t.get("hand_ok", 0) + o["ok"]
    ref = json.load(open(os.path.join(ROOT, "analysis", "agent_redesign", "model_trial.json")))["summary"]
    out["fix_earlier_trial"] = ref
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    return out


def lookups(runs):
    """DeepSeek's code lookups (two runs per question) against the recorded Opus-low Explore grep and grep2 arms of
    lookup_trial.json (the design's control), both graded by lookup_trial.correct at e6f0ef58e; the control's tokens
    recounted from its transcripts as every other arm's."""
    ds = [r for r in runs if r["family"] == "lookup"]
    if not ds:
        return {}
    ctl = json.load(open(os.path.join(ROOT, "analysis", "agent_redesign", "lookup_trial.json")))
    tasks = sorted({r["task"] for r in ds})
    out = {"deepseek": {}, "explore_opus_low": {}}
    for rep in sorted({str(r["rep"]) for r in ds}):
        rr = [r for r in ds if str(r["rep"]) == rep]
        out["deepseek"]["run" + rep] = {"n": len(rr), "correct": sum(bool(r["first_pass"]) for r in rr),
                                        "seconds_per_task": round(sum(r["seconds"] for r in rr) / len(rr), 1),
                                        "lab_tokens_per_task": round(sum(sum((r["usage"] or {}).get(k, 0) for k in
                                            ("in", "out", "reasoning", "cache_read")) for r in rr) / len(rr)),
                                        "our_tokens_per_task": 0}
    for arm in ("grep", "grep2"):
        cc = [ctl[t][arm] for t in tasks if arm in ctl.get(t, {})]
        out["explore_opus_low"][arm] = {"n": len(cc), "correct": sum(bool(c["correct"]) for c in cc),
                                        "our_tokens_per_task": round(sum(transcript(c["agent"])[1] for c in cc) / len(cc))}
    return out


def figure_hard(out):
    """The hard round, task by task: our tokens, wall time and lab tokens per fix of each arm, tasks ordered by
    Sonnet's tokens; an open marker where the fix failed the grader or its Sonnet check disagreed."""
    import plot_style as ps
    hf = out["hardfix"]
    arms = [a for a in ("sonnet", "deepseek+plan", "deepseek") if a in hf]
    rows = {a: {p["task"]: p for p in hf[a]["per_task"]} for a in arms}
    tasks = sorted(rows["sonnet"], key=lambda t: rows["sonnet"][t]["ours"])
    runs = [json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"))]
    bad = {(r["task"], r["arm"]) for r in runs if r["family"] == "hardcheck" and not r["agree"]}
    col = {"sonnet": ps.C.blue, "deepseek+plan": ps.C.vermillion, "deepseek": ps.C.green}
    mk = {"sonnet": "o", "deepseek+plan": "s", "deepseek": "^"}
    lbl = {"sonnet": "Sonnet", "deepseek+plan": "DeepSeek+plan", "deepseek": "DeepSeek"}
    fig, axs = ps.figure(ncols=2, nrows=2)
    axs = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]
    x = list(range(len(tasks)))
    for ax, key, ylab, scale in ((axs[0], "ours", "our tokens per fix ($10^3$)", 1e3),
                                 (axs[1], "min", "wall time per fix (min)", 1),
                                 (axs[2], "lab", "lab tokens per fix ($10^3$)", 1e3)):
        for a in arms:
            if key == "lab" and a == "sonnet":
                continue
            ys = [rows[a][t][key] / scale for t in tasks]
            ax.plot(x, ys, mk[a], color=col[a], label=lbl[a], ms=6)
            for i, t in enumerate(tasks):
                if (t, a) in bad or not rows[a][t]["first"]:
                    ax.plot([i], [ys[i]], mk[a], mfc="white", mec=col[a], ms=11)
        ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.set_xticks(x, tasks, rotation=60, ha="right")
    ps.legend_cell(axs[3], axs[0])
    ps.save(fig, os.path.join(OUT, "figs", "lab_llm_hard"))


def figure(out):
    import plot_style as ps
    fx = out["fix"]
    arms = [a for a in ORDER if a in fx]
    x = range(len(arms))
    lab = [NAME[a] for a in arms]
    fig, axs = ps.figure(ncols=2, nrows=2)
    a, b, c, d = axs[0][0], axs[0][1], axs[1][0], axs[1][1]
    a.bar([i - 0.2 for i in x], [fx[k]["passed_first"] / fx[k]["n"] for k in arms], 0.4, color=ps.C.blue,
          label="first attempt")
    a.bar([i + 0.2 for i in x], [fx[k]["passed_final"] / fx[k]["n"] for k in arms], 0.4, color=ps.C.vermillion,
          label="after one rework")
    a.set_ylabel("share of tool fixes passed")
    a.set_ylim(0, 1.0)
    ps.legend(a, "upper right")
    b.bar(x, [fx[k]["minutes_per_task"] for k in arms], 0.6, color=ps.C.blue)
    b.set_yscale("log")
    b.set_ylabel("wall time per tool fix (min)")
    from matplotlib.ticker import ScalarFormatter
    b.yaxis.set_major_formatter(ScalarFormatter()); b.yaxis.set_minor_formatter(ScalarFormatter())
    c.bar(x, [fx[k]["our_tokens_per_task"] / 1e3 for k in arms], 0.6, color=ps.C.blue)
    c.set_ylabel("our tokens per tool fix ($10^3$)")
    for ax in (a, b, c):
        ax.set_xticks(list(x), lab, rotation=30, ha="right")
    tr = out["triage"]
    tarms = [k for k in ("deepseek", "qwen") if k in tr]
    d.bar([i - 0.2 for i in range(len(tarms))], [tr[k]["passed_first"] / tr[k]["n"] for k in tarms], 0.4,
          color=ps.C.blue, label="cause and step or error line")
    d.bar([i + 0.2 for i in range(len(tarms))], [tr[k].get("hand_ok", 0) / tr[k]["n"] for k in tarms], 0.4,
          color=ps.C.vermillion, label="reason and action, read by hand")
    ps.legend(d, "upper left")
    d.set_xticks(list(range(len(tarms))), [NAME[k] for k in tarms])
    d.set_ylabel("share of failed-run logs read right")
    d.set_ylim(0, 1.0)
    ps.save(fig, os.path.join(OUT, "figs", "lab_llm_pass"))


def main():
    out = summary()
    print(json.dumps(out))              # one line: verify_claims reads the last {...} line
    if "--json" in sys.argv:            # the claims rerun: numbers only, figures left as drawn
        return
    figure(out)
    if "hardfix" in out:
        figure_hard(out)


if __name__ == "__main__":
    main()
