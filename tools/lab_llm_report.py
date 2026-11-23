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
NAME = {"sonnet_timed": "Sonnet", "deepseek+plan": "DeepSeek with\nSonnet's plan", "qwen+plan": "Qwen with\nSonnet's plan",
        "deepseek": "DeepSeek alone", "qwen": "Qwen alone"}
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
    last = {(r["task"], r["arm"]): r for r in runs if r["family"] == "fix"}
    rows = [r for r in runs if r["family"] == "triage"] + list(last.values())
    planner = {t: transcript(a) for t, a in json.load(open(os.path.join(OUT, "planners.json"))).items()}
    s = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        c = s[r["family"]][r["arm"]]
        u, u2 = r.get("usage") or {}, r.get("usage_rework") or {}
        sec = r["seconds"] + r.get("seconds_rework", 0)
        ours = r.get("our_tokens", 0)
        if r["arm"].endswith("+plan"):
            sec += planner[r["task"]][0]; ours = planner[r["task"]][1]
        c["first"].append(bool(r["first_pass"])); c["final"].append(bool(r.get("final_pass")))
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
            if f == "triage":
                out[f][a]["passed_by_class"] = {k: [sum(p for p, q in zip(c["first"], c["by_class"]) if q == k),
                                                    c["by_class"].count(k)] for k in set(c["by_class"])}
    hr = os.path.join(OUT, "hand_read.json")
    if os.path.exists(hr):                         # the owner's read of why and resubmit_as_is against the ack note
        for o in json.load(open(hr))["reads"]:
            t = out["triage"][o["arm"]]
            t["hand_ok"] = t.get("hand_ok", 0) + o["ok"]
    ref = json.load(open(os.path.join(ROOT, "analysis", "agent_redesign", "model_trial.json")))["summary"]
    out["fix_earlier_trial"] = ref
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    return out


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
    a.set_ylabel("share of the 10 tool fixes passed")
    a.set_ylim(0, 1.0)
    ps.legend(a, "upper right")
    b.bar(x, [fx[k]["minutes_per_task"] for k in arms], 0.6, color=ps.C.blue)
    b.set_yscale("log")
    b.set_ylabel("wall time per tool fix (min)")
    c.bar(x, [fx[k]["our_tokens_per_task"] / 1e3 for k in arms], 0.6, color=ps.C.blue)
    c.set_ylabel("our tokens per tool fix ($10^3$)")
    for ax in (a, b, c):
        ax.set_xticks(list(x), lab)
    tr = out["triage"]
    tarms = [k for k in ("deepseek", "qwen") if k in tr]
    d.bar([i - 0.2 for i in range(len(tarms))], [tr[k]["passed_first"] / tr[k]["n"] for k in tarms], 0.4,
          color=ps.C.blue, label="cause and step or error line")
    d.bar([i + 0.2 for i in range(len(tarms))], [tr[k].get("hand_ok", 0) / tr[k]["n"] for k in tarms], 0.4,
          color=ps.C.vermillion, label="reason and action, read by hand")
    ps.legend(d, "lower left")
    d.set_xticks(list(range(len(tarms))), [NAME[k] for k in tarms])
    d.set_ylabel("share of failed-run logs read right")
    d.set_ylim(0, 1.0)
    ps.save(fig, os.path.join(OUT, "figs", "lab_llm_pass"))


def main():
    out = summary()
    print(json.dumps(out, indent=1))
    figure(out)


if __name__ == "__main__":
    main()
