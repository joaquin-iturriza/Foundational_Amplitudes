#!/usr/bin/env python3
"""The lab LLM trial's readout (lab_llm_trial): per family and arm, tasks passed, lab tokens, wall time and our $,
with the Sonnet / Opus-low reference of the same fix tasks (analysis/agent_redesign/model_trial.json); the figure.

    python3 tools/lab_llm_trial.py report        (writes analysis/lab_llm_trial/summary.json and figs/)

Our $ of a lab arm is the Sonnet planner's transcript at list price (model_trial.cost) for +plan arms, 0 for a lab
model alone (lab tokens cost us nothing); the reference's $ is its agents' transcripts, rework included.
Triage arms ran twice: each run is a task attempt; a case counts as passed in a run when that run's answer passed.
"""
import json, os, sys
from collections import defaultdict

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
sys.path.insert(0, ROOT)
import model_trial as mt

OUT = os.path.join(ROOT, "analysis", "lab_llm_trial")
PLANNERS = os.path.join(OUT, "planners.json")       # {task: planner agent id}
NAME = {"sonnet": "Sonnet", "opuslow": "Opus, low effort", "deepseek": "DeepSeek alone", "qwen": "Qwen alone",
        "deepseek+plan": "DeepSeek, Sonnet's plan", "qwen+plan": "Qwen, Sonnet's plan"}


def load():
    runs = [json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"))]
    last = {}
    for r in runs:                                   # a fix run repeated after a harness fault: the last one stands
        if r["family"] == "fix":
            last[(r["task"], r["arm"])] = r
    return [r for r in runs if r["family"] == "triage"] + list(last.values())


def summary():
    rows = load()
    plan_usd = {}
    if os.path.exists(PLANNERS):
        for t, a in json.load(open(PLANNERS)).items():
            plan_usd[t] = mt.cost(a)["usd"]
    s = defaultdict(lambda: defaultdict(lambda: {"n": 0, "first": 0, "final": 0, "sec": [], "tok": [], "usd": []}))
    for r in rows:
        c = s[r["family"]][r["arm"]]
        c["n"] += 1; c["first"] += bool(r["first_pass"]); c["final"] += bool(r.get("final_pass"))
        u, u2 = r["usage"], r.get("usage_rework") or {}
        c["sec"].append(r["seconds"] + r.get("seconds_rework", 0))
        c["tok"].append(sum(u.get(k, 0) + u2.get(k, 0) for k in ("in", "out", "reasoning", "cache_read")))
        c["usd"].append(plan_usd.get(r["task"], 0.0) if r["arm"].endswith("+plan") else 0.0)
    ref = json.load(open(os.path.join(ROOT, "analysis", "agent_redesign", "model_trial.json")))
    for r in ref["runs"]:
        c = s["fix"][r["arm"]]
        c["n"] += 1; c["first"] += r["first_pass"]; c["final"] += r["final_pass"]
        c["usd"].append(sum(mt.cost(a)["usd"] for a in r["agents"]))
    out = {f: {a: {"n": c["n"], "passed_first": c["first"], "passed_final": c["final"],
                   "usd_per_task": round(sum(c["usd"]) / c["n"], 4) if c["usd"] else None,
                   "lab_tokens_per_task": round(sum(c["tok"]) / len(c["tok"])) if c["tok"] else None,
                   "minutes_per_task": round(sum(c["sec"]) / len(c["sec"]) / 60, 2) if c["sec"] else None}
               for a, c in arms.items()} for f, arms in s.items()}
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    return out


def figure(out):
    import plot_style as ps
    fig, (axl, axr) = ps.figure(ncols=2)
    order = ["sonnet", "opuslow", "deepseek+plan", "qwen+plan", "deepseek", "qwen"]
    fx = out.get("fix", {})
    arms = [a for a in order if a in fx]
    for i, a in enumerate(arms):
        c = fx[a]
        axl.bar(i - 0.2, c["passed_first"] / c["n"], 0.4, color=ps.C.blue, label="first attempt" if i == 0 else None)
        axl.bar(i + 0.2, c["passed_final"] / c["n"], 0.4, color=ps.C.vermillion,
                label="after one rework" if i == 0 else None)
    axl.set_xticks(range(len(arms)), [NAME[a].replace(", ", ",\n") for a in arms])
    axl.set_ylabel("tool fixes passed (of 10)")
    axl.set_ylim(0, 1.0)
    ps.legend(axl, "upper right")
    tr = out.get("triage", {})
    tarms = [a for a in ("deepseek", "qwen") if a in tr]
    for i, a in enumerate(tarms):
        axr.bar(i, tr[a]["passed_first"] / tr[a]["n"], 0.5, color=ps.C.blue)
    axr.set_xticks(range(len(tarms)), [NAME[a] for a in tarms])
    axr.set_ylabel("failed-run logs read correctly")
    axr.set_ylim(0, 1.0)
    ps.save(fig, os.path.join(OUT, "figs", "lab_llm_pass"))


def main():
    out = summary()
    print(json.dumps(out, indent=1))
    figure(out)


if __name__ == "__main__":
    main()
