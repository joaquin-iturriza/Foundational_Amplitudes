#!/usr/bin/env python3
"""Ponytail plugin A/B, second round (2026-10-29, the user's "perhaps a bit more testing could be worth it with
ponytail, I'm surprised the reusing didn't show up that much, but maybe the task didn't allow for it. Maybe try more
examples"). Four finished analysis tasks, each replayed four times from its original prompt by a fresh headless
session in its own worktree at the task's parent commit: twice with the plugin on, twice off (labels a to d, the arm
drawn at random per pair). Round one's method otherwise (ponytail_ab.py). Prints one JSON object and writes it to
analysis/ponytail_ab/ponytail_ab2.json.

Inputs, all on the laptop:
  analysis/ponytail_ab/runs2.json            per task: base commit, why existing code could do part of it; per label:
                                             plugin on/off, headless session id, reviewer agent id, blind review label
  analysis/ponytail_ab/diffs2/<t>_<l>.diff    the run's whole change against the base (commits and working tree)
  analysis/ponytail_ab/reviews2/<t>_<l>.txt   the blind code review of that diff: one "REIMPL: <new code> | <original
                                             file:function> | <why>" line per reimplementation, then "REIMPLEMENTED: n"
                                             and "BLOCKING: n" as its last lines
  ~/.claude/projects/-home-joaquin-work-FA-worktrees-zz-ab2-<t>-<l>/<session>.jsonl   the run's transcript

Per run, as round one (ponytail_ab.usage, .added, .measure): tool_calls, reread (input + cache read + cache write over
the distinct API responses, as tools/agent_costs.py counts them), lines_added split into code / tests / text (JSON
excluded), tests_added; and
  reimpl        the reviewer's count of functions or blocks that reimplement code the repo had at the base commit
  blocking      the reviewer's count of blocking findings
  project_uses  distinct names the added .py lines take from modules that exist in the repo at the base commit
                (`from M import a` gives M.a; `import M [as m]` gives each m.<name> used), stdlib and third party not
                counted
Per task and arm: the mean of the two runs and their spread (the absolute difference). Per measure: in how many tasks
the on-off difference of means exceeds the larger of the two arms' spreads.
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ponytail_ab as r1  # noqa: E402  round one's transcript and diff measures

PROJ = os.path.expanduser("~/.claude/projects")
MEASURES = ("tool_calls", "reread_M", "lines_added", "lines_code", "lines_tests", "lines_text", "tests_added",
            "project_uses", "reimpl", "blocking")


def project_modules(base):
    files = r1.git("ls-tree", "-r", "--name-only", base).split()
    return {os.path.basename(f)[:-3] for f in files if f.endswith(".py")} | {f.split("/")[0] for f in files if "/" in f}


def project_uses(add, mods):
    used = set()
    for f, ls in add.items():
        if not f.endswith(".py"):
            continue
        text = "\n".join(ls)
        for m, names in re.findall(r"^\s*from\s+([\w.]+)\s+import\s+\(?([\w ,]+)", text, re.M):
            if m.split(".")[-1] in mods or m.split(".")[0] in mods:
                used |= {"%s.%s" % (m, n.split(" as ")[0].strip()) for n in names.split(",") if n.strip()}
        for m, alias in re.findall(r"^\s*import\s+([\w.]+)(?:\s+as\s+(\w+))?", text, re.M):
            if m.split(".")[-1] in mods or m.split(".")[0] in mods:
                a = alias or m
                used |= {"%s.%s" % (m, n) for n in re.findall(r"(?<![\w.])%s\.(\w+)" % re.escape(a), text)}
    return used


def review(path):
    t = open(path).read()
    get = lambda k: (lambda m: int(m[-1]) if m else None)(re.findall(r"^%s:\s*(\d+)" % k, t, re.M))
    return get("REIMPLEMENTED"), get("BLOCKING"), re.findall(r"^REIMPL: (.*)$", t, re.M)


def main():
    cfg = json.load(open(os.path.join(HERE, "runs2.json")))
    out, tot = {}, dict(reread=0, review_reread=0)
    for t, c in cfg.items():
        mods = project_modules(c["base"])
        runs = {"on": [], "off": []}
        for lab, a in sorted(c["arms"].items()):
            add = r1.added(open(os.path.join(HERE, "diffs2", "%s_%s.diff" % (t, lab))).read())
            tr = os.path.join(PROJ, "-home-joaquin-work-FA-worktrees-zz-ab2-%s-%s" % (t, lab), a["session"] + ".jsonl")
            r = dict(label=lab, **r1.usage(tr), **r1.measure(add))
            r["reread_M"] = round(r["reread"] / 1e6, 2)
            u = project_uses(add, mods)
            r["project_uses"], r["project_used"] = len(u), sorted(u)
            r["reimpl"], r["blocking"], r["reimpl_list"] = review(os.path.join(HERE, "reviews2", "%s_%s.txt" % (t, lab)))
            rf = os.path.join(r1.MAIN, "agent-%s.jsonl" % a["reviewer"])
            r["review_reread"] = r1.usage(rf)["reread"] if os.path.exists(rf) else 0
            r["hit_turn_cap"] = a.get("hit_turn_cap", False)
            runs["on" if a["ponytail"] else "off"].append(r)
            tot["reread"] += r["reread"]
            tot["review_reread"] += r["review_reread"]
        res = {"runs": runs}
        for arm, rs in runs.items():
            res[arm] = {k: round(sum(x[k] for x in rs) / len(rs), 2) for k in MEASURES}
            res[arm + "_spread"] = {k: round(abs(rs[0][k] - rs[1][k]), 2) for k in MEASURES}
        res["beyond_spread"] = {k: abs(res["on"][k] - res["off"][k]) > max(res["on_spread"][k], res["off_spread"][k])
                                for k in MEASURES}
        out[t] = res
    tasks = list(cfg)
    out["summary"] = {k: dict(on_lower=sum(out[t]["on"][k] < out[t]["off"][k] for t in tasks),
                              beyond_spread=sum(out[t]["beyond_spread"][k] for t in tasks),
                              on_sum=round(sum(out[t]["on"][k] for t in tasks), 2),
                              off_sum=round(sum(out[t]["off"][k] for t in tasks), 2)) for k in MEASURES}
    tot["reread_M"] = round(tot["reread"] / 1e6, 1)
    tot["review_reread_M"] = round(tot["review_reread"] / 1e6, 1)
    tot["all_reread_M"] = round((tot["reread"] + tot["review_reread"]) / 1e6, 1)
    out["total"] = tot
    json.dump(out, open(os.path.join(HERE, "ponytail_ab2.json"), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
