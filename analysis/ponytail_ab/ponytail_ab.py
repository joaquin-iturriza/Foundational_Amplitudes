#!/usr/bin/env python3
"""Ponytail plugin A/B (2026-10-29, the user's "we can run our own tests on our work to decide if it works for us or
not"): four finished tasks replayed from their original prompt by a fresh headless session, once with the ponytail
plugin enabled and once without, each in its own worktree at the task's parent commit. This script measures the eight
runs (and the original solution of each task, for reference) and prints one JSON object; it writes it to
analysis/ponytail_ab/ponytail_ab.json too.

Inputs, all on the laptop:
  analysis/ponytail_ab/runs.json        per task: base commit, original commit(s) and agent; per arm: plugin on/off and
                                        the headless session id
  analysis/ponytail_ab/diffs/<t>_<a>.diff   the run's whole change against the base (commits and working tree)
  analysis/ponytail_ab/reviews/<t>_<a>.txt  the code reviewer's report on that diff, blind to the arm; its last
                                        "BLOCKING: n" line is the count of blocking findings
  ~/.claude/projects/-home-joaquin-work-FA-worktrees-zz-ab-<t>-<a>/<session>.jsonl   the run's transcript
  the original agent's transcript under the main session's subagents/ directory

Per run:
  tool_calls        tool_use blocks in the run's assistant messages
  api_calls         distinct API responses (one response is split over several records with the same usage; each is
                    counted once, as tools/agent_costs.py does)
  cache_read, cache_write, input, output   summed over the distinct responses
  reread            input + cache_read + cache_write summed over the responses: the context re-read at every call
  lines_added       non-blank added lines in the diff, JSON files excluded; split into code (.py, .sh outside tests),
                    tests (paths containing "test") and text (everything else: prompts, agent definitions, glossary)
  tests_added       added scenario(...), check(...) calls and def test_... in test files
  reused            existing functions (a method call obj.name( and shell keywords never count) (defined at the base commit in a .py or .sh file the run or the original
                    touched, or in tools/plan.py) that the run's added lines call; reused_like_original: those the
                    original solution's added lines call too
  blocking          the reviewer's count of blocking findings; review_reread: the reviewer's own re-read tokens
"""
import glob, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROJ = os.path.expanduser("~/.claude/projects")
MAIN = os.path.join(PROJ, "-home-joaquin-work-FA", "71602535-0955-42ad-a58f-23dc320798dc", "subagents")
GENERIC = {"main", "run", "check", "write", "read", "bash", "die", "log", "usage", "load", "save", "show",
           "done", "do", "then", "fi", "esac", "in", "lower", "upper", "split", "strip", "get"}


def git(*a):
    return subprocess.run(["git", "-C", ROOT] + list(a), capture_output=True, text=True, check=True).stdout


def usage(path):
    seen, tok = set(), dict(api_calls=0, tool_calls=0, input=0, cache_read=0, cache_write=0, output=0)
    for line in open(path):
        r = json.loads(line)
        if r.get("type") != "assistant":
            continue
        m = r["message"]
        tok["tool_calls"] += sum(1 for x in m.get("content") or [] if isinstance(x, dict) and x.get("type") == "tool_use")
        if m.get("id") in seen:
            continue
        seen.add(m.get("id"))
        u = m.get("usage") or {}
        tok["api_calls"] += 1
        tok["input"] += u.get("input_tokens", 0)
        tok["cache_read"] += u.get("cache_read_input_tokens", 0)
        tok["cache_write"] += u.get("cache_creation_input_tokens", 0)
        tok["output"] += u.get("output_tokens", 0)
    tok["reread"] = tok["input"] + tok["cache_read"] + tok["cache_write"]
    return tok


def added(diff):
    """{file: [added lines]} of a unified diff."""
    out, f = {}, None
    for l in diff.splitlines():
        if l.startswith("+++ "):
            f = l[6:] if l.startswith("+++ b/") else None
            if f:
                out.setdefault(f, [])
        elif l.startswith("+") and f:
            out[f].append(l[1:])
    return out


def defs(base, files):
    names = set()
    for f in set(files) | {"tools/plan.py"}:
        if not f.endswith((".py", ".sh")):
            continue
        try:
            src = git("show", "%s:%s" % (base, f))
        except subprocess.CalledProcessError:
            continue                       # new in this change
        names |= set(re.findall(r"^\s*def (\w+)\(", src, re.M)) | set(re.findall(r"^(\w+)\(\)\s*\{", src, re.M))
    return names - GENERIC


def calls(add, names):
    used = set()
    for f, ls in add.items():
        text = "\n".join(ls)
        for n in names:
            pat = r"(?<![.\w])%s\(" % n if f.endswith(".py") else r"(?:^|[;&|(`]|\$\(|\bthen|\bdo|&&|\|\|)\s*%s\b" % n
            if re.search(pat, text, re.M):
                used.add(n)
    return used


def measure(add):
    code = tests = text = 0
    n_tests = 0
    for f, ls in add.items():
        if f.endswith(".json"):
            continue
        k = sum(1 for l in ls if l.strip())
        if "test" in f:
            tests += k
            n_tests += sum(len(re.findall(r"\b(?:scenario|check)\(|\bdef test_", l)) for l in ls)
        elif f.endswith((".py", ".sh")):
            code += k
        else:
            text += k
    return dict(lines_added=code + tests + text, lines_code=code, lines_tests=tests, lines_text=text,
                tests_added=n_tests)


def main():
    cfg = json.load(open(os.path.join(HERE, "runs.json")))
    out, tot = {}, dict(reread=0, cache_read=0, cache_write=0, output=0, review_reread=0)
    for t, c in cfg.items():
        odiff = "".join(git("show", "--format=", x) for x in c["original_commits"])
        oadd = added(odiff)
        res = {}
        runs = {}
        for arm, a in c["arms"].items():
            d = open(os.path.join(HERE, "diffs", "%s_%s.diff" % (t, arm))).read()
            runs[arm] = (a, added(d))
        names = defs(c["base"], list(oadd) + [f for _, ad in runs.values() for f in ad])
        oused = calls(oadd, names)
        o = dict(measure(oadd), reused=sorted(oused))
        f = os.path.join(MAIN, "agent-%s.jsonl" % c["original_agent"])
        if os.path.exists(f):
            o.update(usage(f))
        res["original"] = o
        for arm, (a, ad) in runs.items():
            tr = os.path.join(PROJ, "-home-joaquin-work-FA-worktrees-zz-ab-%s-%s" % (t, arm), a["session"] + ".jsonl")
            r = dict(ponytail=a["ponytail"], **usage(tr), **measure(ad))
            u = calls(ad, names)
            r["reread_M"] = round(r["reread"] / 1e6, 2)
            r["reused"] = sorted(u)
            r["reused_like_original"] = sorted(u & oused)
            r["n_reused"], r["n_like_original"], r["n_original_reused"] = len(u), len(u & oused), len(oused)
            rf = os.path.join(MAIN, "agent-%s.jsonl" % a["reviewer"])
            r["review_reread"] = usage(rf)["reread"] if os.path.exists(rf) else None
            rev = open(os.path.join(HERE, "reviews", "%s_%s.txt" % (t, arm))).read()
            m = re.findall(r"BLOCKING:\s*(\d+)", rev)
            r["blocking"] = int(m[-1]) if m else None
            r["hit_turn_cap"] = a.get("hit_turn_cap", False)
            res["on" if a["ponytail"] else "off"] = r
            for k in ("reread", "cache_read", "cache_write", "output"):
                tot[k] += r[k]
            tot["review_reread"] += r["review_reread"] or 0
        out[t] = res
    out["total_runs"] = tot
    on = [out[t]["on"] for t in cfg]
    off = [out[t]["off"] for t in cfg]
    out["sum"] = {arm: {k: sum(r[k] for r in rs) for k in ("tool_calls", "reread", "output", "lines_added",
                                                           "lines_code", "tests_added", "blocking", "n_reused",
                                                           "n_like_original", "n_original_reused")}
                  for arm, rs in (("on", on), ("off", off))}
    out["sum"]["on"]["reread_M"] = round(out["sum"]["on"]["reread"] / 1e6, 1)
    out["sum"]["off"]["reread_M"] = round(out["sum"]["off"]["reread"] / 1e6, 1)
    out["total_runs"]["reread_M"] = round(tot["reread"] / 1e6, 1)
    out["total_runs"]["review_reread_M"] = round(tot["review_reread"] / 1e6, 1)
    out["total_runs"]["all_reread_M"] = round((tot["reread"] + tot["review_reread"]) / 1e6, 1)
    json.dump(out, open(os.path.join(HERE, "ponytail_ab.json"), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
