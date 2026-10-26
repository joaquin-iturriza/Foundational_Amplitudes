#!/usr/bin/env python3
"""PreToolUse(Bash): no compute for a plan entry without a recorded design (the sign head search, 2026-10-26: jobs
designed inside a running entry whose measurement could not see the tested change; the user: "it's the second or
third time you bring back results that show nothing and the conclusion is that the way you are testing was wrong").

A submission for a plan entry is
  - `site submit ... --item X`;
  - `site launch add ... --item X` (queued work the supervisor starts);
  - `site run ... --item X -- <cmd>` whose command submits: sweep_manager.py submit, a sweep generator
    (generate_*sweep*.py) with --auto-submit or without --dry-run, sbatch, condor_submit;
  - `scripts/launch_*.sh X ...` (the item is the first argument).
It is refused (exit 2) unless every entry X named has a design recorded with `plan design` in the last DESIGN_H hours
and passed by the design reviewer (`plan design-pass`; a resubmit of the same design within that window passes). The
step agent that records a design runs the design-reviewer subagent itself, then submits on its PASS.
An eval-only re-read of an arm that already ran needs no new design: a command carrying `REREAD=<run id>` (an
environment assignment, ignored by the tools) passes when that run is one of the entry's (registry item or evidence)
and the entry's latest design passed review, whatever its age. What the re-read job does is the step agent's word.
`plan design-pass` and `plan design-fix` are refused unless the tool call comes from the design-reviewer subagent
(the hook input's agent_type), so no step agent or main session passes its own design. `--item '*'` (upkeep) and items that are no plan entry
pass, as does every command that is not a submission. It fails closed like plan_check: an error while checking a
submission refuses it; a command that is no submission is never refused, whatever happens.
Env for tests: the PLAN_* variables of tools/plan.py."""
import json, os, re, shlex, sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEP = set(";&|()")
SUBMITS = re.compile(r"sweep_manager\.py\s+submit\b|\bsbatch\b|\bcondor_submit(?:_dag)?\b")
GENERATOR = re.compile(r"generate_\w*sweeps?\w*\.py")


def segments(cmd):
    try:
        lx = shlex.shlex(cmd, posix=True, punctuation_chars=";&|<>()")
        lx.whitespace_split = True
        toks = list(lx)
    except ValueError:
        return None
    out, cur = [], []
    for t in toks:
        if t and set(t) <= SEP:
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


def items_of(toks):
    out = []
    for i, t in enumerate(toks):
        if t == "--item" and i + 1 < len(toks):
            out.append(toks[i + 1])
        elif t.startswith("--item="):
            out.append(t.split("=", 1)[1])
    return [x for part in out for x in part.split(",") if x]


def remote_submits(text):
    if SUBMITS.search(text):
        return True
    return bool(GENERATOR.search(text)) and ("--auto-submit" in text or "--dry-run" not in text)


def submissions(cmd):
    """The item ids each submission in the command names ([] for a submission with none: site refuses it itself)."""
    segs = segments(cmd)
    if segs is None:                           # unparsable: read the raw text
        if re.search(r"\bsite\s+(?:submit|launch\s+add)\b", cmd) or (
                re.search(r"\bsite\s+run\b", cmd) and remote_submits(cmd)):
            return [re.findall(r"--item[=\s]+['\"]?([^\s'\";&|]+)", cmd)]
        m = re.search(r"scripts/launch_\w+\.sh\s+['\"]?([^\s'\";&|]+)", cmd)
        return [[m.group(1)]] if m else []
    out = []
    for toks in segs:
        for i, t in enumerate(toks):
            if os.path.basename(t) == "site" and i + 1 < len(toks):
                rest, verb = toks[i + 1:], toks[i + 1]
                if verb == "submit" or (verb == "launch" and "add" in rest[:2]):
                    out.append(items_of(rest))
                elif verb == "run" and "--" in rest:
                    k = rest.index("--")
                    if remote_submits(" ".join(rest[k + 1:])):
                        out.append(items_of(rest[:k]))
                break
            if re.search(r"(?:^|/)scripts/launch_\w+\.sh$", t):
                runs = all(re.match(r"\w+=", x) for x in toks[:i]) or toks[i - 1] in ("bash", "sh")
                if runs and i + 1 < len(toks):
                    out.append([toks[i + 1]])
                break
    return out


VERDICT = re.compile(r"tools/plan\.py\b[^;&|\n]*\bdesign-(?:pass|fix)\b")


REREAD = re.compile(r"(?:^|[\s;&|(])REREAD=['\"]?([A-Za-z0-9_.:-]+)")


def reread_ok(plan, e, st, runs):
    x = plan.last_design(e)
    if not runs or not x or not x.get("passed_at"):
        return False
    mine = {r for r, _, _ in plan.runs_of(e, st)} | set((e.get("evidence") or {}).get("runs") or [])
    return all(r in mine for r in runs)


def problems(subs, cmd=""):
    sys.path.insert(0, os.path.join(HERE, "tools"))
    import plan
    d, st = plan.load()
    ids = {e["id"]: e for e in d["entries"]}
    bad = []
    rereads = REREAD.findall(cmd)
    for x in sorted({x for s in subs for x in s if x != "*"}):
        if x in ids and not plan.design_fresh(ids[x]) and not reread_ok(plan, ids[x], st, rereads):
            last = plan.last_design(ids[x])
            old = last and plan.now() - plan.parse(last["at"]) > plan.dt.timedelta(hours=plan.DESIGN_H)
            bad.append("%s (%s)" % (x, "no design recorded" if not last else
                                    "last design recorded %s, more than %d h ago" % (last["at"][:16], plan.DESIGN_H)
                                    if old else "the design recorded %s %s" % (last["at"][:16], (
                                        "was sent back by the design reviewer: " + last.get("fix", "")
                                        if last.get("fix_at") else "waits for the design reviewer's verdict"))))
    return bad, plan.DESIGN_H


def main():
    try:
        inp = json.loads(sys.stdin.read() or "{}")
        cmd = (inp.get("tool_input") or {}).get("command", "") or ""
        subs = submissions(cmd)
    except Exception:
        return 0                                   # not readable as a submission: never refused
    if VERDICT.search(cmd) and inp.get("agent_type") != "design-reviewer":
        print("BLOCKED by design_guard: plan design-pass and design-fix are the design reviewer's verdict; only the "
              "design-reviewer subagent runs them (this call comes from %s). Launch it (Agent tool, subagent_type "
              "design-reviewer) on the entry." % (
                  inp.get("agent_type") or "the main session"), file=sys.stderr)
        return 2
    if not subs:
        return 0
    try:
        bad, hours = problems(subs, cmd)
    except BaseException as ex:                    # plan.py exits through sys.exit on a broken plan
        print("BLOCKED by design_guard: could not check this submission's design (%s: %s). Fix the plan, then "
              "submit again." % (type(ex).__name__, ex), file=sys.stderr)
        return 2
    if not bad:
        return 0
    print("BLOCKED by design_guard: a job for a plan entry needs its design recorded within %d h and passed by the "
          "design reviewer; missing for %s.\nIf the design is recorded, run the design-reviewer subagent on it "
          "(Agent tool, subagent_type design-reviewer, naming the entry); it runs plan design-pass on PASS. An "
          "eval-only re-read of a run of the entry under its passed design carries REREAD=<run id>. "
          "Otherwise record it first, every field (the check says how it was verified, on existing data, that the measure "
          "moves with the tested change; a design whose check fails is not run):\n"
          "  python3 tools/plan.py design <entry> --question \"<what the job answers>\" --measure \"<the number that "
          "answers it>\" --check \"<how it was seen to move with the change, on which data>\" --fixed \"<what is "
          "held fixed>\" --control \"<the matched control>\" --seeds \"<seeds>\" --outcome \"<what yes and what no "
          "look like>\"\nthen submit again." % (hours, ", ".join(bad)), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
