#!/usr/bin/env python3
"""Routed work for the lab's DeepSeek (D541): failed-run log reading and small tool fixes, done by deepseek-v4-flash
through opencode in the lab_llm_trial sandbox, each result checked by a Sonnet mechanical-worker before anything is
acked, committed or resubmitted.

    python3 tools/labllm.py triage RUN [--event E<n>] [--end CLASS]   DeepSeek reads the run's log (site logs)
    python3 tools/labllm.py fix BRIEF_FILE [--test "CMD"]            DeepSeek makes the fix on a copy of HEAD; a diff
    python3 tools/labllm.py check ID --agree|--disagree --note "..."  the checker's verdict on a routed result
    python3 tools/labllm.py stats                                    routed results, checked, agreed

Records: one line per routed result and per check in ROUTED (outside the repo: workers share the checkout).
"""
import argparse, json, os, re, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab_llm_trial as lt

ROUTED = os.path.expanduser("~/.local/share/fa_labllm/routed.jsonl")
MODEL = lt.MODELS["deepseek"]
FIX_TIMEOUT = 540          # one opencode session inside a 10-min tool call; a timeout hands the fix back to Sonnet


def log(rec):
    os.makedirs(os.path.dirname(ROUTED), exist_ok=True)
    with open(ROUTED, "a") as f:
        f.write(json.dumps(rec) + "\n")


def unavailable(u, d):
    """True when DeepSeek never answered (the lab server down: no model step); the caller does the work itself."""
    if u["steps"]:
        return False
    print("labllm: DeepSeek did not answer (no model step; the lab server's errors are in %s/.xdg/data/opencode/log): "
          "do this step yourself, as before the route" % d)
    return True


def rid(kind):
    return "%s%s" % (kind[0].upper(), time.strftime("%Y%m%d%H%M%S"))


def triage(run, event, end):
    i = rid("triage")
    d = lt.fresh("route_" + i)
    out = subprocess.run(["site", "logs", run, "-n", "100000"], capture_output=True, text=True, timeout=300)
    path = os.path.join(d, run + ".log")
    open(path, "w").write(out.stdout)
    if not out.stdout.strip():
        sys.exit("labllm: site logs %s returned nothing (rc %d): read it yourself" % (run, out.returncode))
    ans = os.path.join(d, "answer.json")
    ev, u, sec, rc, se, _ = lt.opencode(d, MODEL, lt.TRIAGE.format(run=run, end=end, log=path, ans=ans), timeout=600)
    if unavailable(u, d):
        log({"id": i, "kind": "triage", "run": run, "event": event, "unavailable": True, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        sys.exit(2)
    try:
        a = json.load(open(ans))
    except (OSError, ValueError):
        a = None
    text = out.stdout
    # what the checker is pointed at: does the quoted line, and the step, occur in the log as written
    quoted = bool(a and a.get("error") and str(a["error"]).strip() in text)
    step = bool(a and a.get("step") is not None and re.search(r"\b%s\b" % re.escape(str(a["step"])), text))
    rec = {"id": i, "kind": "triage", "run": run, "event": event, "end": end, "log": path, "answer": a,
           "quote_in_log": quoted, "step_in_log": step, "seconds": round(sec, 1), "lab_tokens": u, "rc": rc,
           "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    log(rec)
    print(json.dumps({k: rec[k] for k in ("id", "run", "event", "answer", "quote_in_log", "step_in_log", "log")},
                     indent=1))
    print("check it: read the log around the quoted line or step (%s), then python3 tools/labllm.py check %s "
          "--agree|--disagree --note \"...\"; ack or resubmit only after that" % (path, i))


def fix(brief, test):
    i = rid("fix")
    d = lt.fresh("route_" + i)
    repo = os.path.join(d, "repo")
    os.makedirs(repo)
    base = subprocess.run(["git", "-C", lt.ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    subprocess.run("git -C %s archive HEAD | tar -x -C %s" % (lt.ROOT, repo), shell=True, check=True)
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"]):
        subprocess.run(["git", "-C", repo] + cmd, check=True, capture_output=True)
    prompt = ("Work in the repository at %s (a copy; edit it in place, do not commit). Python: %s. %s\n\nMake the "
              "change, run the tests that cover it, end with a short report." % (repo, lt.mt.PY, open(brief).read()))
    if test:
        prompt += "\n\nThe test that must pass: %s" % test
    ev, u, sec, rc, se, _ = lt.opencode(d, MODEL, prompt, timeout=FIX_TIMEOUT)
    if unavailable(u, d):
        log({"id": i, "kind": "fix", "brief": os.path.abspath(brief), "unavailable": True, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        sys.exit(2)
    tres = None
    if test:            # the test runs the model's code: inside the sandbox
        trc, so, se2, _ = lt.sandbox(d, ["bash", "-c", "cd %s && %s" % (repo, test)], 900)
        tres = {"rc": trc, "tail": (so + se2)[-1500:]}
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True)
    diff = subprocess.run(["git", "-C", repo, "diff", "--cached", "HEAD"], capture_output=True, text=True).stdout
    dpath = os.path.join(d, "fix.diff")
    open(dpath, "w").write(diff)
    rec = {"id": i, "kind": "fix", "brief": os.path.abspath(brief), "base": base, "diff": dpath,
           "diff_lines": diff.count("\n"), "test": test, "test_result": tres, "seconds": round(sec, 1),
           "lab_tokens": u, "rc": rc, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    log(rec)
    print(json.dumps({k: rec[k] for k in ("id", "base", "diff", "diff_lines", "test_result", "rc", "seconds")},
                     indent=1))
    print("check it: read %s whole; if right, git apply --3way it in your worktree (base %s), run the test, commit "
          "only on a pass; then python3 tools/labllm.py check %s --agree|--disagree --note \"...\" (empty diff, "
          "timeout or wrong: --disagree, and do the fix yourself)" % (dpath, base[:10], i))


def check(i, ok, note):
    recs = [json.loads(l) for l in open(ROUTED)] if os.path.exists(ROUTED) else []
    if not any(r.get("id") == i for r in recs):
        sys.exit("labllm: no routed result %s in %s" % (i, ROUTED))
    log({"check": i, "agree": ok, "note": note, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print("labllm: %s %s" % (i, "agreed" if ok else "disagreed"))


def stats():
    recs = [json.loads(l) for l in open(ROUTED)] if os.path.exists(ROUTED) else []
    chk = {r["check"]: r["agree"] for r in recs if "check" in r}
    for kind in ("triage", "fix"):
        rs = [r for r in recs if r.get("kind") == kind and not r.get("unavailable")]
        c = [chk[r["id"]] for r in rs if r["id"] in chk]
        print("%-6s routed %d, checked %d, agreed %d" % (kind, len(rs), len(c), sum(c)))
    print("lab server unavailable: %d" % sum(1 for r in recs if r.get("unavailable")))
    open_ = [r["id"] for r in recs if r.get("kind") and not r.get("unavailable") and r["id"] not in chk]
    if open_:
        print("unchecked: " + ", ".join(open_))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("triage"); p.add_argument("run"); p.add_argument("--event"); p.add_argument("--end", default="FAILED")
    p = sp.add_parser("fix"); p.add_argument("brief"); p.add_argument("--test")
    p = sp.add_parser("check"); p.add_argument("id"); p.add_argument("--note", required=True)
    g = p.add_mutually_exclusive_group(required=True); g.add_argument("--agree", action="store_true")
    g.add_argument("--disagree", action="store_true")
    sp.add_parser("stats")
    a = ap.parse_args()
    if a.cmd == "triage":
        triage(a.run, a.event, a.end)
    elif a.cmd == "fix":
        fix(a.brief, a.test)
    elif a.cmd == "check":
        check(a.id, a.agree, a.note)
    else:
        stats()


if __name__ == "__main__":
    main()
