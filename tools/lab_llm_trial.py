#!/usr/bin/env python3
"""Lab LLMs on our subagent work (lab_llm_trial, D530): past tasks with known outcomes run by the lab's models
(deepseek-v4-flash, qwen3.8-27b) through opencode, each run in a bubblewrap sandbox on a scratch copy, graded by a
check of its own (never by a model).

    python3 tools/lab_llm_trial.py cases                       write the log-triage cases (analysis/lab_llm_trial/triage_cases.json)
    python3 tools/lab_llm_trial.py smoke MODEL                 one trivial task through opencode in the sandbox
    python3 tools/lab_llm_trial.py fix TASK ARM [--plan FILE]  one tool-fix task (tools/model_trial.py TRIAL), graded
    python3 tools/lab_llm_trial.py triage CASE ARM             one log-triage case, graded
    python3 tools/lab_llm_trial.py report                      per arm: passed, tokens, wall time; the figure

Families. fix: the ten validated past repo fixes of tools/model_trial.py (same defect text and interface as the
Sonnet / Opus-low trial, D203), graded by the fix commit's own tests on the tree the model leaves (model_trial
run_check + verdict), with one rework round fed the failing check's output, as in that trial. triage: a failed
run's log as the owner got it (run end class from the scheduler), the model writes answer.json; graded against the
log's own decisive line (the divergence guard's step, the AssertionError's message, the time limit).
Arms: <model> (alone) and <model>+plan (brief plus a plan written by a Sonnet subagent that read the code, fix only).
Sandbox: bwrap, / read-only, the run directory and opencode's cache writable, /tmp private, ~/.ssh and the ccorch
config hidden, SSH and Kerberos variables unset (no cluster reachable); network shared (the lab server).
Records: analysis/lab_llm_trial/runs.jsonl, one line per run.
"""
import argparse, glob, hashlib, json, os, re, shutil, sqlite3, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import model_trial as mt

MAIN = os.path.expanduser("~/work/FA")
if not os.path.exists(mt.PY):                      # a worktree has no venv of its own
    mt.PY = os.path.join(MAIN, ".venv-cpu", "bin", "python")
OUT = os.path.join(ROOT, "analysis", "lab_llm_trial")
REC = os.path.join(OUT, "runs.jsonl")
SCR = os.environ.get("LAB_TRIAL_SCRATCH") or "/tmp/claude-%d/lab_llm_trial" % os.getuid()
OPENCODE = os.path.expanduser("~/.local/lib/node_modules/opencode-ai/node_modules/opencode-linux-x64/bin/opencode")
MODELS = {"deepseek": "deepseek-v4-flash", "qwen": "qwen3.8-27b"}
HOME = os.path.expanduser("~")
CCORCH = os.path.join(HOME, ".local", "share", "ccorch")
CONFIG = {
    "$schema": "https://opencode.ai/config.json", "autoupdate": False, "share": "disabled",
    "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny", "external_directory": "allow"},
    "provider": {"lab": {"npm": "@ai-sdk/openai-compatible", "name": "LPNHE lab",
                         "options": {"baseURL": "https://lpnkarim.in2p3.fr/api",
                                     "apiKey": "{file:%s/.config/lpnhe/llm_key}" % HOME},
                         "models": {m: {"name": m} for m in MODELS.values()}}},
}


HIDE = [os.path.join(HOME, x) for x in ("work", ".claude", ".ssh", ".config/ccorch", ".local/share/ccorch")]


def sandbox(run_dir, cmd, timeout, hide=True):
    """(rc, stdout, stderr, seconds) of cmd in bwrap: everything read-only but run_dir and opencode's cache; with
    hide (a model's run) ~/work, the transcripts, the ledger and the keys to the clusters are empty, only the CPU
    venv mounted back (no fix commit, ack note or trial key can be read); the grader runs with hide off."""
    cache = os.path.join(HOME, ".cache", "opencode")
    os.makedirs(cache, exist_ok=True)
    xdg = os.path.join(run_dir, ".xdg")
    for d in ("data", "state", "config"):
        os.makedirs(os.path.join(xdg, d), exist_ok=True)
    cfg = os.path.join(xdg, "opencode.json")
    json.dump(CONFIG, open(cfg, "w"), indent=1)
    b = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
         *sum((["--tmpfs", h] for h in (HIDE if hide else HIDE[2:4])), []),
         *(["--ro-bind"] + [os.path.dirname(os.path.dirname(mt.PY))] * 2 if hide else []),
         "--bind", run_dir, run_dir, "--bind", cache, cache, "--die-with-parent", "--chdir", run_dir]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SSH_", "KRB5", "CLAUDE", "ANTHROPIC"))}
    # the lhe tests write mg5_pipeline_final's OUTPUT_DIR (.lhe_work, manifest.json): a copy in the run dir, so the
    # real ~/datasets is neither needed writable nor touched
    out = os.path.join(run_dir, ".mg5_output")
    if not os.path.exists(out):
        os.makedirs(out)
        shutil.copy(os.path.join(HOME, "datasets", "manifest.json"), out)
    env["MG5_OUTPUT_DIR"] = out
    env.update(XDG_DATA_HOME=xdg + "/data", XDG_STATE_HOME=xdg + "/state", XDG_CONFIG_HOME=xdg + "/config",
               OPENCODE_CONFIG=cfg, PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    try:
        r = subprocess.run(b + cmd, capture_output=True, text=True, timeout=timeout, env=env)
        rc, so, se = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        rc, so, se = "timeout", (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), ""
    return rc, so, se, time.time() - t0


def opencode(run_dir, model, prompt, timeout=2400, session=None):
    """One opencode run; (events, usage {in, out, reasoning, cache_read, steps}, seconds, rc, stderr tail)."""
    cmd = [OPENCODE, "run", "--format", "json", "-m", "lab/" + model]
    if session:
        cmd += ["--session", session]
    rc, so, se, sec = sandbox(run_dir, cmd + [prompt], timeout)
    ev = []
    for line in so.splitlines():
        try:
            ev.append(json.loads(line))
        except ValueError:
            pass
    u = {"in": 0, "out": 0, "reasoning": 0, "cache_read": 0, "steps": 0, "tool_calls": 0}
    for e in ev:
        p = e.get("part") or {}
        if e.get("type") == "step_finish" or p.get("type") == "step-finish":
            t = p.get("tokens") or {}
            u["in"] += t.get("input", 0); u["out"] += t.get("output", 0); u["reasoning"] += t.get("reasoning", 0)
            u["cache_read"] += (t.get("cache") or {}).get("read", 0); u["steps"] += 1
        if e.get("type") == "tool_use" or p.get("type") == "tool":
            u["tool_calls"] += 1
    sid = next((e.get("sessionID") or (e.get("part") or {}).get("sessionID") for e in ev
                if e.get("sessionID") or (e.get("part") or {}).get("sessionID")), None)
    return ev, u, sec, rc, se[-2000:], sid


def record(r):
    os.makedirs(OUT, exist_ok=True)
    with open(REC, "a") as f:
        f.write(json.dumps(r) + "\n")


def fresh(name):
    d = os.path.join(SCR, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


# ---------------------------------------------------------------- smoke
def smoke(arm):
    d = fresh("smoke_" + arm)
    open(os.path.join(d, "numbers.txt"), "w").write("3\n4\n5\n")
    ev, u, sec, rc, se, _ = opencode(d, MODELS[arm], "Write the sum of the numbers in numbers.txt (one per line) "
                                     "into a new file sum.txt, the number only.", timeout=600)
    got = open(os.path.join(d, "sum.txt")).read().strip() if os.path.exists(os.path.join(d, "sum.txt")) else None
    # the sandbox: a write outside the run dir must fail
    rc2, so2, se2, _ = sandbox(d, ["bash", "-c", "touch /usr/zz_probe_sandbox 2>&1; echo write_rc=$?; ls %s/tools "
                                   "%s/runs.db 2>&1 | head -2; ls ~/.ssh | wc -l; %s -c 'import numpy; print(1)'"
                                   % (MAIN, CCORCH, mt.PY)], 60)
    res = {"family": "smoke", "arm": arm, "passed": got == "12", "got": got, "usage": u, "seconds": round(sec, 1),
           "rc": rc, "stderr": se[-500:], "outside_write": so2.strip()}
    print(json.dumps(res))
    return res


# ---------------------------------------------------------------- fix family
def fix_setup(task, arm):
    """A plain copy of the repository at the fix's parent (git archive, its own fresh git repo: no history to read)."""
    c, _, _ = mt.TASKS[task]
    d = fresh("fix_%s_%s" % (task, arm))
    repo = os.path.join(d, "repo")
    os.makedirs(repo)
    subprocess.run("git -C %s archive %s^ | tar -x -C %s" % (ROOT, c, repo), shell=True, check=True)
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"]):
        subprocess.run(["git", "-C", repo] + cmd, check=True, capture_output=True)
    return d, repo


def fix_brief(task, repo):
    return mt.brief(task, "x").replace(mt.wt(task, "x"), repo).replace(mt.PY.replace(MAIN, ROOT), mt.PY)


def fix_grade(task, d, repo):
    """model_trial's check on the tree as left, run inside the sandbox (the tests execute the model's code)."""
    rc, so, se, _ = sandbox(d, [sys.executable, os.path.abspath(__file__), "_grade", task, repo], 1200, hide=False)
    try:
        return json.loads(so.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"passed": False, "tail": (so + se)[-1500:]}


def _grade(task, repo):
    c, tests, _ = mt.TASKS[task]
    ok, out = mt.run_check(repo, c, tests, task=task)
    res = mt.results(out) or {"<rc>": ok}
    print(json.dumps({"passed": mt.verdict(task, res), "failing": sorted(n for n, v in res.items() if not v),
                      "tail": out[-1500:]}))


def fix(task, arm, plan=None):
    model = MODELS[arm.split("+")[0]]
    d, repo = fix_setup(task, arm)
    prompt = fix_brief(task, repo)
    if plan:
        prompt += ("\n\nA senior engineer read the code and wrote this plan for you; follow it:\n\n"
                   + open(plan).read())
    ev, u, sec, rc, se, sid = opencode(d, model, prompt)
    g1 = fix_grade(task, d, repo)
    r = {"family": "fix", "task": task, "arm": arm, "model": model, "first_pass": g1["passed"],
         "usage": u, "seconds": round(sec, 1), "rc": rc, "stderr": se[-500:], "failing_first": g1.get("failing")}
    if not g1["passed"]:                          # one rework round with the check's output, as in D203's trial
        ev2, u2, sec2, rc2, se2, _ = opencode(d, model, "The check that judges the fix failed. Its output:\n\n%s\n\n"
                                              "Fix it, run the tests again, end with a short report." % g1["tail"],
                                              session=sid)
        g2 = fix_grade(task, d, repo)
        r.update(final_pass=g2["passed"], usage_rework=u2, seconds_rework=round(sec2, 1), rc_rework=rc2)
    else:
        r["final_pass"] = True
    r["diff"] = subprocess.run(["git", "-C", repo, "diff", "HEAD", "--stat"], capture_output=True, text=True).stdout[-800:]
    json.dump(ev, open(os.path.join(d, "events.json"), "w"))
    record(r)
    print(json.dumps({k: r[k] for k in ("task", "arm", "first_pass", "final_pass", "seconds")}))


# ---------------------------------------------------------------- triage family
DIVERGED = re.compile(r"Training diverged.*?at step (\d+)")
ASSERT = re.compile(r"^AssertionError: ?(.*)$", re.M)
CASES = os.path.join(OUT, "triage_cases.json")


def cases():
    """Failed-run events of FA acked with a note, whose run has exactly one log fetched home, one per distinct log;
    the key from the log itself: diverged (the guard's step), error (the AssertionError's message) or time_limit."""
    db = sqlite3.connect(os.path.join(CCORCH, "runs.db"))
    rows = db.execute("select id, key, ack_note from events where project='FA' and kind='RUN' and ack_note is not "
                      "null and key not like '%CANCELLED%' and key not like '%COMPLETED%' order by id").fetchall()
    out, seen = [], set()
    for eid, key, note in rows:
        run, end = key.split(":", 1)
        logs = [p for p in glob.glob(os.path.join(CCORCH, "artifacts", "FA", run, "**", "*"), recursive=True)
                if re.search(r"\.(out|log|err)$", p)]
        if len(logs) != 1:
            continue
        text = open(logs[0], errors="replace").read()
        h = hashlib.sha1(re.sub(r"\d", "", text[-3000:]).encode()).hexdigest()
        if h in seen:
            continue
        m, a = DIVERGED.search(text), ASSERT.findall(text)
        if m:
            k = {"class": "diverged", "step": int(m.group(1))}
        elif a:
            toks = sorted(set(re.findall(r"[A-Za-z_][\w.]*_[\w.]+", a[-1])), key=len, reverse=True)
            k = {"class": "error", "token": toks[0] if toks else a[-1][:40]}
        elif end.startswith("TIMEOUT") or "DUE TO TIME LIMIT" in text:
            k = {"class": "time_limit"}
        else:
            continue
        cause = (k["class"], k.get("token")) if k["class"] != "diverged" else h
        if cause in seen:
            continue
        seen.add(h); seen.add(cause)
        out.append({"case": "E%d" % eid, "run": run, "end": end, "log": logs[0], "bytes": len(text), "key": k,
                    "note": note[:300]})
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(CASES, "w"), indent=1)
    from collections import Counter
    print(len(out), Counter(c["key"]["class"] for c in out))


TRIAGE = ("Run {run} of our training project ended {end} on the cluster. Its log is {log} (it may be long: search "
          "it, do not print it whole). Find why it ended. Write {ans} as JSON with keys: \"class\": one of "
          "\"diverged\" (the training's divergence guard stopped it), \"error\" (a code or configuration error "
          "stopped it), \"time_limit\" (the scheduler's time limit cut it), \"other\"; \"step\": the training step "
          "where it diverged (integer, or null); \"error\": the decisive error line verbatim (or null); "
          "\"training_finished\": true or false (did the main training reach its last step); \"resubmit_as_is\": "
          "true or false; \"why\": one sentence. Then stop.")


def triage_grade(k, ans):
    if not isinstance(ans, dict) or ans.get("class") != k["class"]:
        return False
    if k["class"] == "diverged":
        return str(ans.get("step")) == str(k["step"])
    if k["class"] == "error":
        return k["token"] in str(ans.get("error") or "")
    return ans.get("training_finished") is True     # every time_limit case: the main stage had finished


def triage(case, arm):
    c = next(x for x in json.load(open(CASES)) if x["case"] == case)
    d = fresh("triage_%s_%s" % (case, arm))
    log = os.path.join(d, os.path.basename(c["log"]))
    shutil.copy(c["log"], log)
    ans = os.path.join(d, "answer.json")
    ev, u, sec, rc, se, _ = opencode(d, MODELS[arm], TRIAGE.format(run=c["run"], end=c["end"], log=log, ans=ans),
                                     timeout=1200)
    try:
        a = json.load(open(ans))
    except (OSError, ValueError):
        a = None
    r = {"family": "triage", "task": case, "arm": arm, "model": MODELS[arm], "key": c["key"], "answer": a,
         "first_pass": triage_grade(c["key"], a), "usage": u, "seconds": round(sec, 1), "rc": rc,
         "stderr": se[-300:], "log_bytes": c["bytes"]}
    r["final_pass"] = r["first_pass"]
    json.dump(ev, open(os.path.join(d, "events.json"), "w"))
    record(r)
    print(json.dumps({k: r[k] for k in ("task", "arm", "first_pass", "seconds")}))


# ---------------------------------------------------------------- the timed Sonnet reference
def sonnet_timed():
    """The 10 fixes rerun by the Sonnet mechanical-worker (agents in sonnet_timed.json, worktrees
    zz_trial_<task>_sonnettime, model_trial brief and check): first-attempt pass, wall time (first to last call
    of its transcript), tokens (input + output + cache read + cache write, one count per API message), $."""
    from lab_llm_report import transcript
    mt.ROOT = MAIN                                  # the trial worktrees hang off the main checkout
    ids = json.load(open(os.path.join(OUT, "sonnet_timed.json")))
    for task, agent in ids.items():
        sec, tok = transcript(agent)
        c = mt.check(task, "sonnettime")
        record({"family": "fix", "task": task, "arm": "sonnet_timed", "model": "sonnet (mechanical-worker)",
                "agent": agent, "first_pass": c["passed"], "final_pass": c["passed"], "uncommitted": c["uncommitted"],
                "seconds": round(sec, 1), "our_tokens": tok,
                "usd": mt.cost(agent)["usd"], "usage": {}, "rc": 0})
        print(task, c["passed"], round(sec), tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cases", "smoke", "fix", "triage", "_grade", "report", "sonnet_timed"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--plan")
    a = ap.parse_args()
    if a.cmd == "cases":
        cases()
    elif a.cmd == "smoke":
        smoke(a.args[0])
    elif a.cmd == "fix":
        fix(a.args[0], a.args[1], a.plan)
    elif a.cmd == "triage":
        triage(a.args[0], a.args[1])
    elif a.cmd == "_grade":
        _grade(a.args[0], a.args[1])
    elif a.cmd == "sonnet_timed":
        sonnet_timed()
    elif a.cmd == "report":
        import lab_llm_report
        lab_llm_report.main()


if __name__ == "__main__":
    main()
