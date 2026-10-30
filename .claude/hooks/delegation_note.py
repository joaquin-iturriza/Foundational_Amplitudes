#!/usr/bin/env python3
"""PostToolUse(Bash|Edit|Write|MultiEdit|NotebookEdit), main session only: the note-only delegation hook (agent
redesign; Joaquin kept the call limit main_session_guard.py unregistered, D185). The main session talks to the user
and directs agents; it never does an entry's work itself. When it edits code, runs an analysis or a test, or
submits, this hook adds a note to its context (never a refusal) and counts the call in .claude/.main_work.jsonl, which
`site digest` reports. What counts:
  edit      Edit/Write/MultiEdit/NotebookEdit of code or results: *.py, *.sh, *.ipynb, docs/*.tex, and files under
            config/, recipes/, sweep/, models/, analysis/ (not CLAUDE.md, settings, the scratchpad or /tmp)
  analysis  a Bash command running a script under analysis/ or tools/ (the plan, incident, context, settings and
            CLAUDE.md tools excepted), a notebook, or tests (pytest, tests/*.py)
  submit    site submit|launch|queue add, `site run ... --item`, sbatch, condor_submit, scripts/launch_*.sh,
            sweep_manager.py
Python on purpose: a crash exits 1, which blocks nothing. Env for tests: MAIN_WORK_LOG."""
import json, os, re, sys, time

HOOKS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HOOKS)
ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(HOOKS))
LOG = os.environ.get("MAIN_WORK_LOG") or os.path.join(ROOT, ".claude", ".main_work.jsonl")
CODE = re.compile(r"(\.(py|sh|ipynb)$|^docs/[^/]+\.tex$|^(config|recipes|sweep|models|analysis)/)")
TOOLS_OK = r"(plan|hook_incident|agent_context|switch_settings|claudemd_check|review_state)\.py"
ANALYSIS = re.compile(r"(^|[\s;&|(])(python3?|\S*/python3?|bash|sh)\s+(-\S+\s+)*(\./)?(analysis/\S+|tools/(?!%s)\S+\.py)"
                      r"|\bpytest\b|(^|\s)tests/\S+\.py|\bjupyter\b|\bnbconvert\b" % TOOLS_OK)
SUBMIT = re.compile(r"\bsite\s+(submit|launch|queue\s+add)\b|\bsite\s+run\b.*--item\b|\bsbatch\b|\bcondor_submit\b"
                    r"|\bscripts/launch_\S+\.sh\b|\bsweep_manager\.py\b")


def kind(tool, ti):
    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        p = os.path.abspath(ti.get("file_path") or ti.get("notebook_path") or "")
        if not p.startswith(ROOT + os.sep):
            return None, None
        r = os.path.relpath(p, ROOT)
        if r.startswith("worktrees/") and r.count("/") >= 2:
            r = r.split("/", 2)[2]
        if os.path.basename(r) == "CLAUDE.md" or r.startswith(".claude/settings"):
            return None, None
        return ("edit", r) if CODE.search(r) else (None, None)
    if tool == "Bash":
        import cmdparse
        cmd = ti.get("command") or ""
        try:
            text = cmdparse.scan_text(cmd)
        except Exception:
            text = cmd
        first = cmd.strip().splitlines()[0][:160] if cmd.strip() else ""
        if SUBMIT.search(text):
            return "submit", first
        if ANALYSIS.search(text):
            return "analysis", first
    return None, None


def main():
    try:
        inp = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    if not isinstance(inp, dict) or inp.get("agent_id"):
        return 0
    k, what = kind(inp.get("tool_name") or "", inp.get("tool_input") or {})
    if not k:
        return 0
    with open(LOG, "a") as f:
        f.write(json.dumps({"ts": int(time.time()), "session": inp.get("session_id"), "kind": k, "what": what}) + "\n")
    note = ("delegation note: the main session %s (%s). It talks to the user and directs agents; an entry's work goes "
            "to its owner (plan next: resume or launch). Counted in site digest." % (
                {"edit": "edited code or results", "analysis": "ran an analysis or tests", "submit": "submitted work"}[k],
                what))
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": note}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
