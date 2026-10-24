#!/usr/bin/env python3
"""A continuous task leaves .claude/continuous.json only on the user's recorded word (2026-10-24, the user: "this has
to be the fifth time I ask for you to keep working, and you just don't do it"). The list was emptied on 2026-10-22 by
the agent and never refilled, so continuous_guard.py had nothing to enforce while the level work sat idle for hours.

A task is never deleted from the list: it is stopped by setting "stopped": "D<n>", where D<n> is a hold, release or
forbid decision in force (`site decisions --project FA`, which records only the user's own words). continuous_guard.py
skips a stopped task.
  pre   (PreToolUse, Edit|Write): the file an Edit or Write would leave is checked; a task that vanished, or a "stopped"
        that names no hold/release/forbid decision, is refused (exit 2) and nothing is written.
  post  (PostToolUse, Bash): the same check on the file on disk; a violating shell write is undone from git HEAD (exit 2).
Python on purpose: a crash exits 1, which does not block.
"""
import json, os, re, subprocess, sys

ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONF = os.path.join(ROOT, ".claude", "continuous.json")


def tasks(text):
    return {t["name"]: t for t in (json.loads(text or "{}").get("tasks") or [])}


def stop_decisions():
    out = subprocess.run(["site", "decisions", "--project", "FA"], capture_output=True, text=True, timeout=60).stdout
    return set(re.findall(r"^\s*(D\d+)\s+(?:FA\s+)?(?:FA/)?(?:hold|release|forbid)\b", out, re.M))


def violations(old_text, new_text):
    old, new = tasks(old_text), tasks(new_text)
    bad = [f"task '{n}' removed (set \"stopped\": \"D<n>\" with the user's hold/release decision instead)"
           for n in old if n not in new]
    named = {n: t["stopped"] for n, t in new.items() if t.get("stopped") and old.get(n, {}).get("stopped") != t["stopped"]}
    if named:
        ok = stop_decisions()
        bad += [f"task '{n}' stopped under {d}, which is not a hold/release/forbid decision in force" for n, d in named.items()
                if d not in ok]
    return bad


def head_text():
    r = subprocess.run(["git", "-C", ROOT, "show", "HEAD:.claude/continuous.json"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else '{"tasks": []}'


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        inp = json.loads(sys.stdin.read() or "{}")
    except Exception:
        inp = {}
    if mode == "pre":
        ti = inp.get("tool_input") or {}
        if os.path.realpath(ti.get("file_path", "")) != os.path.realpath(CONF):
            return 0
        cur = open(CONF).read() if os.path.exists(CONF) else '{"tasks": []}'
        if "content" in ti:
            new = ti["content"]
        else:
            new = cur
            for e in ti.get("edits") or [ti]:
                a, b = e.get("old_string", ""), e.get("new_string", "")
                new = new.replace(a, b) if e.get("replace_all") else new.replace(a, b, 1)
        bad = violations(cur, new)
        if bad:
            print("BLOCKED by continuous_list_guard:\n  " + "\n  ".join(bad), file=sys.stderr)
            return 2
        return 0
    if mode == "post":
        if not os.path.exists(CONF):
            return 0
        old = head_text()
        bad = violations(old, open(CONF).read())
        if bad:
            open(CONF, "w").write(old)
            print("continuous_list_guard: .claude/continuous.json restored from HEAD:\n  " + "\n  ".join(bad), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
