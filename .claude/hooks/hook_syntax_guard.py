#!/usr/bin/env python3
"""A broken hook script must never go live (2026-10-23: an in-place edit left review_backlog.sh unparseable; every
Bash, Edit and Write call runs it first, so nothing could undo it from inside the session).

Python on purpose: an exception here exits 1, which Claude Code treats as a non-blocking error, while a bash syntax
error exits 2 and blocks every tool. Modes:
  pre   (PreToolUse, Edit|Write): the content an Edit or Write would leave in .claude/hooks/*.sh is checked with
        `bash -n` before it is written; a failure is refused (exit 2) and nothing is written.
  snap  (PreToolUse, Bash): a copy of every hook script that parses is kept in .claude/.hook_backup/ (the last good
        version), since a shell command's effect on a file cannot be checked before it runs.
  post  (PostToolUse, Edit|Write|Bash): every .claude/hooks/*.sh is checked with `bash -n`; one that does not parse is
        put back from .claude/.hook_backup/ (or from git HEAD), and the turn is told (exit 2 with the reason).
HOOK_GUARD_DIR overrides the hooks directory (tests only).
"""
import json, os, shutil, subprocess, sys, tempfile

ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOOKS = os.environ.get("HOOK_GUARD_DIR") or os.path.join(ROOT, ".claude", "hooks")
BACKUP = os.path.join(os.path.dirname(HOOKS), ".hook_backup")


def parses(path):
    return subprocess.run(["bash", "-n", path], capture_output=True, text=True)


def scripts():
    return sorted(os.path.join(HOOKS, f) for f in os.listdir(HOOKS) if f.endswith(".sh"))


def in_hooks(fp):
    return bool(fp) and fp.endswith(".sh") and os.path.dirname(os.path.realpath(fp)) == os.path.realpath(HOOKS)


def pre(inp):
    ti = inp.get("tool_input", {})
    fp = ti.get("file_path", "")
    if not in_hooks(fp):
        return 0
    if "content" in ti:                                  # Write
        new = ti["content"]
    elif "edits" in ti:                                  # MultiEdit: the edits in order
        new = open(fp).read()
        for e in ti["edits"]:
            a, b = e.get("old_string", ""), e.get("new_string", "")
            new = new.replace(a, b) if e.get("replace_all") else new.replace(a, b, 1)
    else:                                                # Edit
        old = open(fp).read()
        a, b = ti.get("old_string", ""), ti.get("new_string", "")
        new = old.replace(a, b) if ti.get("replace_all") else old.replace(a, b, 1)
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as t:
        t.write(new)
    r = parses(t.name); os.unlink(t.name)
    if r.returncode != 0:
        print(f"BLOCKED by hook_syntax_guard: this edit would leave {os.path.basename(fp)} unparseable, and every tool "
              f"call runs it. bash -n says:\n{r.stderr.strip()}\nFix the edit; nothing was written.", file=sys.stderr)
        return 2
    return 0


def snap():
    os.makedirs(BACKUP, exist_ok=True)
    for p in scripts():
        if parses(p).returncode == 0:
            shutil.copy2(p, os.path.join(BACKUP, os.path.basename(p)))
    return 0


def post():
    bad = []
    for p in scripts():
        r = parses(p)
        if r.returncode == 0:
            continue
        b = os.path.join(BACKUP, os.path.basename(p))
        if os.path.exists(b) and parses(b).returncode == 0:
            shutil.copy2(b, p); how = "the last good copy"
        else:
            rel = os.path.relpath(p, ROOT)
            g = subprocess.run(["git", "-C", ROOT, "show", f"HEAD:{rel}"], capture_output=True, text=True)
            if g.returncode != 0:
                bad.append(f"{os.path.basename(p)}: unparseable and no good copy to restore ({r.stderr.strip()})"); continue
            open(p, "w").write(g.stdout); how = "git HEAD"
        bad.append(f"{os.path.basename(p)}: did not parse ({r.stderr.strip()}); restored from {how}")
    if bad:
        print("hook_syntax_guard: a hook script was left broken and has been put back:\n" + "\n".join(bad), file=sys.stderr)
        return 2
    return 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        inp = json.loads(sys.stdin.read() or "{}")
    except Exception:
        inp = {}
    if mode == "pre":
        return pre(inp)
    if mode == "snap":
        return snap()
    if mode == "post":
        return post()
    return 0


if __name__ == "__main__":
    sys.exit(main())
