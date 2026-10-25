#!/usr/bin/env python3
"""The hook registration of the setup redesign (D121), applied by tools/switch_over.sh to .claude/settings.json.
  retired   plan_guard.py, continuous_guard.py, continuous_list_guard.py (their list, .claude/continuous.json, is
            replaced by the plan), slurm_waiter_guard.sh: what they enforced is plan_check.py's
  added     Stop plan_check.py (timeout 300 s, above its item checks' 90 s); PreToolUse plan_files_guard.py
            (Edit|Write|MultiEdit|NotebookEdit and Bash); PreToolUse git_verify_guard.py (Bash and Edit|Write|MultiEdit)
  changed   SessionStart runs context_guard.py for every source, not only after a compaction
Everything else (the correctness guards, review_backlog, reply_check, auto_push...) stays as it is.
    python3 tools/switch_settings.py [--settings .claude/settings.json] [--apply]    (without --apply: the diff)"""
import argparse, copy, difflib, json, os, sys

RETIRED = ("plan_guard.py", "continuous_guard.py", "continuous_list_guard.py", "slurm_waiter_guard.sh")
H = "$CLAUDE_PROJECT_DIR/.claude/hooks/"
ADD = [("Stop", None, 'python3 "%splan_check.py"' % H, 300),
       ("PreToolUse", "Edit|Write|MultiEdit|NotebookEdit", 'python3 "%splan_files_guard.py"' % H, None),
       ("PreToolUse", "Bash", 'python3 "%splan_files_guard.py"' % H, None),
       ("PreToolUse", "Edit|Write|MultiEdit", 'python3 "%sgit_verify_guard.py"' % H, None),
       ("PreToolUse", "Bash", 'python3 "%sgit_verify_guard.py"' % H, 120)]


def switched(s):
    s = copy.deepcopy(s)
    hooks = s.setdefault("hooks", {})
    for ev, groups in hooks.items():
        for g in groups:
            g["hooks"] = [h for h in g["hooks"] if not any(r in h.get("command", "") for r in RETIRED)]
        hooks[ev] = [g for g in groups if g["hooks"]]
    for ev, matcher, cmd, timeout in ADD:
        groups = hooks.setdefault(ev, [])
        if any(cmd == h.get("command") for g in groups if g.get("matcher", "") == (matcher or "") for h in g["hooks"]):
            continue
        h = {"type": "command", "command": cmd}
        if timeout:
            h["timeout"] = timeout
        g = {"matcher": matcher, "hooks": [h]} if matcher else {"hooks": [h]}
        groups.append(g)
    for g in hooks.get("SessionStart", []):
        if g.get("matcher") == "compact" and any("context_guard.py" in h.get("command", "") for h in g["hooks"]):
            g.pop("matcher")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                       ".claude", "settings.json"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    old = json.load(open(a.settings))
    new = switched(old)
    o, n = json.dumps(old, indent=2).splitlines(), json.dumps(new, indent=2).splitlines()
    if not a.apply:
        print("\n".join(difflib.unified_diff(o, n, a.settings, a.settings + " (switched)", lineterm="")))
        return 0
    tmp = a.settings + ".tmp"
    open(tmp, "w").write(json.dumps(new, indent=2) + "\n")
    os.replace(tmp, a.settings)
    print("switch_settings: %s switched" % a.settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
