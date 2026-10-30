#!/usr/bin/env python3
"""The hook registration of the setup redesign (D121), applied by tools/switch_over.sh to .claude/settings.json.
  retired   plan_guard.py, continuous_guard.py, continuous_list_guard.py (their list, .claude/continuous.json, is
            replaced by the plan), slurm_waiter_guard.sh: what they enforced is plan_check.py's
  added     Stop plan_check.py (timeout 300 s, above its item checks' 90 s); PreToolUse plan_files_guard.py
            (Edit|Write|MultiEdit|NotebookEdit and Bash); PreToolUse git_verify_guard.py (Bash and Edit|Write|MultiEdit)
  changed   SessionStart runs context_guard.py for every source, not only after a compaction
Everything else (the correctness guards, review_backlog, reply_check, auto_push...) stays as it is.
    python3 tools/switch_settings.py [--settings .claude/settings.json] [--apply]    (without --apply: the diff)

The agent redesign's switch-overs (impl_order.txt), each from the previous one's registration:
  A   every registration runs through the dispatcher: `X args` becomes `python3 "<dir>/.claude/hooks/dispatch.py" X
      args`, in the project file and in ~/.claude/settings.json (with "$HOME"), nothing added or removed
  B   the hook set: Stop figure_pair_guard.sh, review_backlog.sh check and reply_check.py, PreToolUse(Bash)
      hpo_guard.sh, plot_guard.sh, placement_guard.sh, lock_guard.py, design_guard.py, Pre/PostToolUse(Agent)
      reviewer_resume.py and PostToolUse(Bash) review_backlog.sh bashpost leave; PreToolUse(Bash) submit_guard.py
      (timeout 120) and review_guard.py, Pre/PostToolUse(Agent) review_guard.py come in (project file only)
    python3 tools/switch_settings.py --switch A|B [--patch FILE]     the diff of both files (a patch: patch -d / -p1)
    python3 tools/switch_settings.py --switch A|B --apply            each changed file saved as <file>.pre<A|B> first
    python3 tools/switch_settings.py --revert A|B                    the .pre<A|B> files put back"""
import argparse, copy, difflib, json, os, re, shutil, subprocess, sys, tempfile

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


DISPATCH_P = 'python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/dispatch.py" '
DISPATCH_U = 'python3 "$HOME/.claude/hooks/dispatch.py" '
USER_SETTINGS = os.path.expanduser("~/.claude/settings.json")
HOOK_CMD = re.compile(r'(?:\$CLAUDE_PROJECT_DIR/\.claude/hooks/|~/\.claude/hooks/|\$HOME/\.claude/hooks/)([\w.]+)"?\s*(.*)$')
B_OUT = ("figure_pair_guard.sh", "review_backlog.sh check", "reply_check.py", "placement_guard.sh", "hpo_guard.sh",
         "design_guard.py", "lock_guard.py", "reviewer_resume.py", "review_backlog.sh bashpost")


def _each(s, f):
    s = copy.deepcopy(s)
    for groups in s.get("hooks", {}).values():
        for g in groups:
            for h in g["hooks"]:
                h["command"] = f(h["command"])
    return s


def through_dispatch(s, user=False):
    """A: every hook command run by the dispatcher of its directory; a command already dispatched is kept."""
    def wrap(cmd):
        if "dispatch.py" in cmd:
            return cmd
        m = HOOK_CMD.search(cmd)
        if not m:
            raise SystemExit("switch_settings: cannot read the hook of %r" % cmd)
        return (DISPATCH_U if user else DISPATCH_P) + m.group(1) + ((" " + m.group(2)) if m.group(2) else "")
    return _each(s, wrap)


def hook_set_b(s):
    """B, from A's project file."""
    s = copy.deepcopy(s)
    H = s["hooks"]
    for ev, groups in H.items():
        for g in groups:
            g["hooks"] = [h for h in g["hooks"] if not any(h["command"] == DISPATCH_P + n for n in B_OUT)
                          and not (g.get("matcher") == "Bash" and h["command"] == DISPATCH_P + "plot_guard.sh")]
        H[ev] = [g for g in groups if g["hooks"]]
    for ev, matcher, cmd, timeout in (("PreToolUse", "Bash", "submit_guard.py", 120),
                                      ("PreToolUse", "Bash", "review_guard.py", None),
                                      ("PreToolUse", "Agent", "review_guard.py", None),
                                      ("PostToolUse", "Agent", "review_guard.py", None)):
        if any(h["command"] == DISPATCH_P + cmd for g in H.get(ev, []) if g.get("matcher") == matcher
               for h in g["hooks"]):
            continue
        h = {"type": "command", "command": DISPATCH_P + cmd}
        if timeout:
            h["timeout"] = timeout
        H.setdefault(ev, []).append({"matcher": matcher, "hooks": [h]})
    return s


def _text(obj, path):
    """the file's own JSON style (the user file is ASCII-escaped, the project file is not)."""
    cur = open(path).read()
    ascii_ = json.dumps(json.load(open(path)), indent=2) + "\n" == cur and \
        json.dumps(json.load(open(path)), indent=2, ensure_ascii=False) + "\n" != cur
    return json.dumps(obj, indent=2, ensure_ascii=ascii_) + "\n"


def switch_over(which, project, user):
    """[(path, new text)] of the files the switch-over changes."""
    out = []
    p = json.load(open(project))
    if which == "A":
        out.append((project, _text(through_dispatch(p), project)))
        out.append((user, _text(through_dispatch(json.load(open(user)), user=True), user)))
    elif which == "B":
        out.append((project, _text(hook_set_b(through_dispatch(p)), project)))
    return [(f, t) for f, t in out if t != open(f).read()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--switch", choices=("A", "B"))
    ap.add_argument("--revert", choices=("A", "B"))
    ap.add_argument("--patch", help="--switch: write the diff here as a patch (apply: patch -d / -p1 < FILE)")
    ap.add_argument("--user-settings", default=USER_SETTINGS)
    ap.add_argument("--settings", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                       ".claude", "settings.json"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    if a.revert:
        for f in (a.settings, a.user_settings):
            if os.path.exists(f + ".pre" + a.revert):
                shutil.copy2(f + ".pre" + a.revert, f)
                print("switch_settings: %s restored from %s.pre%s" % (f, f, a.revert))
        return 0
    if a.switch:
        changes = switch_over(a.switch, a.settings, a.user_settings)
        diff = []
        for f, new in changes:
            rel = os.path.abspath(f).lstrip("/")
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as t:
                t.write(new)
            # GNU diff, not difflib: difflib's junk heuristic turns the many "{" and "}" lines into one long hunk
            diff.append(subprocess.run(["diff", "-u", "--label", "a/" + rel, "--label", "b/" + rel, f, t.name],
                                       capture_output=True, text=True).stdout)
            os.unlink(t.name)
        if a.patch:
            open(a.patch, "w").write("".join(diff))
            print("switch_settings: switch-over %s patch written to %s (%d files)" % (a.switch, a.patch, len(changes)))
        elif not a.apply:
            sys.stdout.write("".join(diff))
        if a.apply:
            for f, new in changes:
                shutil.copy2(f, f + ".pre" + a.switch)
                tmp = f + ".tmp"
                open(tmp, "w").write(new)
                os.replace(tmp, f)
                print("switch_settings: %s switched (%s); previous in %s.pre%s" % (f, a.switch, f, a.switch))
        return 0
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
