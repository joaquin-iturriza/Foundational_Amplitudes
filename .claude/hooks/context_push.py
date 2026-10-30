#!/usr/bin/env python3
"""Pushes reference sections into an agent's context (agent redesign D3; design in the scratchpad's
hook_push.draft.txt, kept). The rules files hold rules only; the reference for one kind of work is one section,
`.claude/context/<name>.txt` (FA) or `~/.claude/context/<name>.txt` (cross-project): a header of triggers, `---`, the
text. A section reaches each agent once, the first time a call of it matches a trigger:
  paths:    globs relative to the project (or ~/...), matched against the file a tool reads or writes and against
            every path-like word of a Bash command
  cmds:     Python regexes searched in a Bash command with quoted strings and data heredoc bodies blanked
            (cmdparse.scan_text: a commit message naming sweep_manager.py triggers nothing)
  tools:    tool names (Agent, SendMessage, SendUserFile)
  session:  main: pushed to the main session at SessionStart
Registered as `context_push.py pre` (PreToolUse on acts: Bash, writes, Agent, SendMessage, SendUserFile: the section
arrives before the act), `post` (PostToolUse on reads and acts) and `session` (SessionStart). Output: additionalContext.
Once per agent per section, keyed by agent_id ('main' for the main session) and the section's hash, in
.claude/.context_pushed/<session>.json under a lock: an edited section is pushed again; a compaction or /clear
(SessionStart source compact|clear) re-arms the main session. Never blocks: notes only, a crash exits 1.
Env for tests: CONTEXT_DIRS (':'-separated section directories), CONTEXT_STATE (the state directory)."""
import fcntl, glob, hashlib, json, os, re, sys, time

HOOKS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HOOKS)
ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(HOOKS))
HOME = os.path.expanduser("~")
DIRS = (os.environ.get("CONTEXT_DIRS") or "%s:%s" % (os.path.join(ROOT, ".claude", "context"),
                                                     os.path.join(HOME, ".claude", "context"))).split(":")
STATE = os.environ.get("CONTEXT_STATE") or os.path.join(ROOT, ".claude", ".context_pushed")
KEEP_S = 7 * 24 * 3600


def sections():
    """[(name, {paths, cmds, tools, session}, text)] in file order, project first."""
    out = []
    for d in DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.txt"))):
            if os.path.basename(f).startswith("sections"):     # the combined source, not a section
                continue
            raw = open(f, errors="replace").read()
            if "\n---\n" not in raw:
                continue
            head, text = raw.split("\n---\n", 1)
            h = {}
            for line in head.splitlines():
                k, _, v = line.partition(":")
                if k.strip() in ("paths", "cmds", "tools", "session"):
                    sep = "," if k.strip() in ("paths", "tools") else None
                    h[k.strip()] = [x.strip() for x in (v.split(sep) if sep else [v]) if x.strip()]
            if h.get("cmds"):              # a regex may hold commas: one regex per comma-separated item, as written
                h["cmds"] = [x.strip() for x in re.split(r",\s+(?=\\|\(|\w)", h["cmds"][0]) if x.strip()]
            out.append((os.path.basename(f)[:-4], h, text.strip() + "\n"))
    return out


def glob_re(g):
    g = g.strip()
    home = g.startswith("~/")
    g = g[2:] if home else g
    r, i = "", 0
    while i < len(g):
        if g.startswith("**/", i):
            r, i = r + "(?:.*/)?", i + 3
        elif g.startswith("**", i):
            r, i = r + ".*", i + 2
        elif g[i] == "*":
            r, i = r + "[^/]*", i + 1
        elif g[i] == "?":
            r, i = r + "[^/]", i + 1
        else:
            r, i = r + re.escape(g[i]), i + 1
    return home, re.compile(r + r"\Z")


def rel(p):
    """(relative to the project or None, relative to HOME or None) of a path."""
    p = os.path.normpath(os.path.join(ROOT, os.path.expanduser(p))) if p else ""
    r = os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else None
    # a worktree under the project is the project: worktrees/<name>/<path> matches <path> too
    if r and r.startswith("worktrees/") and r.count("/") >= 2:
        r = r.split("/", 2)[2]
    h = os.path.relpath(p, HOME) if p.startswith(HOME + os.sep) else None
    return r, h


def path_hit(globs, paths):
    for g in globs:
        home, rx = glob_re(g)
        for p in paths:
            r, h = rel(p)
            if (h if home else r) is not None and rx.match(h if home else r):
                return True
    return False


def call_paths(tool, ti):
    ps = [ti.get(k) for k in ("file_path", "notebook_path", "path") if isinstance(ti.get(k), str)]
    if tool == "Glob" and isinstance(ti.get("pattern"), str):
        ps.append(os.path.join(ti.get("path") or ROOT, ti["pattern"]))
    return ps


def matches(tool, ti, secs):
    text, words = "", []
    if tool == "Bash":
        import cmdparse
        try:
            text = cmdparse.scan_text(ti.get("command") or "")
        except Exception:
            text = ti.get("command") or ""
        # every path-like run of the text, inside python -c programs too: a/b/c, or a name with a known suffix
        words = re.findall(r"[\w.~-]*(?:/[\w.*-]+)+|[\w.-]+\.(?:py|sh|yaml|yml|json|txt|tex|toml)\b", text)
    paths = call_paths(tool, ti) + words
    out = []
    for name, h, body in secs:
        hit = tool in (h.get("tools") or []) or path_hit(h.get("paths") or [], paths) or \
            (text and any(re.search(c, text) for c in h.get("cmds") or []))
        if hit:
            out.append((name, body))
    return out


def deliver(session, who, found, reset=False):
    """The sections of `found` not yet delivered to `who` (by hash), marked delivered."""
    os.makedirs(STATE, exist_ok=True)
    for f in glob.glob(os.path.join(STATE, "*.json")):
        if time.time() - os.path.getmtime(f) > KEEP_S:
            try:
                os.unlink(f)
            except OSError:
                pass
    p = os.path.join(STATE, re.sub(r"[^\w-]", "_", session or "nosession") + ".json")
    with open(p + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            st = json.load(open(p))
        except (OSError, ValueError):
            st = {}
        if reset:
            st.pop(who, None)
        mine = st.setdefault(who, {})
        new = []
        for name, body in found:
            h = hashlib.sha1(body.encode()).hexdigest()
            if mine.get(name) != h:
                mine[name] = h
                new.append((name, body))
        tmp = p + ".tmp"
        json.dump(st, open(tmp, "w"))
        os.replace(tmp, p)
    return new


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "post"
    try:
        inp = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    if not isinstance(inp, dict):
        return 0
    secs = sections()
    who = inp.get("agent_id") or "main"
    ev = inp.get("hook_event_name") or {"pre": "PreToolUse", "post": "PostToolUse", "session": "SessionStart"}[mode]
    if mode == "session" or ev == "SessionStart":
        if inp.get("agent_id"):
            return 0
        found = [(n, b) for n, h, b in secs if "main" in " ".join(h.get("session") or [])]
        new = deliver(inp.get("session_id"), "main", found, reset=inp.get("source") in ("compact", "clear"))
        ev = "SessionStart"
    else:
        found = matches(inp.get("tool_name") or "", inp.get("tool_input") or {}, secs)
        new = deliver(inp.get("session_id"), who, found) if found else []
    if new:
        text = "\n\n".join("REFERENCE %s (pushed once; rules stay in CLAUDE.md)\n%s" % (n, b) for n, b in new)
        print(json.dumps({"hookSpecificOutput": {"hookEventName": ev, "additionalContext": text}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
