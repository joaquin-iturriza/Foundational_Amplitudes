#!/usr/bin/env python3
"""The Stop conditions plan_check.py takes over (agent redesign, step B1), each returning (refusals, notes):
a refusal is what the turn can clear (save the pdf, flush the notes, answer the question, start the supervisor or the
waiter); a note waits on the outside (an agent, a job, a cluster, Joaquin) and never blocks.

  figure_pairs     figure_pair_guard.sh ported: a .png or .pdf changed since the last passing check without its
                   counterpart (same pruned trees, same marker .claude/.figpair_last, same ignore list)
  context          context_guard.py's Stop branch: a compaction is wanted and the notes are not flushed
  reply            reply_check.py: does the reply answer the user's message (once per message, passes when the check
                   cannot run)
  ledger           `site guard --stop`: for a plan_check project, the supervisor and the armed waiter (its open events
                   are plan_check's own); ~/.claude/hooks/ledger_guard.sh stop defers to this (LEDGER_MARK)
  opener           (not registered, 2026-10-31: the opener is Joaquin's drift signal; forcing it hides the drift)
  incidents        a hook disabled by the dispatcher (.claude/hooks/disabled.json): a note
While a standalone hook is still registered for Stop in the project's settings (before switch-over B), plan_check
skips its condition here, so nothing is asked twice: registered_at_stop()."""
import hashlib, json, os, re, subprocess, sys

HOOKS = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(HOOKS))
LEDGER_MARK = "ledger: site guard --stop"        # ledger_guard.sh stop exits 0 for a project whose plan_check has it
OPENER = "Joaquin"
PRUNE = {"runs", "sweeps", "data", "conda", ".venv", ".git", "worktrees", "compare_models", "IntrinsicDimDeep",
         "scratchpad"}


def registered_at_stop(name, repo=REPO):
    """the standalone hook `name` is still registered for Stop in the project's settings."""
    try:
        s = json.load(open(os.path.join(repo, ".claude", "settings.json")))
    except (OSError, ValueError):
        return False
    return any(re.search(r"(?:^|[/\s\"])%s(?:\"|\s|$)" % re.escape(name), h.get("command", ""))
               for g in (s.get("hooks") or {}).get("Stop") or [] for h in g.get("hooks") or [])


# ------------------------------------------------------------------ figure pairs
def _pruned(dirpath, d, top):
    rel = os.path.relpath(os.path.join(dirpath, d), top)
    return (d in PRUNE or d.startswith(".venv-") or d.startswith("wt-")) and (
        rel.count(os.sep) == 0 or d in ("scratchpad", "IntrinsicDimDeep"))


def figure_pairs(repo=REPO):
    marker = os.path.join(repo, ".claude", ".figpair_last")
    ignore = os.path.join(repo, ".claude", "figure_pair_ignore.txt")
    if not os.path.isdir(os.path.dirname(marker)):      # no .claude here (a test world with no plan): nothing to do
        return [], []
    if not os.path.exists(marker):                      # first run: a baseline, nothing flagged retroactively
        open(marker, "w").close()
        return [], []
    t0 = os.path.getmtime(marker)
    ign = set()
    if os.path.exists(ignore):
        ign = {l.strip() for l in open(ignore) if l.strip()}
    missing = set()
    for dirpath, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if not _pruned(dirpath, d, repo)]
        for f in files:
            if not f.endswith((".png", ".pdf")):
                continue
            p = os.path.join(dirpath, f)
            try:
                if os.path.getmtime(p) <= t0:
                    continue
            except OSError:
                continue
            rel = os.path.relpath(p, repo)
            if rel in ign or f in ign:
                continue
            stem = rel[:-4]
            for ext in (".png", ".pdf"):
                if not os.path.exists(os.path.join(repo, stem + ext)):
                    missing.add(stem + ext)
    if missing:                                         # the marker stays: the same window is re-checked after the fix
        return ["a figure changed this turn lacks its other format (every figure is saved as both .png and .pdf): "
                + ", ".join(sorted(missing)) + "; generate it, or list a real exception in "
                ".claude/figure_pair_ignore.txt"], []
    open(marker, "w").close()
    return [], []


# ------------------------------------------------------------------ context
def context(inp, repo=REPO):
    wanted = os.path.join(repo, ".claude", ".compact_wanted")
    flushed = os.path.join(repo, ".claude", ".notes_flushed")
    if inp.get("stop_hook_active") or not os.path.exists(wanted) or os.path.exists(flushed):
        return [], []
    sys.path.insert(0, HOOKS)
    import context_guard
    return [context_guard.INSTR], []


# ------------------------------------------------------------------ reply
def reply(inp):
    """reply_check.py's verdict (it keeps its own once-per-message state and passes when it cannot run)."""
    sys.path.insert(0, HOOKS)
    import reply_check
    rc, msg = reply_check.check(inp)
    return ([msg] if rc == 2 else []), []


# ------------------------------------------------------------------ ledger
def ledger_deferred(path=os.path.expanduser("~/.claude/hooks/ledger_guard.sh")):
    """~/.claude/hooks/ledger_guard.sh stop defers to this project's plan_check (it greps for LEDGER_MARK): only then
    does plan_check ask `site guard --stop` itself, so the two never ask twice."""
    try:
        return "LEDGER_MARK" in open(path).read()
    except OSError:
        return False


def ledger(repo=REPO, site=None):
    site = site or ["site"]
    site = [site] if isinstance(site, str) else list(site)
    try:
        r = subprocess.run(site + ["guard", "--stop", "--project-dir", repo], capture_output=True, text=True,
                           timeout=60)
    except Exception as e:
        return ["the site ledger guard could not run (%r); fix it before ending the turn" % (e,)], []
    if r.returncode == 0:
        return [], []
    if r.returncode != 2:
        return ["site ledger guard failed (exit %d): %s" % (r.returncode, (r.stderr or r.stdout).strip()[-300:])], []
    lines = [l.strip() for l in r.stderr.splitlines() if l.strip() and not l.startswith("site ledger: this turn")]
    return [" ".join(lines)] if lines else ["site ledger guard refused without a reason"], []


# ------------------------------------------------------------------ opener
SKIP = ("<system-reminder", "[SYSTEM NOTIFICATION", "Stop hook feedback", "<task-notification", "<command-",
        "<local-command", "Caveat:", "[Request interrupted", "Another Claude session sent")


def _texts(content):
    if isinstance(content, str):
        return [content]
    return [c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text"]


def turn_of(records):
    """(id of the user message that opened the turn, the turn's last assistant text) over transcript records."""
    prompt, last = None, None
    for r in records:
        m = r.get("message") or {}
        if r.get("type") == "user" and not r.get("isMeta") and not r.get("isCompactSummary") \
                and r.get("toolUseResult") is None:
            ts = [t.strip() for t in _texts(m.get("content")) if t.strip()]
            if ts and not ts[0].startswith(SKIP) and not any(isinstance(c, dict) and c.get("type") == "tool_result"
                                                             for c in (m.get("content") or []) if not isinstance(m.get("content"), str)):
                prompt, last = r.get("uuid") or hashlib.sha1(ts[0].encode()).hexdigest(), None
        elif r.get("type") == "assistant":
            tx = "".join(_texts(m.get("content"))).strip()
            if tx:
                last = tx
    return prompt, last


def _tail_records(path, nbytes=4 << 20):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - nbytes))
            data = f.read().split(b"\n")[1:]
    except OSError:
        return []
    out = []
    for line in data:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def starts_with_opener(text):
    return re.sub(r"^[\W_]+", "", text or "").startswith(OPENER)


def opener(inp, state, records=None):
    """refused once per user message: the turn's last text does not start with the name. `state` is a dict kept by
    the caller (key "opener": the prompts already refused)."""
    if inp.get("agent_id") or inp.get("agent_type"):
        return [], []
    if records is None:
        records = _tail_records(inp.get("transcript_path") or "")
    prompt, last = turn_of(records)
    last = last or inp.get("last_assistant_message")
    if not last or starts_with_opener(last):
        return [], []
    if not prompt:            # the turn's user message lies beyond the tail read: once per message cannot be kept
        return [], []
    done = state.setdefault("opener", [])
    key = prompt
    if key in done:
        return [], []
    done.append(key)
    del done[:-200]
    return ["the reply does not start with \"%s\" (CLAUDE.md rule 0): start the message with it" % OPENER], []


# ------------------------------------------------------------------ incidents
def incidents(hooks=HOOKS):
    try:
        off = json.load(open(os.path.join(hooks, "disabled.json")))
    except (OSError, ValueError):
        return [], []
    if not off:
        return [], []
    return [], ["hooks disabled by an incident (python3 tools/hook_incident.py --list; --resolve once fixed): " +
                "; ".join("%s since %s: %s" % (h, (v.get("time") or "")[:16], (v.get("why") or "")[:160])
                          for h, v in sorted(off.items()))]
