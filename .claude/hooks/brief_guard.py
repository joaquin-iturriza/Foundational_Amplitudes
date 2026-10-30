#!/usr/bin/env python3
"""PreToolUse(Agent), main session only (agent redesign D4). A brief to an agent starts from plan/step_prompt.txt
(`plan brief <entry>`); the main session's addition may narrow the task, never override its capacity and scope rules
(CLAUDE.md). An Agent launch is refused (exit 2, the offending lines) when
  - its prompt lacks the template's rules block (between "=== RULES" and "=== END RULES ===") verbatim, the entry id
    filled in; or
  - its added text (the narrowing slot, and anything outside the template) cuts the work: fewer or only N jobs, seeds,
    arms, workers, trials, probes, parents or pools; another horizon or pool; a subset; skipping, or going without or
    instead of, the design, its review or `site pick`.
Subagents' launches are checked too (an owner is never split into more agents for its entry; repo-reviewer
2026-10-30). Exempt: the reviewers with their own definitions (design-reviewer, repo-reviewer, notes-editor, claudemd-keeper), read-only search and planning agents
(Explore, Plan) and the harness's helpers (claude-code-guide, statusline-setup), and a resumed agent (SendMessage is
another tool). A launch whose `model` differs from its subagent_type definition's `model:` (.claude/agents/<type>.md,
then ~/.claude/agents/) is refused for every type, the exempt ones included: the definition is the decided model (three
owners went out on sonnet on 2026-10-30 against definitions saying opus; nobody decided it). No `model` given passes.
A cut the user asked for is quoted in the brief as his words in double quotes: a quote he typed
(tools/plan.py typed_at) is not read, an invented one is.
Python on purpose: a crash exits 1, which blocks nothing."""
import json, os, re, sys

ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATE = os.path.join(ROOT, "plan", "step_prompt.txt")
EXEMPT = {"design-reviewer", "repo-reviewer", "notes-editor", "claudemd-keeper", "Explore", "Plan",
          "claude-code-guide", "statusline-setup"}
BEGIN, END = "=== RULES", "=== END RULES ==="
N = r"(?:\d+|one|two|three|four|five|a few|some|half)"
UNITS = r"(?:jobs?|seeds?|arms?|workers?|trials?|probes?|parents?|pools?|horizons?|energies|cells?|processes)"
CUTS = [r"\b(?:fewer|less)\s+(?:\w+\s+)?" + UNITS + r"\b",
        r"\bonly\s+(?:the\s+first\s+)?" + N + r"\s+(?:of\s+the\s+)?(?:\w+\s+)?" + UNITS + r"\b",
        # "a subset per job" splits the work; a subset of the design's units cuts it (the night's replay, 2026-10-30)
        r"\b(?:run|use|take|only)\s+a\s+subset\b|\ba\s+subset\s+of\s+(?:the\s+)?(?:\w+\s+)?" + UNITS + r"\b",
        r"\b(?:another|other|different|shorter|smaller)\s+(?:horizon|pool)\b",
        r"\bskip(?:ping|s)?\s+(?:the\s+)?(?:design|review|reviewer|probe|site pick)",
        r"\b(?:without|no need for|instead of)\s+(?:a\s+|the\s+)?(?:design|review|reviewer|site pick)",
        r"\b(?:do not|don't|no need to)\s+(?:run|launch|wait for)\s+(?:the\s+)?(?:design\s+)?review"]


def block():
    t = open(TEMPLATE).read()
    i, j = t.find(BEGIN), t.find(END)
    return t[i:j + len(END)] if i >= 0 and j > i else None


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def block_re(b):
    """The block as a regex over whitespace-normalised text, {id} filled with an entry id."""
    return re.compile(re.escape(norm(b)).replace(re.escape("{id}"), r"[A-Za-z0-9_-]+"))


def added_text(prompt, tpl):
    """The text the main session added: the narrowing slot's content and whatever lies outside the template's frame
    (before its first line, after its last)."""
    out = []
    m = re.search(r"never overrides the rules block below\):\n(.*?)\n\s*" + re.escape(BEGIN), prompt, re.S)
    if m:
        out.append(m.group(1))
    first, last = tpl.strip().splitlines()[0][:40], tpl.strip().splitlines()[-1][-40:]
    i, j = prompt.find(first), prompt.rfind(last)
    if i > 0:
        out.append(prompt[:i])
    if j >= 0:
        out.append(prompt[j + len(last):])
    return "\n".join(out)


def users_words(q):
    """A quoted span is exempt only when the user typed it (tools/plan.py typed_at: site's own search, the one
    `site decide` uses). An invented quote is read like any other text (repo-reviewer, 2026-10-30)."""
    try:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import plan
        return bool(plan.typed_at(q))
    except BaseException:        # site's search exits on a quote too short to identify a message: not his words
        return False


def cuts(text):
    text = re.sub(r"\"([^\"\n]*)\"|“([^”]*)”",
                  lambda m: " " if users_words(m.group(1) or m.group(2)) else m.group(0), text)
    bad = []
    for line in text.splitlines():
        if any(re.search(p, line, re.I) for p in CUTS) and line.strip().lower() != "none":
            bad.append(line.strip())
    return bad


def definition_model(agent_type):
    """The `model:` of an agent type's definition frontmatter (project first, then user level); None without one."""
    if not agent_type or not re.fullmatch(r"[A-Za-z0-9_.-]+", agent_type):
        return None
    for d in (os.path.join(ROOT, ".claude", "agents"), os.path.expanduser("~/.claude/agents")):
        try:
            text = open(os.path.join(d, agent_type + ".md")).read()
        except OSError:
            continue
        m = re.match(r"---\n(.*?)\n---", text, re.S)
        mm = re.search(r"^model:\s*(\S+)\s*$", m.group(1), re.M) if m else None
        return mm.group(1) if mm else None
    return None


def model_mismatch(ti):
    want, given = definition_model(ti.get("subagent_type")), ti.get("model")
    if want and given and given != want:
        return ("brief_guard: model %r differs from the %s definition's model %r (.claude/agents/%s.md). Leave `model` "
                "out (the definition's model is used) or pass %r; another model is Joaquin's call.\n"
                % (given, ti.get("subagent_type"), want, ti.get("subagent_type"), want))
    return None


def main():
    try:
        inp = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    if not isinstance(inp, dict) or inp.get("tool_name") not in (None, "Agent"):
        return 0
    ti = inp.get("tool_input") or {}
    bad_model = model_mismatch(ti)
    if bad_model:
        sys.stderr.write(bad_model)
        return 2
    if ti.get("subagent_type") in EXEMPT:
        return 0
    prompt, b = ti.get("prompt") or "", block()
    if b is None:
        return 0
    if not block_re(b).search(norm(prompt)):
        sys.stderr.write("brief_guard: this brief lacks the rules block of plan/step_prompt.txt (CLAUDE.md: a brief to "
                         "an agent starts from it). Build it with `python3 tools/plan.py brief <entry> --add "
                         "\"<narrowing>\"` and pass its output whole as the prompt.\n")
        return 2
    bad = cuts(added_text(prompt, open(TEMPLATE).read()))
    if bad:
        sys.stderr.write("brief_guard: the added text overrides the capacity and scope rules (a cut of the design is "
                         "the user's call, asked before the configs are made):\n" + "\n".join("  " + x for x in bad)
                         + "\nNarrow what the agent does now without cutting the design, or quote the user's words "
                         "that make the cut.\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
