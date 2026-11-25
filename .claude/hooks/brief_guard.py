#!/usr/bin/env python3
"""PreToolUse(Agent), main session only (agent redesign D4). A brief to an agent starts from plan/step_prompt.txt
(`plan brief <entry>`); the main session's addition may narrow the task, never override its capacity and scope rules
(CLAUDE.md). An Agent launch is refused (exit 2, the offending lines) when
  - its prompt lacks the template's rules block (between "=== RULES" and "=== END RULES ===") verbatim, the entry id
    filled in.
No wording check of the added text for cuts (D562 audit: its 4 refusals 2026-11-03..22 were all misfires; the design
reviewer's scope check and submit_guard's PASS requirement hold a cut before any job).
Subagents' launches are checked too (an owner is never split into more agents for its entry; repo-reviewer
2026-10-30). Exempt: the reviewers with their own definitions (design-reviewer, repo-reviewer, notes-editor, claudemd-keeper), read-only search and planning agents
(Explore, Plan) and the harness's helpers (claude-code-guide, statusline-setup), and a resumed agent (SendMessage is
another tool). A launch whose `model` differs from its subagent_type definition's `model:` (.claude/agents/<type>.md,
then ~/.claude/agents/) is refused for every type, the exempt ones included: the definition is the decided model (three
owners went out on sonnet on 2026-10-30 against definitions saying opus; nobody decided it). No `model` given passes.
Also exempt: a cold read, a prompt starting with the line `COLD READ:` that names no `site submit|run|copy|sync`,
`git commit|push` or other writing command.
Python on purpose: a crash exits 1, which blocks nothing."""
import json, os, re, sys

ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATE = os.path.join(ROOT, "plan", "step_prompt.txt")
EXEMPT = {"design-reviewer", "repo-reviewer", "notes-editor", "claudemd-keeper", "Explore", "Plan",
          "claude-code-guide", "statusline-setup"}
BEGIN, END = "=== RULES", "=== END RULES ==="
# a mechanical-worker (D213) may instead carry the short block of plan/mechanical_prompt.txt (`plan brief <entry>
# --mechanical "<step>"`)
MECH_TYPE, MECH_TEMPLATE = "mechanical-worker", os.path.join(ROOT, "plan", "mechanical_prompt.txt")
MBEGIN, MEND = "=== MECHANICAL RULES", "=== END MECHANICAL RULES ==="


def block():
    t = open(TEMPLATE).read()
    i, j = t.find(BEGIN), t.find(END)
    return t[i:j + len(END)] if i >= 0 and j > i else None


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def block_re(b):
    """The block as a regex over whitespace-normalised text, {id} filled with an entry id."""
    return re.compile(re.escape(norm(b)).replace(re.escape("{id}"), r"[A-Za-z0-9_-]+"))



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
    if "TRIAL=model" in (ti.get("description") or ""):
        return None        # the matched model trial (agent_redesign step 4, D203; exemption D205): only the model check
    if want and given and given != want:
        return ("brief_guard: model %r differs from the %s definition's model %r (.claude/agents/%s.md). Leave `model` "
                "out (the definition's model is used) or pass %r; another model is Joaquin's call.\n"
                % (given, ti.get("subagent_type"), want, ti.get("subagent_type"), want))
    return None


COLD = "COLD READ:"
# a cold read (weekly-deck skill, Delivering): a fresh agent given page images only, listing unclear labels; it does
# no entry work, so it carries no rules block, but it may only read: a prompt naming a state-changing command is refused
COLD_WRITES = r"\bsite\s+(?:submit|run|copy|sync|decide|ack)\b|\bgit\s+(?:commit|push|add|checkout|merge)\b|\bplan(?:\.py)?\s+run\b|\bsed\s+-i\b"


def cold_read(prompt):
    """True for a prompt starting with the marker line `COLD READ:` that names no submit, commit or edit command."""
    return prompt.lstrip().startswith(COLD) and not re.search(COLD_WRITES, prompt)


def mech_check(prompt):
    """A mechanical-worker brief: the mechanical rules block verbatim ({id} filled)."""
    try:
        t = open(MECH_TEMPLATE).read()
    except OSError:          # no template, no short block to accept: the owner block is then required
        sys.stderr.write("brief_guard: plan/mechanical_prompt.txt is missing; a mechanical-worker brief needs the "
                         "owner rules block (`plan brief <entry>`).\n")
        return 2
    i, j = t.find(MBEGIN), t.find(MEND)
    mb = t[i:j + len(MEND)]
    m = block_re(mb).search(norm(prompt))
    if not m:
        sys.stderr.write("brief_guard: a mechanical-worker brief carries the mechanical rules block of "
                         "plan/mechanical_prompt.txt (or an owner's full rules block). Build it with `python3 "
                         "tools/plan.py brief <entry> --mechanical \"<the step>\"` and pass its output whole.\n")
        return 2
    return 0


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
    if cold_read(prompt):
        return 0
    if ti.get("subagent_type") == MECH_TYPE and not block_re(b).search(norm(prompt)):
        return mech_check(prompt)
    if not block_re(b).search(norm(prompt)):
        sys.stderr.write("brief_guard: this brief lacks the rules block of plan/step_prompt.txt (CLAUDE.md: a brief to "
                         "an agent starts from it). Build it with `python3 tools/plan.py brief <entry> --add "
                         "\"<narrowing>\"` and pass its output whole as the prompt. A read-only review or lookup that "
                         "writes nothing starts its prompt with the line COLD READ: instead.\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
