#!/usr/bin/env python3
"""A subagent's context, read from its transcript (agent redesign D5): the input + cache read + cache creation tokens
of its last model call, and the compaction boundaries in it. The main session reads it after each hand-back of an
owner; at 300k tokens or more (tools/plan.py CONTEXT_CLOSE) the owner is asked to write its hand-off and close.
    python3 tools/agent_context.py <agent id> [--json]
Only subagent transcripts (<session>/subagents/agent-<id>.jsonl) are read, never the main session's."""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("agent")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    c = plan.agent_context(a.agent)
    if c is None:
        sys.exit("agent_context: no subagent transcript for %s" % a.agent)
    c["close"] = (c["tokens"] or 0) >= plan.CONTEXT_CLOSE
    if a.json:
        print(json.dumps(c))
    else:
        print("%s: %s tokens at %s%s%s" % (a.agent, c["tokens"], c["at"], "; CLOSE (>= %d)" % plan.CONTEXT_CLOSE
                                           if c["close"] else "", "; compactions at " + ", ".join(
                                               map(str, c["compactions"])) if c["compactions"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
