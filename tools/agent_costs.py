#!/usr/bin/env python3
"""Token cost of the agents a session launched, grouped by how they were launched (subagent type, effort, model),
read from the Agent tool calls themselves, never from prompt text (2026-10-29: a keyword filter on prompts called
167 agents mechanical because the step-prompt template mentions collect_run; the launch parameter gives 23).

  python3 tools/agent_costs.py [SESSION_ID] [--after YYYY-MM-DDTHH:MM]   (default: the newest session; --after: only
                                                                        agents launched after that time, UTC)

Per group: agents, median tool calls, median context at start, median tokens re-read over the agent's calls (each
call re-reads the whole context), and the group's share of all agents' re-read tokens."""
import glob, json, os, re, statistics as st, sys
from collections import defaultdict

PROJ = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")


def main(argv):
    after = ""
    if "--after" in argv:
        i = argv.index("--after"); after = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    sid = argv[1] if len(argv) > 1 else os.path.basename(max(glob.glob(PROJ + "/*.jsonl"), key=os.path.getmtime))[:-6]
    launch, ids = {}, {}
    for line in open(os.path.join(PROJ, sid + ".jsonl")):
        r = json.loads(line)
        c = (r.get("message") or {}).get("content")
        if not isinstance(c, list):
            continue
        for x in c:
            if not isinstance(x, dict):
                continue
            if x.get("type") == "tool_use" and x.get("name") == "Agent" and (r.get("timestamp") or "") >= after:
                i = x.get("input") or {}
                launch[x["id"]] = (i.get("subagent_type") or "general-purpose", i.get("effort") or "default",
                                   i.get("model") or "inherit")
            elif x.get("type") == "tool_result" and x.get("tool_use_id") in launch:
                m = re.search(r"agentId: (\w+)", json.dumps(x.get("content")))
                if m:
                    ids[m.group(1)] = launch[x["tool_use_id"]]
    groups = defaultdict(list)
    for aid, key in ids.items():
        f = os.path.join(PROJ, sid, "subagents", "agent-%s.jsonl" % aid)
        if not os.path.exists(f):
            continue
        calls, ctx, seen = 0, [], set()
        for line in open(f):
            r = json.loads(line)
            if r.get("type") != "assistant":
                continue
            m = r["message"]
            if m.get("id") not in seen:          # one API response is split over several records with the same usage
                seen.add(m.get("id"))
                u = m.get("usage") or {}
                ctx.append(u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                           + u.get("cache_creation_input_tokens", 0))
            calls += sum(1 for x in m.get("content") or [] if isinstance(x, dict) and x.get("type") == "tool_use")
        if ctx:
            groups[key].append((calls, ctx[0], sum(ctx)))
    total = sum(s for g in groups.values() for *_, s in g) or 1
    print("%-46s %4s %6s %8s %9s %6s" % ("type, effort, model", "n", "calls", "start", "re-read", "share"))
    for key, g in sorted(groups.items(), key=lambda kv: -sum(s for *_, s in kv[1])):
        print("%-46s %4d %6d %8d %8.1fM %5.0f%%" % (", ".join(key), len(g), st.median(a for a, _, _ in g),
              st.median(b for _, b, _ in g), st.median(c for *_, c in g) / 1e6, 100 * sum(c for *_, c in g) / total))


if __name__ == "__main__":
    main(sys.argv)
