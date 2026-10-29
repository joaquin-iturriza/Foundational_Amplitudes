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


USAGE_KEYS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")


def responses(path, prefilter=True):
    """The API responses of one transcript, in order: [(timestamp of its first record, message id, usage, content
    blocks)]. One response is split over several records with the same id and usage; they are joined here (content
    concatenated, each usage field the largest seen), so each response counts once. usage also carries
    cache_5m / cache_1h, the cache creation split by lifetime when the transcript records it."""
    out, idx = [], {}
    with open(path) as fh:
        for line in fh:
            if prefilter and '"assistant"' not in line:
                continue
            r = json.loads(line)
            if r.get("type") != "assistant":
                continue
            m = r.get("message") or {}
            u = m.get("usage") or {}
            cc = u.get("cache_creation") or {}
            use = {k: u.get(k, 0) or 0 for k in USAGE_KEYS}
            use["cache_5m"] = cc.get("ephemeral_5m_input_tokens", 0) or 0
            use["cache_1h"] = cc.get("ephemeral_1h_input_tokens", 0) or 0
            blocks = [x for x in m.get("content") or [] if isinstance(x, dict)]
            mid = m.get("id") or id(r)
            if mid in idx:
                rec = out[idx[mid]]
                rec[3].extend(blocks)
                for k, v in use.items():
                    rec[2][k] = max(rec[2][k], v)
            else:
                idx[mid] = len(out)
                out.append((r.get("timestamp") or "", mid, use, blocks))
    return out


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
        calls, ctx = 0, []
        for _, _, u, blocks in responses(f):
            ctx.append(u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"])
            calls += sum(1 for x in blocks if x.get("type") == "tool_use")
        if ctx:
            groups[key].append((calls, ctx[0], sum(ctx)))
    total = sum(s for g in groups.values() for *_, s in g) or 1
    print("%-46s %4s %6s %8s %9s %6s" % ("type, effort, model", "n", "calls", "start", "re-read", "share"))
    for key, g in sorted(groups.items(), key=lambda kv: -sum(s for *_, s in kv[1])):
        print("%-46s %4d %6d %8d %8.1fM %5.0f%%" % (", ".join(key), len(g), st.median(a for a, _, _ in g),
              st.median(b for _, b, _ in g), st.median(c for *_, c in g) / 1e6, 100 * sum(c for *_, c in g) / total))


if __name__ == "__main__":
    main(sys.argv)
