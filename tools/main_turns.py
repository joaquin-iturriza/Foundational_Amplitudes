"""What wakes the main session and what it does when woken: each main-session API call attributed to the trigger that
started its segment and the segment classified by what it did; count, context at the segment's first call, weighted
tokens and list-price $ per class, per era (tools/usage_report.py PERIODS). Read-only over the main-session transcripts
of ~/.claude/projects/-home-joaquin-work-FA (the A/B replay folders are not the main session); CPU, under a minute.

    python3 tools/main_turns.py [--out analysis/agent_redesign/main_turns.json] [--samples N]

Segment: starts at a trigger and holds every API message (deduplicated on message id, over all main transcripts) until
the next trigger. A trigger is a user record that is not a tool result and carries text (a new turn, or a Stop hook's
feedback that keeps the turn going); skill text, image captions and local-command caveats are part of the turn.
Messages that arrive mid-turn (queued_command attachments: notifications, hand-backs, typed messages) and the harness's
silent_turn_reminder do not start a segment: the model goes on with its turn; they are counted per class as
absorbed_mid_turn, by the trigger class they would have had. Classified as
  user           origin kind human, or a text with no machine marker (typed messages, /commands, interrupts)
  handback       an agent's message ("[Subagent hand-back]" or any <agent-message>/peer message)
  agent_done     <task-notification> whose task id is an agent id (a + 16 hex): a background subagent finished
  waiter         <task-notification> of a background Bash whose command was `site events ... --wait` (the event waiter),
                 or a Monitor event
  supervisor     <task-notification> of `site supervise`
  bg_other       <task-notification> of any other background command (detach jobs, tests, copies)
  goal_checkin   "Goal check-in" (task-notification summary or meta text)
  stop_hook      "Stop hook feedback:" (a Stop hook refused the turn's end)
  nudge          other harness text: "no visible output", usage-limit notes, artifact auto-react, safety-classifier stop
  compaction     "This session is being continued" (the summary after a compaction)
What the segment did (the first that applies, over its tool calls):
  inline         Edit/Write/NotebookEdit; Bash with git commit|merge|worktree|rm|push, site submit|run|copy|sync|cancel,
                 pytest, python running anything but tools/plan.py, sed -i, tee or a redirect into a file
  read           Read/Grep/Glob/Skill/web/MCP, or other Bash (cat, grep, git log/diff, site runs/logs/status ...)
  orchestration  Agent, SendMessage, TaskStop, Task*, ToolSearch, tools/plan.py, site ack|events|decide|items|supervise|
                 digest|hold|release|pick, tools/detach.py
  none           no tool call
A "nothing to say" wake: a segment not started by the user whose class is none or orchestration and whose visible text
is at most 60 characters (e.g. "Working."); reported with its count and cost, and the subset whose text starts "Working".
"""
import argparse, glob, json, os, re, statistics, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import usage_windows as uw
from usage_report import PERIODS, period_of, ts, W, text_of, TRIAL_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")
AGENT_ID = re.compile(r"<task-id>(a[0-9a-f]{16})</task-id>")
TOOL_USE_ID = re.compile(r"<tool-use-id>(toolu_\w+)</tool-use-id>")
INLINE_BASH = re.compile(r"\bgit (commit|merge|worktree|rm|push)\b|\bsite (submit|run|copy|sync|cancel)\b|\bpytest\b|"
                         r"\bsed -i\b|[^2&]>\s*(?!/tmp/|\$S/|/dev/)[\w./~$-]+\.\w+|\btee\s+(?!/tmp/|\$S/)|"
                         r"(^|[;&|(]\s*)(cp|mv|rm|mkdir|ln)\s")
ORCH_BASH = re.compile(r"(python3?\s+(-I\s+)?)?\S*tools/plan\.py[^;&|]*|\bsite (ack|events|decide|items?|supervise|digest|"
                       r"hold|release|pick)\b[^;&|]*|(python3?\s+)?\S*tools/detach\.py|"
                       r"\bgit (add|commit)\b[^;&|]*--\s+plan/\S*\s*$")
PY_RUN = re.compile(r"\bpython3?\s+(-I\s+|-u\s+)*(?!-)(?!\S*tools/plan\.py)(?!/tmp/|\$S/|\$SP/)(\S+\.py)\b"
                    r"(?![^;&|]*--help)")
QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"' + r"|'[^'\n]*'")


def first_lines(cmd):
    """The command with quoted strings blanked and heredoc bodies dropped: what the shell runs, not what it passes."""
    out, skip = [], None
    for line in cmd.split("\n"):
        if skip is not None:
            if line.strip() == skip:
                skip = None
            continue
        out.append(line)
        m = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
        if m:
            skip = m.group(1)
    return QUOTED.sub('""', "\n".join(out))
PY_INLINE = re.compile(r"\bpython3?\s+(-I\s+)*(-\s|-c\b|-\s*<<|<<)")
PY_WRITES = re.compile(r"open\([^)]*,\s*['\"][wax]b?\+?['\"]|\.write_text\(|json\.dump\(|savefig\(|to_csv\(|np\.save|"
                       r"shutil\.(copy|move)|os\.(remove|rename|replace|unlink)\(|\.write\(|subprocess")
ORCH_TOOLS = {"Agent", "Task", "SendMessage", "TaskStop", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet",
              "TodoWrite", "ToolSearch"}
EDIT_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit"}
READ_TOOLS = {"Read", "Grep", "Glob", "WebFetch", "WebSearch", "Skill"}


def bash_kind(cmd):
    """inline: writes a file or the repo, moves work on a site, or runs an analysis script (python3 <script>.py);
    python run from a heredoc or -c is inline only when its body writes (open(..., 'w'), json.dump, savefig, ...);
    orchestration: plan/ledger/agent bookkeeping, plan-only commits; read: the rest."""
    shell = first_lines(cmd)
    rest = ORCH_BASH.sub(" ", shell)
    if INLINE_BASH.search(rest) or PY_RUN.search(rest) or (PY_INLINE.search(rest) and PY_WRITES.search(cmd)):
        return "inline"
    if ORCH_BASH.search(shell):
        return "orchestration"
    return "read"


def trigger_of(txt, origin, bg_cmd):
    o = (origin or {}).get("kind")
    s = txt.lstrip()
    if s.startswith("Stop hook feedback"):
        return "stop_hook"
    if "Goal check-in" in s[:400]:
        return "goal_checkin"
    if s.startswith("This session is being continued"):
        return "compaction"
    if "<agent-message" in s[:300] or o == "peer" or s.startswith("Another Claude session sent"):
        return "handback"
    if s.startswith("<task-notification>"):
        if "artifact-auto-react" in s[:400]:
            return "nudge"
        if "Monitor event" in s[:600]:
            return "waiter"
        if AGENT_ID.search(s[:200]):
            return "agent_done"
        m = TOOL_USE_ID.search(s[:400])
        cmd = bg_cmd.get(m.group(1), "") if m else ""
        if re.search(r"\bsite events\b.*--wait", cmd):
            return "waiter"
        if re.search(r"\bsite supervise\b", cmd):
            return "supervisor"
        return "bg_other"
    if s.startswith("[Your previous response had no visible output") or s.startswith("[Usage limit") or \
            s.startswith("[Earlier usage-limit") or s.startswith("The user hasn't heard from you") or \
            s.startswith("Your response above was stopped by a safety classifier"):
        return "nudge"
    return "user"


SEEN = set()


def scan(paths):
    seen = SEEN
    segs = []
    bg_cmd = {}
    for p in paths:
        cur = None
        with open(p, errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                t = d.get("timestamp")
                typ = d.get("type")
                trig = None
                if typ == "user":
                    c = (d.get("message") or {}).get("content")
                    is_result = isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result"
                                                            for b in c)
                    if not is_result:
                        txt = text_of(c)
                        s = txt.lstrip()
                        # meta records that are part of a turn, not a trigger (skill text, image captions, caveats)
                        if s and not (d.get("isMeta") and (s.startswith("Base directory for this skill") or
                                                           s.startswith("[Image") or s.startswith("<local-command"))):
                            trig = (trigger_of(txt, d.get("origin"), bg_cmd), txt[:160])
                elif typ == "attachment":
                    a = d.get("attachment") or {}
                    if cur is not None and a.get("type") == "queued_command":
                        pr = a.get("prompt") if isinstance(a.get("prompt"), str) else text_of(a.get("prompt"))
                        cur["absorbed"][trigger_of(pr or "", a.get("origin"), bg_cmd)] += 1
                    elif cur is not None and a.get("type") == "silent_turn_reminder":
                        cur["absorbed"]["silent_turn_reminder"] += 1
                if trig and t:
                    cur = {"trigger": trig[0], "text": trig[1], "t": t, "period": period_of(ts(t)), "msgs": 0,
                           "tokens": 0.0, "usd": 0.0, "ctx0": None, "tools": [], "said": "", "file": os.path.basename(p),
                           "absorbed": defaultdict(int), "per_msg": {},
                           "unpriced": defaultdict(int), "unpriced_tokens": 0.0}
                    segs.append(cur)
                    continue
                if typ != "assistant" or cur is None:
                    continue
                msg = d.get("message") or {}
                mid, u = msg.get("id"), msg.get("usage") or {}
                if mid and u and mid not in seen:
                    seen.add(mid)
                    cc = u.get("cache_creation") or {}
                    c5, c1 = cc.get("ephemeral_5m_input_tokens") or 0, cc.get("ephemeral_1h_input_tokens") or 0
                    usd = uw.cost({"model": msg.get("model"), "in": u.get("input_tokens") or 0,
                                   "out": u.get("output_tokens") or 0, "cw5": c5, "cw1h": c1,
                                   "cwu": max((u.get("cache_creation_input_tokens") or 0) - c5 - c1, 0),
                                   "cr": u.get("cache_read_input_tokens") or 0})
                    tok = sum(w * (u.get(k) or 0) for k, w in W.items())
                    if usd is None:          # no list price for this model: counted and listed, not $0
                        cur["unpriced"][msg.get("model")] += 1
                        cur["unpriced_tokens"] += tok
                        usd = 0.0
                    cur["usd"] += usd
                    cur["tokens"] += tok
                    cur["msgs"] += 1
                    cur["per_msg"][mid] = {"usd": usd, "tokens": tok, "tools": []}
                    if cur["ctx0"] is None:
                        cur["ctx0"] = (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) + \
                                      (u.get("cache_read_input_tokens") or 0)
                for b in msg.get("content") or []:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text":
                        cur["said"] += b.get("text", "")
                    elif b.get("type") == "tool_use":
                        inp = b.get("input") or {}
                        name = b.get("name")
                        cmd = str(inp.get("command") or "")
                        if name == "Bash" and inp.get("run_in_background"):
                            bg_cmd[b.get("id")] = cmd
                        cur["tools"].append((name, cmd))
                        if mid in cur["per_msg"]:
                            cur["per_msg"][mid]["tools"].append((name, cmd))
    for s in segs:
        for m in s["per_msg"].values():
            m["did"] = did_of(m["tools"])
        s["did"] = did_of(s["tools"])
        said = s["said"].strip()
        s["nothing"] = s["trigger"] != "user" and s["did"] in ("none", "orchestration") and len(said) <= 60
        s["working"] = s["nothing"] and said.startswith("Working")
    return segs


def call_kind(name, cmd):
    if name in EDIT_TOOLS:
        return "inline"
    if name == "Bash":
        return bash_kind(cmd)
    if name in READ_TOOLS or name.startswith("mcp__") or name.startswith("Artifact"):
        return "read"
    return "orchestration"


def did_of(tools):
    """inline > read > orchestration > none over the given calls (a segment's, or one API message's)."""
    kinds = [call_kind(n, c) for n, c in tools]
    return next((k for k in ("inline", "read", "orchestration") if k in kinds), "none")


def cell(xs):
    ctx = [x["ctx0"] for x in xs if x["ctx0"] is not None]
    n = sum(1 for x in xs if x["msgs"])
    ab = defaultdict(int)
    for x in xs:
        for k, v in x["absorbed"].items():
            ab[k] += v
    up = defaultdict(int)
    for x in xs:
        for k, v in x["unpriced"].items():
            up[k] += v
    return {"segments": n, "absorbed_mid_turn": dict(ab), "unpriced_calls": dict(up),
            "unpriced_weighted_tokens_M": round(sum(x["unpriced_tokens"] for x in xs) / 1e6, 3), "api_calls": sum(x["msgs"] for x in xs),
            "weighted_tokens_M": round(sum(x["tokens"] for x in xs) / 1e6, 3),
            "usd": round(sum(x["usd"] for x in xs), 2),
            "usd_per_segment": round(sum(x["usd"] for x in xs) / max(1, n), 3),
            "context_at_first_call_median_k": round(statistics.median(ctx) / 1e3, 1) if ctx else None}


def summarise(segs):
    out = {}
    for per, _, _ in PERIODS:
        S = [s for s in segs if s["period"] == per and s["msgs"]]
        if not S:
            continue
        tot_usd = sum(s["usd"] for s in S) or 1
        P = {"total": cell(S), "by_trigger": {}, "by_did": {}, "trigger_x_did": {}}
        for k in sorted({s["trigger"] for s in S}):
            P["by_trigger"][k] = cell([s for s in S if s["trigger"] == k])
            P["by_trigger"][k]["usd_share"] = round(P["by_trigger"][k]["usd"] / tot_usd, 4)
        for k in ("inline", "read", "orchestration", "none"):
            P["by_did"][k] = cell([s for s in S if s["did"] == k])
            P["by_did"][k]["usd_share"] = round(P["by_did"][k]["usd"] / tot_usd, 4)
        for tk in P["by_trigger"]:
            P["trigger_x_did"][tk] = {k: {"n": sum(1 for s in S if s["trigger"] == tk and s["did"] == k),
                                          "usd": round(sum(s["usd"] for s in S if s["trigger"] == tk and s["did"] == k), 2)}
                                      for k in ("inline", "read", "orchestration", "none")}
        msgs = [m for x in S for m in x["per_msg"].values()]
        mu = sum(m["usd"] for m in msgs) or 1
        P["by_call"] = {k: {"api_calls": sum(1 for m in msgs if m["did"] == k),
                            "usd": round(sum(m["usd"] for m in msgs if m["did"] == k), 2),
                            "usd_share": round(sum(m["usd"] for m in msgs if m["did"] == k) / mu, 4)}
                        for k in ("inline", "read", "orchestration", "none")}
        W_ = [x for x in S if x["trigger"] != "user" and x["did"] in ("orchestration", "none")]
        P["orchestration_only_wakes"] = cell(W_)
        P["orchestration_only_wakes"]["usd_share"] = round(P["orchestration_only_wakes"]["usd"] / tot_usd, 4)
        P["nothing_to_say"] = cell([s for s in S if s["nothing"]])
        P["nothing_to_say"]["usd_share"] = round(P["nothing_to_say"]["usd"] / tot_usd, 4)
        P["working_turns"] = cell([s for s in S if s["working"]])
        out[per] = P
    return out


def agent_kinds(seen):
    """Subagent spend per era (of its first call) and agent type (meta.json agentType): count, $ and weighted tokens,
    median context at its first call; for forks, its first-call context over the parent's context at the call that
    launched it (about 1 means the fork carried the parent's whole context)."""
    parent_ctx = {}
    for p in sorted(glob.glob(os.path.join(FOLDER, "*.jsonl"))):
        with open(p, errors="replace") as f:
            for line in f:
                if '"Agent"' not in line and '"Task"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                msg = d.get("message") or {}
                u = msg.get("usage") or {}
                for b in msg.get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") in ("Agent", "Task"):
                        parent_ctx[b.get("id")] = (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) \
                            + (u.get("cache_read_input_tokens") or 0)
    rows = defaultdict(lambda: {"n": 0, "usd": 0.0, "tokens": 0.0, "ctx0": [], "fork_ratio": [],
                                "unpriced": defaultdict(int)})
    for mp in glob.glob(os.path.join(FOLDER, "*", "subagents", "**", "*.meta.json"), recursive=True):
        meta = json.load(open(mp))
        jp = mp[:-len(".meta.json")] + ".jsonl"
        if not os.path.exists(jp):
            continue
        usd = tok = 0.0
        ctx0 = t0 = None
        unpriced = defaultdict(int)
        with open(jp, errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") != "assistant":
                    continue
                msg = d.get("message") or {}
                mid, u = msg.get("id"), msg.get("usage") or {}
                if not (mid and u) or mid in seen:
                    continue
                seen.add(mid)
                cc = u.get("cache_creation") or {}
                c5, c1 = cc.get("ephemeral_5m_input_tokens") or 0, cc.get("ephemeral_1h_input_tokens") or 0
                c = uw.cost({"model": msg.get("model"), "in": u.get("input_tokens") or 0,
                                "out": u.get("output_tokens") or 0, "cw5": c5, "cw1h": c1,
                                "cwu": max((u.get("cache_creation_input_tokens") or 0) - c5 - c1, 0),
                                "cr": u.get("cache_read_input_tokens") or 0})
                if c is None:
                    unpriced[msg.get("model")] += 1
                    c = 0.0
                usd += c
                tok += sum(w * (u.get(k) or 0) for k, w in W.items())
                if ctx0 is None:
                    t0 = d.get("timestamp")
                    ctx0 = (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) + \
                           (u.get("cache_read_input_tokens") or 0)
        if t0 is None:
            continue
        r = rows[(period_of(ts(t0)), meta.get("agentType") or "?")]
        r["n"] += 1
        for k, v in unpriced.items():
            r["unpriced"][k] += v
        r["usd"] += usd
        r["tokens"] += tok
        r["ctx0"].append(ctx0)
        pc = parent_ctx.get(meta.get("toolUseId"))
        if meta.get("agentType") == "fork" and pc:
            r["fork_ratio"].append(ctx0 / pc)
    out = defaultdict(dict)
    for (per, typ), r in sorted(rows.items(), key=lambda x: (str(x[0][0]), -x[1]["usd"])):
        out[per][typ] = {"agents": r["n"], "usd": round(r["usd"], 2), "weighted_tokens_M": round(r["tokens"] / 1e6, 2),
                         "usd_per_agent": round(r["usd"] / r["n"], 2), "unpriced_calls": dict(r["unpriced"]),
                         "context_at_first_call_median_k": round(statistics.median(r["ctx0"]) / 1e3, 1)}
        if r["fork_ratio"]:
            out[per][typ]["fork_ctx_over_parent"] = {"n": len(r["fork_ratio"]),
                                                     "median": round(statistics.median(r["fork_ratio"]), 3),
                                                     "min": round(min(r["fork_ratio"]), 3),
                                                     "max": round(max(r["fork_ratio"]), 3)}
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/agent_redesign/main_turns.json")
    ap.add_argument("--samples", type=int, default=0, help="print N segments of each trigger x did cell (hand check)")
    ap.add_argument("--handcheck", action="store_true",
                    help="redesign era: 20 inline, 10 orchestration, 10 none segments (seed 0) with the call that set the label")
    a = ap.parse_args()
    paths = sorted(p for p in glob.glob(os.path.join(FOLDER, "*.jsonl")) if TRIAL_DIR not in p)
    segs = scan(paths)
    if a.samples:
        import random
        random.seed(0)
        cells = defaultdict(list)
        for s in segs:
            if s["msgs"]:
                cells[(s["trigger"], s["did"])].append(s)
        for k in sorted(cells):
            for s in random.sample(cells[k], min(a.samples, len(cells[k]))):
                print(k, s["t"][:16], "| trig:", s["text"][:100].replace("\n", " "), "| tools:",
                      [(n, c[:60]) for n, c in s["tools"][:3]], "| said:", s["said"].strip()[:60].replace("\n", " "))
    if a.handcheck:
        import random
        random.seed(0)
        R = [x for x in segs if x["period"] == "redesign" and x["msgs"]]
        for k, n in (("inline", 20), ("orchestration", 10), ("none", 10)):
            for x in random.sample([y for y in R if y["did"] == k], min(n, sum(1 for y in R if y["did"] == k))):
                why = next(((nm, c) for nm, c in x["tools"] if call_kind(nm, c) == k), ("-", ""))
                print("HC", k, x["t"][:16], x["trigger"], "$%.2f" % x["usd"], "| set by:", why[0],
                      why[1][:150].replace("\n", " "), "| said:", x["said"].strip()[:50].replace("\n", " "))
    rep = {"definitions": __doc__.split("Segment:")[1].strip(), "files": [os.path.basename(p) for p in paths],
           "by_period": summarise(segs), "agents_by_kind": agent_kinds(SEEN)}
    out = os.path.join(ROOT, a.out)
    json.dump(rep, open(out, "w"), indent=1)
    for per, P in rep["by_period"].items():
        print(per, json.dumps(P["total"]))
        for k, v in sorted(P["by_trigger"].items(), key=lambda x: -x[1]["usd"]):
            print("  trig %-13s" % k, json.dumps(v))
        for k, v in P["by_did"].items():
            print("  did  %-13s" % k, json.dumps(v))
        print("  nothing_to_say", json.dumps(P["nothing_to_say"]))
        print("  working       ", json.dumps(P["working_turns"]))
        print("  by_call", json.dumps(P["by_call"]))
        print("  orch_only_wakes", json.dumps(P["orchestration_only_wakes"]))
    for per, A in rep["agents_by_kind"].items():
        for k, v in A.items():
            print("agents", per, k, json.dumps(v))


if __name__ == "__main__":
    main()
