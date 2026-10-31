"""Switch-over D of the agent redesign (plan entry agent_redesign, impl_order.txt D1 to D8): owners and hand-offs in
the plan, plan next and the stop check by owner, owner context and compactions, the pushed reference sections, the
owner's brief and its guard, the main session's delegation note and the supervisor's known-cause rule. Nothing live
is written: the plan scenarios run in test_setup_scenarios' World (a temporary plan, registry, transcripts and a stub
site), the hooks on temporary state directories; the night's real records (transcripts, ledger) are only read.
    python3 tests/test_redesign_d.py [group ...]"""
import copy, glob, json, os, sqlite3, subprocess, sys, tempfile

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_setup_scenarios import World, E, ago, epoch, scenario, RESULTS, ROOT, J  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "tools"))
FIX = os.path.join(ROOT, "tests", "fixtures", "hooks_20261029")
HOOKS = os.path.join(ROOT, ".claude", "hooks")
OWN = {"PLAN_OWNER_MODE": "1"}
NIGHT = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")


def owner_world(entries, agents=(), events="  no open events", runs=(), acked=()):
    """A World in owner mode. agents: (id, entry, records), records [(ts, kind)] with kind 'work', 'handback',
    'compact' or ('usage', tokens)."""
    w = World(entries, runs=runs, events=events)
    w.env.update(OWN)
    db = sqlite3.connect(w.db)
    db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts INTEGER, project TEXT, item TEXT, "
               "kind TEXT, key TEXT, msg TEXT, acked INTEGER, ack_note TEXT, tell TEXT)")
    for r in acked:
        db.execute("INSERT INTO events (ts,project,item,kind,key,msg,acked,ack_note) VALUES (?,?,?,?,?,?,?,?)",
                   (epoch(5), "FA", "x", "RUN", r + ":COMPLETED", r + " ended", epoch(4), r + ": finished cleanly"))
    db.commit()
    for aid, eid, recs in agents:
        out = [{"type": "user", "timestamp": recs[0][0] if recs else ago(60), "message": {
            "role": "user", "content": "You own one entry.\n\nEntry: %s (t)\nQuestion: q" % eid}}]
        for ts, kind in recs:
            if kind == "compact":
                out.append({"type": "system", "subtype": "compact_boundary", "timestamp": ts})
                continue
            name = "SubagentHandback" if kind == "handback" else "Bash"
            msg = {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": name, "input": {}}]}
            if isinstance(kind, tuple):
                msg["usage"] = {"input_tokens": 2, "cache_read_input_tokens": kind[1] - 1002,
                                "cache_creation_input_tokens": 1000}
            out.append({"type": "assistant", "timestamp": ts, "message": msg})
        open(os.path.join(w.ag, "agent-%s.jsonl" % aid), "w").write("".join(J(r) + "\n" for r in out))
    return w


def owned(eid, agent, state="working", **k):
    return E(eid, owner={"agent": agent, "state": state, "since": ago(90)}, **k)


# ---------------------------------------------------------------- D1: schema, migration, hand-off, brief
def d1():
    import plan_migrate
    before = yaml.safe_load(open(os.path.join(FIX, "plan_before", "plan.yaml")))
    after = plan_migrate.owners(copy.deepcopy(before), "2026-10-30T12:00:00+00:00")
    run = [e for e in after["entries"] if e.get("status") == "running"]
    kept = all(all(a.get(k) == b.get(k) for k in b) for a, b in zip(after["entries"], before["entries"]))
    again = plan_migrate.owners(copy.deepcopy(after), "later")
    scenario("D1: the acked-runs demand's plan migrates: every running entry (%d) gets a hand-off (state, running, "
             "files) and an objective, no owner, every old field kept" % len(run),
             run and kept and all(e["handoff"]["state"] and e["handoff"]["running"] and "owner" not in e for e in run))
    scenario("D1: the migration is idempotent and round-trips through YAML",
             again == after and yaml.safe_load(yaml.safe_dump(after)) == after)
    w = World(after["entries"])
    rc, out = w.plan_cmd("show")
    scenario("D1: the migrated plan reads with plan show", rc == 0, out[-300:])
    briefs = [w.plan_cmd("brief", e["id"]) for e in run]
    scenario("D1: plan brief renders every running entry of it (%d), each with the rules block and its hand-off"
             % len(run), all(rc == 0 and "=== RULES" in o and "Entry: %s" % e["id"] in o and "Hand-off" in o
                             for (rc, o), e in zip(briefs, run)), [o[-300:] for rc, o in briefs if rc][:2])
    w = owner_world([owned("a", "o1")], agents=[("o1", "a", [(ago(30), "work")])])
    r1 = w.plan_cmd("handoff", "a", "--running", "none")
    r2 = w.plan_cmd("handoff", "a", "--state", "probe read")
    r3 = w.plan_cmd("handoff", "a", "--state", "probe read", "--running", "none", "--decisions", "kept 3 TeV: why")
    e = w.entries()["a"]
    scenario("D1: plan handoff refuses a hand-off without --state or --running", r1[0] != 0 and r2[0] != 0, r1 + r2)
    scenario("D1: a hand-off written closes the owner", r3[0] == 0 and e["handoff"]["state"] == "probe read"
             and e["owner"]["state"] == "closed", (r3, e.get("owner")))
    r = w.plan_cmd("finding", "a", "--text", "a compaction inside the turn")
    r2 = w.plan_cmd("finding", "a", "--text", "a compaction inside the turn")
    scenario("D1: plan finding records once", r[0] == 0 and len(w.entries()["a"]["findings"]) == 1, r + r2)
    w = owner_world([owned("a", "o1")], agents=[("o1", "a", [(ago(2), "work")]), ("o2", "a", [(ago(1), "work")])])
    r = w.plan_cmd("owner", "a", "--agent", "o2")
    scenario("D1: a second owner is refused while the first works (one agent per piece of work)",
             r[0] != 0 and w.entries()["a"]["owner"]["agent"] == "o1", r)
    w = World([E("a", next={1: "x", 2: "y"})], state={"a": {"steps": []}}, agents=[("s1", ago(5), "a")])
    w.env.update(OWN)
    r = w.plan_cmd("step", "a", "--agent", "s1", "--next", "2")
    scenario("D1: in owner mode an agent recorded on any step of the entry is its owner",
             r[0] == 0 and (w.entries()["a"].get("owner") or {}).get("agent") == "s1", r)


# ---------------------------------------------------------------- D2 and D6: plan next and the stop check by owner
EV = "  E2300  10-30 03:10 FA  RUN  a  FA-ccin2p3-b1 ended FAILED, broke: Traceback"


def d2_d6():
    w = owner_world([owned("a", "o1")], runs=[("FA-ccin2p3-r%d" % i, "COMPLETED", "a", epoch(90), epoch(70))
                                               for i in range(3)] + [("FA-ccin2p3-live", "RUNNING", "a", epoch(90), None)],
                    acked=["FA-ccin2p3-r%d" % i for i in range(3)],
                    agents=[("o1", "a", [(ago(80), "work"), (ago(60), "handback")])])
    rc, out = w.plan_cmd("next")
    rcc, outc = w.check()
    scenario("D2: the acked-runs demand: runs acked, a live run, the owner stopped -> nothing listed, a note",
             rc == 0 and "RESUME" not in out and "LAUNCH" not in out and "stopped while its work" in out, out)
    scenario("D6: the acked-runs demand -> no refusal", rcc == 0, outc)
    w = owner_world([owned("a", "o1")], events=EV, agents=[("o1", "a", [(ago(300), "work"), (ago(290), "handback")])])
    rc, out = w.plan_cmd("next")
    rcc, outc = w.check()
    scenario("D2: an event of a broke run, its owner stopped -> resume o1 (not a new agent)",
             rc == 1 and "RESUME a o1" in out and "LAUNCH" not in out, out)
    scenario("D6: ... and the stop check refuses until it is resumed", rcc == 2 and "resume its owner o1" in outc, outc)
    w = owner_world([owned("a", "o1")], events=EV, agents=[("o1", "a", [(ago(300), "handback"), (ago(1), "work")])])
    rc, out = w.plan_cmd("next")
    rcc, outc = w.check()
    scenario("D2: an owner working -> nothing (a note naming it)", rc == 0 and "owner o1 is working" in out, out)
    scenario("D6: an owner working with an open event -> no refusal (B9: a turn ending with an agent running ends "
             "with a note)", rcc == 0 and "o1 is working" in outc, outc)
    e = owned("a", "o1", state="closed", handoff={"state": "s", "running": "none", "written": ago(100)})
    w = owner_world([e], events=EV, agents=[("o1", "a", [(ago(300), "handback")])])
    rc, out = w.plan_cmd("next")
    rcc, outc = w.check()
    scenario("D2: an owner closed with a hand-off and an event -> launch from that hand-off",
             rc == 1 and "LAUNCH a" in out and "hand-off of" in out, out)
    scenario("D6: a closed owner with a hand-off and an event -> refusal (launch)", rcc == 2 and "no live owner" in outc,
             outc)
    rc, out = w.plan_cmd("next", "--prompts")
    scenario("D2: plan next --prompts prints the owner's brief for a launch", "=== RULES" in out and "Hand-off" in out,
             out[-400:])
    w = owner_world([E("a")])
    rc, out = w.plan_cmd("next")
    scenario("D2: a running entry with nothing in flight and no owner -> launch", "LAUNCH a" in out, out)
    w = owner_world([owned("a", "o1")], agents=[("o1", "a", [(ago(300), ("usage", 312000)), (ago(290), "handback")])])
    rc, out = w.plan_cmd("next")
    rcc, outc = w.check()
    scenario("D2/D5: an owner at 312000 tokens -> close o1", "CLOSE a o1" in out and "312000" in out, out)
    scenario("D6: closing is a note, not a refusal", rcc == 0, outc)
    w = owner_world([owned("a", "o1")], agents=[("o1", "a", [(ago(1), "work")])])
    w.env["PLAN_EVENTS_CMD"] = "false"
    rcc, outc = w.check()
    scenario("D6: a failed read of site events fails the stop check closed (repo-reviewer)", rcc == 2 and
             "site events failed" in outc, outc)
    w = World([E("a")], events=EV)
    rc, out = w.plan_cmd("next")
    scenario("D2: before switch-over D plan next lists step agents as before", "NEXT a" in out and "LAUNCH" not in out,
             out)
    # repo-reviewer 2026-10-30: a run the supervisor has not sorted yet stays hidden even when it is the step's last
    import plan
    w = World([E("a")], runs=[("FA-ccin2p3-u1", "COMPLETED", "a", epoch(90), epoch(5))])
    db = sqlite3.connect(w.db)
    db.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, ts INTEGER, project TEXT, item TEXT, kind TEXT, key TEXT, "
               "msg TEXT, acked INTEGER, ack_note TEXT, tell TEXT)")
    db.commit()
    plan.RUNS_DB, plan.unsorted_runs = w.db, (lambda ids: set(ids))
    import importlib  # noqa: F401
    r = plan.why_moves(E("a"), {}, [], plan.now(), {"entries": []}, checks={"a": 1}, sweeps={})
    scenario("D2 (review fix): a single ended run the supervisor is still sorting is not listed as ended",
             not r or "runs ended" not in r[0], r)


# ---------------------------------------------------------------- D5: context and compaction
def d5():
    import plan
    files = glob.glob(os.path.join(NIGHT, "*", "subagents", "agent-*.jsonl"))
    bad, past, comp = [], [], []
    for f in files:
        last = None
        for line in open(f, errors="replace"):
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if isinstance(m, dict) and m.get("type") == "assistant" and \
                    isinstance((m.get("message") or {}).get("usage"), dict):
                last = m["message"]["usage"]
        c = plan.agent_context(None, path=f)
        want = None if last is None else sum(int(last.get(k) or 0) for k in (
            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        if c["tokens"] != want:
            bad.append(os.path.basename(f))
        if (c["tokens"] or 0) >= plan.CONTEXT_CLOSE:
            past.append(os.path.basename(f))
        if c["compactions"]:
            comp.append(os.path.basename(f))
    scenario("D5: every subagent transcript of the project (%d) read: context = its last usage record; %d past 300k "
             "would be closed, %d hold a compaction" % (len(files), len(past), len(comp)), files and not bad, bad[:5])
    mains = [os.path.basename(f)[:-6] for f in glob.glob(os.path.join(NIGHT, "*.jsonl"))][:3]
    scenario("D5: the main session's transcript is never read as an owner's (agent_context finds only subagents)",
             mains and all(subprocess.run([sys.executable, os.path.join(ROOT, "tools", "agent_context.py"), m],
                                          capture_output=True, text=True).returncode != 0 for m in mains), mains)
    w = owner_world([owned("a", "o1")], agents=[("o1", "a", [(ago(30), ("usage", 305000)), (ago(20), "compact"),
                                                             (ago(10), "handback")])])
    inp = J({"hook_event_name": "PostToolUse", "tool_name": "SendMessage", "session_id": "s"})
    env = dict(w.env, CLAUDE_PROJECT_DIR=ROOT)
    run = lambda i: subprocess.run([sys.executable, os.path.join(HOOKS, "owner_context.py")], input=i, env=env,
                                   capture_output=True, text=True)
    r1, r2 = run(inp), run(inp)
    e = w.entries()["a"]
    scenario("D5: owner_context records the owner's context (305000) and the compaction as one finding, and says close",
             r1.returncode == 0 and e["owner"].get("context") == 305000 and len(e.get("findings") or []) == 1
             and "hand-off" in r1.stdout, (r1.stdout, r1.stderr, e.get("owner"), e.get("findings")))
    sub = run(J({"hook_event_name": "PostToolUse", "agent_id": "x"}))
    scenario("D5: a subagent's own Agent call is not the main session's (nothing done)",
             sub.returncode == 0 and not sub.stdout, sub.stdout)


GROUPS = [d1, d2_d6, d5]
# ---------------------------------------------------------------- D3: the pushed sections
def push(inp, state, dirs=None):
    env = dict(os.environ, CONTEXT_STATE=state, CLAUDE_PROJECT_DIR=ROOT)
    if dirs:
        env["CONTEXT_DIRS"] = dirs
    r = subprocess.run([sys.executable, os.path.join(HOOKS, "context_push.py"), inp.get("_mode", "post")],
                       input=J({k: v for k, v in inp.items() if k != "_mode"}), env=env, capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        return r.returncode, []
    text = json.loads(r.stdout)["hookSpecificOutput"].get("additionalContext", "")
    return 0, [l.split()[1] for l in text.splitlines() if l.startswith("REFERENCE ")]


def push_run(inp, state, dirs=None, most=4):
    """(sections delivered, held calls) of one PreToolUse call sent again while the hook holds it (a section longer
    than one output arrives over several sends; the call runs only once all it triggers has arrived)."""
    env = dict(os.environ, CONTEXT_STATE=state, CLAUDE_PROJECT_DIR=ROOT)
    if dirs:
        env["CONTEXT_DIRS"] = dirs
    names, held = [], 0
    for _ in range(most):
        r = subprocess.run([sys.executable, os.path.join(HOOKS, "context_push.py"), inp.get("_mode", "post")],
                           input=J({k: v for k, v in inp.items() if k != "_mode"}), env=env, capture_output=True,
                           text=True)
        out = json.loads(r.stdout)["hookSpecificOutput"] if r.stdout.strip() else {}
        names += [l.split()[1] for l in out.get("additionalContext", "").splitlines() if l.startswith("REFERENCE ")]
        if out.get("permissionDecision") != "deny":
            return sorted(set(names)), held
        held += 1
    return sorted(set(names)), -1


def d3():
    st = tempfile.mkdtemp(prefix="push_")
    read = lambda agent, p: {"hook_event_name": "PostToolUse", "tool_name": "Read", "session_id": "s", "agent_id": agent,
                             "tool_input": {"file_path": os.path.join(ROOT, p)}}
    bash = lambda agent, c, mode="pre": {"_mode": mode, "hook_event_name": "PreToolUse" if mode == "pre" else
                                         "PostToolUse", "tool_name": "Bash", "session_id": "s", "agent_id": agent,
                                         "tool_input": {"command": c}}
    a1, a2, b1 = push(read("A", "models/lloca.py"), st), push(read("A", "models/lloca.py"), st), \
        push(read("B", "models/lloca.py"), st)
    scenario("D3: a Read of models/lloca.py by agent A pushes fa-model (and nothing else it does not trigger); a second "
             "Read nothing; agent B again", "fa-model" in a1[1] and a2[1] == [] and "fa-model" in b1[1], (a1, a2, b1))
    a3 = push(read("A", "models/transformer_lloca_mup.py"), st)
    scenario("D3: a Read of the LLoCa hot path by A pushes fa-performance once (fa-model already delivered)",
             a3[1] == ["fa-performance"], a3)
    c = push(bash("C", 'git commit -m "sweep_manager.py: fix" -- x'), st)
    scenario("D3: a commit message naming sweep_manager.py pushes nothing (git sections aside)",
             not any(n.startswith("fa-sweeps") for n in c[1]), c)
    g = push_run(bash("D", "python sweep/generate_sweep.py --config x"), st)
    scenario("D3: python sweep/generate_sweep.py --config x pushes fa-sweeps-and-hpo and fa-canonical-setup before it "
             "runs (held until both arrived)", {"fa-sweeps-and-hpo", "fa-canonical-setup"} <= set(g[0]) and g[1] >= 0, g)
    # an edited section is pushed again (a scratch copy of the sections)
    d = tempfile.mkdtemp(prefix="secs_")
    for f in glob.glob(os.path.join(ROOT, ".claude", "context", "fa-model.txt")):
        open(os.path.join(d, "fa-model.txt"), "w").write(open(f).read())
    st2 = tempfile.mkdtemp(prefix="push_")
    x1 = push(read("A", "models/lloca.py"), st2, d)
    open(os.path.join(d, "fa-model.txt"), "a").write("- one more line\n")
    x2 = push(read("A", "models/lloca.py"), st2, d)
    scenario("D3: a section edited is pushed again to an agent that saw it", x1[1] == x2[1] == ["fa-model"], (x1, x2))
    ss = lambda src: {"hook_event_name": "SessionStart", "source": src, "session_id": "m", "_mode": "session"}
    s1, s2, s3 = push(ss("startup"), st), push(ss("resume"), st), push(ss("compact"), st)
    scenario("D3: SessionStart pushes working-on-the-plan to the main session once; a compaction re-arms it",
             s1[1] == ["working-on-the-plan"] and s2[1] == [] and s3[1] == ["working-on-the-plan"], (s1, s2, s3))
    sub = push(dict(ss("startup"), agent_id="Z"), st)
    ag = push({"_mode": "pre", "hook_event_name": "PreToolUse", "tool_name": "Agent", "session_id": "n",
               "tool_input": {"prompt": "x"}}, st)
    scenario("D3: not to a subagent at its start; an Agent launch by a main session lacking it pushes it",
             sub[1] == [] and "working-on-the-plan" in ag[1], (sub, ag))
    # the night's first command of each subagent that runs lhe_parts (outside a quoted string)
    import re as _re
    sys.path.insert(0, HOOKS)
    from cmdparse import scan_text as scan
    firsts = {}
    for f in glob.glob(os.path.join(NIGHT, "*", "subagents", "agent-*.jsonl")):
        for line in open(f, errors="replace"):
            if "lhe_parts" not in line:
                continue
            try:
                m = json.loads(line)
            except ValueError:
                continue
            for c in (m.get("message") or {}).get("content") or [] if m.get("type") == "assistant" else []:
                if isinstance(c, dict) and c.get("name") == "Bash" and _re.search(r"\blhe_parts\b", scan(
                        c["input"].get("command") or "")):
                    firsts.setdefault(os.path.basename(f), c["input"]["command"])
            if os.path.basename(f) in firsts:
                break
    st3 = tempfile.mkdtemp(prefix="push_")
    got = {k: push_run(bash(k, v), st3) for k, v in firsts.items()}
    miss = {k: v for k, v in got.items() if not {"fa-generation", "fa-catalog-reference"} <= set(v[0]) or v[1] < 0}
    scenario("D3: the first lhe_parts command of each of the night's %d subagents that ran one pushes fa-generation "
             "and fa-catalog-reference before it runs" % len(firsts), firsts and not miss,
             {k: (firsts[k][:120], v) for k, v in list(miss.items())[:3]})


# ---------------------------------------------------------------- D4: the owner's brief and its guard
def brief(prompt, typ="general-purpose", agent=None, model=None, description="x"):
    inp = {"hook_event_name": "PreToolUse", "tool_name": "Agent", "session_id": "s",
           "tool_input": {"subagent_type": typ, "prompt": prompt, "description": description}}
    if model:
        inp["tool_input"]["model"] = model
    if agent:
        inp["agent_id"] = agent
    r = subprocess.run([sys.executable, os.path.join(HOOKS, "brief_guard.py")], input=J(inp),
                       env=dict(os.environ, CLAUDE_PROJECT_DIR=ROOT), capture_output=True, text=True)
    return r.returncode, r.stderr


import importlib.util as _iu
_sp = _iu.spec_from_file_location("brief_guard", os.path.join(HOOKS, "brief_guard.py"))
BG = _iu.module_from_spec(_sp)
_sp.loader.exec_module(BG)


def d4():
    w = World([E("cov", question="does the coverage hold?")])
    rc, b = w.plan_cmd("brief", "cov")
    rcn, bn = w.plan_cmd("brief", "cov", "--add", "Take the 3 TeV block of ee4_epem_cgsxwm first.")
    scenario("D4: a brief built by plan brief passes, with or without a narrowing", brief(b)[0] == 0 and
             brief(bn)[0] == 0, (brief(b), brief(bn)))
    scenario("D4: a brief without the rules block is refused", brief("Entry: cov\nDo the thing.")[0] == 2)
    cut = brief(w.plan_cmd("brief", "cov", "--add", "Run only 2 seeds and skip the design review.")[1])
    scenario("D4: an addition that cuts seeds and skips the review is refused, naming the line",
             cut[0] == 2 and "only 2 seeds" in cut[1], cut)
    open(os.path.join(w.tr, "s.jsonl"), "a").write(J({"type": "user", "timestamp": ago(30), "entrypoint": "cli",
                                                      "message": {"content": "ok, only 2 seeds for now"}}) + "\n")
    env = dict(w.env, CLAUDE_PROJECT_DIR=ROOT)

    def brief_w(text):
        inp = {"hook_event_name": "PreToolUse", "tool_name": "Agent", "session_id": "s",
               "tool_input": {"subagent_type": "general-purpose", "prompt": text}}
        r = subprocess.run([sys.executable, os.path.join(HOOKS, "brief_guard.py")], input=J(inp), env=env,
                           capture_output=True, text=True)
        return r.returncode, r.stderr
    quoted = brief_w(w.plan_cmd("brief", "cov", "--add", 'Joaquin: "only 2 seeds for now"; narrow to the 3 TeV block.')[1])
    invented = brief_w(w.plan_cmd("brief", "cov", "--add", 'run "only two probes" for now')[1])
    scenario("D4: a cut quoted as words the user typed passes; an invented quote is refused (repo-reviewer)",
             quoted[0] == 0 and invented[0] == 2, (quoted, invented))
    sub = brief("Entry: cov\nhelp me", agent="o1")
    scenario("D4: a subagent's launch of a helper without the rules block is refused (one agent per entry)",
             sub[0] == 2, sub)
    tamper = b.replace("never a reason to shrink the job", "a reason to shrink the job")
    scenario("D4: a rules block edited in the brief is refused", brief(tamper)[0] == 2)
    scenario("D4: reviewers are exempt, launched by the main session or by an owner",
             brief("x", "design-reviewer")[0] == 0 and brief("x", "repo-reviewer")[0] == 0 and
             brief("x", "design-reviewer", agent="o1")[0] == 0)
    # 2026-10-30: three owners launched as compact-worker with model sonnet, against the definition's opus
    son, dflt, rev = brief(b, "compact-worker", model="sonnet"), brief(b, "compact-worker"), \
        brief("x", "design-reviewer", model="opus")
    scenario("D4: a launch whose model differs from its definition's is refused (compact-worker + sonnet); no model, or "
             "the definition's own (design-reviewer + opus), passes",
             son[0] == 2 and "sonnet" in son[1] and dflt[0] == 0 and rev[0] == 0, (son, dflt, rev))
    tri = brief(b, "compact-worker", model="sonnet", agent="o1", description="zz_trial t1 TRIAL=model")
    scenario("D4: a model-trial launch (TRIAL=model in its description, D203/D205) may take another model "
             "(compact-worker + sonnet passes; without TRIAL=model it is refused)", tri[0] == 0 and son[0] == 2, tri)
    # the night's launches: each main-session addition to a step prompt, replayed in the narrowing slot
    import re as _re
    norm = lambda x: " ".join(x.split())
    legacy = {norm(l) for l in open(os.path.join(ROOT, "plan", "step_prompt_legacy.txt")).read().splitlines()}
    adds, n = [], 0
    for f in glob.glob(os.path.join(NIGHT, "*.jsonl")):
        for line in open(f, errors="replace"):
            if '"Agent"' not in line:
                continue
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("type") != "assistant" or not str(m.get("timestamp", "")) >= "2026-10-28T20":
                continue
            for c in (m.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("name") == "Agent":
                    p = c["input"].get("prompt") or ""
                    if "Entry: " in p:
                        n += 1
                        # the main session wrote its prompts from the template, often shortened: its own lines are
                        # those not in the template (field values excepted: they follow a "Name: " of the template)
                        own = [l for l in p.splitlines() if l.strip() and norm(l) not in legacy
                               and not _re.match(r"\s*(Entry|Question|Why you were started|Your step|Feeds|Done when|"
                                                 r"Evidence so far|  if |  \d+: |  \")", l)]
                        if own:
                            adds.append("\n".join(own))
    verdicts = [(a, brief(w.plan_cmd("brief", "cov", "--add", a)[1])) for a in adds]
    refused = [v[1] for a, v in verdicts if v[0] == 2]
    print("      the night: %d step launches, %d with an addition, %d refused when replayed:" % (n, len(adds), len(refused)))
    for err in refused:
        print("        - " + " | ".join(l.strip() for l in err.splitlines()[1:-1])[:300])
    scenario("D4: the night's step launches replayed (%d additions): the guard runs on each and lists what it refuses"
             % len(adds), n > 0)


# ---------------------------------------------------------------- the note-only delegation hook
def delegation():
    log = os.path.join(tempfile.mkdtemp(prefix="mw_"), "mw.jsonl")
    def call(tool, ti, agent=None):
        inp = {"hook_event_name": "PostToolUse", "tool_name": tool, "session_id": "s", "tool_input": ti}
        if agent:
            inp["agent_id"] = agent
        r = subprocess.run([sys.executable, os.path.join(HOOKS, "delegation_note.py")], input=J(inp),
                           env=dict(os.environ, CLAUDE_PROJECT_DIR=ROOT, MAIN_WORK_LOG=log), capture_output=True, text=True)
        return r.returncode, r.stdout
    cases = [("Edit", {"file_path": ROOT + "/tools/lhe_parts.py"}, True),
             ("Write", {"file_path": ROOT + "/worktrees/wt-x/analysis/a.py"}, True),
             ("Bash", {"command": "python3 analysis/catalog_v4/validate_v4.py --all"}, True),
             ("Bash", {"command": ".venv-cpu/bin/pytest tests/test_amp.py"}, True),
             ("Bash", {"command": "site pick FA --jobs 4 && site submit ccin2p3 FA scripts/x.sh --item cov"}, True),
             ("Edit", {"file_path": ROOT + "/CLAUDE.md"}, False),
             ("Bash", {"command": "python3 tools/plan.py next"}, False),
             ("Bash", {"command": "git commit -m 'site submit fix' -- x"}, False),
             ("Bash", {"command": "site events --project FA"}, False)]
    got = [(t, ti, want, call(t, ti)) for t, ti, want in cases]
    scenario("D: the delegation hook notes the main session's code edits, analyses, tests and submissions, never "
             "refuses, and lets plan, git and reading calls be", all(rc == 0 and bool(out) == want
                                                                    for t, ti, want, (rc, out) in got),
             [(ti, want, out[:80]) for t, ti, want, (rc, out) in got if bool(out) != want])
    n = len(open(log).read().splitlines())
    sub = call("Edit", {"file_path": ROOT + "/tools/x.py"}, agent="o1")
    scenario("D: each noted call is counted once for site digest (5), a subagent's never",
             n == 5 and sub[1] == "" and len(open(log).read().splitlines()) == 5, n)


# ---------------------------------------------------------------- the supervisor's known-cause rule
def known_cause():
    import importlib.util, sqlite3
    sp = importlib.util.spec_from_file_location("site_runends", os.path.expanduser("~/.local/bin/site_runends.py"))
    R = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(R)
    lists = os.path.join(ROOT, "analysis", "catalog_v4", "data", "complete_runs.json")
    keys = set(json.load(open(lists)))
    rules = R.rules_of({"run_ends": {"known_causes": [{"line": r"\[FAIL\]", "pool": r"\[FAIL\] ([a-z0-9_]+)",
                                                       "lists": ["complete_runs.json"], "why": "w"}]}})
    rd = lambda p: keys if p == "complete_runs.json" else set()
    c = sqlite3.connect(os.path.expanduser("~/.local/share/ccorch/runs.db"))
    rows = c.execute("SELECT id, msg FROM events WHERE project='FA' AND kind='RUN' AND msg LIKE '%broke: [FAIL]%'").fetchall()
    import re as _re
    acked, kept = [], []
    for eid, msg in rows:
        m = _re.search(r"broke: (.*?)\. Read `site logs", msg)
        line = m.group(1) if m else msg
        (acked if R.known_cause(line, rules, rd) else kept).append((eid, line[:80]))
    pools_ok = all(_re.search(r"\[FAIL\] ([a-z0-9_]+)", l).group(1) in keys for _, l in acked)
    pools_out = all(not _re.search(r"\[FAIL\] ([a-z0-9_]+)", l) or _re.search(r"\[FAIL\] ([a-z0-9_]+)", l).group(1)
                    not in keys for _, l in kept)
    scenario("D: known cause: of the ledger's %d broke [FAIL] run ends, %d name a pool on the rebuild list and would be "
             "acked with that note, %d stay open" % (len(rows), len(acked), len(kept)), rows and pools_ok and pools_out,
             (acked[:2], kept[:2]))
    log = {"head": "", "tail": "[FAIL] dsh_bb_bbgg part 3: 120 of 1000 events\nCANCELLED AT x DUE TO TIME LIMIT",
           "matches": []}
    log = dict(log, tail="[FAIL] dsh_bb_bbgg part 3: 120 of 1000 events\n")
    d1 = R.decide_run_end("r1", "FAILED", "scripts/job_lhe_parts.sh", "cov", log, rules, read_list=rd)
    d2 = R.decide_run_end("r2", "FAILED", "scripts/job_lhe_parts.sh", "cov",
                          dict(log, tail="[FAIL] zz_not_listed part 1\n"), rules, read_list=rd)
    d3 = R.decide_run_end("r1", "FAILED", "scripts/job_lhe_parts.sh", "cov", log, R.rules_of({}), read_list=rd)
    scenario("D: decide_run_end acks a listed pool's failure with the note, keeps an unlisted one open, and changes "
             "nothing without the rule", "ack" in d1 and "rebuild list" in d1["ack"] and "event" in d2 and "event" in d3,
             (d1, d2, d3))


GROUPS += [d3, d4, delegation, known_cause]



def main():
    only = sys.argv[1:]
    for g in GROUPS:
        if not only or g.__name__ in only:
            g()
    print("%d/%d scenarios pass" % (sum(RESULTS), len(RESULTS)))
    return 0 if RESULTS and all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
