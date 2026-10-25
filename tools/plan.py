#!/usr/bin/env python3
"""The plan: the only source of truth for open work (setup redesign, D121).

Two files in the MAIN checkout (resolved through git's common dir, so worktrees and step agents share one plan):
  plan/plan.yaml     durable content, in git: id, title, question, status (running | waiting_user | held | done),
                     quotes [{said_at, text, decision}], decisions, next, done_when (a shell check, exit 0 = done),
                     notes_ref (a \\label in docs/results.tex), match (the site item's globs), evidence {runs, commit,
                     paths}, dead_ends [{tried, why, at}], asks [{ask, at, answered_at, decision}], ask + waiting_since
                     + checked_until (waiting_user), hold_decision (held), writeup (done)
  plan/.state.json   runtime state, gitignored: per entry the step agents launched [{agent, at, reason}] and runs
Both change only under an exclusive lock (plan/plan.yaml.lock), written through a temp file in the same directory.

  plan show [--compact] | where
  plan add <id> --title --question --next --match GLOBS (--decision D | --ask Q)   (done_when set apart, see check)
  plan item <id> --decision D             create or attach the entry's site work item (same id, its match globs)
  plan check <id> --done-when CMD         the machine check that finishes it; refused if it passes in an empty dir
  plan ref <id> --label L                 notes_ref: an existing \\label in docs/results.tex
  plan answer <id> --kind K --quote Q [--set title|question|next=V]   the user's words: typed after the entry's last
                                          ask or step (after the hold, for a release), then `site decide`, then the
                                          entry; all or nothing
  plan ask <id> --ask Q --searched S | unanswered <id> | step <id> --agent <subagent id> | run <id> RUN...
  plan evidence <id> [--run R] [--commit C] [--path P] | dead-end <id> --tried T --why W | set-next <id> T
  plan next [--prompts]                   deterministic: what can move now, with each step agent's exact prompt
  plan done <id> --writeup C [--override-runs WHY]
  plan view [--write]
Work is done by step agents: `plan next` lists what can move, each step agent does one step, writes evidence, dead
ends and the next step through this CLI, and exits. A running entry is in flight while a registry job of it is live,
a unit of its site item's check is active, or a step agent recorded for it started within STEP_MIN minutes. A step is
a real subagent started with that entry's step prompt ("Entry: <id>"), one step per subagent, re-validated against its
transcript on every read (a step written into the state file by hand does not count). A waiting entry moves only
through plan answer or plan unanswered. The plan files change only through this CLI (plan_files_guard.py).
Env for tests: PLAN_FILE, PLAN_STATE, PLAN_RUNS_DB, PLAN_SITE, PLAN_SITE_MODULE, PLAN_TRANSCRIPTS, PLAN_AGENTS,
PLAN_ROOT."""
import argparse, contextlib, datetime as dt, fcntl, fnmatch, glob, importlib.machinery, importlib.util, json, os, re
import shlex, shutil, sqlite3, subprocess, sys, tempfile

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _test_env():
    """The PLAN_* variables that point the tool at a test world (tests/test_setup_scenarios.py). Honoured only with
    PLAN_TEST=1 and PLAN_ROOT under the temp directory, so a variable left in a shell never redirects the real plan."""
    if os.environ.get("PLAN_TEST") != "1":
        return {}
    tmp = os.path.realpath(tempfile.gettempdir())
    root = os.path.realpath(os.environ.get("PLAN_ROOT") or "/")
    if not root.startswith(tmp + os.sep):
        sys.exit("plan: PLAN_TEST=1 needs PLAN_ROOT under %s" % tmp)
    return {k: v for k, v in os.environ.items() if k.startswith("PLAN_")}


ENV = _test_env()


def main_root():
    """The main checkout: plan files live there whichever worktree runs the command."""
    if ENV.get("PLAN_ROOT"):
        return ENV["PLAN_ROOT"]
    r = subprocess.run(["git", "-C", HERE, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                       capture_output=True, text=True)
    common = r.stdout.strip()
    return os.path.dirname(common) if r.returncode == 0 and os.path.basename(common) == ".git" else HERE


ROOT = main_root()
PLAN_FILE = ENV.get("PLAN_FILE") or os.path.join(ROOT, "plan", "plan.yaml")
STATE_FILE = ENV.get("PLAN_STATE") or os.path.join(os.path.dirname(PLAN_FILE), ".state.json")
CHECK_CACHE = os.path.join(os.path.dirname(STATE_FILE), ".check_cache.json")
RUNS_DB = ENV.get("PLAN_RUNS_DB") or os.path.expanduser("~/.local/share/ccorch/runs.db")
SITE = shlex.split(ENV.get("PLAN_SITE", "site"))
PROJECT = "FA"
STATUSES = ("running", "waiting_user", "held", "done")
SETTABLE = ("title", "question", "next")          # status, done_when and notes_ref have their own commands
KINDS = ("approve", "scope", "prefer", "hold", "release", "forbid")
ASK_MARGIN_MIN = 15          # a question written in the same reply as `plan ask`, or just before it, counts
STEP_MIN = 30                # a step agent started within this many minutes is the entry's work in flight
CHECK_CACHE_MIN = 20         # an item check (sweep units on the sites) is rerun at most this often
# seconds per item check; the Stop hook (plan_check) is registered with 300
CHECK_TIMEOUT = int(ENV.get("PLAN_CHECK_TIMEOUT", 90))
BEGIN, END = "% plan:begin (generated by tools/plan.py view; edit plan/plan.yaml)", "% plan:end"


# ---------------------------------------------------------------- the site tool's own rules, imported, never copied
_SITE = None


def S():
    global _SITE
    if _SITE is None:
        path = ENV.get("PLAN_SITE_MODULE") or shutil.which("site")
        loader = importlib.machinery.SourceFileLoader("site_tool", path)
        mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("site_tool", loader))
        loader.exec_module(mod)
        _SITE = mod
    return _SITE


def terminal(state):
    w = (state or "").split()[:1]
    return bool(w) and w[0] in S().TERMINAL


def live(state):
    """A job still going: not terminal, not UNKNOWN (sacct returned nothing), not HELD."""
    w = (state or "").split()[:1]
    return bool(w) and w[0] not in S().TERMINAL and w[0] not in ("UNKNOWN", "HELD")


# ---------------------------------------------------------------- files, under one lock
def now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def parse(s):
    if not s:
        return None
    if isinstance(s, (int, float)):
        return dt.datetime.fromtimestamp(s, dt.timezone.utc)
    t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _read():
    d = (yaml.safe_load(open(PLAN_FILE)) if os.path.exists(PLAN_FILE) else None) or {}
    d.setdefault("entries", [])
    st = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}
    return d, st


def _write_atomic(path, text):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".plan_tmp_")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


@contextlib.contextmanager
def transaction(write=True):
    """The plan under its lock. Nothing is created where no plan exists (the main checkout before the switch-over):
    an untracked plan/ there makes `site sync` refuse every launch (2026-10-25, four pp launches)."""
    if not os.path.exists(PLAN_FILE):
        if write:
            sys.exit("plan: no plan at %s (it is created by tools/plan_migrate.py at the switch-over)" % PLAN_FILE)
        yield {"entries": []}, {}
        return
    with open(PLAN_FILE + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
        d, st = _read()
        yield d, st
        if write:
            _write_atomic(PLAN_FILE, "# The plan: the only source of truth for open work. Edit through tools/plan.py.\n"
                          + yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=120))
            _write_atomic(STATE_FILE, json.dumps(st, indent=1, sort_keys=True))
            commit_plan()


def commit_plan():
    """Commit plan/plan.yaml alone, under the lock, after every write that changed it: a tracked file left dirty in
    the main checkout makes `site sync` refuse every launch. Outside a git checkout (a test world) nothing is done."""
    top = os.path.dirname(PLAN_FILE)
    g = lambda *a: subprocess.run(["git", "-C", top] + list(a), capture_output=True, text=True)
    if g("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        return
    rel = os.path.basename(PLAN_FILE)
    tracked = g("ls-files", "--error-unmatch", rel).returncode == 0
    if tracked and g("diff", "--quiet", "HEAD", "--", rel).returncode == 0:
        return
    if not tracked:
        g("add", "--", rel)
    import time
    for wait in (0, 1, 2, 4, 8):                 # an index.lock or a merge in progress clears within seconds
        time.sleep(wait)
        r = g("commit", "-q", "-m", "plan: %s" % " ".join(COMMAND), "--only", "--", rel)
        busy = "index.lock" in r.stderr or "during a merge" in r.stderr or "MERGE_HEAD" in r.stderr
        if r.returncode == 0 or not busy:
            break
    if r.returncode != 0:
        sys.stderr.write("plan: recorded, but committing %s failed (%s); run `python3 tools/plan.py commit` once "
                         "git is free, before any launch\n" % (
            PLAN_FILE, (r.stderr or r.stdout).strip()[-300:]))
        sys.exit(3)


COMMAND = ["?"]


def load():
    with transaction(write=False) as (d, st):
        return d, st


def entry(d, eid):
    for e in d["entries"]:
        if e["id"] == eid:
            return e
    sys.exit("plan: no entry %r" % eid)


def words(s):
    return set(re.findall(r"[a-z0-9]{5,}", (s or "").lower()))


def overlap(ask, text):
    w = words(ask)
    return bool(w) and len(w & words(text)) >= 0.6 * len(w)


# ---------------------------------------------------------------- registry
def db():
    return sqlite3.connect(RUNS_DB) if os.path.exists(RUNS_DB) else None


def runs_of(e, st):
    """(id, state, finished) of every run of the entry: its recorded runs, its evidence runs and every run filed under
    item == id, read from the whole registry (no row limit)."""
    c = db()
    if c is None:
        return []
    ids = sorted(set((st.get(e["id"]) or {}).get("runs", []) + ((e.get("evidence") or {}).get("runs") or [])))
    q = "SELECT id, state, finished FROM runs WHERE project=? AND (item=?%s)" % (
        " OR id IN (%s)" % ",".join("?" * len(ids)) if ids else "")
    return c.execute(q, [PROJECT, e["id"]] + ids).fetchall()


def decision(did):
    c = db()
    r = c.execute("SELECT kind, in_force, covers, said_at, quote FROM decisions WHERE id=? AND project=?",
                  (did, PROJECT)).fetchone() if c and did else None
    return {"kind": r[0], "in_force": bool(r[1]), "covers": [x.strip() for x in (r[2] or "").split(",")],
            "said_at": parse(r[3]), "quote": r[4] or ""} if r else None


def item_decision(eid):
    c = db()
    r = c.execute("SELECT decision FROM items WHERE project=? AND id=?", (PROJECT, eid)).fetchone() if c else None
    return (r or [None])[0]


def names_it(dd, did, eid):
    """The decision is about this entry: its covers name it (or a glob does), or it is the entry's site item's."""
    return covers_name(dd, eid) or did == item_decision(eid)


def quote_problem(q, eid):
    """Why a quote on an entry is not the user's recorded words for it (None: it is). Its decision is in the registry
    with that text (`site decide` stored it only after finding it in a message the user typed), is about this entry,
    carries the same said_at, and was said after the question or step it answers. Coverage is read from the covers
    recorded with the quote (a release edits the hold's covers in the registry afterwards)."""
    did = q.get("decision")
    dd = decision(did)
    if not dd:
        return "its decision %s is not in the registry" % did
    a, b = S()._norm(q.get("text") or ""), S()._norm(dd["quote"])
    if not a or not (a in b or b in a):
        return "%s's recorded words are not %r" % (did, (q.get("text") or "")[:60])
    snap = dict(dd, covers=q["covers"]) if q.get("covers") is not None else dd    # covers as recorded: site decide
    if not names_it(snap, did, eid):                                              # edits them on a later hold/release
        return "%s does not cover %s (covers: %s)" % (did, eid, ", ".join(snap["covers"])[:120])
    if dd["said_at"] and parse(q.get("said_at")) and abs((dd["said_at"] - parse(q["said_at"])).total_seconds()) > 1:
        return "the quote is dated %s, %s says %s" % (q["said_at"][:16], did, dd["said_at"].isoformat()[:16])
    if q.get("answers_after") and dd["said_at"] and dd["said_at"] <= parse(q["answers_after"]):
        return "%s was said before what it answers (%s)" % (did, q["answers_after"][:16])
    return None


def approval_problem(e):
    """A running entry needs an approval in force about it (approve, scope or release)."""
    for did in e.get("decisions") or []:
        dd = decision(did)
        if dd and dd["in_force"] and dd["kind"] in ("approve", "scope", "release") and names_it(dd, did, e["id"]):
            return None
    return "no approval in force names it (decisions: %s)" % (", ".join(e.get("decisions") or []) or "none")


def _check_cmd(eid):
    c = db()
    row = c.execute("SELECT check_cmd FROM items WHERE project=? AND id=?", (PROJECT, eid)).fetchone() if c else None
    return (row or [None])[0]


def _run_check(eid, cmd):
    """(active units, note): None when the check could not run (a site unreachable, a timeout), with the reason."""
    try:
        r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True,
                           timeout=max(1, min(CHECK_TIMEOUT, budget())))
        js = [l for l in r.stdout.splitlines() if l.strip().startswith("{")]
        if r.returncode != 0 or not js:
            return None, "exit %d: %s" % (r.returncode, (r.stderr or "no JSON").strip()[-160:])
        return sum(int(u.get("active") or 0) for u in (json.loads(js[-1]).get("units") or {}).values()), None
    except subprocess.TimeoutExpired:
        return None, "no answer in %d s" % CHECK_TIMEOUT
    except (ValueError, AttributeError, TypeError) as ex:
        return None, "bad output (%s)" % ex


def item_checks(eids):
    """{entry: active units of its site item's check} (sweep trials submitted by sweep_manager or a feeder never
    reach runs.db). The checks reach the sites: they run in parallel, each within CHECK_TIMEOUT (below the Stop
    hook's), and the result, a failure included, is cached for CHECK_CACHE_MIN. A check that cannot run gives None:
    the entry counts as in flight and a note says so (an unreachable site never blocks the turn)."""
    try:
        cache = json.load(open(CHECK_CACHE))
    except (OSError, ValueError):
        cache = {}
    out, todo = {}, {}
    for eid in eids:
        cmd = _check_cmd(eid)
        hit = cache.get(eid)
        if not cmd:
            out[eid] = 0
        elif hit and hit.get("cmd") == cmd and (now() - parse(hit["at"])).total_seconds() < 60 * CHECK_CACHE_MIN:
            out[eid] = hit["active"]
            if hit["active"] is None:
                NOTES.append("%s: its item check could not run (%s, since %s); counted as in flight" % (
                    eid, hit.get("note"), hit.get("failing_since", hit["at"])[:16]))
        else:
            todo[eid] = cmd
    if todo:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(len(todo)) as ex:
            res = dict(zip(todo, ex.map(lambda k: _run_check(k, todo[k]), todo)))
        for eid, (active, note) in res.items():
            out[eid] = active
            prev = cache.get(eid) or {}
            cache[eid] = {"cmd": todo[eid], "at": now().isoformat(), "active": active, "note": note}
            if active is None:
                cache[eid]["failing_since"] = prev.get("failing_since") if prev.get("active") is None and \
                    prev.get("failing_since") else now().isoformat()
                NOTES.append("%s: its item check could not run (%s, since %s); counted as in flight" % (
                    eid, note, cache[eid]["failing_since"][:16]))
        if os.path.exists(PLAN_FILE):                 # never create plan/ where no plan exists
            _write_atomic(CHECK_CACHE, json.dumps(cache))
    return out


def item_check_active(eid):
    return item_checks([eid])[eid]


# ---------------------------------------------------------------- transcripts (the user's typed words, site's rules)
def slug():
    return re.sub(r"[^A-Za-z0-9]", "-", os.path.realpath(ROOT))


def transcript_files(since=None):
    if ENV.get("PLAN_TRANSCRIPTS"):
        files = glob.glob(os.path.join(ENV["PLAN_TRANSCRIPTS"], "*.jsonl"))
    else:
        files = glob.glob(os.path.join(S().TRANSCRIPTS, slug() + "*", "*.jsonl"))   # main sessions, not subagents
    if since:
        files = [f for f in files if os.path.getmtime(f) >= since.timestamp()]
    return [f for f in files if S()._interactive(f)]


def messages(since=None):
    """(time, role, text) of the interactive sessions after `since`: role "user" for the user's typed words (site's
    _typed_texts), "assistant" for the text the user was sent."""
    out = []
    for f in transcript_files(since):
        for line in open(f, errors="replace"):
            try:
                m = json.loads(line)
                ts = parse(m.get("timestamp"))
            except (ValueError, TypeError):
                continue
            if not ts or (since and ts < since):
                continue
            if m.get("type") == "assistant":
                text = " ".join(x.get("text", "") for x in (m.get("message") or {}).get("content") or []
                                if isinstance(x, dict) and x.get("type") == "text")
                if text.strip():
                    out.append((ts, "assistant", text))
            else:
                for t in S()._typed_texts(m):
                    out.append((ts, "user", t))
    return sorted(out, key=lambda x: x[0])


def typed_at(quote, since=None):
    """When the user typed `quote` (site's own search; in tests over the fixture transcripts)."""
    if ENV.get("PLAN_TRANSCRIPTS"):
        q = S()._norm(quote)
        return sorted(ts for ts, role, t in messages(since) if role == "user" and q in S()._norm(t))
    return sorted(parse(x) for x in S().user_said(quote, PROJECT) if x and x != "?")


_AGENTS = {}


def subagent(agent_id):
    """(started, first prompt) of the subagent's transcript, or None when no such subagent exists. The first prompt is
    the user record that starts it (a fork's is the Agent call that spawned it)."""
    if agent_id in _AGENTS:
        return _AGENTS[agent_id]
    base = ENV.get("PLAN_AGENTS")
    pat = os.path.join(base, "agent-%s.jsonl" % agent_id) if base else \
        os.path.join(S().TRANSCRIPTS, slug() + "*", "*", "subagents", "agent-%s.jsonl" % agent_id)
    out = None
    for f in glob.glob(pat) if re.fullmatch(r"[A-Za-z0-9_-]+", agent_id or "") else []:
        started, prompt = None, None
        for line in open(f, errors="replace"):
            try:
                m = json.loads(line)
            except ValueError:
                continue
            started = started or parse(m.get("timestamp"))
            if prompt is None:
                c = (m.get("message") or {}).get("content")
                if m.get("type") == "user":
                    prompt = c if isinstance(c, str) else " ".join(x.get("text", "") for x in c or []
                                                                   if isinstance(x, dict))
                elif isinstance(c, list):
                    prompt = next((x["input"].get("prompt") for x in c if isinstance(x, dict) and
                                   x.get("type") == "tool_use" and x.get("name") == "Agent"), None)
            if started and prompt is not None:
                break
        if started:
            out = (started, prompt or "")
            break
    _AGENTS[agent_id] = out
    return out


def subagent_started(agent_id):
    r = subagent(agent_id)
    return r[0] if r else None


def names_entry(prompt, eid):
    """The step prompt of this entry is in the subagent's first prompt (plan/step_prompt.txt's "Entry: <id>" line)."""
    return re.search(r"(?m)^\s*Entry: %s(?=\s|\(|$)" % re.escape(eid), prompt or "") is not None


def step_problem(st, eid, s):
    """Why a recorded step is not a real step agent of this entry (None: it is). Run on every read, so a step
    written into the state file by hand does not count."""
    r = subagent(s.get("agent"))
    if r is None:
        return "no subagent %s in this project's transcripts" % s.get("agent")
    if abs((r[0] - parse(s.get("at"))).total_seconds()) > 1:
        return "recorded at %s, but subagent %s started %s" % (s.get("at"), s["agent"], r[0].isoformat())
    if not names_entry(r[1], eid):
        return "subagent %s was not started with %s's step prompt (no \"Entry: %s\" line)" % (s["agent"], eid, eid)
    uses = [k for k, v in entries_state(st) for x in v.get("steps") or [] if x.get("agent") == s["agent"]]
    if len(uses) > 1:
        return "subagent %s is recorded as %d steps (%s)" % (s["agent"], len(uses), ", ".join(uses))
    return None


def entries_state(st):
    """The per-entry part of the state file (keys starting with _ are the tool's own: _claims_verified)."""
    return [(k, v or {}) for k, v in st.items() if not k.startswith("_")]


def bad_steps(st):
    return [(eid, s, why) for eid, v in entries_state(st) for s in v.get("steps") or []
            for why in [step_problem(st, eid, s)] if why]


# ---------------------------------------------------------------- what can move now
def last_step(st, eid):
    """The latest recorded step that is a real step agent of the entry."""
    steps = [parse(s["at"]) for s in (st.get(eid) or {}).get("steps") or [] if not step_problem(st, eid, s)]
    return max(steps) if steps else None


def mark(d, st, eid):
    """The entry's last step, or the switch-over (plan.yaml's migrated_at, in git) when it is later."""
    ts = [x for x in (last_step(st, eid), parse(d.get("migrated_at"))) if x]
    return max(ts) if ts else None


def last_event(e, st):
    """The entry's latest ask or step: a quote for it must be newer."""
    ts = [parse(x.get("at")) for x in e.get("asks") or []] + [last_step(st, e["id"])]
    ts = [t for t in ts if t]
    return max(ts) if ts else None


NOTES = []                    # what the check could not see (an unreachable site): reported, never blocking
DEADLINE = [None]             # monotonic time by which the Stop hook's checks must end (plan_check sets it)


def budget():
    """Seconds left for checks that reach out (item checks, the claims check); large outside the Stop hook."""
    import time
    return 10 ** 6 if DEADLINE[0] is None else max(0.0, DEADLINE[0] - time.monotonic())


def why_moves(e, st, msgs, t, d=None, checks=None):
    """(reason, since) when the entry can move now, else None. `since` is when the reason arose."""
    s, ls = e.get("status"), mark(d or {}, st, e["id"])
    if s == "running":
        rows = runs_of(e, st)
        ended = [(r, fin) for r, state, fin in rows if terminal(state) and fin and (not ls or fin > ls.timestamp())]
        if ended:
            return "runs ended: " + ", ".join(r for r, _ in ended), parse(max(f for _, f in ended))
        if any(live(state) for r, state, fin in rows):
            return None
        active = (checks or {}).get(e["id"], 0) if checks is not None else item_check_active(e["id"])
        if active is None or active > 0:          # None: the check could not run; counted in flight, with a note
            return None
        if ls and 0 <= (t - ls).total_seconds() / 60 <= STEP_MIN:
            return None
        # the reason arose when the last step's window lapsed or the last run ended, whichever is later
        fins = [parse(fin) for r, state, fin in rows if fin]
        marks = [x for x in [ls + dt.timedelta(minutes=STEP_MIN) if ls else None] + fins if x]
        return "nothing in flight: " + (e.get("next") or "take the next step"), (max(marks) if marks else None)
    if s == "waiting_user":
        since = max([x for x in (parse(e.get("waiting_since")), parse(e.get("checked_until"))) if x]
                    or [dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)])
        said = [ts for ts, role, text in msgs if role == "user" and ts > since]
        if said:
            return ("the user wrote at %s after the question: plan answer with their words, or plan unanswered if "
                    "it does not answer it" % said[-1].isoformat()[:16]), said[-1]
    return None


ASKS_FIRST_MIN = 60          # with no scan recorded yet, re-asks are looked for in the last hour


def asks_mark(st):
    """Where the scan for answered questions asked again starts: the end of the previous scan (state _asks_scanned)."""
    return parse(st.get("_asks_scanned")) or now() - dt.timedelta(minutes=ASKS_FIRST_MIN)


def oldest_open(d, st=None):
    """The earliest time the transcripts are read from: the oldest OPEN question (less the margin) or the end of the
    previous re-ask scan, whichever is earlier."""
    ts = [parse(e.get("waiting_since")) - dt.timedelta(minutes=ASK_MARGIN_MIN) for e in d["entries"]
          if e.get("status") == "waiting_user" and e.get("waiting_since")]
    ts.append(asks_mark(st or {}))
    return min(ts)


def next_steps(d, st, t=None, msgs=None):
    """Deterministic: [(entry, reason, launched)]. A running entry's step is launched when a step agent was recorded
    after the reason arose; a waiting entry is never launched by a step (only plan answer or unanswered clear it)."""
    t = t or now()
    msgs = messages(oldest_open(d, st)) if msgs is None else msgs
    checks = item_checks([e["id"] for e in d["entries"] if e.get("status") == "running"])
    out = []
    for e in d["entries"]:
        r = why_moves(e, st, msgs, t, d, checks)
        if r:
            reason, since = r
            ls = last_step(st, e["id"])
            launched = e.get("status") == "running" and ls is not None and since is not None and ls >= since
            out.append((e, reason, launched))
    return out


def step_prompt(e, reason):
    tpl = open(os.path.join(HERE, "plan", "step_prompt.txt")).read()
    q = "\n".join('  "%s" (%s)' % (x.get("text"), (x.get("said_at") or "")[:16]) for x in e.get("quotes") or [])
    de = "\n".join("  %s: %s" % (x["tried"], x["why"]) for x in e.get("dead_ends") or []) or "  none"
    return tpl.format(id=e["id"], title=e.get("title"), question=e.get("question"), reason=reason,
                      next=e.get("next"), done_when=e.get("done_when"), quotes=q or "  none", dead_ends=de,
                      evidence=json.dumps(e.get("evidence") or {}))


# ---------------------------------------------------------------- site writes
def site_call(*args):
    r = subprocess.run(SITE + list(args), capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def item_exists(eid):
    c = db()
    return bool(c and c.execute("SELECT 1 FROM items WHERE project=? AND id=?", (PROJECT, eid)).fetchone())


def ensure_item(e, did):
    """The site work item of the same id with the entry's globs, so `site submit --item <id>` and work_gate pass."""
    if item_exists(e["id"]):
        return True, ""
    rc, out, err = site_call("item", "add", PROJECT, e["id"], "--decision", did, "--title", e.get("title") or e["id"],
                             "--scope", e.get("question") or e["id"], "--match", e.get("match") or e["id"] + "*",
                             "--notes", "docs/results.tex")
    return rc == 0, (err or out).strip()[-300:]


def rollback(did, why):
    """Undo a decision this call just recorded, when the rest of the call failed (all or nothing)."""
    c = sqlite3.connect(RUNS_DB)
    c.execute("UPDATE decisions SET in_force=0, note=? WHERE id=? AND project=?", ("rolled back by plan: " + why, did,
                                                                                  PROJECT))
    c.commit()


def covers_name(dd, eid):
    return any(x == eid or (any(ch in x for ch in "*?[") and fnmatch.fnmatch(eid, x)) for x in dd["covers"])


def trivial_check(cmd, timeout=120):
    """A check that passes in an empty directory with a bare environment does not look at the work."""
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(cmd, shell=True, cwd=tmp, env={"PATH": "/usr/bin:/bin"}, capture_output=True,
                           timeout=timeout)
    return r.returncode == 0


def labels():
    p = os.path.join(ROOT, "docs", "results.tex")
    return set(re.findall(r"\\label\{([^}]+)\}", open(p).read())) if os.path.exists(p) else set()


# ---------------------------------------------------------------- the numbers on trunk, checked as state
# seconds for verify_claims over the new trunk commits inside the Stop hook (below plan_check's 300)
CLAIMS_TIMEOUT = float(ENV.get("PLAN_CLAIMS_TIMEOUT", 240))
CLAIMS_CACHE = os.path.join(os.path.dirname(STATE_FILE), ".claims_cache.json")   # passed paragraphs, by content


def _set_state(key, value):
    """One key of the state file, under the lock, without rewriting plan.yaml."""
    with open(PLAN_FILE + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}
        st[key] = value
        _write_atomic(STATE_FILE, json.dumps(st, indent=1, sort_keys=True))


def claims_verified():
    """(last verified commit, trunk's head): read only, nothing run (auto_push.sh pushes trunk up to the first)."""
    g = lambda *a: subprocess.run(["git", "-C", ROOT] + list(a), capture_output=True, text=True)
    st = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}
    return st.get("_claims_verified"), g("rev-parse", "-q", "--verify", "trunk^{commit}").stdout.strip()


def claims_problems(limit=True):
    """verify_claims over every trunk commit since the last verified one (_claims_verified in the state file). This
    is the gate on the numbers in docs/results.tex, whatever way a commit reached trunk (hooks skipped, a merge, a
    rebase); the git hooks and git_verify_guard only catch a problem earlier. On a pass the watermark moves to trunk's
    head. With `limit` (the Stop hook) a check that does not finish in the time left blocks as "claims check
    incomplete", never a silent timeout of the hook; `plan claims` runs it with no limit, and paragraphs that passed
    are kept by content hash, so each run builds on the last."""
    if not os.path.exists(PLAN_FILE):
        return []
    g = lambda *a: subprocess.run(["git", "-C", ROOT] + list(a), capture_output=True, text=True)
    head = g("rev-parse", "-q", "--verify", "trunk^{commit}").stdout.strip()
    if not head:
        return []
    last = load()[1].get("_claims_verified")
    if last == head:
        return []
    if not last or g("cat-file", "-e", last + "^{commit}").returncode != 0:
        _set_state("_claims_verified", head)          # first run: nothing before the switch-over is re-litigated
        return []
    base = last if g("merge-base", "--is-ancestor", last, head).returncode == 0 else \
        g("merge-base", last, head).stdout.strip()
    left = min(CLAIMS_TIMEOUT, budget()) if limit else None
    incomplete = ["claims check incomplete: verify_claims over trunk %s..%s did not finish in the %d s left; run "
                  "`python3 tools/plan.py claims` (no time limit; passed paragraphs are kept) to finish it"
                  % (base[:8], head[:8], left or 0)]
    if left is not None and left < 5:
        return incomplete
    import signal
    pr = subprocess.Popen([sys.executable, os.path.join(ROOT, "tools", "verify_claims.py"), "--root", ROOT, "--rev",
                           "%s..%s" % (base, head), "--cache", CLAIMS_CACHE], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = pr.communicate(timeout=left)
    except subprocess.TimeoutExpired:
        os.killpg(pr.pid, signal.SIGKILL)               # the scripts it runs too, not only verify_claims
        pr.communicate()
        return incomplete
    r = subprocess.CompletedProcess(pr.args, pr.returncode, out, err)
    if r.returncode == 0:
        _set_state("_claims_verified", head)
        return []
    return ["numbers committed to docs/results.tex on trunk since %s fail the claims check (fix them in a new commit):"
            % base[:8]] + ["  " + l for l in r.stdout.splitlines() if l.startswith("CLAIM")]


DONE_RECHECK_H = 24           # a done entry's finishing checks are rerun at most this often


def done_problems(d, st):
    """[(entry, why)]: done entries that no longer pass what plan done checked. Cached for DONE_RECHECK_H hours under
    a key of every field the check reads, so any write to the entry reruns it. A regression is a suggestion (plan next
    lists `plan reopen`), never a standing block."""
    try:
        cache = json.load(open(CHECK_CACHE))
    except (OSError, ValueError):
        cache = {}
    out, changed = [], False
    for e in d["entries"]:
        if e.get("status") != "done":
            continue
        import hashlib
        runs = sorted(map(list, runs_of(e, st)))
        key = "done:%s:%s" % (e["id"], hashlib.sha256(json.dumps([e, runs], sort_keys=True, default=str).encode())
                              .hexdigest()[:16])
        hit = cache.get(key)
        if hit and (now() - parse(hit["at"])).total_seconds() < 3600 * DONE_RECHECK_H:
            why = hit["why"]
        else:
            try:
                why = done_refusals(e, st, e.get("writeup") or "?", e.get("runs_override"), timeout=min(30, budget()))
            except subprocess.TimeoutExpired:
                NOTES.append("%s: its done check did not finish; it runs again next time" % e["id"])
                continue
            cache[key], changed = {"at": now().isoformat(), "why": why}, True
        if why:
            out.append((e, "; ".join(why)))
    if changed and os.path.exists(PLAN_FILE):
        _write_atomic(CHECK_CACHE, json.dumps(cache))
    return out


# ---------------------------------------------------------------- commands
def cmd_reopen(a):
    """A done entry back to running (its write-up kept in its history), e.g. when plan next says it regressed."""
    with transaction() as (d, st):
        e = entry(d, a.id)
        if e.get("status") != "done":
            sys.exit("plan: %s is %s, not done" % (a.id, e.get("status")))
        e.setdefault("history", []).append({"at": now().isoformat(), "was": "done", "writeup": e.pop("writeup", None),
                                            "why": a.why})
        e["status"] = "running"
    print("plan: %s reopened" % a.id)


def cmd_commit(a):
    """Commit plan/plan.yaml when a write left it uncommitted (an index.lock or a merge in progress)."""
    with transaction(write=False):
        commit_plan()


def cmd_claims(a):
    """The claims check of trunk since the last verified commit, with no time limit (the Stop hook runs it within its
    budget). --verified prints the last verified commit and runs nothing (auto_push.sh)."""
    if a.verified:
        print(claims_verified()[0] or "")
        return 0
    bad = claims_problems(limit=False)
    for l in bad + ["note: " + n for n in NOTES]:
        print(l)
    return 1 if bad else 0

def cmd_where(a):
    print("plan file %s\nstate     %s\nregistry  %s" % (PLAN_FILE, STATE_FILE, RUNS_DB))


def cmd_show(a):
    d, st = load()
    for e in d["entries"]:
        s = e.get("status")
        if a.compact and s == "done":
            continue
        extra = ""
        if s == "running":
            ls = last_step(st, e["id"])
            lr = [r for r, state, fin in runs_of(e, st) if live(state)]
            extra = ("live runs " + ", ".join(lr)) if lr else ("last step %s" % ls.isoformat()[:16] if ls else "no step")
        elif s == "waiting_user":
            extra = "asked %s: %s (searched first: %s)" % ((e.get("waiting_since") or "?")[:16], e.get("ask", ""),
                                                           e.get("searched") or "not recorded")
        elif s == "held":
            extra = "held by %s" % e.get("hold_decision")
        print("- [%s] %s (%s): %s" % (s, e.get("title"), e["id"], extra))
        for k in (("next",) if a.compact else ("question", "next", "done_when", "notes_ref")):
            if e.get(k):
                print("    %s: %s" % (k, e[k]))
        if not a.compact:
            for q in e.get("quotes") or []:
                print("    user %s: \"%s\"" % ((q.get("said_at") or "")[:16], q.get("text")))
            for x in e.get("dead_ends") or []:
                print("    dead end: %s (%s)" % (x["tried"], x["why"]))


SEARCHED_MIN = 40             # characters: what was looked for before asking, named, not a token


def _searched_ok(searched):
    """A question goes to the user only once what could settle it was looked for: the CLAUDE.md rules (canonical
    setup, HPO rules) and the decisions in force (`site decisions --project FA`), and why none settles it. (The lr
    scalings entry waited three weeks on a question CLAUDE.md's HPO rule 2 and the measured lr surface settled.)"""
    if len((searched or "").strip()) < SEARCHED_MIN:
        sys.exit("plan: --searched must say what was looked for before asking (the CLAUDE.md rules and decisions read, "
                 "and why none settles it), in at least %d characters" % SEARCHED_MIN)


def _ask_ok(ask):
    if len(words(ask)) < 3:
        sys.exit("plan: the question needs at least three words of five letters or more, to be found in a message")


def cmd_add(a):
    if bool(a.decision) == bool(a.ask):
        sys.exit("plan: an entry starts approved (--decision D<n>, whose covers name it) or as a question (--ask)")
    if a.ask:
        _ask_ok(a.ask)
        _searched_ok(a.searched)
    with transaction() as (d, st):
        if any(e["id"] == a.id for e in d["entries"]):
            sys.exit("plan: entry %s exists" % a.id)
        e = {"id": a.id, "title": a.title, "question": a.question, "status": "running", "quotes": [],
             "decisions": [], "next": a.next, "done_when": None, "notes_ref": None, "match": a.match,
             "evidence": {"runs": [], "commit": None, "paths": []}, "dead_ends": [], "asks": []}
        if a.decision:
            dd = decision(a.decision)
            if not dd or not dd["in_force"] or dd["kind"] not in ("approve", "scope", "release"):
                sys.exit("plan: %s is not an approval in force" % a.decision)
            if not covers_name(dd, a.id):
                sys.exit("plan: %s's covers (%s) do not name %s: record the user's approval for it with plan answer"
                         % (a.decision, ", ".join(dd["covers"])[:200], a.id))
            ok, why = ensure_item(e, a.decision)
            if not ok:
                sys.exit("plan: site item %s not created, entry not added: %s" % (a.id, why))
            e["decisions"].append(a.decision)
        else:
            t = now().isoformat()
            e.update(status="waiting_user", ask=a.ask, waiting_since=t, searched=a.searched)
            e["asks"].append({"ask": a.ask, "at": t, "searched": a.searched})
        d["entries"].append(e)
    print("plan: added %s (%s); set its machine check with plan check" % (a.id, e["status"]))


def cmd_item(a):
    with transaction() as (d, st):
        e = entry(d, a.id)
        if a.match:
            e["match"] = a.match
        dd = decision(a.decision)
        if not dd or not dd["in_force"] or dd["kind"] not in ("approve", "scope", "release") or not covers_name(dd, a.id):
            sys.exit("plan: %s is not an approval in force naming %s" % (a.decision, a.id))
        ok, why = ensure_item(e, a.decision)
        if not ok:
            sys.exit("plan: site item %s not created: %s" % (a.id, why))
        if a.decision not in e.setdefault("decisions", []):
            e["decisions"].append(a.decision)
    print("plan: site item %s attached" % a.id)


def cmd_check(a):
    if trivial_check(a.done_when):
        sys.exit("plan: %r passes in an empty directory: it checks nothing about the work" % a.done_when)
    r = subprocess.run(a.done_when, shell=True, cwd=ROOT, capture_output=True, timeout=600)
    if r.returncode == 0:
        sys.exit("plan: %r already passes in the checkout: it would finish the entry with nothing done; check "
                 "what this entry still has to produce" % a.done_when)
    with transaction() as (d, st):
        entry(d, a.id)["done_when"] = a.done_when
    print("plan: %s finishes when `%s` passes" % (a.id, a.done_when))


def cmd_ref(a):
    if a.label not in labels():
        sys.exit("plan: no \\label{%s} in docs/results.tex" % a.label)
    with transaction() as (d, st):
        entry(d, a.id)["notes_ref"] = a.label


def cmd_answer(a):
    """The user's words: checked first (typed, and after what they answer), then `site decide`, then the entry. A
    failure after site decide rolls the decision back, so nothing half-applies."""
    sets = dict(s.split("=", 1) for s in a.set or [])
    bad = [k for k in sets if k not in SETTABLE]
    if bad:
        sys.exit("plan: cannot set %s here (settable: %s; status through ask/hold/release/done, done_when through "
                 "plan check, notes_ref through plan ref)" % (", ".join(bad), ", ".join(SETTABLE)))
    with transaction() as (d, st):
        e = entry(d, a.id)
        times = typed_at(a.quote)
        if not times:
            sys.exit("plan: not recorded, the entry is unchanged: these words are not in any message the user typed")
        if e.get("status") == "waiting_user":
            since, what = parse(e.get("waiting_since")), "the question was asked"
        elif a.kind == "release" and e.get("status") == "held":
            since, what = (decision(e.get("hold_decision")) or {}).get("said_at"), "the hold was said"
        else:
            since, what = last_event(e, st), "the entry's last ask or step"
        after = [t for t in times if not since or t > since]
        if not after:
            sys.exit("plan: not recorded: the user typed these words before %s (%s)" % (
                what, since.isoformat()[:16] if since else "?"))
        if a.kind in ("hold", "release") and not item_exists(e["id"]):
            sys.exit("plan: %s has no site item to %s; attach one first (plan item)" % (e["id"], a.kind))
        rc, out, err = site_call("decide", PROJECT, "--kind", a.kind, "--quote", a.quote, "--covers", e["id"])
        m = re.search(r"\b(D\d+)\b", out)
        if rc != 0 or not m:
            sys.exit("plan: not recorded, the entry is unchanged (site decide: %s)" % (err or out).strip()[-300:])
        did = m.group(1)
        dd = decision(did)
        said = (dd or {}).get("said_at")
        if not said or (since and said <= since):
            rollback(did, "said_at %s is not after %s" % (said, since))
            sys.exit("plan: site decide dated the words %s, not after %s (%s); %s rolled back, the entry is unchanged"
                     % (said.isoformat()[:16] if said else "?", what, since.isoformat()[:16] if since else "?", did))
        if a.kind == "approve":
            ok, why = ensure_item(e, did)
            if not ok:
                rollback(did, "item not created")
                sys.exit("plan: site item %s not created (%s); %s rolled back, the entry is unchanged" % (
                    e["id"], why, did))
        e.setdefault("quotes", []).append({"said_at": said.isoformat(), "text": a.quote, "decision": did,
                                           "answers_after": since.isoformat() if since else None,
                                           "covers": dd["covers"]})
        e.setdefault("decisions", []).append(did)
        for k, v in sets.items():
            e[k] = v
        if a.kind in ("hold", "forbid"):
            e["status"], e["hold_decision"] = "held", did
        elif a.kind == "release" and e.get("status") == "held":
            e["status"] = "running"
            e.pop("hold_decision", None)
        elif e.get("status") == "waiting_user":
            e["status"] = "running"
        if e["status"] != "waiting_user":
            for x in e.get("asks") or []:
                x.setdefault("answered_at", after[0].isoformat())
                x.setdefault("decision", did)
            for k in ("ask", "waiting_since", "checked_until"):
                e.pop(k, None)
    print("plan: %s recorded as %s; entry %s now %s" % (a.kind, did, e["id"], e["status"]))


def answered_asks(d):
    return [(e, x) for e in d["entries"] for x in e.get("asks") or [] if x.get("answered_at")]


def cmd_ask(a):
    _ask_ok(a.ask)
    _searched_ok(a.searched)
    with transaction() as (d, st):
        e = entry(d, a.id)
        if e.get("status") not in ("running", "waiting_user"):
            sys.exit("plan: %s is %s" % (a.id, e.get("status")))
        again = [(o, x) for o, x in answered_asks(d) if overlap(x["ask"], a.ask) or overlap(a.ask, x["ask"])]
        if again:
            o, x = again[0]
            sys.exit("plan: already answered: %r on %s, %s (%s); act on that answer" % (
                x["ask"], o["id"], x["answered_at"][:16], x.get("decision")))
        t = now().isoformat()
        e.update(status="waiting_user", ask=a.ask, waiting_since=t, searched=a.searched)
        e.pop("checked_until", None)
        e.setdefault("asks", []).append({"ask": a.ask, "at": t, "searched": a.searched})
    print("plan: %s waits on the user from %s" % (a.id, t))


def cmd_unanswered(a):
    """The user's messages since the question do not answer it (it stays open)."""
    with transaction() as (d, st):
        e = entry(d, a.id)
        if e.get("status") != "waiting_user":
            sys.exit("plan: %s is not waiting on the user" % a.id)
        e["checked_until"] = now().isoformat()


def cmd_step(a):
    """A step agent (a real subagent, its id) was launched for a running entry after the entry's reason arose."""
    with transaction() as (d, st):
        e = entry(d, a.id)
        if e.get("status") != "running":
            sys.exit("plan: %s is %s; a waiting entry moves through plan answer or unanswered, not a step" % (
                a.id, e.get("status")))
        rec = subagent(a.agent)
        if rec is None:
            sys.exit("plan: no subagent %s in this project's transcripts" % a.agent)
        started, prompt = rec
        if not names_entry(prompt, a.id):
            sys.exit("plan: subagent %s was not started with %s's step prompt (plan next --prompts; its \"Entry: %s\" "
                     "line)" % (a.agent, a.id, a.id))
        used = [k for k, v in entries_state(st) for x in v.get("steps") or [] if x.get("agent") == a.agent]
        if used:
            sys.exit("plan: subagent %s is already recorded as a step of %s; one subagent is one step" % (
                a.agent, ", ".join(used)))
        r = why_moves(e, st, [], now(), d)
        since = r[1] if r else None
        if since and started < since:
            sys.exit("plan: subagent %s started %s, before the reason arose (%s)" % (
                a.agent, started.isoformat()[:16], since.isoformat()[:16]))
        st.setdefault(a.id, {}).setdefault("steps", []).append({"agent": a.agent, "at": started.isoformat(),
                                                               "reason": r[0] if r else ""})
    print("plan: step on %s recorded (%s, started %s)" % (a.id, a.agent, started.isoformat()[:16]))


def cmd_run(a):
    with transaction() as (d, st):
        entry(d, a.id)
        rs = st.setdefault(a.id, {}).setdefault("runs", [])
        rs += [r for r in a.runs if r not in rs]


def cmd_evidence(a):
    with transaction() as (d, st):
        ev = entry(d, a.id).setdefault("evidence", {"runs": [], "commit": None, "paths": []})
        ev["runs"] = sorted(set((ev.get("runs") or []) + (a.run or [])))
        ev["paths"] = sorted(set((ev.get("paths") or []) + (a.path or [])) - set(a.drop or []))
        if a.commit:
            ev["commit"] = a.commit


def cmd_set_next(a):
    with transaction() as (d, st):
        entry(d, a.id)["next"] = a.text


def cmd_dead_end(a):
    with transaction() as (d, st):
        entry(d, a.id).setdefault("dead_ends", []).append({"tried": a.tried, "why": a.why, "at": now().isoformat()})


def cmd_next(a):
    """What can move now, and the exact prompt of each step agent to launch (exit 1 while anything is unlaunched)."""
    d, st = load()
    todo = [(e, why) for e, why, launched in next_steps(d, st) if not launched]
    for e, why in todo:
        print("NEXT %s: %s" % (e["id"], why))
        if a.prompts and e.get("status") == "running":
            print(step_prompt(e, why))
            print("-" * 60)
    for e, why in done_problems(d, st):
        print("SUGGEST %s: marked done, but %s; plan reopen %s --why ..., or fix it (plan evidence --drop ...)" % (
            e["id"], why, e["id"]))
    print("plan: %d step(s) to launch" % len(todo))
    return 1 if todo else 0


def done_refusals(e, st, commit, override_runs=None, timeout=600):
    """Why the entry cannot be marked done (empty: it can)."""
    out = []
    chk = e.get("done_when")
    if not chk:
        out.append("no done_when check (plan check)")
    elif trivial_check(chk, min(timeout, 120)):
        out.append("done_when %r checks nothing about the work" % chk)
    else:
        r = subprocess.run(chk, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            out.append("done_when fails (%s): %s" % (chk, (r.stderr or r.stdout).strip()[-200:]))
    ev = e.get("evidence") or {}
    if not (ev.get("paths") or ev.get("runs")):
        out.append("no evidence recorded (plan evidence)")
    miss = [p for p in ev.get("paths") or [] if not os.path.exists(os.path.join(ROOT, p))]
    if miss:
        out.append("evidence paths missing: " + ", ".join(miss))
    ref = e.get("notes_ref")
    if not ref or ref not in labels():
        out.append("notes_ref %r is not a \\label in docs/results.tex (plan ref)" % ref)
    show = subprocess.run(["git", "-C", ROOT, "show", "--format=%ct", "--unified=0", commit, "--", "docs/results.tex"],
                          capture_output=True, text=True)
    lines = show.stdout.splitlines()
    added = "\n".join(l[1:] for l in lines[1:] if l.startswith("+") and not l.startswith("+++"))
    names = [n for n in (ref, e.get("title")) if n]
    if show.returncode != 0 or not added:
        out.append("%s adds nothing to docs/results.tex" % commit)
    elif not any(n.lower() in added.lower() for n in names):
        out.append("%s's addition to docs/results.tex names neither %s" % (commit, " nor ".join(map(repr, names))))
    ls = last_step(st, e["id"])
    if show.returncode == 0 and lines and lines[0].strip().isdigit() and ls and int(lines[0]) < ls.timestamp():
        out.append("%s was committed before the entry's last step (%s)" % (commit, ls.isoformat()[:16]))
    rows = runs_of(e, st)
    unfinished = [r for r, s, fin in rows if live(s)]
    unknown = [r for r, s, fin in rows if not live(s) and not terminal(s)]
    if unfinished:
        out.append("runs still going: " + ", ".join(unfinished))
    if unknown and not override_runs:
        out.append("runs in an unknown or held state: %s (pass --override-runs with the reason)" % ", ".join(unknown))
    return out


def cmd_done(a):
    with transaction() as (d, st):
        e = entry(d, a.id)
        why = done_refusals(e, st, a.writeup, a.override_runs)
        if why:
            sys.exit("plan: %s is not done:\n  " % a.id + "\n  ".join(why))
        e["status"], e["writeup"] = "done", a.writeup
        if a.override_runs:
            e["runs_override"] = a.override_runs
    print("plan: %s done (%s)" % (a.id, a.writeup))


def render(d):
    tex = lambda s: re.sub(r"([_&%#$])", r"\\\1", str(s or ""))
    out = [BEGIN, "\\begin{itemize}"]
    for e in d["entries"]:
        st = {"running": "running", "waiting_user": "waiting on the user", "held": "held"}.get(e.get("status"))
        if st:
            body = tex(e.get("question")) + (" Next: " + tex(e["next"]) if e.get("next") else "")
            if e.get("status") == "waiting_user":
                body += " Asked: " + tex(e.get("ask")) + " Searched first: " + tex(e.get("searched") or "not recorded")
            out.append("  \\item \\textbf{%s} (%s). %s" % (tex(e.get("title")), st, body))
    return "\n".join(out + ["\\end{itemize}", END])


def cmd_view(a):
    v = render(load()[0])
    if not a.write:
        return print(v)
    p = os.path.join(ROOT, "docs", "results.tex")
    s = open(p).read()
    i, j = s.find(BEGIN), s.find(END)
    if i < 0 or j < 0:
        sys.exit("plan: no plan markers in docs/results.tex")
    _write_atomic(p, s[:i] + v + s[j + len(END):])


def main(argv=None):
    p = argparse.ArgumentParser(prog="plan")
    sub = p.add_subparsers(dest="cmd", required=True)
    A = lambda *n, **k: (n, k)

    def cmd(name, f, *args):
        s = sub.add_parser(name)
        for n, k in args:
            s.add_argument(*n, **k)
        s.set_defaults(f=f)
    cmd("where", cmd_where)
    cmd("show", cmd_show, A("--compact", action="store_true"))
    cmd("add", cmd_add, A("id"), A("--title", required=True), A("--question", required=True),
        A("--next", required=True), A("--match", required=True), A("--decision"), A("--ask"), A("--searched"))
    cmd("item", cmd_item, A("id"), A("--decision", required=True), A("--match"))
    cmd("check", cmd_check, A("id"), A("--done-when", required=True))
    cmd("ref", cmd_ref, A("id"), A("--label", required=True))
    cmd("answer", cmd_answer, A("id"), A("--quote", required=True), A("--kind", required=True, choices=KINDS),
        A("--set", action="append"))
    cmd("ask", cmd_ask, A("id"), A("--ask", required=True), A("--searched", required=True))
    cmd("unanswered", cmd_unanswered, A("id"))
    cmd("step", cmd_step, A("id"), A("--agent", required=True))
    cmd("run", cmd_run, A("id"), A("runs", nargs="+"))
    cmd("evidence", cmd_evidence, A("id"), A("--run", action="append"), A("--commit"), A("--path", action="append"),
        A("--drop", action="append", help="a path no longer evidence (pruned, moved)"))
    cmd("set-next", cmd_set_next, A("id"), A("text"))
    cmd("dead-end", cmd_dead_end, A("id"), A("--tried", required=True), A("--why", required=True))
    cmd("next", cmd_next, A("--prompts", action="store_true"))
    cmd("done", cmd_done, A("id"), A("--writeup", required=True), A("--override-runs"))
    cmd("view", cmd_view, A("--write", action="store_true"))
    cmd("claims", cmd_claims, A("--verified", action="store_true"))
    cmd("reopen", cmd_reopen, A("id"), A("--why", required=True))
    cmd("commit", cmd_commit)
    a = p.parse_args(argv)
    COMMAND[:] = [a.cmd] + ([a.id] if getattr(a, "id", None) else [])
    return a.f(a) or 0


if __name__ == "__main__":
    sys.exit(main())
