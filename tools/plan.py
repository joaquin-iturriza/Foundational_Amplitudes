#!/usr/bin/env python3
"""The plan: the only source of truth for open work (setup redesign, D121).

Two files in the MAIN checkout (resolved through git's common dir, so worktrees and step agents share one plan):
  plan/plan.yaml     durable content, in git: id, title, question, status (running | waiting_user | held | done),
                     quotes [{said_at, text, decision}], decisions, next, done_when (a shell check, exit 0 = done),
                     notes_ref (a \\label in docs/results.tex), match (the site item's globs), evidence {runs, commit,
                     paths}, dead_ends [{tried, why, at}], asks [{ask, at, answered_at, decision}], the open question: ask +
                     waiting_since + searched + blocks (what waits on the answer) + checked_until, on a running entry,
                     or with holds_all (why everything waits) on a waiting_user one, hold_decision (held), writeup
                     (done), designs [{at, question, prior, measure, check, fixed, control, seeds, outcome, passed_at, reviewer,
                     fix_at, fix}]
  plan/.state.json   runtime state, gitignored: per entry the step agents launched [{agent, at, reason}], the design
                     reviewers launched [{agent, at, design_at}] and runs
Both change only under an exclusive lock (plan/plan.yaml.lock), written through a temp file in the same directory.

  plan show [<id>] [--compact] | where   (one entry in full with an id; --help on every subcommand)
  plan add <id> --title --question --next --match GLOBS (--decision D | --ask Q)   (done_when set apart, see check)
  plan item <id> --decision D             create or attach the entry's site work item (same id, its match globs)
  plan check <id> --done-when CMD         the machine check that finishes it; refused if it passes in an empty dir
  plan ref <id> --label L                 notes_ref: an existing \\label in docs/results.tex
  plan answer <id> --kind K --quote Q [--set title|question|next=V]   the user's words: typed after the entry's last
                                          ask or step (after the hold, for a release), then `site decide`, then the
                                          entry; all or nothing
  plan ask <id> --ask Q --searched S [--blocks B | --holds-all WHY]   an open question on the entry: it stays running
                                          and routed, its step prompt names the question and what waits on it (B);
                                          --holds-all makes the whole entry wait on the user (WHY: why nothing in
                                          it can move without the answer)
  plan design <id> --question Q --prior P --measure M --check C --fixed F --control K --seeds S --outcome O
                                          the design of the next job, before it is submitted: the question it
                                          answers, the prior work found on that question (each with its result
                                          and how this design differs, or "none found" with the search terms),
                                          the measurement that answers it, how it was checked on existing
                                          data that the measurement moves with the tested change, what is held
                                          fixed, the matched control, the seeds, the outcome read as yes and as no;
                                          design_guard.py refuses a submission for an entry without one recorded
                                          in DESIGN_H h and passed by the design reviewer, which the step agent
                                          that recorded it runs itself before submitting
  plan review <id> --agent <subagent id>  the fallback: a design with no verdict REVIEW_FALLBACK_MIN after it was
                                          recorded (its step agent gone) is listed by plan next as a design to
                                          review; the main session launches the design reviewer
                                          (.claude/agents/design-reviewer.md) from that prompt and records it here
  plan design-pass <id> --reviewer NOTE | design-fix <id> --reasons R    the reviewer's verdict on the latest design
                                          (run by the reviewer itself; design_guard.py refuses them from anyone else)
  plan unanswered <id> | step <id> --agent <subagent id> [--next N]
  plan run <id> [RUN...] [--sweep SITE:NAME] [--next N]   a registry run, or a sweep sweep_manager submitted (never in
                                          runs.db); --next: the step it belongs to
  plan set-next <id> "<step>" [--also "<independent step>" ...] [--next N]   the entry's next steps (with --next N:
                                          replace step N only) | add-next <id> "<step>" | drop-next <id> <N>
  plan sweeps [--fresh]                   every recorded sweep's state on its site, and the unrecorded ones recorded
  plan evidence <id> [--run R] [--commit C] [--path P] | dead-end <id> --tried T --why W
  plan next [--prompts]                   deterministic: what can move now, with each step agent's exact prompt
  plan done <id> --writeup C [--override-runs WHY]
  plan view [--write]
Work is done by step agents: `plan next` lists what can move, each step agent does one step, writes evidence, dead
ends and the next step through this CLI, and exits. A running entry is in flight while a registry job of it is live,
a sweep recorded on it has a job in its site's queue, a unit of its site item's check is active, or a step agent
recorded for it started within STEP_MIN minutes. A sweep no entry records, live or submitted within SWEEP_RECENT_H
hours, is recorded on the one running entry whose match globs name it whenever plan next or the Stop check runs;
one that fits several blocks until recorded by hand. An entry holds one or several independent next steps (numbered,
the numbers stable while others are added or dropped): each is listed, launched and in flight on its own, so a free step
is listed while another step of the entry runs; its runs, sweeps and step agents are the ones recorded with its number
(--next N), and anything recorded without one belongs to the first step. A step is
a real subagent started with that entry's step prompt ("Entry: <id>"), one step per subagent, re-validated against its
transcript on every read (a step written into the state file by hand does not count). A waiting entry moves only
through plan answer or plan unanswered; a running entry with an open question keeps moving on everything the
question does not block (CLAUDE.md 0g: only what needs the answer waits). The plan files change only through this CLI (plan_files_guard.py).
Env for tests: PLAN_FILE, PLAN_STATE, PLAN_RUNS_DB, PLAN_SITE, PLAN_SITE_MODULE, PLAN_TRANSCRIPTS, PLAN_AGENTS,
PLAN_ROOT, PLAN_SWEEP_SITES."""
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


# ---------------------------------------------------------------- an entry's independent next steps
def nexts(e):
    """[(number, text)] of the entry's next steps: one string (step 1, as every entry held before), or {number: text}
    when it holds several independent ones. The numbers are stable: a step agent's number stays its own while other
    steps are added or dropped."""
    n = e.get("next")
    if isinstance(n, dict):
        return sorted((int(k), v) for k, v in n.items() if v)
    if isinstance(n, list):
        return [(i + 1, v) for i, v in enumerate(n) if v]
    return [(1, n)] if n else []


def step_keys(e):
    return [k for k, _ in nexts(e)] or [1]


def step_text(e, k):
    return dict(nexts(e)).get(k)


def put_nexts(e, items):
    """Store [(number, text)]: a lone step 1 as a plain string (the one-step form), else {number: text}."""
    items = [(k, t) for k, t in items if t]
    e["next"] = None if not items else items[0][1] if len(items) == 1 and items[0][0] == 1 else dict(items)


def step_of(e, k):
    """The step a run or sweep recorded with number k belongs to: k while the entry holds it, else (recorded without
    a number, or its step dropped) the first step, so no job of the entry goes unwatched."""
    keys = step_keys(e)
    return k if k in keys else keys[0]


def rec_of(e, x):
    """The step a recorded step agent belongs to: its number, or the first step when recorded without one (before
    numbered steps); None when its step was dropped (its agent no longer holds any step of the entry)."""
    k = x.get("next")
    if k is None:
        return step_keys(e)[0]
    return k if k in step_keys(e) else None


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


# ---------------------------------------------------------------- sweeps (sweep_manager's jobs never reach runs.db)
SWEEP_RECENT_H = 24      # an unrecorded sweep submitted this recently is recorded even if it has already ended


def note(n):
    if n not in NOTES:
        NOTES.append(n)


def sweep_name(x):
    """A sweep's name from its name or directory (sweep_manager's sweep_name_from_dir); "" for nothing."""
    return os.path.basename(os.path.normpath(x)) if (x or "").strip(" /.") else ""


def sweep_sites():
    """The sites whose sweep_manager registry is read for sweeps no entry records: sites/sites.yaml's SLURM sites
    (sweep_manager submits with sbatch; lxplus's legacy HTCondor DAG path keeps no registry)."""
    if "PLAN_SWEEP_SITES" in ENV:
        return [x for x in ENV["PLAN_SWEEP_SITES"].split(",") if x]
    try:
        y = yaml.safe_load(open(os.path.join(HERE, "sites", "sites.yaml"))) or {}
    except OSError:
        return []
    return sorted(k for k, v in y.items()
                  if isinstance(v, dict) and (v.get("cluster") or {}).get("scheduler") == "slurm")


def sweeps_of(st, eid):
    return list((st.get(eid) or {}).get("sweeps") or [])


def want_of(st, eids):
    """{site: names} of the sweeps recorded on these entries."""
    out = {}
    for eid in eids:
        for x in sweeps_of(st, eid):
            out.setdefault(x["site"], set()).add(x["name"])
    return out


def _read_site_sweeps(site, names):
    """(sweeps, note): `sweep_manager.py state` on the site, read only. `site run` names no work item, so the site's
    sbatch shim refuses anything it might submit."""
    try:
        r = subprocess.run(SITE + ["run", site, PROJECT, "--", "python", "sweep/sweep_manager.py", "state"]
                           + sorted(names), capture_output=True, text=True,
                           timeout=max(1, min(CHECK_TIMEOUT, budget())))
    except subprocess.TimeoutExpired:
        return None, "no answer in %d s" % CHECK_TIMEOUT
    line = [l for l in r.stdout.splitlines() if l.startswith("SWEEP_STATE ")]
    if r.returncode != 0 or not line:
        return None, "exit %d: %s" % (r.returncode, (r.stderr or r.stdout or "no SWEEP_STATE line").strip()[-160:])
    try:
        return dict(json.loads(line[-1][len("SWEEP_STATE "):])["sweeps"]), None
    except (ValueError, KeyError, TypeError) as ex:
        return None, "bad output (%s)" % ex


def site_sweeps(want, all_sites=True, fetch=True, fresh=False):
    """{site: {"sweeps": {name: facts | None (not in the registry)}, "ended": {name: first seen ended}} or None (could
    not be read)} for the sites of `want` ({site: names to report even if ended}), and with all_sites every
    sweep_sites() site. Read in parallel and cached CHECK_CACHE_MIN per site like the item checks, a failure included;
    an unreadable site gets a note and its recorded sweeps count as in flight. The end of a sweep is the first read
    that found none of its jobs in the queue, kept across reads. fetch=False reads the cache only (plan show)."""
    try:
        cache = json.load(open(CHECK_CACHE))
    except (OSError, ValueError):
        cache = {}
    sites = set(want) | (set(sweep_sites()) if all_sites else set())
    out, todo = {}, {}
    for site in sorted(sites):
        names = set(want.get(site) or ())
        hit = cache.get("sweeps:" + site)
        ok = hit and not fresh and names <= set(hit.get("names") or []) and \
            (now() - parse(hit["at"])).total_seconds() < 60 * CHECK_CACHE_MIN
        if ok or (hit and not fetch):
            data = hit.get("data")
            out[site] = {"sweeps": data, "ended": hit.get("ended") or {}} if data is not None else None
            if hit.get("data") is None:
                note("the sweep registry on %s could not be read (%s, since %s): its recorded sweeps count as in flight"
                     % (site, hit.get("note"), (hit.get("failing_since") or hit["at"])[:16]))
        elif fetch:
            todo[site] = names | set((hit or {}).get("names") or [])
        else:
            out[site] = None
    if todo:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(len(todo)) as ex:
            res = dict(zip(todo, ex.map(lambda k: _read_site_sweeps(k, todo[k]), todo)))
        try:
            cache = json.load(open(CHECK_CACHE))          # the item checks may have written it meanwhile
        except (OSError, ValueError):
            cache = {}
        for site, (data, why) in res.items():
            prev = cache.get("sweeps:" + site) or {}
            ended = dict(prev.get("ended") or {})
            rec = {"at": now().isoformat(), "names": sorted(todo[site]), "data": data, "note": why}
            if data is None:
                rec["failing_since"] = prev.get("failing_since") if prev.get("data") is None and \
                    prev.get("failing_since") else now().isoformat()
                note("the sweep registry on %s could not be read (%s, since %s): its recorded sweeps count as in flight"
                     % (site, why, rec["failing_since"][:16]))
            else:
                for name, f in data.items():
                    if f is None or f.get("live"):
                        ended.pop(name, None)
                    else:
                        ended.setdefault(name, now().isoformat())
            rec["ended"] = ended
            cache["sweeps:" + site] = rec
            out[site] = {"sweeps": data, "ended": ended} if data is not None else None
        if os.path.exists(PLAN_FILE):
            _write_atomic(CHECK_CACHE, json.dumps(cache))
    return out


def sweep_label(name, site, f):
    return "%s (%s, %s of %s trials submitted, %s results)" % (
        name, site, f.get("submitted"), "?" if f.get("scripts") is None else f["scripts"],
        "?" if f.get("results") is None else f["results"])


def sweep_status(eid, st, sw, keep=None):
    """(ended [(label, ended at, recorded at)], live [label], unread [label]) of the sweeps recorded on the entry (those
    `keep` takes, when given). One missing from its site's registry is listed as ended (from when it was recorded), so
    a step agent looks at it."""
    ended, lv, unread = [], [], []
    for x in sweeps_of(st, eid):
        if keep and not keep(x):
            continue
        site, name, rec = x["site"], x["name"], parse(x.get("at"))
        data = sw.get(site)
        if data is None:
            unread.append("%s (%s)" % (name, site))
            continue
        f = data["sweeps"].get(name)
        if f is None:
            ended.append(("%s (%s: not in its sweep registry)" % (name, site), rec, rec))
        elif f.get("live"):
            lv.append("%s (%s, %d jobs queued or running)" % (name, site, f["live"]))
        else:
            ended.append((sweep_label(name, site, f), parse(data["ended"].get(name)) or now(), rec))
    return ended, lv, unread


def unrecorded_sweeps(d, st, sw):
    """[(site, name, [entry ids it fits], facts)] of the sweeps on the sites, live or submitted within SWEEP_RECENT_H
    hours, that no entry records. It fits the running entries whose match globs name it, narrowed to the work items
    its submission named (sweep_manager's stamp) when that names some of them."""
    recorded = {(x.get("site"), x.get("name")) for k, v in entries_state(st) for x in v.get("sweeps") or []}
    running = [e for e in d["entries"] if e.get("status") == "running"]
    cut = now().timestamp() - 3600 * SWEEP_RECENT_H
    out = []
    for site in sorted(sw):
        for name, f in sorted(((sw[site] or {}).get("sweeps") or {}).items()):
            if not f or (site, name) in recorded or not (f.get("live") or (f.get("submitted_at") or 0) >= cut):
                continue
            fits = [e["id"] for e in running if any(fnmatch.fnmatchcase(name, g.strip())
                                                    for g in (e.get("match") or "").split(",") if g.strip())]
            stamped = [x for x in f.get("items") or [] if x in {e["id"] for e in running}]
            narrowed = [x for x in fits if x in stamped] or (stamped if len(stamped) == 1 and not fits else [])
            out.append((site, name, narrowed or fits, f))
    return out


def reconcile_sweeps():
    """Every sweep a site runs that no entry records is recorded on the one running entry it fits (plan next and the
    Stop check run this first). Returns the blocking problems: a sweep that fits several entries (record it with plan
    run <entry> --sweep). A live one that fits none is noted."""
    if not os.path.exists(PLAN_FILE):
        return []
    d, st = load()
    running = [e["id"] for e in d["entries"] if e.get("status") == "running"]
    found = unrecorded_sweeps(d, st, site_sweeps(want_of(st, running)))
    probs, todo = [], []
    for site, name, fits, f in found:
        if len(fits) == 1:
            todo.append((fits[0], site, name, f))
        elif fits:
            probs.append("sweep %s on %s runs with no record on the plan and fits %s: record it on its entry with "
                         "`python3 tools/plan.py run <entry> --sweep %s:%s`"
                         % (name, site, ", ".join(fits), site, name))
        elif f.get("live"):
            note("sweep %s on %s has %d jobs in the queue and no running entry's match names it" % (name, site,
                                                                                                    f["live"]))
    if todo:
        with transaction() as (d2, st2):
            for eid, site, name, f in todo:
                lst = st2.setdefault(eid, {}).setdefault("sweeps", [])
                if not any(x.get("site") == site and x.get("name") == name for x in lst):
                    at = parse(f.get("submitted_at")) or now()
                    lst.append({"site": site, "name": name, "at": at.isoformat(), "found": now().isoformat()})
        for eid, site, name, f in todo:
            note("sweep %s on %s ran with no record on the plan: recorded on %s" % (name, site, eid))
    return probs


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


def working_steps(st, eid, t, e=None, key=None):
    """(started, last record) of each recorded step agent of the entry (with `key`, of that step of it) that has not
    handed back and wrote to its transcript within STEP_MIN of t."""
    out = []
    for x in (st.get(eid) or {}).get("steps") or []:
        if key is not None and rec_of(e, x) != key:
            continue
        a = None if step_problem(st, eid, x) else subagent_activity(x.get("agent"))
        # no lower bound: an agent writing while the check runs has records later than t (now() drops the
        # microseconds), and that is the most active agent there is (2026-10-26: an open event of an entry blocked
        # the turn while its step agent was reading those very runs)
        if a and not a[1] and (t - a[0]).total_seconds() / 60 <= STEP_MIN:
            out.append((parse(x.get("at")), a[0]))
    return out


def step_working(st, eid, t):
    """A recorded step agent of the entry has not handed back and wrote to its transcript within STEP_MIN of t: the
    entry's open events are its to ack."""
    return bool(working_steps(st, eid, t))


def run_submitted(ids):
    """{id: submitted (epoch)} of these runs in the registry."""
    c = db()
    if c is None or not ids:
        return {}
    return dict(c.execute("SELECT id, submitted FROM runs WHERE project=? AND id IN (%s)" % ",".join("?" * len(ids)),
                          [PROJECT] + list(ids)).fetchall())


def subagent_activity(agent_id):
    """(last, handed_back): the time of the subagent's last transcript record, and whether its last assistant record
    is a SubagentHandback call (its report is delivered and its run is over); None with no transcript. A step agent
    waiting on a long batch on a site keeps writing, so it is in flight until it hands back or goes quiet."""
    base = ENV.get("PLAN" + "_AGENTS")
    pat = os.path.join(base, "agent-%s.jsonl" % agent_id) if base else \
        os.path.join(S().TRANSCRIPTS, slug() + "*", "*", "subagents", "agent-%s.jsonl" % agent_id)
    for f in glob.glob(pat) if re.fullmatch(r"[A-Za-z0-9_-]+", agent_id or "") else []:
        last, handed = None, False
        for line in open(f, errors="replace"):
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if not isinstance(m, dict):
                continue
            last = parse(m.get("timestamp")) or last
            if m.get("type") == "assistant":
                c = (m.get("message") or {}).get("content")
                handed = isinstance(c, list) and any(isinstance(x, dict) and x.get("type") == "tool_use"
                                                     and x.get("name") == "SubagentHandback" for x in c)
        if last:
            return last, handed
    return None


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
def last_step(st, eid, e=None, key=None):
    """The latest recorded step that is a real step agent of the entry (with `key`, of that step of it)."""
    steps = [parse(s["at"]) for s in (st.get(eid) or {}).get("steps") or [] if not step_problem(st, eid, s)
             and (key is None or rec_of(e, s) == key)]
    return max(steps) if steps else None


def mark(d, st, eid, e=None, key=None):
    """The entry's (or its step's) last step, or the switch-over (plan.yaml's migrated_at, in git) when it is later."""
    ts = [x for x in (last_step(st, eid, e, key), parse(d.get("migrated_at"))) if x]
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


def why_moves(e, st, msgs, t, d=None, checks=None, sweeps=None, key=None):
    """(reason, since) when the entry's step `key` (default its first) can move now, else None. `since` is when the
    reason arose. Each next step is in flight on its own: its runs and sweeps (recorded with its number; those recorded
    without one are the first step's), its step agents, and the item check's active units for the steps that hold the
    entry's sweeps (the first step when none does)."""
    key = step_of(e, key)
    s, ls = e.get("status"), mark(d or {}, st, e["id"], e, key)
    if s == "running":
        tags = (st.get(e["id"]) or {}).get("run_steps") or {}
        rows = [r for r in runs_of(e, st) if step_of(e, tags.get(r[0])) == key]
        mine = lambda x: step_of(e, x.get("next")) == key
        ended = [(r, fin) for r, state, fin in rows if terminal(state) and fin and (not ls or fin > ls.timestamp())]
        if sweeps is None:
            sweeps = site_sweeps(want_of(st, [e["id"]]), all_sites=False) if sweeps_of(st, e["id"]) else {}
        sw_end, sw_live, sw_unread = sweep_status(e["id"], st, sweeps, mine)
        # a sweep that ended after the last step, as (label, ended, recorded) in epoch seconds
        sended = [(lab, f.timestamp(), r.timestamp() if r else 0) for lab, f, r in sw_end if not ls or f > ls]
        # an ended run a working step agent owns is its to handle: one it submitted (after its step started) or one
        # that ended before its last transcript record; once it hands back, the run is listed again (it ended after
        # the step started, so after `ls`). Hidden only when every ended run is owned: when one is not, the reason
        # lists them all, owned ones included, since the step it starts moves `ls` past every one of them
        work = working_steps(st, e["id"], t, e, key) if ended or sended else []
        if work:
            sub = run_submitted([r for r, _ in ended])
            owned = lambda subm, fin: any(subm >= s0.timestamp() or fin <= last.timestamp() for s0, last in work)
            if all(owned(sub.get(r) or 0, fin) for r, fin in ended) and all(owned(r, f) for _, f, r in sended):
                return None
        if ended or sended:
            why = (["runs ended: " + ", ".join(r for r, _ in ended)] if ended else []) + \
                (["sweep ended: " + ", ".join(lab for lab, _, _ in sended)] if sended else [])
            return "; ".join(why), parse(max([f for _, f in ended] + [f for _, f, _ in sended]))
        if any(live(state) for r, state, fin in rows) or sw_live or sw_unread:
            return None
        holders = {step_of(e, x.get("next")) for x in sweeps_of(st, e["id"])} or {step_keys(e)[0]}
        active = 0 if key not in holders else \
            (checks or {}).get(e["id"], 0) if checks is not None else item_check_active(e["id"])
        if active is None or active > 0:          # None: the check could not run; counted in flight, with a note
            return None
        if ls and 0 <= (t - ls).total_seconds() / 60 <= STEP_MIN:
            return None
        # a step agent still writing its transcript is in flight; one that handed back is not
        acts = [a for a in (subagent_activity(x.get("agent")) for x in (st.get(e["id"]) or {}).get("steps") or []
                            if not step_problem(st, e["id"], x) and rec_of(e, x) == key) if a]
        if working_steps(st, e["id"], t, e, key):
            return None
        # the reason arose when the last step's window lapsed, its agent handed back or went quiet for the window, or
        # the last run ended, whichever is later
        fins = [parse(fin) for r, state, fin in rows if fin] + [f for _, f, _ in sw_end]
        quiet = [last if handed else last + dt.timedelta(minutes=STEP_MIN) for last, handed in acts]
        marks = [x for x in [ls + dt.timedelta(minutes=STEP_MIN) if ls else None] + fins + quiet if x]
        return "nothing in flight: " + (step_text(e, key) or "take the next step"), (max(marks) if marks else None)
    if s == "waiting_user":
        return replied(e, msgs)
    return None


def replied(e, msgs):
    """(reason, when) if the user wrote after the entry's open question (and after the last plan unanswered), else
    None. Never blocks the stop; plan next lists it until plan answer or plan unanswered."""
    if not open_question(e):
        return None
    since = max([x for x in (parse(e.get("waiting_since")), parse(e.get("checked_until"))) if x]
                or [dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)])
    said = [ts for ts, role, text in msgs if role == "user" and ts > since]
    if said:
        return ("the user wrote at %s after the question %r: plan answer with their words, or plan unanswered "
                "if it does not answer it (never blocks the stop; listed here until one of them)" % (
                    said[-1].isoformat()[:16], e.get("ask"))), said[-1]
    return None


ASKS_FIRST_MIN = 60          # with no scan recorded yet, re-asks are looked for in the last hour


def asks_mark(st):
    """Where the scan for answered questions asked again starts: the end of the previous scan (state _asks_scanned)."""
    return parse(st.get("_asks_scanned")) or now() - dt.timedelta(minutes=ASKS_FIRST_MIN)


def open_question(e):
    """The entry's question to the user not yet answered: on a waiting entry (it holds everything) or on a running one
    (it holds only what `blocks` names; the rest of the entry keeps moving)."""
    return bool(e.get("ask") and e.get("waiting_since") and e.get("status") in ("running", "waiting_user"))


def oldest_open(d, st=None):
    """The earliest time the transcripts are read from: the oldest OPEN question (less the margin) or the end of the
    previous re-ask scan, whichever is earlier."""
    ts = [parse(e.get("waiting_since")) - dt.timedelta(minutes=ASK_MARGIN_MIN) for e in d["entries"]
          if open_question(e)]
    ts.append(asks_mark(st or {}))
    return min(ts)


def next_steps(d, st, t=None, msgs=None):
    """Deterministic: [(entry, reason, launched)], one per step that can move (next_steps_keyed has the step's number)."""
    return [x[:3] for x in next_steps_keyed(d, st, t, msgs)]


def next_steps_keyed(d, st, t=None, msgs=None):
    """Deterministic: [(entry, reason, launched, step number or None)]. Each next step of a running entry is listed on
    its own and is launched when a step agent was recorded for that step after its reason arose; a waiting entry is
    never launched by a step (only plan answer or unanswered clear it)."""
    t = t or now()
    msgs = messages(oldest_open(d, st)) if msgs is None else msgs
    running = [e["id"] for e in d["entries"] if e.get("status") == "running"]
    checks = item_checks(running)
    sweeps = site_sweeps(want_of(st, running), all_sites=False)
    out = []
    for e in d["entries"]:
        # a fallback: the step agent that records a design runs the design reviewer itself; a design left without a
        # verdict REVIEW_FALLBACK_MIN after it was recorded, its step agent gone, goes to the main session
        if e.get("status") == "running" and design_pending(e) and \
                (t - parse(last_design(e)["at"])).total_seconds() / 60 > REVIEW_FALLBACK_MIN and \
                not step_working(st, e["id"], t):
            launched, why = review_launched(st, e, t)
            out.append((e, "design to review: the design recorded %s has no verdict%s; launch the design reviewer" % (
                (last_design(e).get("at") or "")[:16], why), launched, None))
            continue
        for k in step_keys(e) if e.get("status") == "running" else [None]:
            r = why_moves(e, st, msgs, t, d, checks, sweeps, key=k)
            if r:
                reason, since = r
                ls = last_step(st, e["id"], e, k) if k is not None else None
                launched = e.get("status") == "running" and ls is not None and since is not None and ls >= since
                out.append((e, reason, launched, k if len(nexts(e)) > 1 else None))
    return out


PROMPT_TAIL = 8               # dead ends and evidence runs / paths a step prompt carries; plan show <id> has them all


def entry_quotes(e):
    """The user's words on the entry, one per line: its quotes, and the quote of each decision it has none of."""
    qs = [(x.get("text"), x.get("said_at")) for x in e.get("quotes") or []]
    have = {x.get("decision") for x in e.get("quotes") or []}
    for did in e.get("decisions") or []:                # an entry added under a decision has no quote of its own
        dd = decision(did) if did not in have else None
        if dd and dd["quote"]:
            qs.append((dd["quote"], dd["said_at"].isoformat() if dd["said_at"] else ""))
    return "\n".join('  "%s" (%s)' % (t, (w or "")[:16]) for t, w in qs)


def step_prompt(e, reason, key=None):
    """The step agent's prompt for the entry's step `key` (default its first): {next} is that step, {others} the
    entry's other next steps, which other step agents take in parallel."""
    tpl = open(os.path.join(HERE, "plan", "step_prompt.txt")).read()
    key = step_of(e, key)
    others = "\n".join("  %d: %s" % (k, x) for k, x in nexts(e) if k != key) or "  none"
    q = entry_quotes(e)
    # the last PROMPT_TAIL dead ends and evidence items only: an entry's whole history made one prompt 17k characters
    dl = e.get("dead_ends") or []
    de = "\n".join("  %s: %s" % (x["tried"], x["why"]) for x in dl[-PROMPT_TAIL:]) or "  none"
    if len(dl) > PROMPT_TAIL:
        de = "  (%d earlier dead ends not shown: python3 tools/plan.py show %s)\n" % (len(dl) - PROMPT_TAIL, e["id"]) + de
    ev, cut = dict(e.get("evidence") or {}), []
    for k in ("runs", "paths"):
        if len(ev.get(k) or []) > PROMPT_TAIL:
            cut.append("%d earlier %s" % (len(ev[k]) - PROMPT_TAIL, k))
            ev[k] = ev[k][-PROMPT_TAIL:]
    evidence = json.dumps(ev) + ("\n  (%s not shown: python3 tools/plan.py show %s)" % (" and ".join(cut), e["id"])
                                 if cut else "")
    oq = "  none"
    if open_question(e):
        oq = ('  asked %s: "%s"\n  Waits on the answer: %s. Do not do or decide that part; work on the rest of the '
              'entry.' % ((e.get("waiting_since") or "")[:16], e["ask"], e.get("blocks") or
                          "not recorded (only what the question itself decides)"))
    return tpl.format(id=e["id"], title=e.get("title"), question=e.get("question"), reason=reason,
                      next=step_text(e, key), n=key, others=others, done_when=e.get("done_when"),
                      quotes=q or "  none", dead_ends=de,
                      evidence=evidence, open_question=oq,
                      design=design_text(last_design(e)))


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
                           "%s..%s" % (base, head), "--cache", CLAIMS_CACHE, "--script-timeout", "0"],
                          stdout=subprocess.PIPE,
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


def done_problems(d, st, limit=True, only=None):
    """[(entry, why)]: done entries that no longer pass what plan done checked. Cached for DONE_RECHECK_H hours under
    a key of every field the check reads, so any write to the entry reruns it. A regression is a suggestion (plan next
    lists `plan reopen`), never a standing block. With `limit` (the Stop hook) a check that runs out of time is cached
    for the day like a pass and noted once; `plan recheck` runs it with no limit."""
    try:
        cache = json.load(open(CHECK_CACHE))
    except (OSError, ValueError):
        cache = {}
    out, changed = [], False
    for e in d["entries"]:
        if e.get("status") != "done" or (only and e["id"] != only):
            continue
        import hashlib
        runs = sorted(map(list, runs_of(e, st)))
        key = "done:%s:%s" % (e["id"], hashlib.sha256(json.dumps([e, runs], sort_keys=True, default=str).encode())
                              .hexdigest()[:16])
        hit = cache.get(key)
        if limit and hit and (now() - parse(hit["at"])).total_seconds() < 3600 * DONE_RECHECK_H:
            why = hit["why"]
        else:
            try:
                why = done_refusals(e, st, e.get("writeup") or "?", e.get("runs_override"),
                                    timeout=min(30, budget()) if limit else None, record=e.get("record"))
            except subprocess.TimeoutExpired:
                # cached for the day like a pass, noted once: retrying it at every stop only repeats the note
                cache[key], changed = {"at": now().isoformat(), "why": [], "unfinished": True}, True
                NOTES.append("%s: its daily done check did not finish within the stop's time; `python3 tools/plan.py "
                             "recheck %s` runs it with no limit" % (e["id"], e["id"]))
                continue
            cache[key], changed = {"at": now().isoformat(), "why": why}, True
        if why:
            out.append((e, "; ".join(why)))
    if changed and os.path.exists(PLAN_FILE):
        _write_atomic(CHECK_CACHE, json.dumps(cache))
    return out


# ---------------------------------------------------------------- commands
def cmd_recheck(a):
    """The done entries' finishing checks with no time limit (those the Stop's budget could not finish); the result is
    cached, so the next stops read it."""
    d, st = load()
    bad = done_problems(d, st, limit=False, only=a.id)
    for e, why in bad:
        print("REGRESSED %s: %s; plan reopen %s --why ..., or fix it" % (e["id"], why, e["id"]))
    print("plan: %d done entr%s rechecked, %d regressed" % (
        sum(1 for e in d["entries"] if e.get("status") == "done" and (not a.id or e["id"] == a.id)),
        "y" if a.id else "ies", len(bad)))
    return 1 if bad else 0


def cmd_kind(a):
    """setup: the entry is about the setup (its record may be CLAUDE.md or a tool, plan done --record); work: the
    default, written up in docs/results.tex."""
    with transaction() as (d, st):
        e = entry(d, a.id)
        if a.kind == "work":
            e.pop("kind", None)
        else:
            e["kind"] = a.kind
    print("plan: %s is %s" % (a.id, a.kind))


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
    if a.id and not any(e["id"] == a.id for e in d["entries"]):
        sys.exit("plan: no entry %r (plan show lists them)" % a.id)
    for e in d["entries"]:
        s = e.get("status")
        if (a.compact and s == "done") or (a.id and e["id"] != a.id):
            continue
        extra = ""
        if s == "running":
            ls = last_step(st, e["id"])
            lr = [r for r, state, fin in runs_of(e, st) if live(state)]
            extra = ("live runs " + ", ".join(lr)) if lr else ("last step %s" % ls.isoformat()[:16] if ls else "no step")
            if open_question(e):
                extra += "; open question asked %s: %s (blocks: %s; searched first: %s)" % (
                    e["waiting_since"][:16], e["ask"], e.get("blocks") or "not recorded",
                    e.get("searched") or "not recorded")
            if sweeps_of(st, e["id"]):
                ended, lv, unread = sweep_status(e["id"], st, site_sweeps(want_of(st, [e["id"]]), all_sites=False,
                                                                          fetch=False))
                extra += "; sweeps: " + ", ".join(["live " + x for x in lv] + ["ended " + x[0] for x in ended]
                                                  + [x + " not read yet" for x in unread])
        elif sweeps_of(st, e["id"]):
            extra = "sweeps " + ", ".join("%s (%s)" % (x["name"], x["site"]) for x in sweeps_of(st, e["id"]))
        elif s == "waiting_user":
            extra = "asked %s: %s (searched first: %s; holds all: %s)" % (
                (e.get("waiting_since") or "?")[:16], e.get("ask", ""), e.get("searched") or "not recorded",
                e.get("holds_all") or "not recorded")
        elif s == "held":
            extra = "held by %s" % e.get("hold_decision")
        print("- [%s] %s (%s): %s" % (s, e.get("title"), e["id"], extra))
        for k in (("next",) if a.compact else ("question", "next", "done_when", "notes_ref")):
            if k == "next" and len(nexts(e)) > 1:
                for n, x in nexts(e):
                    print("    next %d: %s" % (n, x))
            elif e.get(k):
                print("    %s: %s" % (k, e[k]))
        if not a.compact:
            for q in e.get("quotes") or []:
                print("    user %s: \"%s\"" % ((q.get("said_at") or "")[:16], q.get("text")))
            for x in e.get("dead_ends") or []:
                print("    dead end: %s (%s)" % (x["tried"], x["why"]))
            if a.id and e.get("evidence"):
                print("    evidence: " + json.dumps(e["evidence"]))
            if last_design(e):
                print("    design:\n" + design_text(last_design(e), "      "))


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


DESIGN_FIELDS = ("question", "prior", "measure", "check", "fixed", "control", "seeds", "outcome")
DESIGN_H = 24                 # hours a recorded design covers submissions of its entry (design_guard.py)
CHECK_MIN = 40                # characters: how the measure was seen to move with the change, named, not a token
PRIOR_MIN = 40                # characters: the prior work found, its result and how this design differs
PRIOR_TERMS = 3               # search terms a "none found" names
# a bare "none" is no claim of an empty search: "none of the AdamW seeds reached ..." names prior work
NONE_FOUND = re.compile(r"\bnone found\b|\bnothing found\b|\bno prior( work)?( found)?\b", re.I)


def last_design(e):
    return (e.get("designs") or [None])[-1]


def design_fresh(e, hours=DESIGN_H):
    """The entry's latest design, if recorded within `hours` and passed by the design reviewer; else None."""
    x = last_design(e)
    t = parse(x.get("at")) if x else None
    return x if t and x.get("passed_at") and now() - t <= dt.timedelta(hours=hours) else None


REVIEW_FALLBACK_MIN = 30     # minutes a design may wait for its verdict before plan next lists it for review


def design_pending(e):
    """The latest design has no verdict yet (after REVIEW_FALLBACK_MIN, plan next lists it as a design to review)."""
    x = last_design(e)
    return bool(x) and not x.get("passed_at") and not x.get("fix_at")


REVIEW_HEAD = "Design review."


def review_prompt(e):
    x = last_design(e)
    de = "\n".join("  %s: %s" % (y["tried"], y["why"]) for y in e.get("dead_ends") or []) or "  none"
    qs = entry_quotes(e) or "  none"
    return ("%s Review the latest recorded design of one plan entry (your instructions: "
            ".claude/agents/design-reviewer.md).\n\nEntry: %s (%s)\nQuestion of the entry: %s\nThe user's words on "
            "it (the design answers every one that names a comparison):\n%s\nThe design, recorded "
            "%s:\n%s\nEvidence so far: %s\nDead ends so far:\n%s\n\nOn PASS run `python3 tools/plan.py design-pass "
            "%s --reviewer \"<your note>\"`; on FIX run `python3 tools/plan.py design-fix %s --reasons \"<what to "
            "change>\"`. Then report the verdict and its reasons." % (
                REVIEW_HEAD, e["id"], e.get("title"), e.get("question"), qs, (x.get("at") or "")[:16],
                design_text(x), json.dumps(e.get("evidence") or {}), de, e["id"], e["id"]))


def review_launched(st, e, t):
    """(launched, note): a design reviewer recorded after the latest design that is still working (it has not
    handed back and wrote within STEP_MIN), so the entry waits on its verdict."""
    at = parse(last_design(e).get("at"))
    revs = [x for x in (st.get(e["id"]) or {}).get("reviews") or [] if x.get("design_at") == last_design(e).get("at")
            and parse(x.get("at")) >= at]
    for x in revs:
        a = subagent_activity(x.get("agent"))
        if a and not a[1] and 0 <= (t - a[0]).total_seconds() / 60 <= STEP_MIN:
            return True, ""
    return False, (" (the reviewer %s ended without design-pass or design-fix)" % revs[-1]["agent"]) if revs else ""


def design_text(x, indent="  "):
    if not x:
        return indent + "none recorded (a job needs one first: python3 tools/plan.py design, step 2)"
    verdict = ("passed review %s (%s)" % (x["passed_at"][:16], x.get("reviewer")) if x.get("passed_at") else
               "review asked for a fix %s: %s" % (x["fix_at"][:16], x.get("fix")) if x.get("fix_at") else
               "not reviewed yet")
    return "\n".join(["%srecorded %s, %s" % (indent, (x.get("at") or "")[:16], verdict)] +
                      ["%s%s: %s" % (indent, k, x.get(k) if x.get(k) is not None else "not recorded")
                       for k in DESIGN_FIELDS])


def _prior_ok(prior):
    """A design names the work already done on its question before it proposes more (2026-10-26: a pp-trained arm
    existed, was written up and named by the entry's decisions as the reference, yet a comparison was written without
    it and a rerun proposed; the user: "you completely ignored previous work done ... and you were about to run the
    same thing again, that's completely wasteful and can't happen ever again"). Either each prior result with how
    this design differs, or "none found" with the terms searched (in docs/results.tex, plan show, site decisions)."""
    if NONE_FOUND.search(prior):
        rest = NONE_FOUND.sub(" ", prior)
        terms = [t for t in re.split(r"[\s,;:/()\"']+", rest) if len(t) >= 2 and t.lower() not in (
            "searched", "search", "terms", "term", "for", "in", "the", "and", "of", "on", "with", "docs", "results.tex",
            "plan", "show", "site", "decisions")]
        if len(terms) < PRIOR_TERMS:
            sys.exit("plan: --prior says none found but names no search terms: list what was searched for in "
                     "docs/results.tex, plan show and site decisions (process, arm, data, quantity names), at least "
                     "%d terms" % PRIOR_TERMS)
    elif len(prior) < PRIOR_MIN:
        sys.exit("plan: --prior must name the prior work found on this question (searched in docs/results.tex, plan "
                 "show, site decisions), each with its result and how this design differs, or \"none found\" with the "
                 "search terms; at least %d characters" % PRIOR_MIN)


def cmd_design(a):
    """A job answers its question only if its measurement can see the tested change (the sign head search,
    2026-10-26: an objective blind to the sign, a comparison that changed two things at once, untuned arms; the user:
    "it's the second or third time you bring back results that show nothing and the conclusion is that the way you
    are testing was wrong"). So the design is written down before the job, every field of it."""
    vals = {k: (getattr(a, k) or "").strip() for k in DESIGN_FIELDS}
    empty = [k for k, v in vals.items() if not v]
    if empty:
        sys.exit("plan: a design needs every field; empty: %s" % ", ".join("--" + k for k in empty))
    _prior_ok(vals["prior"])
    if len(vals["check"]) < CHECK_MIN:
        sys.exit("plan: --check must say how it was verified, on existing data, that the measure moves with the "
                 "tested change (which runs or pools, what moved and by how much), in at least %d characters"
                 % CHECK_MIN)
    with transaction() as (d, st):
        e = entry(d, a.id)
        e.setdefault("designs", []).append(dict(at=now().isoformat(), **vals))
    print("plan: design recorded on %s; now run the design-reviewer subagent on it (Agent tool, subagent_type "
          "design-reviewer); submissions for it pass design_guard for %d h once it passes" % (a.id, DESIGN_H))


def cmd_design_pass(a):
    with transaction() as (d, st):
        x = last_design(entry(d, a.id))
        if not x:
            sys.exit("plan: %s has no design to pass" % a.id)
        if not a.reviewer.strip():
            sys.exit("plan: --reviewer needs the reviewer's note")
        x.update(passed_at=now().isoformat(), reviewer=a.reviewer.strip())
        x.pop("fix_at", None), x.pop("fix", None)
    print("plan: the latest design of %s passed review" % a.id)


def cmd_design_fix(a):
    with transaction() as (d, st):
        x = last_design(entry(d, a.id))
        if not x:
            sys.exit("plan: %s has no design to review" % a.id)
        if not a.reasons.strip():
            sys.exit("plan: --reasons needs what to change")
        x.update(fix_at=now().isoformat(), fix=a.reasons.strip())
        x.pop("passed_at", None), x.pop("reviewer", None)
    print("plan: the latest design of %s needs a fix; the next step records a new one" % a.id)


def cmd_review(a):
    """The design reviewer (a real subagent started from the entry's review prompt) launched on the latest design."""
    with transaction() as (d, st):
        e = entry(d, a.id)
        x = last_design(e)
        if not design_pending(e):
            sys.exit("plan: %s has no design waiting for review" % a.id)
        rec = subagent(a.agent)
        if rec is None:
            sys.exit("plan: no subagent %s in this project's transcripts" % a.agent)
        started, prompt = rec
        if not names_entry(prompt, a.id) or REVIEW_HEAD not in (prompt or ""):
            sys.exit("plan: subagent %s was not started with %s's review prompt (plan next --prompts)" % (a.agent, a.id))
        if started < parse(x["at"]):
            sys.exit("plan: subagent %s started before the design was recorded" % a.agent)
        st.setdefault(a.id, {}).setdefault("reviews", []).append({"agent": a.agent, "at": started.isoformat(),
                                                                 "design_at": x["at"]})
    print("plan: design reviewer on %s recorded (%s)" % (a.id, a.agent))


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
            e["quotes"].append({"said_at": dd["said_at"].isoformat() if dd["said_at"] else None, "text": dd["quote"],
                                "decision": a.decision, "covers": dd["covers"]})
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
    """Set the check that finishes an entry. --replace changes a DONE entry's check (e.g. one too slow for the daily
    recheck): it must pass now, since the entry is done; only a trivially true check is refused."""
    if trivial_check(a.done_when):
        sys.exit("plan: %r passes in an empty directory: it checks nothing about the work" % a.done_when)
    status = entry(load()[0], a.id).get("status")
    if a.stale and (a.replace or status == "done"):
        sys.exit("plan: --stale is for an open entry whose check names the wrong files; a done one takes --replace")
    if a.stale:
        # the old check fails, the new one passes, and the swap is recorded with its reason for the user to see:
        # a check that names the wrong deliverable (a figure split into panels) is fixed, never quietly loosened
        if not a.why:
            sys.exit("plan: --stale needs --why (what in the old check was wrong)")
        old_dw = entry(load()[0], a.id).get("done_when") or ""
        if old_dw and subprocess.run(old_dw, shell=True, cwd=ROOT, capture_output=True, timeout=600).returncode == 0:
            sys.exit("plan: %s's current check passes: it is not stale" % a.id)
        if subprocess.run(a.done_when, shell=True, cwd=ROOT, capture_output=True, timeout=600).returncode != 0:
            sys.exit("plan: the new check fails too: the work is not there yet")
        with transaction() as (d, st):
            e = entry(d, a.id)
            e.setdefault("history", []).append({"at": now().isoformat(), "was_done_when": old_dw,
                                                "stale_because": a.why})
            e["done_when"] = a.done_when
        print("plan: %s's stale check replaced (%s); tell the user, then plan done" % (a.id, a.why))
        return
    if a.replace and status != "done":
        sys.exit("plan: --replace is for a done entry's check; %s is %s (plan check without --replace, or --stale"
                 " for a check naming the wrong files)" % (a.id, status))
    if not a.replace and status == "done":
        sys.exit("plan: %s is done; change its check with --replace" % a.id)
    r = subprocess.run(a.done_when, shell=True, cwd=ROOT, capture_output=True, timeout=600)
    if a.replace and r.returncode != 0:
        sys.exit("plan: %r fails now, and %s is done: reopen it (plan reopen) rather than give it a failing check"
                 % (a.done_when, a.id))
    if not a.replace and r.returncode == 0:
        sys.exit("plan: %r already passes in the checkout: it would finish the entry with nothing done; check "
                 "what this entry still has to produce" % a.done_when)
    with transaction() as (d, st):
        e = entry(d, a.id)
        if a.replace and e.get("done_when"):
            e.setdefault("history", []).append({"at": now().isoformat(), "was_done_when": e["done_when"]})
        e["done_when"] = a.done_when
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
        if open_question(e):
            since, what = parse(e.get("waiting_since")), "the question was asked"
        elif a.kind == "release" and e.get("status") == "held":
            since, what = (decision(e.get("hold_decision")) or {}).get("said_at"), "the hold was said"
        else:
            since, what = last_event(e, st), "the entry's last ask or step"
        after = [t for t in times if not since or t > since]
        if not after:
            sys.exit("plan: not recorded: the user typed these words before %s (%s)" % (
                what, since.isoformat()[:16] if since else "?"))
        made_item = False
        if a.kind == "release" and not item_exists(e["id"]):
            sys.exit("plan: %s has no site item to release; attach one first (plan item)" % e["id"])
        if a.kind in ("hold", "forbid") and not item_exists(e["id"]):
            # an entry paused before it had an item: the item is filed under the entry's own approval or scope in
            # force (site item add takes no hold), then the hold puts it on hold; undone if the hold fails
            under = next((x for x in e.get("decisions") or [] for dx in [decision(x)]
                          if dx and dx["in_force"] and dx["kind"] in ("approve", "scope", "release")), None)
            if not under:
                sys.exit("plan: %s has no site item and no approval or scope in force to file one under (an item "
                         "needs one); record the user's scope for it first" % e["id"])
            ok, why = ensure_item(e, under)
            if not ok:
                sys.exit("plan: site item %s not created (%s); nothing recorded" % (e["id"], why))
            made_item = True
        rc, out, err = site_call("decide", PROJECT, "--kind", a.kind, "--quote", a.quote, "--covers", e["id"])
        m = re.search(r"\b(D\d+)\b", out)
        if rc != 0 or not m:
            if made_item:
                site_call("item", "done", PROJECT, e["id"])
            sys.exit("plan: not recorded, the entry is unchanged%s (site decide: %s)" % (
                "; the site item made for the hold is closed" if made_item else "", (err or out).strip()[-300:]))
        did = m.group(1)
        dd = decision(did)
        said = (dd or {}).get("said_at")
        if not said or (since and said <= since):
            if made_item:
                site_call("item", "done", PROJECT, e["id"])
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
            for k in ("ask", "waiting_since", "checked_until", "searched", "blocks", "holds_all"):
                e.pop(k, None)
    print("plan: %s recorded as %s; entry %s now %s" % (a.kind, did, e["id"], e["status"]))


def answered_asks(d):
    return [(e, x) for e in d["entries"] for x in e.get("asks") or [] if x.get("answered_at")]


HOLDS_ALL_MIN = 40           # characters: why nothing in the entry can move without the answer


def cmd_ask(a):
    """An open question on the entry. Only what needs the answer waits (CLAUDE.md 0g): the entry stays running, routed
    by plan next, and its step prompt names the question and what it blocks. --holds-all (the whole entry waits)
    must say why nothing else in it can move. (catalog_coverage, 2026-10-26: a question on one process's top width
    froze the entry's independent work for 8 h.)"""
    _ask_ok(a.ask)
    _searched_ok(a.searched)
    if a.holds_all is not None and a.blocks:
        sys.exit("plan: --blocks names the part that waits; --holds-all makes everything wait: pass one")
    if a.holds_all is not None and len(a.holds_all.strip()) < HOLDS_ALL_MIN:
        sys.exit("plan: --holds-all must say why nothing else in the entry can move without the answer, in at least "
                 "%d characters (otherwise pass --blocks: only what needs the answer waits)" % HOLDS_ALL_MIN)
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
        for k in ("checked_until", "blocks", "holds_all"):
            e.pop(k, None)
        e.update(ask=a.ask, waiting_since=t, searched=a.searched)
        rec = {"ask": a.ask, "at": t, "searched": a.searched}
        if a.holds_all is not None:
            e.update(status="waiting_user", holds_all=a.holds_all)
            rec["holds_all"] = a.holds_all
        else:
            e["status"] = "running"
            if a.blocks:
                e["blocks"] = rec["blocks"] = a.blocks
        e.setdefault("asks", []).append(rec)
    if e["status"] == "waiting_user":
        print("plan: %s waits on the user from %s (everything: %s)" % (a.id, t, a.holds_all))
    else:
        print("plan: %s keeps running; question open from %s, waiting on it: %s" % (
            a.id, t, a.blocks or "not recorded"))


def cmd_unanswered(a):
    """The user's messages since the question do not answer it (it stays open)."""
    with transaction() as (d, st):
        e = entry(d, a.id)
        if not open_question(e):
            sys.exit("plan: %s has no open question" % a.id)
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
        if not names_entry(prompt, a.id) or (prompt or "").startswith(REVIEW_HEAD):
            sys.exit("plan: subagent %s was not started with %s's step prompt (plan next --prompts; its \"Entry: %s\" "
                     "line)" % (a.agent, a.id, a.id))
        used = [k for k, v in entries_state(st) for x in v.get("steps") or [] if x.get("agent") == a.agent]
        if used:
            sys.exit("plan: subagent %s is already recorded as a step of %s; one subagent is one step" % (
                a.agent, ", ".join(used)))
        key = a.next
        if key is None:
            if len(nexts(e)) > 1:
                sys.exit("plan: %s has %d next steps (%s); name the one this agent takes with --next N, as plan next "
                         "--prompts listed it" % (a.id, len(nexts(e)), ", ".join(str(k) for k in step_keys(e))))
            key = step_keys(e)[0]
        elif key not in step_keys(e):
            sys.exit("plan: %s has no next step %d (it has %s)" % (a.id, key, ", ".join(str(k) for k in step_keys(e))))
        m = re.search(r"(?m)^Your step \(number (\d+)", prompt or "")
        if m and int(m.group(1)) != key:
            sys.exit("plan: subagent %s was started with the prompt of step %s, not %d" % (a.agent, m.group(1), key))
        r = why_moves(e, st, [], now(), d, key=key)
        since = r[1] if r else None
        if since and started < since:
            sys.exit("plan: subagent %s started %s, before the reason arose (%s)" % (
                a.agent, started.isoformat()[:16], since.isoformat()[:16]))
        st.setdefault(a.id, {}).setdefault("steps", []).append({"agent": a.agent, "at": started.isoformat(),
                                                               "reason": r[0] if r else "", "next": key})
    print("plan: step %d on %s recorded (%s, started %s)" % (key, a.id, a.agent, started.isoformat()[:16]))


def cmd_run(a):
    """Registry runs, and sweeps (--sweep SITE:NAME, a sweep name or dir): sweep_manager's jobs are not in runs.db, so
    a sweep is in flight while its site's queue holds a job of it. A sweep the site's registry does not know is
    refused; an unreadable site records it, counted in flight until read."""
    if not a.runs and not a.sweep:
        sys.exit("plan: name a run, or a sweep with --sweep SITE:NAME")
    specs = []
    for x in a.sweep or []:
        site, _, name = x.partition(":")
        if not site or not sweep_name(name):
            sys.exit("plan: --sweep takes SITE:NAME (e.g. ccin2p3:muon_test_search_adamw), not %r" % x)
        specs.append((site, sweep_name(name)))
    if specs:
        if os.path.exists(PLAN_FILE):
            entry(load()[0], a.id)                    # before reaching the site
        want = {}
        for site, name in specs:
            want.setdefault(site, set()).add(name)
        sw = site_sweeps(want, all_sites=False, fresh=True)
        missing = ["%s on %s" % (n, site) for site, n in specs if sw.get(site) is not None
                   and sw[site]["sweeps"].get(n) is None]
        if missing:
            sys.exit("plan: not recorded: no sweep %s in the site's sweep_manager registry" % ", ".join(missing))
    with transaction() as (d, st):
        e = entry(d, a.id)
        if a.next is not None and a.next not in step_keys(e):
            sys.exit("plan: %s has no next step %d (it has %s)" % (a.id, a.next, ", ".join(str(k) for k in step_keys(e))))
        rs = st.setdefault(a.id, {}).setdefault("runs", [])
        rs += [r for r in a.runs if r not in rs]
        if a.next is not None:
            st[a.id].setdefault("run_steps", {}).update({r: a.next for r in a.runs})
        sl = st[a.id].setdefault("sweeps", [])
        for site, name in specs:
            x = next((x for x in sl if x.get("site") == site and x.get("name") == name), None)
            if x is None:
                x = {"site": site, "name": name, "at": now().isoformat()}
                sl.append(x)
            if a.next is not None:
                x["next"] = a.next
    for site, name in specs:
        print("plan: sweep %s (%s) recorded on %s" % (name, site, a.id))
    for n in NOTES:
        print("note: " + n)


def cmd_sweeps(a):
    """Records the sweeps no entry records (as plan next does), then prints every recorded sweep of an open entry with
    its state on its site."""
    probs = reconcile_sweeps()
    d, st = load()
    eids = [e["id"] for e in d["entries"] if e.get("status") != "done"]
    sw = site_sweeps(want_of(st, eids), all_sites=False, fresh=a.fresh)
    for eid in eids:
        ended, lv, unread = sweep_status(eid, st, sw)
        for lab in lv:
            print("%s: live %s" % (eid, lab))
        for lab, f, r in ended:
            print("%s: ended %s, seen ended %s" % (eid, lab, f.isoformat()[:16]))
        for lab in unread:
            print("%s: not read %s" % (eid, lab))
    for l in probs + ["note: " + n for n in NOTES]:
        print(l)
    return 1 if probs else 0


def cmd_evidence(a):
    with transaction() as (d, st):
        ev = entry(d, a.id).setdefault("evidence", {"runs": [], "commit": None, "paths": []})
        ev["runs"] = sorted(set((ev.get("runs") or []) + (a.run or [])))
        ev["paths"] = sorted(set((ev.get("paths") or []) + (a.path or [])) - set(a.drop or []))
        if a.commit:
            ev["commit"] = a.commit


def used_keys(e, st):
    """Every step number the entry holds or its state still names (step agents, runs, sweeps): a new step never
    takes a number an old record points at."""
    v = st.get(e["id"]) or {}
    return set(step_keys(e)) | {x.get("next") for x in (v.get("steps") or []) + (v.get("sweeps") or [])
                                if x.get("next") is not None} | set((v.get("run_steps") or {}).values())


def cmd_set_next(a):
    """Without --next: the entry's next steps become TEXT and every --also (independent steps); a lone TEXT on an entry
    of one step replaces it in place, otherwise they are numbered from 1 when no record names a number, else past every
    number a step agent, run or sweep record names (an old record never lands on a new step). With --next N: step N
    becomes TEXT (a step agent replaces only its own step), the --also ones added beside it."""
    with transaction() as (d, st):
        e = entry(d, a.id)
        if a.next is None and not a.also and len(nexts(e)) <= 1:
            put_nexts(e, [(step_keys(e)[0], a.text)])
        elif a.next is None:
            v = st.get(e["id"]) or {}
            named = v.get("run_steps") or any(x.get("next") is not None
                                              for x in (v.get("steps") or []) + (v.get("sweeps") or []))
            k0 = max(used_keys(e, st)) if named else 0
            put_nexts(e, list(enumerate([a.text] + (a.also or []), k0 + 1)))
        else:
            if a.next not in dict(nexts(e)) and not (a.next == 1 and not nexts(e)):
                sys.exit("plan: %s has no next step %d (it has %s)" % (a.id, a.next,
                                                                      ", ".join(str(k) for k, _ in nexts(e)) or "none"))
            items = dict(nexts(e))
            items[a.next] = a.text
            k0 = max(used_keys(e, st) | set(items))
            items.update({k0 + i: x for i, x in enumerate(a.also or [], 1)})
            put_nexts(e, sorted(items.items()))
    print_nexts(a.id)


def cmd_add_next(a):
    """One more independent next step, numbered past every number the entry used."""
    with transaction() as (d, st):
        e = entry(d, a.id)
        items = dict(nexts(e))
        k = max(used_keys(e, st) | set(items) | {0}) + 1
        items[k] = a.text
        put_nexts(e, sorted(items.items()))
    print_nexts(a.id)


def cmd_drop_next(a):
    """Step N is finished and nothing follows it: it leaves the list; the others keep their numbers."""
    with transaction() as (d, st):
        e = entry(d, a.id)
        items = dict(nexts(e))
        if a.n not in items:
            sys.exit("plan: %s has no next step %d (it has %s)" % (a.id, a.n, ", ".join(str(k) for k in items) or "none"))
        del items[a.n]
        put_nexts(e, sorted(items.items()))
    print_nexts(a.id)


def print_nexts(eid):
    e = entry(load()[0], eid)
    for k, x in nexts(e):
        print("plan: %s next %d: %s" % (eid, k, x))
    if not nexts(e):
        print("plan: %s has no next step" % eid)


def cmd_dead_end(a):
    with transaction() as (d, st):
        entry(d, a.id).setdefault("dead_ends", []).append({"tried": a.tried, "why": a.why, "at": now().isoformat()})


def cmd_next(a):
    """What can move now, and the exact prompt of each step agent to launch (exit 1 while anything is unlaunched)."""
    probs = reconcile_sweeps()
    d, st = load()
    msgs = messages(oldest_open(d, st))
    todo = [(e, why, k) for e, why, launched, k in next_steps_keyed(d, st, msgs=msgs) if not launched]
    for p in probs:
        print("UNRECORDED " + p)
    for e, why, k in todo:
        if k is None:
            print("NEXT %s: %s" % (e["id"], why))
        else:
            print("NEXT %s step %d of %d (plan step %s --agent <id> --next %d): %s" % (
                e["id"], k, len(nexts(e)), e["id"], k, why))
        if a.prompts and e.get("status") == "running":
            print(review_prompt(e) if why.startswith("design to review") else step_prompt(e, why, k))
            print("-" * 60)
    for e in d["entries"]:                        # a running entry's open question: a reply is listed, never a step
        r = replied(e, msgs) if e.get("status") == "running" else None
        if r:
            print("REPLY %s: %s" % (e["id"], r[0]))
    for e, why in done_problems(d, st):
        print("SUGGEST %s: marked done, but %s; plan reopen %s --why ..., or fix it (plan evidence --drop ...)" % (
            e["id"], why, e["id"]))
    for n in NOTES:
        print("note: " + n)
    stop_notes = st.get("_stop_notes") or []      # the last stop's notes: the stop check never shows them to the user
    for n in stop_notes:
        if n not in NOTES:
            print("note: " + n)
    if stop_notes:
        _set_state("_stop_notes", [])
    print("plan: %d step(s) to launch" % len(todo))
    return 1 if todo or probs else 0


def done_refusals(e, st, commit, override_runs=None, timeout=600, record=None):   # timeout None: no limit
    """Why the entry cannot be marked done (empty: it can). Work is written up in docs/results.tex (its notes_ref
    label, a commit adding to it); a setup entry (kind: setup, plan kind) may instead have its record elsewhere, a
    CLAUDE.md rule or a tool, named by `record`, which the write-up commit must change."""
    out = []
    if record and e.get("kind") != "setup":
        out.append("--record is for setup entries only (plan kind %s setup); work is written up in docs/results.tex"
                   % e["id"])
        record = None
    chk = e.get("done_when")
    if not chk:
        out.append("no done_when check (plan check)")
    elif trivial_check(chk, min(timeout or 120, 120)):
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
    if record:
        show = subprocess.run(["git", "-C", ROOT, "show", "--format=%ct", "--name-only", commit, "--", record],
                              capture_output=True, text=True)
        lines = show.stdout.splitlines()
        if not os.path.exists(os.path.join(ROOT, record)):
            out.append("the record %s does not exist" % record)
        if show.returncode != 0 or not any(l.strip() == os.path.normpath(record) for l in lines[1:]):
            out.append("%s does not change the record %s" % (commit, record))
    else:
        ref = e.get("notes_ref")
        if not ref or ref not in labels():
            out.append("notes_ref %r is not a \\label in docs/results.tex (plan ref)" % ref)
        show = subprocess.run(["git", "-C", ROOT, "show", "--format=%ct", "--unified=0", commit, "--",
                               "docs/results.tex"], capture_output=True, text=True)
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
        why = done_refusals(e, st, a.writeup, a.override_runs, record=a.record)
        if sweeps_of(st, a.id):
            ended, lv, unread = sweep_status(a.id, st, site_sweeps(want_of(st, [a.id]), all_sites=False, fresh=True))
            if lv:
                why.append("sweeps still going: " + ", ".join(lv))
            if unread and not a.override_runs:
                why.append("sweeps whose site could not be read: %s (pass --override-runs with the reason)"
                           % ", ".join(unread))
        if why:
            sys.exit("plan: %s is not done:\n  " % a.id + "\n  ".join(why))
        e["status"], e["writeup"] = "done", a.writeup
        if a.record:
            e["record"] = a.record
        if a.override_runs:
            e["runs_override"] = a.override_runs
    print("plan: %s done (%s)" % (a.id, a.writeup))


def render(d):
    tex = lambda s: re.sub(r"([_&%#$])", r"\\\1", str(s or ""))
    out = [BEGIN, "\\begin{itemize}"]
    for e in d["entries"]:
        st = {"running": "running", "waiting_user": "waiting on the user", "held": "held"}.get(e.get("status"))
        if st:
            body = tex(e.get("question")) + (" Next: " + tex("; ".join(x for _, x in nexts(e))) if nexts(e) else "")
            if open_question(e):
                body += " Asked: " + tex(e.get("ask")) + " Searched first: " + tex(e.get("searched") or "not recorded")
                body += (" Waits on it: everything, " + tex(e["holds_all"])) if e.get("holds_all") else \
                    (" Waits on it: " + tex(e["blocks"]) if e.get("blocks") else "")
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

    def cmd(name, f, help, *args):
        s = sub.add_parser(name, help=help, description=help)
        for n, k in args:
            s.add_argument(*n, **k)
        s.set_defaults(f=f)
    ID = A("id", help="the plan entry's id")
    cmd("where", cmd_where, "print the paths of the plan file, the state file and the run registry")
    cmd("show", cmd_show, "print the entries; with an id, that one entry in full (read an entry this way, never by "
        "grepping plan/plan.yaml)",
        A("id", nargs="?", help="one entry, with its whole evidence, dead ends, quotes and design"),
        A("--compact", action="store_true", help="skip done entries and print only each entry's next step"))
    cmd("add", cmd_add, "add a new entry (running with --decision, waiting on the user with --ask)", ID,
        A("--title", required=True, help="a plain name of the work"),
        A("--question", required=True, help="the question the entry answers"),
        A("--next", required=True, help="the next step"),
        A("--match", required=True, help="comma-separated globs of the runs/sweeps its site work item covers"),
        A("--decision", help="the decision (D<n>) of the user's that authorises it"),
        A("--ask", help="the question for the user it waits on, when no decision authorises it yet"),
        A("--searched", help="with --ask: what was read before asking, and why none of it settles the question"))
    cmd("item", cmd_item, "create or attach the entry's site work item (same id, its match globs)", ID,
        A("--decision", required=True, help="the decision (D<n>) that authorises the item"),
        A("--match", help="comma-separated globs; default the entry's match"))
    cmd("check", cmd_check, "set the entry's done_when shell check (refused if it passes in an empty dir)", ID,
        A("--done-when", required=True, help="shell command, exit 0 = the entry is finished"),
        A("--replace", action="store_true", help="change a done entry's check (it must pass now)"),
        A("--stale", action="store_true", help="an open entry's check names the wrong files: the old one must fail, the new one pass (needs --why)"),
        A("--why", help="with --stale: what in the old check was wrong"))
    cmd("recheck", cmd_recheck, "rerun the done entries' done_when checks with no time limit",
        A("id", nargs="?", help="only this entry"))
    cmd("ref", cmd_ref, "set the entry's notes_ref", ID,
        A("--label", required=True, help="an existing \\label in docs/results.tex"))
    cmd("answer", cmd_answer, "record the user's answer: site decide, then the entry, all or nothing", ID,
        A("--quote", required=True, help="the user's exact words"),
        A("--kind", required=True, choices=KINDS, help="the kind of decision the words make"),
        A("--set", action="append", metavar="FIELD=VALUE", help="title|question|next=VALUE to change with it "
          "(repeatable)"))
    cmd("ask", cmd_ask, "put an open question for the user on the entry", ID,
        A("--ask", required=True, help="the question, as sent to the user"),
        A("--searched", required=True, help="what was read before asking (CLAUDE.md rules, site decisions) and why "
          "none settles it"),
        A("--blocks", help="what in the entry waits on the answer; the rest keeps moving"),
        A("--holds-all", metavar="WHY", help="the whole entry waits on the user; say why nothing else can move"))
    cmd("design", cmd_design, "record the design of the entry's next job (needed, and passed by the design reviewer, "
        "before any submission)", ID,
        *[A("--" + k, required=True, help={
            "question": "the question the job answers",
            "prior": "the prior work on the same question, searched in docs/results.tex, plan show <entry> and site "
                     "decisions --project FA (process, arm, data, quantity names): each with its result and how this "
                     "design differs; or \"none found\" plus the search terms",
            "measure": "the measurement that answers it",
            "check": "how existing data showed the measurement moves with the tested change",
            "fixed": "what is held fixed",
            "control": "the matched control",
            "seeds": "the seeds",
            "outcome": "how the outcome reads as yes and as no"}.get(k)) for k in DESIGN_FIELDS])
    cmd("design-pass", cmd_design_pass, "the design reviewer's PASS on the latest design (the reviewer runs it)", ID,
        A("--reviewer", required=True, help="the reviewer's note"))
    cmd("design-fix", cmd_design_fix, "the design reviewer's FIX on the latest design (the reviewer runs it)", ID,
        A("--reasons", required=True, help="what must change before the job goes"))
    cmd("review", cmd_review, "record the design reviewer the main session launched for a design left without a "
        "verdict", ID, A("--agent", required=True, help="the reviewer subagent's id"))
    cmd("unanswered", cmd_unanswered, "the user's messages since the open question do not answer it (it stays open)",
        ID)
    cmd("step", cmd_step, "record the step agent launched for a running entry", ID,
        A("--agent", required=True, help="the step subagent's id"),
        A("--next", type=int, help="the number of the next step it takes (needed when the entry holds several)"))
    cmd("run", cmd_run, "record jobs of the entry: registry runs or sweeps", ID,
        A("runs", nargs="*", help="registry run ids (site runs)"),
        A("--sweep", action="append", help="SITE:NAME of a sweep submitted through sweep_manager (repeatable)"),
        A("--next", type=int, help="the number of the next step the jobs belong to (default: the first step)"))
    cmd("sweeps", cmd_sweeps, "every recorded sweep's state on its site; records the unrecorded ones",
        A("--fresh", action="store_true", help="read the sites now, not the cache"))
    cmd("evidence", cmd_evidence, "add evidence to the entry", ID,
        A("--run", action="append", help="a run id (repeatable)"),
        A("--commit", help="the commit that holds the evidence"),
        A("--path", action="append", help="a repo path (repeatable)"),
        A("--drop", action="append", help="a path no longer evidence (pruned, moved)"))
    cmd("set-next", cmd_set_next, "set the entry's next steps: independent pieces of work are separate steps (--also), "
        "never chained with 'then'", ID, A("text", help="the next step"),
        A("--also", action="append", help="another independent next step, run in parallel (repeatable)"),
        A("--next", type=int, help="replace only next step N (a step agent's own step), keeping the others"))
    cmd("add-next", cmd_add_next, "add an independent next step to the entry", ID, A("text", help="the step"))
    cmd("drop-next", cmd_drop_next, "remove next step N (finished, nothing follows it)", ID,
        A("n", type=int, help="the step's number"))
    cmd("dead-end", cmd_dead_end, "record something tried that did not work", ID,
        A("--tried", required=True, help="what was tried"), A("--why", required=True, help="why it failed"))
    cmd("next", cmd_next, "what can move now (exit 1 while a listed step is not launched)",
        A("--prompts", action="store_true", help="print each step agent's exact prompt"))
    cmd("done", cmd_done, "close the entry: done_when passes and the write-up is committed", ID,
        A("--writeup", required=True, help="the commit of the write-up"),
        A("--override-runs", metavar="WHY", help="close despite runs in an unknown or unreadable state; say why"),
        A("--record", help="setup entries: the file that is their record (CLAUDE.md, a tool), changed by the write-up"))
    cmd("kind", cmd_kind, "mark the entry setup (its record may be CLAUDE.md or a tool) or work (written up in "
        "docs/results.tex)", ID, A("kind", choices=("setup", "work"), help="setup or work"))
    cmd("view", cmd_view, "render the open entries as the plan block of docs/results.tex",
        A("--write", action="store_true", help="write it into docs/results.tex instead of printing"))
    cmd("claims", cmd_claims, "the claims check of trunk since the last verified commit, no time limit",
        A("--verified", action="store_true", help="print the last verified commit and run nothing"))
    cmd("reopen", cmd_reopen, "a done entry back to running", ID, A("--why", required=True, help="why it reopens"))
    cmd("commit", cmd_commit, "commit plan/plan.yaml when a write left it uncommitted")
    a = p.parse_args(argv)
    COMMAND[:] = [a.cmd] + ([a.id] if getattr(a, "id", None) else [])
    return a.f(a) or 0


if __name__ == "__main__":
    sys.exit(main())
