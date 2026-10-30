#!/usr/bin/env python3
"""Build lhe_pp pools as one-run part jobs pulled from a task queue on the site's filesystem (catalog_coverage, the
default-sampled share; mg5_pipeline_final "One-run part jobs").

    plan    SPEC.yaml... --queue Q [--seed 42] [--rates analysis/catalog_v4/data/lhe_rates.json] [--cores NAME=C ...]
            one task per MadEvent run of every lhe_pp pool of the specs not yet built (its part file not there), named
            so a sorted listing is longest first (the cost estimate: the class's measured core-seconds per event times
            the run's events); --cores runs that process's tasks on C cores each (MadEvent multicore; a run that is
            slow on one core, e.g. a stalling one whose every run asks the whole pool), estimated at core-s / C; rerunnable: a task already queued, claimed or whose part exists is not queued again
    work    --queue Q [--workers W] [--hours H]
            W slots, each claims the longest task that still fits the job's remaining time (claims are atomic renames,
            so any number of jobs drain one queue), runs it, and the job whose part completes a process assembles its
            pools (a lock per standalone directory, so no two jobs label or compile the same backend at once).
            Claims of jobs no longer in the queue (squeue) are put back first. Exits when nothing fits or is left.
    layout  --queue Q        one JSON line: current (every spec's layout is the recipes' now, not parked, no plan in
                             progress), why not, and the task counts with claimed_dead (claims of workers gone from
                             the scheduler; read by the supervisor before it resubmits workers)
    plan refuses (agent_redesign C5): a replan of a queue whose claims live workers hold unless it is parked first; a
            process missing from --probe (default analysis/catalog_v4/data/lhe_probe_yields.json) or asking more than
            2x the events it keeps there, unless --exclude'd or accepted (lhe_yield_accepted.json, written by accept-yield)
    accept-yield NAME=D<n>...   (laptop) record the user's decision accepting a low-yield process, checked against
                             the decision ledger: in force, and its quote or covers name the process
    park    --queue Q        its unclaimed tasks moved to Q/parked/<time> (its workers end with what they hold); parked
                             (Q/parked/.active) until a plan of the queue completes
    plan ... --skip-claimed Q2...   runs claimed in Q2 are not planned again; --exclude NAME... held processes
    plan ... --grid-cores C   an lhe.gridpack pool: first one gridpack task per energy (mg.lhe_build_gridpack, on C
            cores, at the head of the queue), then its runs, each claimed only once its energy's gridpack is stored
            ("needs"; a worker whose only runs left wait on a gridpack another job is making looks again every POLL_S)
            and drawn from it (mg.lhe_build_part), written as the same part files; the assembly is unchanged
    status  --queue Q        counts per state, and the processes assembled, as one JSON line
    yields  --queue Q        events asked and returned per run of every process with a short part, as one JSON line
            (recipes/gen_catalog_v4_default.py sizes those pools' n_runs from it; plan then links their parts, _adopt)
    shares  --queue Q [--requeue]   e+e- processes: events per energy in their parts against the equal share; --requeue
            moves the assembled marker of an assembled one whose shares are off aside, so its rebuild is assembled
    gc      --queue Q [--dry-run]
            the MadEvent working directories ($SCRATCH/lhe_work/parts/<process>__<run>; a worker on node-local $TMPDIR
            deletes its own, a failed run's logs copied to Q/logs/<process>__<k> first) of the queue's runs that are
            not running: a run whose part file holds its events loses the whole directory, any other (failed, killed,
            not started) keeps only its log files. A manifest of every path removed, with its size, is written under
            Q/gc/ before anything is deleted. A worker does the same for each run it ends (tidy_run).
    workdirs [--min-age H] [--apply]   every MadEvent working directory left on $SCRATCH/lhe_work, of any queue or
            layout (parts/<process>__<run>) or of a failed series build (<process>_madevent): sizes, ages, saved parts,
            any .npz inside checked against its lhe_parts copy; --apply deletes those older than H hours whose .npz all
            have copies (manifest under $SCRATCH/lhe_work/gc first)

The queue holds  todo/<prio>__<process>__<k>.json (a gridpack task: 00000000__<process>__g<j>.json)  claimed/<same>.<jobid>  done/  failed/  assembled/<process>.
Every pool is the one prebuild_recipes.py would build for the same spec and seed (same recipe ids).
"""
import argparse, glob, json, math, os, re, shutil, socket, statistics, subprocess, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import yaml  # noqa: E402
import datagen  # noqa: E402
import mg5_pipeline_final as mg  # noqa: E402

COUNT_KEY = {"train": "n_train", "val": "n_val", "test": "n_test"}
ASSEMBLE_S = 900          # time kept free for an assembly after the last part of a process (labels + compile)


def _items(spec_paths, seed):
    """process -> [(recipe, out)] of the lhe_pp pools of the specs not yet built (as prebuild_recipes plans them)."""
    out = {}
    for sp in spec_paths:
        for p in yaml.safe_load(open(sp))["processes"]:
            name = p["name"]
            if not mg.is_lhe_pp(name):
                continue
            for role in ("val", "test", "train"):
                n = int(p[COUNT_KEY[role]])
                if n <= 0:
                    continue
                rec = mg.variable_energy_recipe(name, float(p["sqrts"][0]), float(p["sqrts"][1]), n, role=role,
                                                seed=seed)
                path = mg.recipe_output_path(rec, datagen.dest_for_role(role))
                if not datagen._is_cached(path, mg.recipe_id(rec)):
                    out.setdefault(name, []).append((rec, path))
    return out


def _rate(name, rates, med):
    """Core-seconds per event of a process's MadEvent run: its own measured rate, else that of the pool it reuses or
    whose standalone it shares, else the median of its final-state multiplicity (an estimate)."""
    cfg = mg.PROCESSES[name]
    par = cfg.get("parent") or ""
    for n in (name, cfg.get("reuse"), cfg.get("standalone_of"), par.replace("v4_", "pp13v4_", 1) if par else None):
        if n and n in rates:
            return rates[n]["core_s_per_event"], "measured"
    return med.get(int(cfg["nfinal"]), max(med.values())), "multiplicity median"


def _dirs(q):
    return {k: os.path.join(q, k) for k in ("todo", "claimed", "done", "failed", "assembled", "locks", "specs")}


PROBE_YIELDS = os.path.join(ROOT, "analysis", "catalog_v4", "data", "lhe_probe_yields.json")
YIELD_FACTOR = 2.0        # events asked over events kept above this refuses the plan (agent_redesign C5)
# low-yield processes the user accepted, each with the decision checked against the ledger when it was recorded
# (accept-yield runs on the laptop, where the ledger is; plan reads this committed file on the site)
ACCEPTED_YIELDS = os.path.join(ROOT, "analysis", "catalog_v4", "data", "lhe_yield_accepted.json")
LEDGER = os.path.expanduser("~/.local/share/ccorch/runs.db")
ACCEPT_KINDS = ("approve", "scope", "prefer", "release")


def cmd_accept_yield(a):
    """NAME=D<n>...: record that the user's decision D<n> accepts a process that keeps under 1/YIELD_FACTOR of the
    events it asks. Refused unless D<n> is an FA decision in force (approve/scope/prefer/release) whose quote or covers
    name the process. Writes ACCEPTED_YIELDS (commit it; plan reads it on the site)."""
    import sqlite3
    if not os.path.exists(a.ledger):
        sys.exit("lhe_parts accept-yield: no decision ledger at %s (run it on the laptop)" % a.ledger)
    c = sqlite3.connect("file:%s?mode=ro" % a.ledger, uri=True)
    path = a.accepted or ACCEPTED_YIELDS
    acc = json.load(open(path)) if os.path.exists(path) else {}
    bad = []
    for x in a.names:
        name, _, did = x.partition("=")
        r = c.execute("SELECT kind, quote, covers FROM decisions WHERE id=? AND project='FA' AND in_force=1",
                      (did,)).fetchone()
        if not r or r[0] not in ACCEPT_KINDS:
            bad.append("%s: %s is not an FA decision in force (%s)" % (name, did, "/".join(ACCEPT_KINDS)))
        elif not any(re.search(r"(?<![\w])%s(?![\w])" % re.escape(name), t or "") for t in (r[1], r[2])):
            bad.append("%s: %s neither quotes nor covers it" % (name, did))
        else:
            acc[name] = {"decision": did, "quote": r[1], "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if bad:
        sys.exit("lhe_parts accept-yield: refused, nothing recorded:\n  " + "\n  ".join(bad))
    with open(path, "w") as f:
        json.dump(acc, f, indent=1, sort_keys=True)
    print(json.dumps({"accepted": sorted(acc)}))


def live_claims(q):
    """Claims in queue q held by jobs still in the scheduler (a claim of a job unknown to squeue counts as live when
    squeue cannot be read: the replan guard errs on the side of refusing)."""
    D = os.path.join(q, "claimed")
    if not os.path.isdir(D):
        return []
    live = _my_jobs()
    out = []
    for f in os.listdir(D):
        job = f.rsplit(".", 1)[-1]
        if live is None or job in live or not job.isdigit():
            out.append(f)
    return out


PARK_MARK = os.path.join("parked", ".active")    # written by park, removed by the next plan that completes


def parked(q):
    """A queue is parked from `park` until a plan of it completes (its live workers only finish what they hold);
    tasks parked earlier and then replanned do not make it parked again."""
    return os.path.exists(os.path.join(q, PARK_MARK))


def plan_refusals(a, items):
    """The generation rule's checks before a plan writes anything (C5). Returns the reasons to refuse ([] = go).
      - a queue already planned whose claims live workers hold is replanned only once parked (`park`): workers started
        under the old layout failed 908 tasks with IndexError after the 2026-10-29 replan (85e93d0f)
      - with --probe: every process is in the probe file (kept fraction of events per process, tools/lhe_probe.py
        summary), and none asks more than YIELD_FACTOR x the events it keeps, unless it is excluded or accepted in
        ACCEPTED_YIELDS (accept-yield, which checks the decision against the ledger)"""
    why = []
    if os.path.isdir(os.path.join(a.queue, "specs")) and os.listdir(os.path.join(a.queue, "specs")):
        held = live_claims(a.queue)
        if held and not parked(a.queue):
            why.append("queue %s is being replanned while live workers hold %d claims (e.g. %s): park it first "
                       "(lhe_parts park --queue %s), or plan a new queue with --skip-claimed %s"
                       % (a.queue, len(held), held[0], a.queue, a.queue))
    probe = getattr(a, "probe", None)
    if probe:
        y = json.load(open(probe))
        accfile = getattr(a, "accepted", None) or ACCEPTED_YIELDS
        accept = json.load(open(accfile)) if os.path.exists(accfile) else {}
        names = [n for n in sorted(items) if n not in (getattr(a, "exclude", None) or [])]
        missing = [n for n in names if n not in y]
        if missing:
            why.append("%d process(es) missing from the probe file %s (probe them first): %s"
                       % (len(missing), probe, ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else "")))
        low = [(n, y[n]) for n in names if n in y and (y[n] <= 0 or 1.0 / y[n] > YIELD_FACTOR) and n not in accept]
        if low:
            why.append("%d process(es) ask more than %gx the events they keep (kept fraction in %s): %s; fix their "
                       "layout, exclude them, or record the user's decision accepting them (on the laptop: "
                       "python3 tools/lhe_parts.py accept-yield NAME=D<n>..., then commit and sync)"
                       % (len(low), YIELD_FACTOR, probe, ", ".join("%s (kept %.3f, asked/kept %s)" % (
                           n, v, "inf" if v <= 0 else "%.1f" % (1.0 / v)) for n, v in low)))
    return why


def cmd_layout(a):
    """One JSON line for the supervisor: is the queue's layout the one workers submitted now would build?
    current = every spec's events identity equals the recipes' now (layout_current), the queue is not parked and no
    plan is in progress (Q/.planning). With the counts of `status`, and claimed_dead: claims of workers no longer in
    the scheduler (a worker that died at its limit), work still to do that only a worker started now puts back (0
    when squeue cannot be read)."""
    D = _dirs(a.queue)
    c = {k: len([f for f in os.listdir(D[k]) if ".json" in f and not f.endswith(".err")]) if os.path.isdir(D[k]) else 0
         for k in ("todo", "claimed", "done", "failed")}
    c["claimed_dead"] = max(0, c["claimed"] - len(live_claims(a.queue)))
    out = dict(c, queue=os.path.basename(os.path.normpath(a.queue)), current=True, why="")
    stale = []
    for f in sorted(os.listdir(D["specs"])) if os.path.isdir(D["specs"]) else []:
        name = f[:-5]
        try:
            layout_current(name, _spec(D, name))
        except StaleLayout:
            stale.append(name)
        except Exception as e:                     # a process the recipes no longer have
            stale.append("%s (%s)" % (name, type(e).__name__))
    if stale:
        out.update(current=False, why="%d process(es) laid out otherwise in the recipes now, e.g. %s"
                   % (len(stale), stale[0]), stale=stale[:50])
    elif parked(a.queue):
        out.update(current=False, why="parked")
    elif os.path.exists(os.path.join(a.queue, ".planning")):
        out.update(current=False, why="a plan is in progress")
    print(json.dumps(out))


def cmd_plan(a):
    items_pre = _items(a.specs, a.seed)
    why = plan_refusals(a, items_pre)
    if why:
        sys.exit("lhe_parts plan: refused:\n  " + "\n  ".join(why))
    D = _dirs(a.queue)
    for d in D.values():
        os.makedirs(d, exist_ok=True)
    mark = os.path.join(a.queue, ".planning")
    open(mark, "w").write(str(os.getpid()))
    try:
        _plan(a, D, items_pre)
    finally:
        os.remove(mark)
    if os.path.exists(os.path.join(a.queue, PARK_MARK)):    # planned again: no longer parked
        os.remove(os.path.join(a.queue, PARK_MARK))


def _plan(a, D, items):
    rates = json.load(open(a.rates))          # tools/lhe_rates.py; no invented fallback rates
    by = {}
    for r in rates.values():
        by.setdefault(int(r["nfinal"]), []).append(r["core_s_per_event"])
    med = {k: statistics.median(v) for k, v in by.items()}
    src_cost = {}
    # items: read once by cmd_plan, before its checks
    cores = {k: int(v) for k, v in (c.split("=") for c in a.cores)}
    # the spec of each process (one per process: the assembly rebuilds its items from it)
    # tasks queued or claimed (a done one is known by its part file; a failed one is queued again: plan retries it)
    known = {f.split("__", 1)[1].split(".json")[0] for d in ("todo", "claimed") for f in os.listdir(D[d])}
    # runs another queue's live workers hold (--skip-claimed Q2...: a queue being drained while this one takes over its
    # work), so no run is made twice; a part the old worker leaves short or failed is planned again on a later plan
    for q2 in getattr(a, "skip_claimed", None) or []:
        known |= {f.split("__", 1)[1].split(".json")[0] for f in os.listdir(os.path.join(q2, "claimed"))}
    n_new = n_have = n_grid = 0
    n_adopt = 0
    for name, its in sorted(items.items()):
        if name in (getattr(a, "exclude", None) or []):
            continue                                  # held out of this queue (plan --exclude)
        n_adopt += _adopt(D, name, its, [os.path.join(q2, "specs") for q2 in getattr(a, "skip_claimed", None) or []])
        with open(os.path.join(D["specs"], name + ".json"), "w") as f:
            json.dump({"seed": a.seed, "items": [[r, p] for r, p in its]}, f)
        rate, src = _rate(name, rates, med)
        c = cores.get(name, 1)
        parts = mg.lhe_part_paths(name, its)
        # lhe.gridpack: a gridpack task per energy first (on grid_cores cores, queued at the head: prio 0), its runs
        # claimed only once its gridpack exists ("needs"); a gridpack whose runs are all made is not queued again
        grids = mg.lhe_gridpack_paths(name, its)
        gc_ = int(getattr(a, "grid_cores", 1) or 1)
        for j, (tar, g) in enumerate(grids):
            tag = f"{name}__g{j:02d}"
            mine = [p for p, r in parts if mg.lhe_gridpack_of_run(r, [x for _, x in grids]) == j]
            if os.path.exists(tar) or tag in known or all(os.path.exists(p) for p in mine):
                continue
            # an estimate: the class's core-seconds per event times the grid's events, on gc_ cores
            est = rate * g[1] / gc_
            sc = src_cost.setdefault(src, [0, 0.0]); sc[0] += 1; sc[1] += est * gc_ / 3600
            with open(os.path.join(D["todo"], f"{0:08d}__{tag}.json"), "w") as f:
                json.dump({"process": name, "kind": "gridpack", "j": j, "est_s": est, "events": g[1],
                           "gridpack": tar, "cores": gc_}, f)
            n_grid += 1
        for k, (part, run) in enumerate(parts):
            est = rate * run[1] / c
            sc = src_cost.setdefault(src, [0, 0.0]); sc[0] += 1; sc[1] += est * c / 3600
            tag = f"{name}__{k:03d}"
            if os.path.exists(part) or tag in known:
                n_have += 1
                continue
            prio = max(0, 99999999 - int(est))
            t = {"process": name, "k": k, "est_s": est, "events": run[1], "part": part, "cores": c}
            if grids:
                j = mg.lhe_gridpack_of_run(run, [x for _, x in grids])
                t["needs"], t["needs_task"] = grids[j][0], f"{0:08d}__{name}__g{j:02d}.json"
            with open(os.path.join(D["todo"], f"{prio:08d}__{tag}.json"), "w") as f:
                json.dump(t, f)
            n_new += 1
    # an estimate: the measured core-seconds per event of the class's pool (or its multiplicity's median) times the
    # events asked, one-core time taken as cores x wall of the measured multicore run (no parallel loss)
    print(json.dumps({"processes": len(items), "tasks_new": n_new, "gridpack_tasks_new": n_grid, "tasks_present": n_have, "parts_adopted": n_adopt,
                      "est_core_h": round(sum(v[1] for v in src_cost.values()), 1),
                      "est_by_rate_source": {k: {"tasks": v[0], "core_h": round(v[1], 1)} for k, v in src_cost.items()},
                      "est_longest_task_wall_h": round(max([json.load(open(f))["est_s"] for f in
                                                        glob.glob(os.path.join(D["todo"], "*.json"))] or [0]) / 3600,
                                                   2)}))


def _adopt(D, name, its, other_specs=()):
    """Hard-link the part files of a process's previous plan (Q/specs) to its new names when the new layout changes only
    the number of runs (lhe.n_runs raised or cut; with lhe.run_events unchanged, also the margin; no lhe.energies): run
    k below both counts has the same run name, iseed and events, so the same MadEvent run. Nothing is deleted. Returns
    the parts linked."""
    # the process's previous plan: this queue's spec, else that of a queue it takes over (--skip-claimed)
    sf = next((f for f in [os.path.join(D["specs"], name + ".json")] +
               [os.path.join(d, name + ".json") for d in other_specs] if os.path.exists(f)), None)
    if sf is None:
        return 0
    old = [r for r, _ in json.load(open(sf))["items"]]
    new = [r for r, _ in its]
    # n_runs may go up (more runs) or down (the leading runs made already hold the blocks); with lhe.run_events the
    # events each run asks are fixed by it, so n_requested (margin) may change too; anything else is another layout
    fixed_ev = bool(new[0]["lhe"].get("run_events")) and new[0]["lhe"].get("run_events") == old[0]["lhe"].get("run_events")
    drop = ("n_runs", "n_requested") if fixed_ev else ("n_runs",)
    # lhe.complete_runs (identity "runs": "complete") changes which runs the assembly uses, not a run's events
    drop += ("runs",)
    strip = lambda r: {**r, "lhe": {k: v for k, v in r["lhe"].items() if k not in drop and k != "block"}}
    if len(old) != len(new) or any(strip(a) != strip(b) for a, b in zip(old, new)) or new[0]["lhe"].get("energies"):
        return 0
    o_n, n_n = int(old[0]["lhe"].get("n_runs", 1)), int(new[0]["lhe"].get("n_runs", 1))
    o_id, n_id = mg._lhe_events_id(old[0]), mg._lhe_events_id(new[0])
    if o_id == n_id or n_n == o_n and not fixed_ev:
        return 0
    n = 0
    for k, (part, run) in enumerate(mg.lhe_part_paths(name, its)[:min(o_n, n_n)]):
        src = os.path.join(os.path.dirname(part), os.path.basename(part).replace(f"{name}_{n_id}_", f"{name}_{o_id}_", 1))
        if os.path.exists(src) and not os.path.exists(part):
            os.link(src, part)
            n += 1
    return n


def _my_jobs():
    try:
        r = subprocess.run(["squeue", "--me", "-h", "-o", "%A"], capture_output=True, text=True, timeout=60)
        return set(r.stdout.split()) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _requeue_stale(D):
    live = _my_jobs()
    if live is None:
        return 0
    n = 0
    for l in os.listdir(D["locks"]):          # an assembly lock of a job that died: freed, _assemble_pending redoes it
        try:
            owner = open(os.path.join(D["locks"], l, "owner")).read().strip()
        except OSError:
            continue                       # being created this instant
        if owner.isdigit() and owner not in live and owner != os.environ.get("SLURM_JOB_ID"):
            shutil.rmtree(os.path.join(D["locks"], l), ignore_errors=True)
    for f in os.listdir(D["claimed"]):
        job = f.rsplit(".", 1)[-1]
        if not job.isdigit():
            continue                       # a claim of a worker run by hand (local<pid>): never in squeue
        if job not in live and job != os.environ.get("SLURM_JOB_ID"):
            try:
                os.rename(os.path.join(D["claimed"], f), os.path.join(D["todo"], f.rsplit(".", 1)[0]))
                n += 1
            except OSError:
                pass
    return n


def _claim(D, left_s, free=1):
    me = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
    for f in sorted(os.listdir(D["todo"])):
        if not f.endswith(".json"):
            continue
        try:
            t = json.load(open(os.path.join(D["todo"], f)))
        except (OSError, json.JSONDecodeError):
            continue
        if 1.5 * t["est_s"] + ASSEMBLE_S > left_s or int(t.get("cores", 1)) > free:
            continue
        if t.get("needs") and not os.path.exists(t["needs"]):
            continue                     # a gridpack run whose gridpack is not made yet
        dst = os.path.join(D["claimed"], f"{f}.{me}")
        try:
            os.rename(os.path.join(D["todo"], f), dst)
        except OSError:
            continue                     # another job took it
        t["_file"], t["_claimed"] = f, dst
        return t
    return None


LOG_NAMES = ("ME5_debug", "MG5_debug")    # MadEvent's and MG5's crash logs; with every *.log file, what a failed run keeps


def _local_root():
    """The node-local scratch ($TMPDIR) when the job has one, else None."""
    t = os.environ.get("TMPDIR")
    return os.path.realpath(t) if t and os.path.isdir(t) else None


def parts_work_root():
    """The MadEvent work tree of part runs, the only tree tidy_run deletes in: $SCRATCH/lhe_work/parts, or a tree
    under the node-local $TMPDIR (scripts/job_lhe_parts.sh puts the run directories there, so the shared disk holds
    only part files and logs). Refuses any other place."""
    got = os.path.realpath(os.path.join(mg.lhe_work_dir(), "parts"))
    s = os.environ.get("SCRATCH")
    if s and got == os.path.realpath(os.path.join(s, "lhe_work", "parts")):
        return got
    loc = _local_root()
    if loc and got.startswith(loc + os.sep):
        return got
    raise RuntimeError(f"the MadEvent work tree {got} is neither $SCRATCH/lhe_work/parts nor under $TMPDIR: "
                       f"nothing deleted")


def is_local(root):
    loc = _local_root()
    return bool(loc) and os.path.realpath(root).startswith(loc + os.sep)


def _under(root, path):
    r = os.path.realpath(path)
    if not r.startswith(root + os.sep) or os.path.islink(path):
        raise RuntimeError(f"refusing to delete {path}: not under {root}")
    return r


def part_ok(path, n_asked=None):
    """A run's events are saved: its part file loads and holds events (all it was asked for, when n_asked is given)."""
    try:
        with np.load(path) as z:
            n = len(z["mom"])
    except Exception:
        return False
    return n > 0 and (n_asked is None or n >= int(n_asked))


def part_need(name, run):
    """Events run `run` of `name` must return for its part to count as done: all it asked when the pool is split into
    runs sized to return their events (lhe.run_events), else any (runs that each ask the whole pool and stall at a few
    thousand, lhe.n_runs sized from what one run returned: short by design, the assembly checks their sum)."""
    l = mg.PROCESSES[name]["lhe"]
    # lhe.complete_runs: any part counts as a run made (a short one is not retried: its iseed would stop short again);
    # the assembly uses only the runs that returned every event they asked
    if l.get("complete_runs"):
        return None
    # lhe.accept_short: a pool whose runs come back short at random (MadEvent "fail to reach target"), n_runs sized
    # from the events its runs returned (recipes/gen_catalog_v4_default.py SHORT_YIELDS): any pp part counts; an e+e-
    # part (lhe.energies) counts when it holds its run's part of its energy's equal share of the blocks
    if l.get("accept_short") and l.get("energies"):
        need = sum(int(l.get(k, 0)) for k in ("n_val", "n_test", "n_train"))
        share = int(math.ceil(need / len(l["energies"])))
        runs_e = int(l.get("n_runs", 1))
        if l.get("energy_runs"):
            j = [float(e) for e in l["energies"]].index(2.0 * float(run[2]["ebeam1"]))
            runs_e = int(l["energy_runs"][j])
        return int(math.ceil(share / runs_e))
    return run[1] if l.get("run_events") and not l.get("accept_short") else None


def _du(path):
    n = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                n += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return n


def _keep(fname):
    return fname.endswith(".log") or fname in LOG_NAMES


def _removals(wd, saved):
    """([(path, bytes)] of the files tidying wd removes, [files kept]): every file when its events are saved, else all
    but its log files."""
    rm, kept = [], []
    for dp, _, fs in os.walk(wd):
        for f in fs:
            fp = os.path.join(dp, f)
            if not saved and _keep(f) and not os.path.islink(fp):
                kept.append(fp)
                continue
            try:
                rm.append((fp, os.lstat(fp).st_size))
            except OSError:
                pass
    return rm, kept


def tidy_plan(wd, saved):
    """(bytes removed, files kept) of tidying the run directory wd: everything when its events are saved, else all but
    its log files."""
    if saved:
        return _du(wd), []
    rm, kept = _removals(wd, saved)
    return sum(b for _, b in rm), kept


def tidy_apply(root, wd, saved):
    """Delete the run directory wd (saved) or all of it but its log files. Only under root."""
    _under(root, wd)
    if saved:
        shutil.rmtree(wd, ignore_errors=True)
        return
    for dp, ds, fs in os.walk(wd, topdown=False):
        for f in fs:
            fp = os.path.join(dp, f)
            if not (_keep(f) and not os.path.islink(fp)):
                try:
                    os.unlink(fp)
                except OSError:
                    pass
        for x in ds:
            xp = os.path.join(dp, x)
            if os.path.islink(xp):
                os.unlink(xp)
            else:
                try:
                    os.rmdir(xp)          # only when empty: a directory holding a kept log stays
                except OSError:
                    pass


def run_workdir(name, run):
    return os.path.join(mg.lhe_work_dir(), "parts", f"{name}__{run[3]}")


def tidy_run(name, items, k, q):
    """A worker's end of run k: its directory deleted once its part holds all the events it asked, else reduced to its
    logs. Each path removed and its size are appended first to Q/gc/worker_<job>.jsonl; when that write fails nothing
    is deleted. Returns 'saved', 'logs' or None (nothing deleted). Never raises: tidying never fails a task."""
    try:
        part, run = mg.lhe_part_paths(name, items)[k]
        return _tidy(run_workdir(name, run), part_ok(part, part_need(name, run)), f"{name}__{k:03d}", q)
    except Exception as e:
        print(f"  [TIDY] {name} part {k}: not tidied ({type(e).__name__}: {e})", flush=True)
        return None


def _tidy(wd, saved, tag, q):
    """tidy_run's work on the directory wd of task tag: deleted when saved, else reduced to its logs; each removal in
    Q/gc/worker_<job>.jsonl first. Returns 'saved', 'logs' or None. Never raises."""
    try:
        if not os.path.isdir(wd):
            return None
        root = parts_work_root()
        _under(root, wd)
        rm, _ = _removals(wd, saved)
        os.makedirs(os.path.join(q, "gc"), exist_ok=True)
        man = os.path.join(q, "gc", f"worker_{os.environ.get('SLURM_JOB_ID', f'local{os.getpid()}')}.jsonl")
        with open(man, "a") as f:
            for fp, b in rm:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": tag,
                                    "saved": saved, "path": fp, "bytes": b}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if is_local(root) and not saved:
            # a node-local directory vanishes with the job: its logs go to Q/logs/<run> on the shared disk, the rest
            # is deleted with it
            dst = os.path.join(q, "logs", tag)
            os.makedirs(dst, exist_ok=True)
            for fp in _removals(wd, False)[1]:
                shutil.copy2(fp, os.path.join(dst, os.path.relpath(fp, wd).replace(os.sep, "__")))
            tidy_apply(root, wd, True)
            return "logs"
        tidy_apply(root, wd, saved)
        return "saved" if saved else "logs"
    except Exception as e:
        print(f"  [TIDY] {tag}: not tidied ({type(e).__name__}: {e})", flush=True)
        return None


def _spec(D, name):
    s = json.load(open(os.path.join(D["specs"], name + ".json")))
    return [(r, p) for r, p in s["items"]]


class StaleLayout(RuntimeError):
    """The worker's process table (mg.PROCESSES, read from the recipes when it started) gives another run layout than
    the queue's spec: the queue was planned again under newer recipes while this worker ran."""


STALE = "StaleLayout"


def layout_current(name, items):
    """Refuses (StaleLayout) when the recipes this process would write now differ from the spec's in their events'
    identity (lhe.n_runs, energy_runs, run_events, cards...): lhe_part_paths takes its identity from the spec and its
    runs from mg.PROCESSES, so a worker started before a replan would build or assemble the old layout's run k under
    the new name (another energy at the same path, or an index past the old runs)."""
    r = items[0][0]
    now = mg.lhe_pp_recipe(name, r["sqrts_min"], r["sqrts_max"], r["n_events"], role=r["role"], seed=r["seed"])
    if mg._lhe_events_id(now) != mg._lhe_events_id(r):
        raise StaleLayout(f"{name}: this worker's recipes give another layout than the queue's spec (planned again "
                          f"since it started); task returned to todo")


def _run_gridpack_task(q, t, items):
    """A gridpack task: the j-th gridpack of the process made and stored (mg.lhe_build_gridpack); its work directory
    tidied as a run's (deleted once the gridpack is stored, else reduced to its logs, Q/logs on node-local disk)."""
    name, j = t["process"], int(t["j"])
    tar, g = mg.lhe_gridpack_paths(name, items)[j]
    try:
        mg.lhe_build_gridpack(name, items, j, mg_cores=int(t.get("cores", 1)), keep_dir=True)
    finally:
        _tidy(os.path.join(mg.lhe_work_dir(), "parts", f"{name}__{g[3]}"), os.path.exists(tar),
              f"{name}__g{j:02d}", q)
    return f"gridpack {g[3]} made"


def _run_task(q, t):
    """Pool worker: the part, then the assembly if this part completed the process. Never raises."""
    D = _dirs(q)
    name = t["process"]
    t0 = time.time()
    try:
        items = _spec(D, name)
        layout_current(name, items)
        if t.get("kind") == "gridpack":
            return t, _run_gridpack_task(q, t, items) + f" in {time.time() - t0:.0f}s", None
        try:
            mg.lhe_build_part(name, items, t["k"], mg_cores=int(t.get("cores", 1)), keep_dir=True)
            part, run = mg.lhe_part_paths(name, items)[t["k"]]
            if not part_ok(part, part_need(name, run)):   # a short part of a run_events pool is a failed run
                raise RuntimeError(f"{name}: part {t['k']} written but holds fewer events than the {run[1]} asked: "
                                   f"{part}")
        finally:
            if "_file" in t:                     # a queued run (not the assembly-only call of _assemble_pending)
                tidy_run(name, items, t["k"], q)
        msg = f"part {t['k']} in {time.time() - t0:.0f}s"
        parts = mg.lhe_part_paths(name, items)
        if all(os.path.exists(p) for p, _ in parts) and not os.path.exists(os.path.join(D["assembled"], name)):
            lock = os.path.join(D["locks"], os.path.basename(mg.lhe_standalone_dir(name)))
            try:
                os.mkdir(lock)
            except FileExistsError:
                return t, msg + "; assembly left to the lock holder", None
            with open(os.path.join(lock, "owner"), "w") as f:
                f.write(os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}"))
            if os.path.exists(os.path.join(D["assembled"], name)):     # assembled between the check and the lock
                shutil.rmtree(lock, ignore_errors=True)
                return t, msg, None
            try:
                res = mg.lhe_assemble(name, items, write_sidecars=False)
                for out, full in res:
                    mg.write_recipe(out, full)
                with open(os.path.join(D["assembled"], name), "w") as f:
                    f.write("\n".join(o for o, _ in res) + "\n")
                msg += f"; assembled {len(res)} pools"
            finally:
                shutil.rmtree(lock, ignore_errors=True)
        return t, msg, None
    except BaseException as e:
        return t, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"


def _assemble_pending(D, q):
    """Processes whose parts are all there and nobody assembled (an assembler that died, or a lock that was held)."""
    out = []
    for sf in glob.glob(os.path.join(D["specs"], "*.json")):
        name = os.path.basename(sf)[:-5]
        if os.path.exists(os.path.join(D["assembled"], name)):
            continue
        try:
            if all(os.path.exists(p) for p, _ in mg.lhe_part_paths(name, _spec(D, name))):
                out.append(name)
        except Exception:
            continue
    return out


POLL_S = 60               # how often a worker with runs waiting on a gridpack made elsewhere looks again


def _waiting_on_gridpack(D):
    """Whether a run task in todo/ waits on a gridpack that a live job is making now (its gridpack task in claimed/;
    claims of jobs gone from squeue are put back to todo/ first). A gridpack task still in todo/ is not waited for:
    this job would have claimed it had it fit."""
    # a claim of a job gone from the scheduler (killed while making the gridpack) goes back to todo first, so this
    # worker claims the gridpack itself instead of waiting on it
    _requeue_stale(D)
    claimed = {f.rsplit(".", 1)[0] for f in os.listdir(D["claimed"])}
    for f in os.listdir(D["todo"]):
        if not f.endswith(".json"):
            continue
        try:
            t = json.load(open(os.path.join(D["todo"], f)))
        except (OSError, json.JSONDecodeError):
            continue
        if t.get("needs") and not os.path.exists(t["needs"]) and t.get("needs_task") in claimed:
            return True
    return False


def _label(t):
    return f"gridpack {t['j']}" if t.get("kind") == "gridpack" else f"part {t['k']}"


def cmd_work(a):
    D = _dirs(a.queue)
    end = time.time() + a.hours * 3600
    print(f"host {socket.gethostname()} job {os.environ.get('SLURM_JOB_ID')} workers {a.workers} "
          f"requeued {_requeue_stale(D)} stale claims", flush=True)
    n_ok = n_fail = n_stale = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        live = {}
        while True:
            while True:
                free = a.workers - sum(int(t.get("cores", 1)) for t in live.values())
                t = _claim(D, end - time.time(), free) if free > 0 else None
                if t is None:
                    break
                live[ex.submit(_run_task, a.queue, t)] = t
            blocked = _waiting_on_gridpack(D)
            if not live:
                if blocked and end - time.time() > ASSEMBLE_S:
                    time.sleep(POLL_S)       # runs whose gridpack another job is making: claimed once it is stored
                    continue
                break
            done, _ = wait(list(live), timeout=POLL_S if blocked else None, return_when=FIRST_COMPLETED)
            for fut in done:
                t = live.pop(fut)
                _, msg, err = fut.result()
                if err is not None and err.startswith(STALE):
                    # planned again under newer recipes: the task goes back, this worker claims nothing more
                    try:
                        os.rename(t["_claimed"], os.path.join(D["todo"], t["_file"]))
                    except OSError:
                        pass
                    n_stale += 1
                    end = 0.0
                    print(f"  [STALE] {t['process']} {_label(t)}: {err.splitlines()[0]}", flush=True)
                    continue
                dst = D["done"] if err is None else D["failed"]
                try:
                    os.rename(t["_claimed"], os.path.join(dst, t["_file"]))
                except OSError:
                    pass
                if err is None:
                    n_ok += 1
                    print(f"  [OK] {t['process']} {msg}", flush=True)
                else:
                    n_fail += 1
                    with open(os.path.join(D["failed"], t["_file"] + ".err"), "w") as f:
                        f.write(err)
                    print(f"  [FAIL] {t['process']} {_label(t)}: {err.splitlines()[0]}", flush=True)
        # assemblies nobody did (their lock holder died or the last part ran in a job that lost the lock race)
        for name in (_assemble_pending(D, a.queue) if end - time.time() > ASSEMBLE_S and not n_stale else []):
            t, msg, err = _run_task(a.queue, {"process": name, "k": 0})
            print(f"  [ASSEMBLE] {name}: {msg or err.splitlines()[0]}", flush=True)
    print(f"LHE_PARTS_DONE ok {n_ok} failed {n_fail} stale {n_stale} left {len(os.listdir(D['todo']))}", flush=True)
    sys.exit(1 if n_fail or n_stale else 0)


def cmd_gc(a):
    """Tidy the work directories of the queue's runs that no live job is running (tidy_plan), manifest first."""
    D = _dirs(a.queue)
    root = parts_work_root()
    live = _my_jobs()
    if live is None:
        raise SystemExit("squeue unavailable: cannot tell running claims, nothing deleted")
    live = {"jobs": live}

    def claimed_now(tag):
        """Whether a live job (or a hand-run worker, local<pid>) holds run tag (<process>__<k>) at this moment:
        claimed/ read again, squeue asked again for a claim by a job not in the last answer."""
        for f in os.listdir(D["claimed"]):
            if f.rsplit(".", 1)[0].split("__", 1)[1][:-len(".json")] != tag:
                continue
            job = f.rsplit(".", 1)[-1]
            if not job.isdigit() or job in live["jobs"]:
                return True
            live["jobs"] = _my_jobs()
            if live["jobs"] is None or job in live["jobs"]:   # squeue unavailable: counted as running
                live["jobs"] = live["jobs"] or set()
                return True
        return False
    todo = []
    stats = {"saved": [0, 0], "logs": [0, 0], "running_skipped": 0, "absent": 0, "short_parts": 0}
    for sf in sorted(glob.glob(os.path.join(D["specs"], "*.json"))):
        name = os.path.basename(sf)[:-5]
        items = _spec(D, name)
        # each run's directory (saved: its part holds its events), and each gridpack's (saved: the gridpack is stored)
        dirs = [(run_workdir(name, run), f"{name}__{k:03d}", part, run)
                for k, (part, run) in enumerate(mg.lhe_part_paths(name, items))]
        dirs += [(os.path.join(mg.lhe_work_dir(), "parts", f"{name}__{g[3]}"), f"{name}__g{j:02d}", tar, None)
                 for j, (tar, g) in enumerate(mg.lhe_gridpack_paths(name, items))]
        for wd, tag, part, run in dirs:
            if not os.path.isdir(wd):
                stats["absent"] += 1
                continue
            if claimed_now(tag):
                stats["running_skipped"] += 1
                continue
            _under(root, wd)
            if run is None:
                saved = os.path.exists(part)
            else:
                saved = part_ok(part, part_need(name, run))   # a short part of a run_events pool keeps its logs
                if not saved and part_ok(part):
                    stats["short_parts"] += 1
            n, kept = tidy_plan(wd, saved)
            kind = "saved" if saved else "logs"
            stats[kind][0] += 1; stats[kind][1] += n
            todo.append({"path": wd, "run": tag, "bytes": n, "removed": "all" if saved else "all but the kept logs",
                         "kept": kept})
    os.makedirs(os.path.join(a.queue, "gc"), exist_ok=True)
    man = os.path.join(a.queue, "gc", f"manifest_{time.strftime('%Y%m%d_%H%M%S')}{'_dry' if a.dry_run else ''}.json")
    with open(man, "w") as f:
        json.dump({"root": root, "dry_run": a.dry_run, "entries": todo}, f)
    skipped = []                                 # claimed between the plan and its delete (a gc runs for hours)
    if not a.dry_run:
        for e in todo:
            if claimed_now(e["run"]):
                skipped.append(e["path"])
                continue
            tidy_apply(root, e["path"], e["removed"] == "all")
        with open(man[:-len(".json")] + "_skipped.json", "w") as f:
            json.dump({"skipped_claimed_at_delete": skipped}, f)
    print(json.dumps({"manifest": man, "dry_run": a.dry_run, "dirs": len(todo),
                      "gb": round(sum(e["bytes"] for e in todo) / 1e9, 2),
                      "saved_dirs": stats["saved"][0], "saved_gb": round(stats["saved"][1] / 1e9, 2),
                      "logs_only_dirs": stats["logs"][0], "logs_only_gb": round(stats["logs"][1] / 1e9, 2),
                      "running_skipped": stats["running_skipped"], "claimed_at_delete_skipped": len(skipped),
                      "no_dir": stats["absent"], "short_parts_kept_logs": stats["short_parts"]}))


def _newest_mtime(path):
    t = os.lstat(path).st_mtime
    for dp, ds, fs in os.walk(path):
        for f in ds + fs:
            try:
                t = max(t, os.lstat(os.path.join(dp, f)).st_mtime)
            except OSError:
                pass
    return t


def _run_card(wd):
    """(iseed, nevents) read from the run card of the MadEvent directory under the run directory wd, None each when
    absent."""
    out = {"iseed": None, "nevents": None}
    for c in glob.glob(os.path.join(wd, "*_madevent", "Cards", "run_card.dat")):
        for line in open(c, errors="replace"):
            m = re.match(r"\s*(\d+)\s*=\s*(iseed|nevents)\b", line.split("!")[0])
            if m:
                out[m.group(2)] = int(m.group(1))
    return out["iseed"], out["nevents"]


def workdir_audit(min_age_h=12.0):
    """Every MadEvent working directory left on the shared disk, whatever queue or layout made it: the run
    directories $SCRATCH/lhe_work/parts/<process>__<run> (a queue's gc reaches only the runs of its current layout, so
    a run of a replanned layout or of no queue is never tidied) and the series builds' $SCRATCH/lhe_work/<process>_madevent
    (build_lhe_pp left them when a build failed). Per directory: bytes, age, the part files it would have made, and every
    .npz inside it with whether lhe_parts holds an identical-size copy that loads. A directory is deletable when it is
    older than min_age_h (no job writes it) and holds no .npz without its copy."""
    s = os.environ.get("SCRATCH")
    if not s:
        raise SystemExit("SCRATCH not set: nothing audited")
    work = os.path.realpath(os.path.join(s, "lhe_work"))
    pdir = mg.lhe_parts_dir()
    now = time.time()
    dirs = [(os.path.join(work, "parts", x), "run") for x in sorted(os.listdir(os.path.join(work, "parts")))
            if os.path.isdir(os.path.join(work, "parts", x))] if os.path.isdir(os.path.join(work, "parts")) else []
    dirs += [(os.path.join(work, x), "series") for x in sorted(os.listdir(work))
             if x.endswith("_madevent") and os.path.isdir(os.path.join(work, x))] if os.path.isdir(work) else []
    out = []
    for wd, kind in dirs:
        base = os.path.basename(wd)
        process, run = (base.split("__", 1) + [None])[:2] if kind == "run" else (base[:-len("_madevent")], None)
        iseed, nev = _run_card(wd)
        npz = []
        for dp, _, fs in os.walk(wd):
            for f in fs:
                if f.endswith(".npz"):
                    fp = os.path.join(dp, f)
                    cp = os.path.join(pdir, process, f)
                    ok = (os.path.exists(cp) and os.path.getsize(cp) == os.path.getsize(fp) and part_ok(cp))
                    npz.append({"path": fp, "copy": cp, "copied": ok})
        parts = []
        if run:
            parts = glob.glob(os.path.join(pdir, process, f"{process}_*_{run}_seed{iseed if iseed else '*'}_ev*.npz"))
            parts += glob.glob(os.path.join(pdir, process, f"{process}_*_{run}_seed*_gridpack.tar.gz"))
        events = bool(glob.glob(os.path.join(wd, "**", "*.lhe*"), recursive=True))
        age_h = (now - _newest_mtime(wd)) / 3600
        out.append({"path": wd, "kind": kind, "process": process, "run": run, "bytes": _du(wd),
                    "age_h": round(age_h, 1), "iseed": iseed, "nevents": nev, "events_in_dir": events,
                    "part_saved": any(part_ok(p) or p.endswith(".tar.gz") for p in parts),
                    "npz": npz, "deletable": age_h >= min_age_h and all(n["copied"] for n in npz)})
    return work, out


def cmd_workdirs(a):
    """workdirs [--min-age H] [--apply]: workdir_audit over the shared tree, a manifest of every directory and its
    size under $SCRATCH/lhe_work/gc/ first; with --apply each deletable directory is removed (re-checked just before)."""
    work, out = workdir_audit(a.min_age)
    os.makedirs(os.path.join(work, "gc"), exist_ok=True)
    man = os.path.join(work, "gc", f"workdirs_{time.strftime('%Y%m%d_%H%M%S')}{'' if a.apply else '_dry'}.json")
    with open(man, "w") as f:
        json.dump({"root": work, "apply": a.apply, "min_age_h": a.min_age, "dirs": out}, f)
    freed = 0
    if a.apply:
        for e in out:
            if not e["deletable"]:
                continue
            _under(work, e["path"])
            if (time.time() - _newest_mtime(e["path"])) / 3600 < a.min_age:
                continue                                   # written since the audit
            shutil.rmtree(e["path"], ignore_errors=True)
            freed += 0 if os.path.exists(e["path"]) else e["bytes"]
    gb = lambda xs: round(sum(e["bytes"] for e in xs) / 1e9, 2)
    by = lambda k: [e for e in out if e["kind"] == k]
    npz = [n for e in out for n in e["npz"]]
    print(json.dumps({"manifest": man, "apply": a.apply, "dirs": len(out), "gb": gb(out),
                      "run_dirs": len(by("run")), "run_gb": gb(by("run")),
                      "series_dirs": len(by("series")), "series_gb": gb(by("series")),
                      "deletable_dirs": sum(e["deletable"] for e in out), "deletable_gb": gb([e for e in out if e["deletable"]]),
                      "npz_inside": len(npz), "npz_without_copy": [n["path"] for n in npz if not n["copied"]],
                      "run_dirs_with_part_saved": sum(e["part_saved"] for e in by("run")),
                      "run_dirs_events_no_part": sum(e["events_in_dir"] and not e["part_saved"] for e in by("run")),
                      "younger_than_min_age": sum(e["age_h"] < a.min_age for e in out),
                      "freed_gb": round(freed / 1e9, 2)}))


def cmd_park(a):
    """Move a queue's unclaimed tasks (todo/) to parked/<time>/, so its live workers claim nothing more and end with the
    runs they hold; nothing deleted. Their work is planned in another queue (plan --skip-claimed this queue)."""
    D = _dirs(a.queue)
    dst = os.path.join(a.queue, "parked", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dst, exist_ok=True)
    open(os.path.join(a.queue, PARK_MARK), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
    n = 0
    for f in os.listdir(D["todo"]):
        try:
            os.rename(os.path.join(D["todo"], f), os.path.join(dst, f))
            n += 1
        except OSError:
            pass                                  # claimed this instant
    print(json.dumps({"parked": n, "to": dst, "claimed_left": len(os.listdir(D["claimed"]))}))


def cmd_status(a):
    D = _dirs(a.queue)
    c = {k: len([f for f in os.listdir(D[k]) if ".json" in f and not f.endswith(".err")])     # tasks, not subdirs
         for k in ("todo", "claimed", "done", "failed")}
    c["assembled"] = len(os.listdir(D["assembled"]))
    c["processes"] = len(os.listdir(D["specs"]))
    print(json.dumps(c))


def cmd_yields(a):
    """process -> {asked, got} per run (got null: no part file) for every process of the queue with a short part."""
    D = _dirs(a.queue)
    out = {}
    for sf in sorted(glob.glob(os.path.join(D["specs"], "*.json"))):
        name = os.path.basename(sf)[:-5]
        asked, got = [], []
        for part, run in mg.lhe_part_paths(name, _spec(D, name)):
            asked.append(int(run[1]))
            try:
                with np.load(part) as z:
                    got.append(int(len(z["mom"])))
            except (FileNotFoundError, OSError, KeyError, ValueError):
                got.append(None)
        if any(g is not None and g < q for g, q in zip(got, asked)):
            out[name] = {"asked": asked, "got": got}
    print(json.dumps(out, sort_keys=True))


def cmd_shares(a):
    """Every e+e- process of the queue (lhe.energies): events per energy in its part files against the equal share of
    its blocks, ceil(need / energies); one JSON line {process: {assembled, counts, share, ok}}. --requeue moves the
    assembled marker of each assembled process whose shares are off to Q/assembled_superseded/<process>.<time> (its
    pools are rebuilt when its layout changes; nothing deleted)."""
    D = _dirs(a.queue)
    out = {}
    for sf in sorted(glob.glob(os.path.join(D["specs"], "*.json"))):
        name = os.path.basename(sf)[:-5]
        items = _spec(D, name)
        parts = mg.lhe_part_paths(name, items)
        E = mg.lhe_energy_of_runs([r for _, r in parts])
        if not E or E[0] is None:
            continue
        cnt = dict.fromkeys(E, 0)
        for (part, _), e in zip(parts, E):
            try:
                with np.load(part) as z:
                    cnt[e] += int(len(z["mom"]))
            except (FileNotFoundError, OSError, KeyError, ValueError):
                pass
        need = max(r["lhe"]["block"][1] for r, _ in items)
        share = int(math.ceil(need / len(cnt)))
        out[name] = {"assembled": os.path.exists(os.path.join(D["assembled"], name)),
                     "counts": {f"{e:g}": c for e, c in cnt.items()}, "share": share,
                     "ok": min(cnt.values()) >= share}
    if a.requeue:
        sup = os.path.join(a.queue, "assembled_superseded")
        os.makedirs(sup, exist_ok=True)
        for name, r in out.items():
            if r["assembled"] and not r["ok"]:
                os.rename(os.path.join(D["assembled"], name),
                          os.path.join(sup, f"{name}.{time.strftime('%Y%m%d_%H%M%S')}"))
                r["requeued"] = True
    print(json.dumps(out, sort_keys=True))


def part_energy_ok(path, run):
    """An e+e- part (its run has ebeam1) holds events at its run's energy: the beams' energies sum to 2 x ebeam1."""
    if "ebeam1" not in run[2]:
        return True
    with np.load(path) as z:
        m = z["mom"]
    want = 2.0 * float(run[2]["ebeam1"])
    return len(m) == 0 or abs(float(np.median(m[:, 0, 0] + m[:, 1, 0])) - want) <= 1e-6 * want


def cmd_requeue(a):
    """Tasks a stale worker lost (StaleLayout): for every process not assembled, each run whose part file is missing
    and has no task in todo/ or claimed/ gets its latest task from failed/ (only when its error's first line matches
    --match) or done/ (done but its part missing: built under another layout) moved back to todo/, its .err moved to
    failed_requeued/. --energies: an e+e- part at another energy than its run's is moved to
    <parts>/<process>/wrong_energy/, its pool's assembled marker to assembled_superseded/, and its run requeued.
    Nothing deleted; prints what it does, does it with --apply."""
    D = _dirs(a.queue)
    pat = re.compile(a.match) if a.match else None
    live = {f.split("__", 1)[1].split(".json")[0] for d in ("todo", "claimed") for f in os.listdir(D[d])}
    tasks = {}
    for d in ("done", "failed"):
        for f in os.listdir(D[d]):
            if f.endswith(".json") and f.count("__") == 2:
                tag = f.split("__", 1)[1][:-5]
                p = os.path.join(D[d], f)
                if tag not in tasks or os.path.getmtime(p) > os.path.getmtime(tasks[tag]):
                    tasks[tag] = p
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = {"requeued_failed": 0, "requeued_done": 0, "wrong_energy": [], "unassembled": [], "no_task": 0,
           "failed_not_matching": 0}
    moves = []
    for sf in sorted(glob.glob(os.path.join(D["specs"], "*.json"))):
        name = os.path.basename(sf)[:-5]
        parts = mg.lhe_part_paths(name, _spec(D, name))
        wrong = []
        if a.energies:
            for k, (part, run) in enumerate(parts):
                if os.path.exists(part) and not part_energy_ok(part, run):
                    wrong.append(k)
                    dst = os.path.join(os.path.dirname(part), "wrong_energy", f"{os.path.basename(part)}.{stamp}")
                    moves.append((part, dst))
                    out["wrong_energy"].append(os.path.basename(part))
        marker = os.path.join(D["assembled"], name)
        if wrong and os.path.exists(marker):
            moves.append((marker, os.path.join(a.queue, "assembled_superseded", f"{name}.{stamp}")))
            out["unassembled"].append(name)
        elif os.path.exists(marker):
            continue
        for k, (part, _) in enumerate(parts):
            tag = f"{name}__{k:03d}"
            if (os.path.exists(part) and k not in wrong) or tag in live:
                continue
            src = tasks.get(tag)
            if src is None:
                out["no_task"] += 1
                continue
            if os.path.basename(os.path.dirname(src)) == "failed":
                try:
                    first = (open(src + ".err").read().splitlines() or [""])[0]
                except OSError:
                    first = ""
                if pat is None or not pat.search(first):
                    out["failed_not_matching"] += 1
                    continue
                if os.path.exists(src + ".err"):
                    moves.append((src + ".err", os.path.join(a.queue, "failed_requeued", os.path.basename(src) + f".err.{stamp}")))
                out["requeued_failed"] += 1
            else:
                out["requeued_done"] += 1
            moves.append((src, os.path.join(D["todo"], os.path.basename(src))))
    if a.apply:
        for s, d in moves:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            os.rename(s, d)
    out["applied"] = bool(a.apply)
    print(json.dumps(out))


def cmd_shortbuilt(a):
    """Every lhe_pp pool on disk built from a MadEvent run that stopped short of its ask (catalog_coverage: such a run
    biases its events, analysis/catalog_v4/short_run_bias.py): per process, its sidecar's events per run against the
    events each run of its layout asked (mg.lhe_runs at the sidecar's iseed and n_requested; a layout whose run seeds no
    longer match the sidecar's is listed as layout_unknown). One JSON object {process: {...}} to --out."""
    import datagen
    out, n_seen = {}, 0
    for name in sorted(mg.PROCESSES):
        if not mg.is_lhe_pp(name):
            continue
        cfg = mg.PROCESSES[name]
        side = path = None
        for role in ("train", "test", "val"):
            lo, hi = mg.lhe_layout(cfg)[role]
            if hi - lo <= 0:
                continue
            rec = mg.lhe_pp_recipe(name, 13000, 13000, hi - lo, role=role, seed=a.seed)
            path = mg.recipe_output_path(rec, datagen.dest_for_role(role))
            try:
                side = json.load(open(path + ".recipe.json"))
                if side.get("lhe_run"):
                    break
            except (FileNotFoundError, ValueError):
                side = None
        if not side or not side.get("lhe_run"):
            continue
        n_seen += 1
        lr, sl = side["lhe_run"], side.get("lhe") or {}
        got = [int(x) for x in lr.get("events_per_run") or []]
        try:
            runs = mg.lhe_runs(name, int(sl["iseed"]), int(sl["n_requested"]))
        except Exception as e:
            out[name] = {"layout_unknown": f"{type(e).__name__}: {e}"[:200], "pool": path}
            continue
        if sl.get("runs") == "complete":                  # lhe.complete_runs: the sidecar lists only the runs used
            gone = {int(x["iseed"]) for x in lr.get("runs_dropped_short") or []}
            runs = [r for r in runs if int(r[0]) not in gone]
        if [r[0] for r in runs] != [int(x) for x in lr.get("iseeds") or []] or len(runs) != len(got):
            out[name] = {"layout_unknown": "the layout's run seeds differ from the sidecar's", "pool": path,
                         "events_per_run": got}
            continue
        asked = [int(r[1]) for r in runs]
        short = [k for k, (g, q) in enumerate(zip(got, asked)) if g < q]
        if short:
            out[name] = {"built": sl.get("build", "serial"), "runs": len(got), "short_runs": len(short),
                         "events_short": sum(got[k] for k in short), "events": sum(got),
                         "asked_per_run": sorted(set(asked)), "pool": path,
                         "recipe_current": side.get("recipe_id") == mg.recipe_id(rec)}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=0, sort_keys=True)
    print(json.dumps({"pools_with_sidecar": n_seen, "built_from_short_runs": sum("short_runs" in v for v in out.values()),
                      "layout_unknown": sorted(k for k, v in out.items() if "layout_unknown" in v), "out": a.out}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("specs", nargs="+"); p.add_argument("--queue", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cores", nargs="*", default=[], help="NAME=C: that process's runs on C cores each")
    p.add_argument("--skip-claimed", nargs="*", default=[], help="queues whose claimed runs are not planned again")
    p.add_argument("--exclude", nargs="*", default=[], help="processes not planned in this queue (held)")
    p.add_argument("--grid-cores", type=int, default=8,
                   help="cores of each gridpack task (lhe.gridpack pools; MadEvent multicore integration)")
    p.add_argument("--probe", default=PROBE_YIELDS, help="kept fraction per process (tools/lhe_probe.py summary); every "
                   "planned process must be in it and keep at least 1/%g of the events it asks" % YIELD_FACTOR)
    ay = sub.add_parser("accept-yield"); ay.add_argument("names", nargs="+", metavar="NAME=D<n>")
    ay.add_argument("--ledger", default=LEDGER); ay.add_argument("--accepted")
    ly = sub.add_parser("layout"); ly.add_argument("--queue", required=True)
    pk = sub.add_parser("park"); pk.add_argument("--queue", required=True)
    p.add_argument("--rates", default=os.path.join(ROOT, "analysis/catalog_v4/data/lhe_rates.json"))
    w = sub.add_parser("work"); w.add_argument("--queue", required=True)
    w.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    w.add_argument("--hours", type=float, default=1.0)
    s = sub.add_parser("status"); s.add_argument("--queue", required=True)
    g = sub.add_parser("gc"); g.add_argument("--queue", required=True); g.add_argument("--dry-run", action="store_true")
    wdp = sub.add_parser("workdirs"); wdp.add_argument("--min-age", type=float, default=12.0)
    wdp.add_argument("--apply", action="store_true")
    y = sub.add_parser("yields"); y.add_argument("--queue", required=True)
    sh = sub.add_parser("shares"); sh.add_argument("--queue", required=True); sh.add_argument("--requeue", action="store_true")
    rq = sub.add_parser("requeue"); rq.add_argument("--queue", required=True)
    rq.add_argument("--match", default=None, help="regex on a failed task's error (first line); none: no failed task")
    rq.add_argument("--energies", action="store_true"); rq.add_argument("--apply", action="store_true")
    sb = sub.add_parser("shortbuilt"); sb.add_argument("--out", required=True); sb.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    {"shortbuilt": cmd_shortbuilt, "plan": cmd_plan, "layout": cmd_layout, "accept-yield": cmd_accept_yield, "park": cmd_park, "work": cmd_work, "status": cmd_status, "gc": cmd_gc, "workdirs": cmd_workdirs, "yields": cmd_yields,
     "shares": cmd_shares, "requeue": cmd_requeue}[a.cmd](a)


if __name__ == "__main__":
    main()
