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
    status  --queue Q        counts per state, and the processes assembled, as one JSON line
    yields  --queue Q        events asked and returned per run of every process with a short part, as one JSON line
            (recipes/gen_catalog_v4_default.py sizes those pools' n_runs from it; plan then links their parts, _adopt)
    gc      --queue Q [--dry-run]
            the MadEvent working directories ($SCRATCH/lhe_work/parts/<process>__<run>) of the queue's runs that are
            not running: a run whose part file holds its events loses the whole directory, any other (failed, killed,
            not started) keeps only its log files. A manifest of every path removed, with its size, is written under
            Q/gc/ before anything is deleted. A worker does the same for each run it ends (tidy_run).

The queue holds  todo/<prio>__<process>__<k>.json  claimed/<same>.<jobid>  done/  failed/  assembled/<process>.
Every pool is the one prebuild_recipes.py would build for the same spec and seed (same recipe ids).
"""
import argparse, glob, json, os, shutil, socket, statistics, subprocess, sys, time, traceback
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


def cmd_plan(a):
    D = _dirs(a.queue)
    for d in D.values():
        os.makedirs(d, exist_ok=True)
    rates = json.load(open(a.rates))          # tools/lhe_rates.py; no invented fallback rates
    by = {}
    for r in rates.values():
        by.setdefault(int(r["nfinal"]), []).append(r["core_s_per_event"])
    med = {k: statistics.median(v) for k, v in by.items()}
    src_cost = {}
    items = _items(a.specs, a.seed)
    cores = {k: int(v) for k, v in (c.split("=") for c in a.cores)}
    # the spec of each process (one per process: the assembly rebuilds its items from it)
    # tasks queued or claimed (a done one is known by its part file; a failed one is queued again: plan retries it)
    known = {f.split("__", 1)[1].split(".json")[0] for d in ("todo", "claimed") for f in os.listdir(D[d])}
    n_new = n_have = 0
    n_adopt = 0
    for name, its in sorted(items.items()):
        n_adopt += _adopt(D, name, its)
        with open(os.path.join(D["specs"], name + ".json"), "w") as f:
            json.dump({"seed": a.seed, "items": [[r, p] for r, p in its]}, f)
        rate, src = _rate(name, rates, med)
        c = cores.get(name, 1)
        for k, (part, run) in enumerate(mg.lhe_part_paths(name, its)):
            est = rate * run[1] / c
            sc = src_cost.setdefault(src, [0, 0.0]); sc[0] += 1; sc[1] += est * c / 3600
            tag = f"{name}__{k:03d}"
            if os.path.exists(part) or tag in known:
                n_have += 1
                continue
            prio = max(0, 99999999 - int(est))
            with open(os.path.join(D["todo"], f"{prio:08d}__{tag}.json"), "w") as f:
                json.dump({"process": name, "k": k, "est_s": est, "events": run[1], "part": part, "cores": c}, f)
            n_new += 1
    # an estimate: the measured core-seconds per event of the class's pool (or its multiplicity's median) times the
    # events asked, one-core time taken as cores x wall of the measured multicore run (no parallel loss)
    print(json.dumps({"processes": len(items), "tasks_new": n_new, "tasks_present": n_have, "parts_adopted": n_adopt,
                      "est_core_h": round(sum(v[1] for v in src_cost.values()), 1),
                      "est_by_rate_source": {k: {"tasks": v[0], "core_h": round(v[1], 1)} for k, v in src_cost.items()},
                      "est_longest_task_wall_h": round(max([json.load(open(f))["est_s"] for f in
                                                        glob.glob(os.path.join(D["todo"], "*.json"))] or [0]) / 3600,
                                                   2)}))


def _adopt(D, name, its):
    """Hard-link the part files of a process's previous plan (Q/specs) to its new names when the new layout only adds
    runs (lhe.n_runs raised, nothing else of the recipes changed, no lhe.energies): run k < the old n_runs has the same
    run name, iseed and events, so the same MadEvent run. Nothing is deleted. Returns the parts linked."""
    sf = os.path.join(D["specs"], name + ".json")
    if not os.path.exists(sf):
        return 0
    old = [r for r, _ in _spec(D, name)]
    new = [r for r, _ in its]
    strip = lambda r: {**r, "lhe": {k: v for k, v in r["lhe"].items() if k != "n_runs"}}
    if len(old) != len(new) or any(strip(a) != strip(b) for a, b in zip(old, new)) or new[0]["lhe"].get("energies"):
        return 0
    o_n, n_n = int(old[0]["lhe"].get("n_runs", 1)), int(new[0]["lhe"].get("n_runs", 1))
    o_id, n_id = mg._lhe_events_id(old[0]), mg._lhe_events_id(new[0])
    if o_id == n_id or n_n <= o_n:
        return 0
    n = 0
    for k, (part, run) in enumerate(mg.lhe_part_paths(name, its)[:o_n]):
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
        dst = os.path.join(D["claimed"], f"{f}.{me}")
        try:
            os.rename(os.path.join(D["todo"], f), dst)
        except OSError:
            continue                     # another job took it
        t["_file"], t["_claimed"] = f, dst
        return t
    return None


LOG_NAMES = ("ME5_debug", "MG5_debug")    # MadEvent's and MG5's crash logs; with every *.log file, what a failed run keeps


def parts_work_root():
    """$SCRATCH/lhe_work/parts, the only tree tidy_run deletes in; refuses when the work dir is set elsewhere."""
    s = os.environ.get("SCRATCH")
    if not s:
        raise RuntimeError("SCRATCH not set: no work tree to tidy")
    want = os.path.realpath(os.path.join(s, "lhe_work", "parts"))
    got = os.path.realpath(os.path.join(mg.lhe_work_dir(), "parts"))
    if got != want:
        raise RuntimeError(f"the MadEvent work tree {got} is not {want}: nothing deleted")
    return want


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
    # lhe.accept_short: a pool whose runs come back short at random (MadEvent "fail to reach target"), n_runs sized
    # from the events its runs returned (recipes/gen_catalog_v4_default.py SHORT_YIELDS): any part counts
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
        wd = run_workdir(name, run)
        if not os.path.isdir(wd):
            return None
        root = parts_work_root()
        _under(root, wd)
        saved = part_ok(part, part_need(name, run))
        rm, _ = _removals(wd, saved)
        os.makedirs(os.path.join(q, "gc"), exist_ok=True)
        man = os.path.join(q, "gc", f"worker_{os.environ.get('SLURM_JOB_ID', f'local{os.getpid()}')}.jsonl")
        with open(man, "a") as f:
            for fp, b in rm:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": f"{name}__{k:03d}",
                                    "saved": saved, "path": fp, "bytes": b}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        tidy_apply(root, wd, saved)
        return "saved" if saved else "logs"
    except Exception as e:
        print(f"  [TIDY] {name} part {k}: not tidied ({type(e).__name__}: {e})", flush=True)
        return None


def _spec(D, name):
    s = json.load(open(os.path.join(D["specs"], name + ".json")))
    return [(r, p) for r, p in s["items"]]


def _run_task(q, t):
    """Pool worker: the part, then the assembly if this part completed the process. Never raises."""
    D = _dirs(q)
    name = t["process"]
    t0 = time.time()
    try:
        items = _spec(D, name)
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


def cmd_work(a):
    D = _dirs(a.queue)
    end = time.time() + a.hours * 3600
    print(f"host {socket.gethostname()} job {os.environ.get('SLURM_JOB_ID')} workers {a.workers} "
          f"requeued {_requeue_stale(D)} stale claims", flush=True)
    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        live = {}
        while True:
            while True:
                free = a.workers - sum(int(t.get("cores", 1)) for t in live.values())
                t = _claim(D, end - time.time(), free) if free > 0 else None
                if t is None:
                    break
                live[ex.submit(_run_task, a.queue, t)] = t
            if not live:
                break
            done, _ = wait(list(live), return_when=FIRST_COMPLETED)
            for fut in done:
                t = live.pop(fut)
                _, msg, err = fut.result()
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
                    print(f"  [FAIL] {t['process']} part {t['k']}: {err.splitlines()[0]}", flush=True)
        # assemblies nobody did (their lock holder died or the last part ran in a job that lost the lock race)
        for name in (_assemble_pending(D, a.queue) if end - time.time() > ASSEMBLE_S else []):
            t, msg, err = _run_task(a.queue, {"process": name, "k": 0})
            print(f"  [ASSEMBLE] {name}: {msg or err.splitlines()[0]}", flush=True)
    print(f"LHE_PARTS_DONE ok {n_ok} failed {n_fail} left {len(os.listdir(D['todo']))}", flush=True)
    sys.exit(1 if n_fail else 0)


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
        for k, (part, run) in enumerate(mg.lhe_part_paths(name, items)):
            wd = run_workdir(name, run)
            if not os.path.isdir(wd):
                stats["absent"] += 1
                continue
            tag = f"{name}__{k:03d}"
            if claimed_now(tag):
                stats["running_skipped"] += 1
                continue
            _under(root, wd)
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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("specs", nargs="+"); p.add_argument("--queue", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cores", nargs="*", default=[], help="NAME=C: that process's runs on C cores each")
    p.add_argument("--rates", default=os.path.join(ROOT, "analysis/catalog_v4/data/lhe_rates.json"))
    w = sub.add_parser("work"); w.add_argument("--queue", required=True)
    w.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    w.add_argument("--hours", type=float, default=1.0)
    s = sub.add_parser("status"); s.add_argument("--queue", required=True)
    g = sub.add_parser("gc"); g.add_argument("--queue", required=True); g.add_argument("--dry-run", action="store_true")
    y = sub.add_parser("yields"); y.add_argument("--queue", required=True)
    a = ap.parse_args()
    {"plan": cmd_plan, "work": cmd_work, "status": cmd_status, "gc": cmd_gc, "yields": cmd_yields}[a.cmd](a)


if __name__ == "__main__":
    main()
