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

The queue holds  todo/<prio>__<process>__<k>.json  claimed/<same>.<jobid>  done/  failed/  assembled/<process>.
Every pool is the one prebuild_recipes.py would build for the same spec and seed (same recipe ids).
"""
import argparse, glob, json, os, shutil, socket, statistics, subprocess, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
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
    for name, its in sorted(items.items()):
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
    print(json.dumps({"processes": len(items), "tasks_new": n_new, "tasks_present": n_have,
                      "est_core_h": round(sum(v[1] for v in src_cost.values()), 1),
                      "est_by_rate_source": {k: {"tasks": v[0], "core_h": round(v[1], 1)} for k, v in src_cost.items()},
                      "est_longest_task_wall_h": round(max([json.load(open(f))["est_s"] for f in
                                                        glob.glob(os.path.join(D["todo"], "*.json"))] or [0]) / 3600,
                                                   2)}))


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
        mg.lhe_build_part(name, items, t["k"], mg_cores=int(t.get("cores", 1)))
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


def cmd_status(a):
    D = _dirs(a.queue)
    c = {k: len([f for f in os.listdir(D[k]) if not f.endswith(".err")]) for k in ("todo", "claimed", "done", "failed")}
    c["assembled"] = len(os.listdir(D["assembled"]))
    c["processes"] = len(os.listdir(D["specs"]))
    print(json.dumps(c))


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
    a = ap.parse_args()
    {"plan": cmd_plan, "work": cmd_work, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
