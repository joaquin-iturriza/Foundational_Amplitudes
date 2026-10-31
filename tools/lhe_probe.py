#!/usr/bin/env python3
"""Probe of the lhe part work still to do (catalog_coverage step 7): what a MadEvent run of each remaining process
returns, how much of it its pool keeps, what it costs in time and disk, and what disk a CPU node has.

    list   --queue Q...                     the processes of the queues not assembled, one per line
    parts  --queue Q... --out F.json [--match RE]   from the part files already made: per process (assembled or not) the runs
                                            with a part, their events asked and returned, MadEvent seconds and cores;
                                            for a default-share pool (dsh_) the fraction of its events the mixture
                                            keeps (mg5_pipeline_final.default_share_momenta's accept: the class's
                                            sqrt(s_hat) window and fiducial cuts in the partonic frame) and what each
                                            part of that accept removes alone; an e+e- pool's kept fraction per energy
    run    --process P [--events N] [--timeout S] [--work DIR] [--patch K=V ...] [--energy E]
                                            one MadEvent run of P (its first run's card and patches, a probe iseed) in
                                            DIR (default $TMPDIR): seconds of the MG5 output and of generate_events,
                                            events returned, the fraction kept (as above), the run directory's largest
                                            size while it ran; one JSON line. The directory is deleted afterwards.
    summary LOG...                          the PROBE_JSON lines of probe job logs (site logs <run> -n 100000) per class
                                            (beam: pp or e+e-; pool: default share, pp13v4 or ee4; final-state
                                            multiplicity): runs, timeouts, median and minimum yield, median kept
                                            fraction, median seconds per kept event, median MG5 output seconds, peak
                                            run-directory size (median, max); variants (--patch) apart
    kept LOG... --out F.json [--yields-out Y.json] [--by-energy-out B.json] [--rates-out R.json]
                                            the default-share runs' kept fractions per process and e+e- energy; with
                                            --yields-out each probed process's yield too (a gridpack pool's worst
                                            energy's), with --by-energy-out its yield and complete share per energy
    mixcheck --queue Q...                   every assembled default-share pool read as the arms' mixture reads it
    runsize --queue Q... [--max-run N] [--min-events M]   default-share pp parts from runs of at most N events
                                            against the reuse pool's events (runs of 10k+), KS on sqrt(s_hat), max |eta|,
                                            max pT
    runlevel --queue Q... [--match RE]     the runsize pass with the run as the unit: reuse pools' short runs, rows'
                                            runs from the shuffle, within-run correlation, run-level permutation test
    node                                    the node's disks (TMPDIR, /tmp, SCRATCH), cores and memory, one JSON line
"""
import argparse, glob, json, math, os, re, shutil, signal, socket, subprocess, sys, time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import yaml  # noqa: E402
import mg5_pipeline_final as mg  # noqa: E402
from tools import lhe_parts as lp  # noqa: E402

TRAIN_RECIPE = os.path.join(ROOT, "recipes", "catalog_v4_train.yaml")


def _queues(qs):
    return [q if os.path.isabs(q) else os.path.join(os.environ["SCRATCH"], "lhe_queue", q) for q in qs]


def remaining(qs):
    out = []
    for q in _queues(qs):
        D = lp._dirs(q)
        for sf in sorted(glob.glob(os.path.join(D["specs"], "*.json"))):
            name = os.path.basename(sf)[:-5]
            if not os.path.exists(os.path.join(D["assembled"], name)):
                out.append((q, name))
    return out


_WIN = None


def accept_of(name):
    """(sqrts_lo, sqrts_hi, m_finals, pdg_ids, cuts) the mixture applies to a dsh pool's events (its parent class's
    training window and cuts), None for a pool that keeps every event (pp13v4, ee4)."""
    global _WIN
    cfg = mg.PROCESSES[name]
    if not name.startswith("dsh_"):
        return None
    if _WIN is None:
        _WIN = {e["name"]: e["sqrts"] for e in yaml.safe_load(open(TRAIN_RECIPE))["processes"]}
    par = mg.PROCESSES[cfg["parent"]]
    cuts = par.get("fiducial_cuts") or mg.V4_CUTS[mg.v4_cut_kind(par["pdg_ids"])]
    lo, hi = _WIN[cfg["parent"]]
    return float(lo), float(hi), [float(m) for m in par["m_finals"]], [int(p) for p in par["pdg_ids"]], cuts


def kept(P, acc):
    """Fraction of events P (lab frame, as in a part file) the accept keeps, and the fraction each piece removes alone:
    sqrt(s_hat) below / above the window, the fiducial cuts (partonic frame)."""
    if acc is None or len(P) == 0:
        return {"kept": 1.0 if len(P) else None}
    lo, hi, m_f, pdg, cuts = acc
    Q = mg.to_partonic_frame(P)
    q = Q[:, 0] + Q[:, 1]
    sq = np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 0.0, None))
    cut = mg.fiducial_pass_mask(Q, np.asarray(m_f, float), np.asarray(pdg, int), cuts)
    # the same cuts in the lab frame: what MadEvent's card could apply (pT, Delta R and pair masses are boost
    # invariant along the beams, |eta| is not)
    cut_lab = mg.fiducial_pass_mask(np.asarray(P, float), np.asarray(m_f, float), np.asarray(pdg, int), cuts)
    ok = (sq >= lo) & (sq <= hi) & cut
    return {"kept": float(ok.mean()), "below_window": float((sq < lo).mean()), "above_window": float((sq > hi).mean()),
            "cuts_partonic": float((~cut).mean()), "cuts_lab": float((~cut_lab).mean()),
            "cuts_partonic_not_lab": float((~cut & cut_lab).mean())}


def cmd_list(a):
    for q, n in remaining(a.queue):
        print(n)


def cmd_parts(a):
    out = {}
    for q in _queues(a.queue):
        D = lp._dirs(q)
        for sf in sorted(glob.glob(os.path.join(D["specs"], "*.json"))):
            name = os.path.basename(sf)[:-5]
            try:
                parts = mg.lhe_part_paths(name, lp._spec(D, name))
            except Exception as e:                       # a layout this checkout no longer gives
                out[name] = {"queue": os.path.basename(q), "error": f"{type(e).__name__}: {e}"}
                continue
            if a.match and not re.search(a.match, name):
                continue
            asked, got, secs, cores, Ps = [], [], [], [], []
            byE = {}                                     # e+e-: up to 5 parts per collider energy
            for p, run in parts:
                try:
                    with np.load(p) as z:
                        m = np.asarray(z["mom"])
                        got.append(len(m)); secs.append(float(z["seconds"][1])); cores.append(int(z["mg_cores"]))
                        asked.append(int(run[1]))
                        if len(Ps) < 20:
                            Ps.append(m)
                        if run[2].get("ebeam1") is not None and len(byE.setdefault(2 * float(run[2]["ebeam1"]), [])) < 5:
                            byE[2 * float(run[2]["ebeam1"])].append(m)
                except (FileNotFoundError, OSError, KeyError, ValueError):
                    continue
            acc = accept_of(name)
            r = {"queue": os.path.basename(q), "assembled": os.path.exists(os.path.join(D["assembled"], name)),
                 "runs": len(parts), "runs_with_part": len(got), "asked": sum(asked), "got": sum(got),
                 "yield": (sum(got) / sum(asked)) if asked else None,
                 "core_s_per_returned_event": (sum(s * c for s, c in zip(secs, cores)) / sum(got)) if sum(got) else None,
                 "run_events": parts[0][1][1], "nfinal": int(mg.PROCESSES[name]["nfinal"])}
            if Ps:
                r.update(kept(np.concatenate(Ps), acc))
            if byE and acc is not None:
                # the pool takes equal shares of the energies, so what it keeps is the mean over them of each one's
                r["kept_by_energy"] = {f"{E:g}": kept(np.concatenate(ms), acc)["kept"] for E, ms in sorted(byE.items())}
                r["energies"] = [float(e) for e in mg.PROCESSES[name]["lhe"]["energies"]]
            out[name] = r
    with open(a.out, "w") as f:
        json.dump(out, f, indent=0, sort_keys=True)
    print(json.dumps({"processes": len(out), "out": a.out}))


def _du(path):
    try:
        return int(subprocess.run(["du", "-sb", path], capture_output=True, text=True, timeout=60).stdout.split()[0])
    except Exception:
        return None


def _keep(keep, name, fname, m, aq, banner):
    """Save one run's events and banner as KEEP/<process>/<fname> (atomic rename); returns the path."""
    d = os.path.join(os.path.expandvars(keep), name)      # KEEP may name $SCRATCH (a job's --env holds it literally)
    os.makedirs(d, exist_ok=True)
    kp = os.path.join(d, fname)
    np.savez(kp + ".tmp.npz", mom=m, aqcd=aq, banner=np.array(banner))
    os.replace(kp + ".tmp.npz", kp)
    return kp


def cmd_gridpack(a):
    """One gridpack of a process at the precision of a run of --grid-events (mg.lhe_make_gridpack, on --cores cores),
    then --runs runs of --events events drawn from it with seeds seed_base+1.., --cores at a time, each kept as
    KEEP/<process>/<process>_gpack<grid iseed>_seed<s>_ev<n>.npz with the LHE header as its banner. Cards and patches as
    the part jobs run them (the pool's first run's patches, --energy for an e+e- run). One GRIDPACK_JSON line per run."""
    from concurrent.futures import ThreadPoolExecutor
    name = a.process
    cfg = mg.PROCESSES[name]
    base = a.work or os.environ.get("TMPDIR") or "/tmp"
    wd = os.path.join(base, f"lhe_gpack_{name}_{os.getpid()}")
    os.makedirs(wd, exist_ok=True)
    log = os.path.join(wd, f"{name}.mg5.log")
    head = {"process": name, "host": socket.gethostname(), "grid_events": a.grid_events, "events_asked": a.events}
    try:
        t0 = time.time()
        ev, _ = mg.lhe_prepare(name, wd, log, madevent=True, standalone=False)
        runs = mg.lhe_runs(name, mg.lhe_iseed(name, 42), mg.lhe_n_requested(cfg))
        patches = dict(runs[0][2])
        if a.energy:
            patches.update({"ebeam1": a.energy / 2.0, "ebeam2": a.energy / 2.0})
        giseed = mg.lhe_iseed(name, 7777 + a.seed_base)
        tar = mg.lhe_make_gridpack(name, ev, a.grid_events, giseed, log, mg_cores=a.cores, patches=patches)
        head.update(grid_iseed=giseed, s_gridpack=round(time.time() - t0, 1), patches=patches,
                    gridpack_bytes=os.path.getsize(tar))
        txt = open(log).read()
        head["grid_fail_to_reach_target"] = "fail to reach target" in txt
        print("GRIDPACK_JSON " + json.dumps(dict(head, stage="grid")), flush=True)

        def one(s):
            r = dict(head, seed=s)
            rw = os.path.join(wd, f"run_{s}")
            t1 = time.time()
            try:
                lhe = mg.lhe_run_gridpack(tar, rw, a.events, s, os.path.join(wd, f"run_{s}.log"), timeout=a.timeout)
                m, aq, _ = mg.read_lhe_pp(lhe, cfg["pdg_ids"])
                r.update(events_returned=len(m), s_run=round(time.time() - t1, 1))
                if a.keep:
                    r["kept_events_file"] = _keep(a.keep, name, f"{name}_gpack{giseed}_seed{s}_ev{int(a.events)}.npz",
                                                  m, aq, mg.lhe_gridpack_header(lhe))
            except Exception as e:
                r["error"] = f"{type(e).__name__}: {e}"
            finally:
                shutil.rmtree(rw, ignore_errors=True)
            print("GRIDPACK_JSON " + json.dumps(r), flush=True)
            return r
        with ThreadPoolExecutor(max(1, a.cores)) as ex:
            out = list(ex.map(one, range(a.seed_base + 1, a.seed_base + 1 + a.runs)))
        print("GRIDPACK_SUMMARY " + json.dumps({"process": name, "runs": len(out),
                                                "complete": sum(r.get("events_returned") == a.events for r in out),
                                                "errors": sum("error" in r for r in out)}), flush=True)
    except Exception as e:
        print("GRIDPACK_JSON " + json.dumps(dict(head, stage="grid", error=f"{type(e).__name__}: {e}")), flush=True)
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def _kv(x):
    k, v = x.split("=", 1)
    return k, (float(v) if "." in v else int(v))


def _keep_logs(keep, name, stem, rw, run_log):
    """A gridpack run's logs saved as KEEP/<process>/<stem>.logs.tar.gz: its run.sh output and, per integration
    channel, results.dat and log.txt (the iterations, points and unweighted events of the run's event generation)."""
    import tarfile
    d = os.path.join(os.path.expandvars(keep), name)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, stem + ".logs.tar.gz")
    with tarfile.open(out, "w:gz") as t:
        if os.path.exists(run_log):
            t.add(run_log, arcname="run.log")
        for f in sorted(glob.glob(os.path.join(rw, "madevent", "SubProcesses", "P*", "G*", "*"))):
            if os.path.basename(f) in ("results.dat", "log.txt") or os.path.basename(f).startswith("log"):
                t.add(f, arcname=os.path.relpath(f, rw))
    return out


def cmd_gridpool(a):
    """The generation rule's probe of lhe.gridpack pools (catalog_coverage step 7): for each --process, each of its
    gridpacks built as the queue builds it (mg.lhe_build_gridpack, at its path: the queue then reuses it), then --runs
    runs of the pool's run size drawn from it at probe seeds (lhe_iseed(process, 9999 + s), not the pool's), --cores at
    a time. One PROBE_JSON line per run, read by `summary` and `kept --yields-out` like a MadEvent probe run's (yield,
    kept fraction, seconds per kept event; "gridpack": true, the energy in its patches); one GRIDPACK_JSON line per
    gridpack (stage grid: seconds, bytes). --events asks runs of that size instead of the pool's run size (a run-size probe
    on the pool's own gridpacks), --energy keeps only that energy's gridpack, --seed-base B takes run seeds B+1, B+2, ..
    instead of the hashed probe seeds (which collide: lhe_iseed is mod 20000), so probes of several sizes laid out on
    disjoint ranges never share a seed, and --keep saves each run's
    events as KEEP/<process>/<process>_gpack<grid iseed>_seed<iseed>_ev<n>.npz (short_run_bias.py sizecheck)."""
    import datagen
    from concurrent.futures import ThreadPoolExecutor
    for name in a.process:
        cfg = mg.PROCESSES[name]
        if not cfg["lhe"].get("gridpack"):
            print("GRIDPACK_JSON " + json.dumps({"process": name, "stage": "grid", "error": "no lhe.gridpack"}),
                  flush=True)
            continue
        items = []
        for role, (lo, hi) in mg.lhe_layout(cfg).items():
            if hi > lo:
                rec = mg.variable_energy_recipe(name, 13000, 13000, hi - lo, role=role, seed=42)
                items.append((rec, mg.recipe_output_path(rec, datagen.dest_for_role(role))))
        base = a.work or os.environ.get("TMPDIR") or "/tmp"
        wd = os.path.join(base, f"lhe_gridpool_{name}_{os.getpid()}")
        os.makedirs(wd, exist_ok=True)
        run_ev = int(a.events or cfg["lhe"]["run_events"])
        xi = dict(_kv(x) for x in (a.ximprove or []))
        try:
            for j, (tar, g) in enumerate(mg.lhe_gridpack_paths(name, items)):
                if a.energy and abs(2.0 * float(g[2].get("ebeam1", 0.0)) - a.energy) > 1e-6:
                    continue
                head = {"process": name, "host": socket.gethostname(), "grid": g[3], "grid_iseed": g[0],
                        "grid_events": g[1], "patches": g[2]}
                t0 = time.time()
                try:
                    had = os.path.exists(tar)
                    mg.lhe_build_gridpack(name, items, j, mg_cores=a.cores, work_root=wd)
                    print("GRIDPACK_JSON " + json.dumps(dict(head, stage="grid", reused=had,
                                                             s_gridpack=round(time.time() - t0, 1),
                                                             gridpack_bytes=os.path.getsize(tar))), flush=True)
                except Exception as e:
                    print("GRIDPACK_JSON " + json.dumps(dict(head, stage="grid", error=f"{type(e).__name__}: {e}")),
                          flush=True)
                    continue

                def one(s, tar=tar, g=g):
                    seed = a.seed_base + s if a.seed_base else mg.lhe_iseed(name, 9999 + s)
                    r = {"process": name, "seed": seed, "events_asked": run_ev, "patches": g[2], "gridpack": True,
                         "grid": g[3], "host": socket.gethostname()}
                    rw = os.path.join(wd, f"run_{g[3]}_{s}")
                    t1 = time.time()
                    try:
                        lhe = mg.lhe_run_gridpack(tar, rw, run_ev, seed, os.path.join(wd, f"run_{g[3]}_{s}.log"),
                                                  timeout=a.timeout, ximprove=xi)
                        m, aq, _ = mg.read_lhe_pp(lhe, cfg["pdg_ids"])
                        r["s_generate"] = round(time.time() - t1, 1)
                        if a.keep:
                            r["kept_events_file"] = _keep(a.keep, name, f"{name}_gpack{g[0]}_seed{seed}_ev{run_ev}.npz",
                                                          m, aq, mg.lhe_gridpack_header(lhe))
                        r["events_returned"] = len(m)
                        r["yield"] = len(m) / run_ev
                        r.update(kept(m, accept_of(name)))
                        k = r.get("kept") or 0.0
                        r["s_per_kept_event"] = round(r["s_generate"] / (len(m) * k), 4) if len(m) and k else None
                    except Exception as e:
                        r["error"] = f"{type(e).__name__}: {e}"
                    finally:
                        if a.keep_logs:
                            r["logs_file"] = _keep_logs(a.keep_logs, name, f"{name}_gpack{g[0]}_seed{seed}_ev{run_ev}",
                                                        rw, os.path.join(wd, f"run_{g[3]}_{s}.log"))
                        shutil.rmtree(rw, ignore_errors=True)
                    print("PROBE_JSON " + json.dumps(r), flush=True)
                    return r
                with ThreadPoolExecutor(max(1, a.cores)) as ex:
                    list(ex.map(one, range(1, a.runs + 1)))
        finally:
            shutil.rmtree(wd, ignore_errors=True)


GRID_TIME_FACTOR = 4.3    # measured gridpack seconds over the rate estimate: the gridpack probe, 2026-10-30 (1.5 to 4.3)


def _gridpool_measured(logs):
    """From gridpool job logs: per pool its built gridpacks' seconds ({grid: s}), its probe runs (s or None for an
    error), and, for a job cut by its time limit, the pool it was on and the seconds it had spent there (a lower bound
    on that pool: start of the job plus what the log's finished pools took, to the cancel line)."""
    import datetime as dt
    m = {}
    for f in logs:
        start = end = None; spent = 0.0; last = None
        for l in open(f, errors="replace"):
            if l.startswith("GRIDPACK_JSON ") or l.startswith("PROBE_JSON "):
                d = json.loads(l.split(" ", 1)[1]); q = m.setdefault(d["process"], {"grid": {}, "runs": []})
                last = d["process"]
                if l.startswith("GRIDPACK_JSON") and "s_gridpack" in d:
                    q["grid"][d["grid"] + json.dumps(d.get("patches"), sort_keys=True)] = d["s_gridpack"]
                    spent += d["s_gridpack"]
                elif l.startswith("PROBE_JSON"):
                    q["runs"].append(d.get("s_generate"))
            k = re.search(r"\* Date:\s+(.*CET \d{4})", l)
            if k and start is None:
                start = dt.datetime.strptime(k.group(1).replace(" CET", ""), "%a %b %d %I:%M:%S %p %Y")
            k = re.search(r"\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)[.\d]*\] error: .*DUE TO TIME LIMIT", l)
            if k:
                end = dt.datetime.fromisoformat(k.group(1))
        if end and start:
            m.setdefault("__cut__", []).append({"log": f, "after": last, "s_job": (end - start).total_seconds()})
    return m


def cmd_gridpool_plan(a):
    """The gridpool probe's batch: the lhe.gridpack pools of --complete (a JSON list, default
    analysis/catalog_v4/data/complete_runs.json) dealt longest first to --jobs jobs of --cores cores (each pool's
    estimate: per gridpack, the class's core-seconds per event (lhe_parts._rate) times its grid events over the cores,
    times GRID_TIME_FACTOR, plus --runs runs of its run size at --cores at a time). One JSON line: the jobs' pools and
    estimated hours (an estimate).
    --logs (gridpool logs of an earlier pass): a pool with every gridpack built and every probe run returned is done
    and left out; a built gridpack costs nothing; a run that errored (its timeout) costs --timeout; an unbuilt
    gridpack's estimate is scaled by the largest measured ratio (gridpack seconds over the rate estimate) of its
    final-state multiplicity instead of GRID_TIME_FACTOR (the first gridpool pass measured 0.1 to 18.4 against 4.3:
    the multiplicity-median rate does not predict a 5-body gridpack); a pool a time limit cut is at least the seconds
    it had run ("floor_h"). Jobs >= pools gives one pool per job, so no pool waits behind another's estimate error."""
    import statistics
    rates = json.load(open(a.rates))
    by = {}
    for r in rates.values():
        by.setdefault(int(r["nfinal"]), []).append(r["core_s_per_event"])
    med = {k: statistics.median(v) for k, v in by.items()}
    meas = _gridpool_measured(a.logs) if a.logs else {}
    pools = json.load(open(a.complete))
    per_grid, nf = {}, {}
    for n in pools:
        l = mg.PROCESSES[n]["lhe"]
        r, _ = lp._rate(n, rates, med)
        per_grid[n] = (r * int(l["gridpack"]) / a.cores, math.ceil(a.runs / a.cores) * r * int(l["run_events"]))
        nf[n] = int(mg.PROCESSES[n]["nfinal"])
    factor = {}
    for n, q in meas.items():
        if n in per_grid and q["grid"]:
            g = max(q["grid"].values())
            factor[nf[n]] = max(factor.get(nf[n], 0.0), g / per_grid[n][0])
    est, done, floors = {}, [], {}
    for n in pools:
        l = mg.PROCESSES[n]["lhe"]
        E = len(l.get("energies") or [1])
        q = meas.get(n, {"grid": {}, "runs": []})
        if len(q["grid"]) >= E and len(q["runs"]) >= a.runs * E and all(x is not None for x in q["runs"]):
            done.append(n); continue
        g1, r1 = per_grid[n]
        f = factor.get(nf[n], GRID_TIME_FACTOR) if meas else GRID_TIME_FACTOR
        bad = sum(1 for x in q["runs"] if x is None)
        est[n] = (E - len(q["grid"])) * f * g1 + (E * r1 if (len(q["runs"]) < a.runs * E or bad) else 0.0) \
            + (a.timeout if bad else 0.0)
    for c in meas.get("__cut__", []):
        # pools run in the order given and their runs in parallel: the job's finished wall is each named pool's
        # gridpacks plus its longest run; the pool cut is the last named one if it is unfinished, else the next
        order = (a.order or {}).get(c["log"]) or []
        named = [n for n in order if n in meas]
        spent = sum(sum(meas[n]["grid"].values()) + max([x or 0.0 for x in meas[n]["runs"]] or [0.0]) for n in named)
        cut = c["after"] if c["after"] not in done else (order[len(named)] if len(named) < len(order) else None)
        if cut:
            floors[cut] = round((c["s_job"] - spent) / 3600, 2)
    for n, h in floors.items():
        if n in est:
            est[n] = max(est[n], h * 3600)
    jobs = [[0.0, []] for _ in range(a.jobs)]
    for n in sorted(est, key=lambda x: -est[x]):
        j = min(jobs, key=lambda x: x[0])
        j[0] += est[n]; j[1].append(n)
    jobs = [j for j in jobs if j[1]]
    print(json.dumps({"pools": len(est), "done": len(done), "factor_by_nfinal": {k: round(v, 1) for k, v in factor.items()},
                      "floor_h": floors,
                      "jobs": [{"h": round(t / 3600, 2), "pools": ps} for t, ps in jobs],
                      "est_total_job_h": round(sum(t for t, _ in jobs) / 3600, 1),
                      "est_longest_job_h": round(max(t for t, _ in jobs) / 3600, 2) if jobs else 0.0}))


def cmd_run(a):
    name = a.process
    cfg = mg.PROCESSES[name]
    base = a.work or os.environ.get("TMPDIR") or "/tmp"
    wd = os.path.join(base, f"lhe_probe_{name}_{os.getpid()}")
    os.makedirs(wd, exist_ok=True)
    log = os.path.join(wd, f"{name}.mg5.log")
    res = {"process": name, "seed": a.seed, "extra_patches": a.patch, "host": socket.gethostname(), "work": wd, "events_asked": a.events}
    t0 = time.time()
    try:
        ev, _ = mg.lhe_prepare(name, wd, log, madevent=True, standalone=False)
        res["s_output"] = round(time.time() - t0, 1)
        res["du_after_output"] = _du(ev)
        iseed = mg.lhe_iseed(name, 9999 + a.seed)              # a probe seed, not one of the pool's runs
        runs = mg.lhe_runs(name, mg.lhe_iseed(name, 42), mg.lhe_n_requested(cfg))
        patches = dict(runs[0][2])
        if a.energy:                                               # an e+e- pool's run at another of its energies
            patches.update({"ebeam1": a.energy / 2.0, "ebeam2": a.energy / 2.0})
        run_card, param_card = mg.lhe_cards(cfg)
        shutil.copy2(param_card, f"{ev}/Cards/param_card.dat")
        shutil.copy2(run_card, f"{ev}/Cards/run_card.dat")
        extra = dict(kv.split("=", 1) for kv in a.patch)              # --patch k=v: a run-card value on top
        patches.update(extra)
        mg.patch_card(f"{ev}/Cards/run_card.dat", {"nevents": int(a.events), "iseed": int(iseed), **patches})
        mg.add_card_params(f"{ev}/Cards/run_card.dat", extra)
        mg._set_me5_config(f"{ev}/Cards/me5_configuration.txt",
                           {"run_mode": 0, "nb_core": 1, "automatic_html_opening": "False"})
        res["patches"] = patches
        t0 = time.time()
        with open(log, "a") as lf:
            pr = subprocess.Popen([f"{ev}/bin/generate_events", "probe", "-f"], cwd=ev, stdout=lf,
                                  stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS="1"),
                                  start_new_session=True)
            peak = 0
            while pr.poll() is None:
                if time.time() - t0 > a.timeout:
                    os.killpg(pr.pid, signal.SIGKILL)
                    pr.wait()
                    res["timeout"] = True
                    break
                peak = max(peak, _du(ev) or 0)
                time.sleep(15)
        peak = max(peak, _du(ev) or 0)
        res["s_generate"] = round(time.time() - t0, 1)
        res["du_peak"] = peak
        lhe = f"{ev}/Events/probe/unweighted_events.lhe.gz"
        if os.path.exists(lhe):
            m, aq, _ = mg.read_lhe_pp(lhe, cfg["pdg_ids"])
            res["events_returned"] = len(m)
            if a.keep:          # the run's events and banner kept (analysis/catalog_v4/short_run_bias.py run sizes)
                bs = sorted(glob.glob(f"{ev}/Events/probe/*_banner.txt"))
                res["kept_events_file"] = _keep(a.keep, name, f"{name}_probe_seed{iseed}_ev{int(a.events)}.npz",
                                                m, aq, open(bs[0]).read() if bs else "")
            res["yield"] = len(m) / a.events
            res.update(kept(m, accept_of(name)))
            k = res.get("kept") or 0.0
            res["s_per_kept_event"] = round(res["s_generate"] / (len(m) * k), 4) if len(m) and k else None
        else:
            res["events_returned"] = 0
        txt = open(log).read()
        res["fail_to_reach_target"] = "fail to reach target" in txt
        res["second_refine_skipped"] = "No need for second refine" in txt
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    print("PROBE_JSON " + json.dumps(res), flush=True)


def _class(name):
    cfg = mg.PROCESSES[name]
    beam = "e+e-" if abs(int(cfg["pdg_ids"][0])) in (11, 13) else "pp"
    pool = name.split("_", 1)[0]
    return f"{beam} {pool} 2->{int(cfg['nfinal'])}"


def cmd_summary(a):
    import statistics as st
    rows = []
    for f in a.logs:
        for line in open(f):
            if "PROBE_JSON " in line:
                rows.append(json.loads(line.split("PROBE_JSON ", 1)[1]))
    by = {}
    for r in rows:
        key = _class(r["process"]) + ("  " + " ".join(r.get("extra_patches") or []) if r.get("extra_patches") else "")
        by.setdefault(key, []).append(r)
    med = lambda xs: round(st.median(xs), 4) if xs else None
    out = {}
    for k, rs in sorted(by.items()):
        ok = [r for r in rs if not r.get("timeout") and "error" not in r]
        out[k] = {"runs": len(rs), "timeouts": sum(bool(r.get("timeout")) for r in rs),
                  "errors": sum("error" in r for r in rs),
                  "yield_median": med([r["yield"] for r in ok if "yield" in r]),
                  "yield_min": min([r["yield"] for r in ok if "yield" in r], default=None),
                  "short_runs": sum(r.get("yield", 0) < 1 for r in ok),
                  "kept_median": med([r["kept"] for r in ok if r.get("kept") is not None]),
                  "kept_min": min([r["kept"] for r in ok if r.get("kept") is not None], default=None),
                  "s_per_kept_event_median": med([r["s_per_kept_event"] for r in ok if r.get("s_per_kept_event")]),
                  "s_generate_median": med([r["s_generate"] for r in ok if "s_generate" in r]),
                  "s_output_median": med([r["s_output"] for r in rs if "s_output" in r]),
                  "du_peak_mb_median": med([r["du_peak"] / 1e6 for r in rs if r.get("du_peak")]),
                  "du_peak_mb_max": round(max([r["du_peak"] / 1e6 for r in rs if r.get("du_peak")], default=0), 1)}
    out["_all"] = {"runs": len(rows), "processes": len({r["process"] for r in rows}),
                   "timeouts": sorted(r["process"] for r in rows if r.get("timeout")),
                   "errors": sorted(f"{r['process']}: {r['error'][:80]}" for r in rows if "error" in r)}
    print(json.dumps(out, indent=1))


def cmd_kept(a):
    """{process: {kept, by_energy: {E: kept}}} of the default-share (dsh_) runs in probe logs: the fraction of a run's
    events its pool keeps, per e+e- collider energy (2 x ebeam1 of the run's patches), the mean over the runs of one
    energy; read by recipes/gen_catalog_v4_default.py (accept_margin)."""
    acc, ys, by_e, rt = {}, {}, {}, {}
    for f in a.logs:
        for line in open(f):
            if "PROBE_JSON " not in line:
                continue
            r = json.loads(line.split("PROBE_JSON ", 1)[1])
            # the yield at SMALL_RUN events per run (recipes/gen_catalog_v4_default.py reads it as that); a run that
            # returned nothing is left out (read as no measurement)
            # a gridpack pool's runs (gridpool): its yield at its own run size, every energy
            if r.get("yield") and r.get("gridpack") and not r.get("error"):
                ys.setdefault(r["process"], []).append(float(r["yield"]))
            # a gridpack pool's runs per collider energy (2 x ebeam1; "all" for pp): yield and the share of runs that
            # returned all they asked (the assembly keeps only those); a run that errored counts as returning nothing
            # the wall of each run over the events it asked (one core per run): a timed-out run counts its timeout,
            # a lower bound of its cost
            if r.get("gridpack") and int(r.get("events_asked") or 0) > 0:
                s_run = r.get("s_generate")
                if r.get("error") and "timed out after" in str(r["error"]):
                    s_run = float(re.search(r"timed out after ([0-9.]+)", str(r["error"])).group(1))
                if s_run:
                    rt.setdefault(r["process"], []).append((float(s_run), int(r["events_asked"]), bool(r.get("error"))))
            if r.get("gridpack"):
                pe = r.get("patches") or {}
                E = f"{2.0 * float(pe['ebeam1']):g}" if "ebeam1" in pe else "all"
                got = 0 if r.get("error") else int(r.get("events_returned") or 0)
                asked = int(r.get("events_asked") or 0)
                be = by_e.setdefault(r["process"], {}).setdefault(E, [0, 0, 0, 0])
                be[0] += got; be[1] += asked; be[2] += 1; be[3] += int(asked > 0 and got >= asked)
            elif (r.get("yield") and int(r.get("events_asked", 0)) == a.small_run and not r.get("extra_patches")
                    and not r.get("timeout") and "ebeam1" not in (r.get("patches") or {})):
                ys.setdefault(r["process"], []).append(float(r["yield"]))
            if not r["process"].startswith("dsh_") or r.get("kept") is None or r.get("extra_patches"):
                continue
            E = 2.0 * float(r["patches"]["ebeam1"]) if "ebeam1" in (r.get("patches") or {}) else None
            acc.setdefault(r["process"], {}).setdefault(E, []).append(float(r["kept"]))
    out = {}
    for n, by in sorted(acc.items()):
        e = {f"{E:g}": round(sum(v) / len(v), 4) for E, v in by.items() if E is not None}
        allk = [x for v in by.values() for x in v]
        out[n] = {"kept": round(sum(e.values()) / len(e), 4) if e else round(sum(allk) / len(allk), 4),
                  **({"by_energy": e} if e else {})}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    # per pool and energy: yield (events returned over asked, errored runs as 0) and complete share
    energy = {n: {E: {"yield": round(v[0] / v[1], 4) if v[1] else 0.0, "complete": round(v[3] / v[2], 4), "runs": v[2]}
                  for E, v in sorted(by.items())} for n, by in sorted(by_e.items())}
    if a.yields_out:          # the events a probe run returned over those it asked, per process (mean over its runs);
        # a gridpack pool's yield is its worst energy's, so the plan's check and the run sizing see the energy that
        # falls short, not the pool's average (ee4_epem_cgsxwm: 0.90 on average, 0.52 at 3000 GeV)
        y = {n: round(sum(v) / len(v), 4) for n, v in sorted(ys.items())}
        y.update({n: min(e["yield"] for e in by.values()) for n, by in energy.items()})
        with open(a.yields_out, "w") as f:
            json.dump(dict(sorted(y.items())), f, indent=1)
    if getattr(a, "rates_out", None):
        # lhe_parts plan --rates: each probed gridpack pool's slowest run, seconds per event asked on one core
        # (core_s_per_event, the rates file's unit), so a task's estimate is the wall of its pool's slowest probe run
        rates = {}
        for n, v in sorted(rt.items()):
            s_ev = max(s / k for s, k, _ in v)
            rates[n] = {"core_s_per_event": round(s_ev, 5), "mg_cores": 1, "nfinal": int(mg.PROCESSES[n]["nfinal"]),
                        "runs": len(v), "runs_timed_out": sum(e for _, _, e in v),
                        "s_run_max": round(max(s for s, _, _ in v), 1), "source": "probe logs (lhe_probe kept)"}
        with open(a.rates_out, "w") as f:
            json.dump(rates, f, indent=1)
    if getattr(a, "by_energy_out", None):
        with open(a.by_energy_out, "w") as f:
            json.dump(energy, f, indent=1)
    low = {n: v["kept"] for n, v in out.items() if v["kept"] < 1.0 / 1.1}
    print(json.dumps({"processes": len(out), "kept_below_1_over_1.1": low}))


def cmd_mixcheck(a):
    """Every assembled default-share pool of the queues read as the arms' mixture reads it (default_share_momenta, the
    largest arm's rows of each role, the class's window and cuts): ok, or the rows it is short. One JSON object."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gdef", os.path.join(ROOT, "recipes", "gen_catalog_v4_default.py"))
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    out = {}
    for q in _queues(a.queue):
        D = lp._dirs(q)
        for name in sorted(os.listdir(D["assembled"])):
            if not name.startswith("dsh_"):
                continue
            acc = accept_of(name)
            for role, k in (("train", "n_train"), ("val", "n_val"), ("test", "n_test")):
                need = int(round(g.F_MAX * g.SIZES[k]))
                try:
                    mg.default_share_momenta(name, role, g.SEED, (0, need), accept=acc)
                except Exception as e:
                    out.setdefault(name, {})[role] = str(e)[:240]
    print("MIXCHECK_JSON " + json.dumps({"failing": out, "n_failing": len(out)}), flush=True)


def _sqrt_shat(P):
    q = P[:, 0] + P[:, 1]
    return np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 0.0, None))


def _runsize_feats():
    """The run-size test's features of momenta P (n, k, 4), lab frame as generated."""
    return {"sqrt_shat": _sqrt_shat,
            "max_abs_eta": lambda P: np.abs(np.arcsinh(P[:, 2:, 3] / np.clip(np.hypot(P[:, 2:, 1], P[:, 2:, 2]),
                                                                            1e-9, None))).max(1),
            "max_pt": lambda P: np.hypot(P[:, 2:, 1], P[:, 2:, 2]).max(1)}


def _pool_row_runs(name, path):
    """(sidecar, run index of each row of the pool at `path`, runs asked, runs got) of a serially built or assembled
    pp pool: _lhe_finish concatenates its runs in order (events_per_run) and shuffles with SeedSequence([seed, iseed]),
    so a row's run is recoverable. Rows' runs None when the sidecar lacks what that needs."""
    side = json.load(open(path + ".recipe.json"))
    lr, lh = side.get("lhe_run") or {}, side.get("lhe") or {}
    per = [int(x) for x in lr.get("events_per_run") or []]
    asked = None
    if lh.get("iseed") is not None and lh.get("n_requested"):
        runs = mg.lhe_runs(name, int(lh["iseed"]), int(lh["n_requested"]))
        by_seed = {int(r[0]): int(r[1]) for r in runs}
        asked = [by_seed.get(int(i)) for i in lr.get("iseeds") or []]
    n_lhe = int(lr.get("n_lhe_events") or 0)
    blk = lh.get("block")
    if not per or n_lhe != sum(per) or not blk or side.get("seed") is None:
        return side, None, asked, per
    order = np.random.default_rng(np.random.SeedSequence([int(side["seed"]), int(lh["iseed"])])).permutation(n_lhe)
    run_of = np.repeat(np.arange(len(per)), per)
    return side, run_of[order[int(blk[0]):int(blk[1])]], asked, per


def cmd_runlevel(a):
    """The run-size pass's differences with the run as the unit (catalog_coverage, 2026-10-30): per default-share pp
    pool the runsize pass compares, (1) its reuse pool's runs from the sidecar (asked against returned: short, deep
    under 0.3 of the ask); (2) the reconstruction of each pool row's run (_pool_row_runs) checked on the dsh pool's own
    train block against its parts (share of rows found in the part of their reconstructed run); (3) per feature,
    within-run correlation on each side (Kruskal-Wallis across runs: p small = runs differ beyond their events'
    scatter, so an event-level test overstates significance); (4) the run-level permutation test of the dsh runs
    against the reuse pool's runs (short_run_bias.run_perm, binned KS); (5) with short reuse runs, the event KS of the
    dsh events against the reuse pool's complete-run rows only."""
    from scipy.stats import ks_2samp, kruskal
    sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_v4"))
    from short_run_bias import run_perm
    rx = re.compile(a.match) if a.match else None
    feats = _runsize_feats()
    out = {}
    for q in _queues(a.queue):
        D = lp._dirs(q)
        for sf in sorted(glob.glob(os.path.join(D["specs"], "dsh_*.json"))):
            name = os.path.basename(sf)[:-5]
            if rx and not rx.search(name):
                continue
            cfg = mg.PROCESSES[name]
            per_ask = int(cfg["lhe"].get("run_events") or 0)
            if not per_ask or per_ask > a.max_run or cfg["lhe"].get("energies") or name in out:
                continue
            units, ids = [], []
            for p, r in mg.lhe_part_paths(name, lp._spec(D, name)):
                try:
                    with np.load(p) as z:
                        units.append(np.asarray(z["mom"], dtype=np.float64)); ids.append(int(r[0]))
                except (FileNotFoundError, OSError, KeyError, ValueError):
                    pass
            if not units or sum(len(u) for u in units) < a.min_events:
                continue
            reuse = cfg.get("reuse")
            pools = mg.default_share_pools(name, "train", 42)
            ref = [(nm, path) for nm, _, path in pools if nm == reuse and os.path.exists(path)]
            if not ref:
                continue
            r = {"reuse": reuse, "dsh_runs": len(units), "dsh_events": int(sum(len(u) for u in units)),
                 "dsh_runs_short": int(sum(len(u) < per_ask for u in units))}
            # (2) the row-run reconstruction, checked on the dsh pool's own train block against its parts
            own = [path for nm, _, path in pools if nm == name and os.path.exists(path)]
            if own:
                try:
                    side, rr, _, per = _pool_row_runs(name, own[0])
                    used = [int(i) for i in (side.get("lhe_run") or {}).get("iseeds") or []]
                    if rr is not None and len(used) == len(per):
                        x = np.load(own[0], mmap_mode="r")
                        k = int(cfg["nfinal"]) + 2
                        sel = np.random.default_rng(0).choice(len(x), min(500, len(x)), replace=False)
                        key = lambda m: tuple(np.round(np.asarray(m[:8], dtype=np.float64), 3))
                        sets = {i: {key(m.reshape(-1)) for m in u} for i, u in zip(ids, units)}
                        hit = sum(key(np.asarray(x[j, :4 * k], dtype=np.float64)) in sets.get(used[rr[j]], ())
                                  for j in sel)
                        r["recon_check"] = {"rows": int(len(sel)), "in_their_run": round(hit / len(sel), 4)}
                except (OSError, ValueError, KeyError) as e:
                    r["recon_check"] = {"error": str(e)[:200]}
            nm, path = ref[0]
            x = np.load(path, mmap_mode="r")
            k = int(mg.PROCESSES[nm]["nfinal"]) + 2
            B = np.asarray(x[:, :4 * k], dtype=np.float64).reshape(len(x), k, 4)
            try:
                side, rr, asked, per = _pool_row_runs(nm, path)
            except (OSError, ValueError, KeyError) as e:
                side, rr, asked, per = {}, None, None, []
                r["reuse_sidecar_error"] = str(e)[:200]
            short = [i for i, (g, s) in enumerate(zip(per, asked or [])) if s is not None and g < s]
            r["reuse_sidecar"] = {"runs": len(per), "events_per_run": per, "asked_per_run": asked,
                                  "runs_short": len(short),
                                  "runs_deep": sum(1 for i in short if per[i] < 0.3 * asked[i]),
                                  "rows_known_run": rr is not None and len(rr) == len(B)}
            if rr is not None and len(rr) == len(B):
                r["reuse_runs_in_block"] = int(len(np.unique(rr)))
                if short:
                    r["reuse_rows_from_short_runs"] = int(np.isin(rr, short).sum())
            for f, fn in feats.items():
                fa = [fn(u) for u in units]
                fb = fn(B)
                t = {"ks_event": round(float(ks_2samp(np.concatenate(fa), fb).statistic), 4),
                     "p_event": float(ks_2samp(np.concatenate(fa), fb).pvalue),
                     "kw_p_dsh": float(kruskal(*fa).pvalue) if len(fa) > 1 else None}
                if rr is not None and len(rr) == len(B):
                    ub = [fb[rr == i] for i in np.unique(rr) if (rr == i).sum() > 0]
                    t["kw_p_reuse"] = float(kruskal(*ub).pvalue) if len(ub) > 1 else None
                    rp = run_perm(fa, ub, nperm=a.nperm)
                    if rp:
                        t["run_level"] = rp
                    if short:
                        c = fb[~np.isin(rr, short)]
                        if len(c) >= 1000:
                            s2 = ks_2samp(np.concatenate(fa), c)
                            t["vs_reuse_complete_rows"] = {"n": int(len(c)), "ks": round(float(s2.statistic), 4),
                                                           "p": float(s2.pvalue)}
                if a.null and len(fa) >= 2 * a.null:
                    # calibration on a known null: the dsh pool's first --null runs against its others
                    x0, x1 = fa[:a.null], fa[a.null:]
                    rp0 = run_perm(x0, x1, nperm=a.nperm)
                    t["null_dsh_split"] = {"p_event": float(ks_2samp(np.concatenate(x0), np.concatenate(x1)).pvalue),
                                           "p_runs": rp0["p_runs"] if rp0 else None}
                r[f] = t
            out[name] = r
            print(f"[runlevel] {name}: " + json.dumps(r, separators=(",", ":"))[:600], flush=True)
    print("RUNLEVEL_JSON " + json.dumps(out), flush=True)


def cmd_runlevel_summary(a):
    """One JSON line from runlevel's RUNLEVEL_JSON (saved as a file): per reference kind (pp13v4 reuse pool, same
    generation as the dsh pool; or a catalog_v3 pp13 pool, other model, QED order and cards), the pools, the
    reconstruction check, the reuse runs short, the event-level tests under 0.05 (and expected), the dsh side's
    Kruskal-Wallis across runs under 0.05, and every run-level test (reuse pools of 2+ runs) with its p."""
    d = json.load(open(a.json))
    F = ("sqrt_shat", "max_abs_eta", "max_pt")
    out = {}
    for kind, sel in (("pp13v4", lambda r: r["reuse"].startswith("pp13v4_")),
                      ("catalog_v3_pp13", lambda r: not r["reuse"].startswith("pp13v4_"))):
        g = {n: r for n, r in d.items() if sel(r)}
        pe = [r[f]["p_event"] for r in g.values() for f in F]
        kd = [r[f]["kw_p_dsh"] for r in g.values() for f in F if r[f].get("kw_p_dsh") is not None]
        kr = [r[f]["kw_p_reuse"] for r in g.values() for f in F if r[f].get("kw_p_reuse") is not None]
        runs = {}
        for n, r in g.items():
            k = r["reuse_sidecar"]["runs"]
            runs[k] = runs.get(k, 0) + 1
        out[kind] = {"pools": len(g),
                     "recon_min": min((r.get("recon_check") or {}).get("in_their_run", 0) for r in g.values()),
                     "reuse_pools_by_runs": runs,
                     "reuse_runs_short": sum(r["reuse_sidecar"]["runs_short"] for r in g.values()),
                     "event_tests": len(pe), "event_p_lt_05": sum(p < 0.05 for p in pe),
                     "event_expected": round(0.05 * len(pe), 2), "event_p_min": min(pe),
                     "event_bonferroni": round(0.05 / len(pe), 6),
                     "kw_dsh_tests": len(kd), "kw_dsh_lt_05": sum(p < 0.05 for p in kd),
                     "kw_reuse_tests": len(kr), "kw_reuse_lt_05": sum(p < 0.05 for p in kr),
                     "run_level": {n: {f: [r[f]["run_level"]["p_runs"], r[f]["p_event"]] for f in F
                                       if r[f].get("run_level")} for n, r in g.items()
                                   if any(r[f].get("run_level") for f in F)},
                     "reuse_events_per_run": {n: r["reuse_sidecar"]["events_per_run"] for n, r in g.items()
                                              if r["reuse_sidecar"]["runs"]}}
    print("RUNLEVEL_SUMMARY " + json.dumps(out), flush=True)


def cmd_runsize(a):
    """Whether MadEvent runs of few events give the events runs of many give: for each default-share pp pool whose runs
    ask at most --max-run events and whose parts hold at least --min-events, its parts' events against its reuse pool's
    (runs of 10k+ events): two-sample Kolmogorov-Smirnov on sqrt(s_hat), on the largest |eta| of the finals and on the
    largest final pT, all in the lab frame as generated. same_physics says whether the reuse pool has the same generate
    line, model and cards (a pp13v4 pool does; a reused catalog_v3 pp13 pool has the minimal QED order, the sm model and
    catalog_v3's cards, so a difference there is not the run size's). Neither side is known to be the unbiased one: the
    large runs are the ones that stop short of their target."""
    from scipy.stats import ks_2samp
    import datagen
    out = {}
    for q in _queues(a.queue):
        D = lp._dirs(q)
        for sf in sorted(glob.glob(os.path.join(D["specs"], "dsh_*.json"))):
            name = os.path.basename(sf)[:-5]
            cfg = mg.PROCESSES[name]
            per = int(cfg["lhe"].get("run_events") or 0)
            if not per or per > a.max_run or cfg["lhe"].get("energies"):
                continue
            ms = []
            for p, _ in mg.lhe_part_paths(name, lp._spec(D, name)):
                try:
                    with np.load(p) as z:
                        ms.append(np.asarray(z["mom"]))
                except (FileNotFoundError, OSError, KeyError, ValueError):
                    pass
            if not ms or sum(len(m) for m in ms) < a.min_events:
                continue
            reuse = cfg.get("reuse")
            refs = []
            for nm, rec, path in mg.default_share_pools(name, "train", 42)[:1]:
                if nm == reuse and os.path.exists(path):
                    x = np.load(path, mmap_mode="r")
                    k = int(mg.PROCESSES[nm]["nfinal"]) + 2
                    refs.append(np.asarray(x[:, :4 * k], dtype=np.float64).reshape(len(x), k, 4))
            if not refs:
                continue
            A, B = np.concatenate(ms), refs[0]
            feat = _runsize_feats()
            rc = mg.PROCESSES[reuse]
            same = (list(cfg["mg5_generate"]) == list(rc["mg5_generate"])
                    and cfg["lhe"].get("model") == rc["lhe"].get("model")
                    and [mg._file_sha256(c) for c in mg.lhe_cards(cfg)] == [mg._file_sha256(c) for c in mg.lhe_cards(rc)])
            r = {"run_events": per, "n_small": len(A), "reuse": reuse, "n_reuse": len(B), "same_physics": same}
            for f, fn in feat.items():
                t = ks_2samp(fn(A), fn(B))
                r[f] = {"ks": round(float(t.statistic), 4), "p": float(t.pvalue)}
            out[name] = r
    print("RUNSIZE_JSON " + json.dumps(out), flush=True)


def cmd_node(a):
    out = {"host": socket.gethostname(), "nproc": os.cpu_count(), "TMPDIR": os.environ.get("TMPDIR")}
    for k, p in (("tmpdir", os.environ.get("TMPDIR")), ("tmp", "/tmp"), ("scratch", os.environ.get("SCRATCH"))):
        if p and os.path.isdir(p):
            st = os.statvfs(p)
            out[k] = {"path": p, "free_gb": round(st.f_bavail * st.f_frsize / 1e9, 1),
                      "size_gb": round(st.f_blocks * st.f_frsize / 1e9, 1),
                      "df": subprocess.run(["df", "-hT", p], capture_output=True, text=True).stdout.strip()}
    try:
        out["mem_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError):
        pass
    print("NODE_JSON " + json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list"); s.add_argument("--queue", nargs="+", required=True)
    s = sub.add_parser("parts"); s.add_argument("--queue", nargs="+", required=True); s.add_argument("--out", required=True)
    s.add_argument("--match", default=None, help="only the processes whose name matches this regex")
    s = sub.add_parser("run"); s.add_argument("--process", required=True); s.add_argument("--events", type=int, default=1000)
    s.add_argument("--timeout", type=float, default=720.0); s.add_argument("--work", default=None)
    s.add_argument("--energy", type=float, default=None, help="e+e-: the collider energy (default: the first run's)")
    s.add_argument("--seed", type=int, default=0, help="probe seed offset (iseed of seed 9999 + this)")
    s.add_argument("--patch", nargs="*", default=[], help="k=v run-card values (a hidden one is added to the card)")
    s.add_argument("--keep", default=None, help="save the run's events and banner under KEEP/<process>/")
    s = sub.add_parser("gridpack"); s.add_argument("--process", required=True)
    s.add_argument("--grid-events", type=int, required=True, help="events of the run whose precision the grid has")
    s.add_argument("--runs", type=int, required=True); s.add_argument("--events", type=int, default=1000)
    s.add_argument("--seed-base", type=int, default=0); s.add_argument("--cores", type=int, default=1)
    s.add_argument("--energy", type=float, default=None); s.add_argument("--keep", default=None)
    s.add_argument("--work", default=None); s.add_argument("--timeout", type=float, default=1800.0)
    s = sub.add_parser("gridpool"); s.add_argument("--process", nargs="+", required=True)
    s.add_argument("--runs", type=int, default=16, help="probe runs per gridpack at the pool's run size")
    s.add_argument("--cores", type=int, default=1); s.add_argument("--work", default=None)
    s.add_argument("--timeout", type=float, default=1800.0)
    s.add_argument("--events", type=int, default=None, help="run size (default: the pool's lhe.run_events)")
    s.add_argument("--energy", type=float, default=None, help="e+e-: only the gridpack of this collider energy")
    s.add_argument("--seed-base", type=int, default=0, help="run seeds this+1, this+2, .. (default: hashed probe seeds)")
    s.add_argument("--keep", default=None, help="save each run's events and banner under KEEP/<process>/")
    s.add_argument("--keep-logs", default=None, help="save each run's logs under KEEP_LOGS/<process>/ (_keep_logs)")
    s.add_argument("--ximprove", nargs="*", default=None,
                   help="key=value: the gridpack event-generation limits of each run (mg.lhe_patch_ximprove)")
    s = sub.add_parser("gridpool-plan"); s.add_argument("--jobs", type=int, required=True)
    s.add_argument("--cores", type=int, default=16); s.add_argument("--runs", type=int, default=16)
    s.add_argument("--complete", default=os.path.join(ROOT, "analysis/catalog_v4/data/complete_runs.json"))
    s.add_argument("--rates", default=os.path.join(ROOT, "analysis/catalog_v4/data/lhe_rates.json"))
    s.add_argument("--logs", nargs="*", default=None, help="gridpool logs of an earlier pass (_gridpool_measured)")
    s.add_argument("--order", type=lambda f: json.load(open(f)), default=None,
                   help="JSON {log path: [pools in the order that job ran them]}: names the pool a time limit cut")
    s.add_argument("--timeout", type=float, default=1800.0, help="a probe run's timeout (gridpool --timeout)")
    sub.add_parser("node")
    s = sub.add_parser("mixcheck"); s.add_argument("--queue", nargs="+", required=True)
    s = sub.add_parser("runsize"); s.add_argument("--queue", nargs="+", required=True)
    s.add_argument("--max-run", type=int, default=2000); s.add_argument("--min-events", type=int, default=5000)
    s = sub.add_parser("runlevel"); s.add_argument("--queue", nargs="+", required=True)
    s.add_argument("--max-run", type=int, default=2000); s.add_argument("--min-events", type=int, default=5000)
    s.add_argument("--match", default=None); s.add_argument("--nperm", type=int, default=2000)
    s.add_argument("--null", type=int, default=0, help="calibrate: the dsh pool's first N runs against its others")
    s = sub.add_parser("runlevel-summary"); s.add_argument("--json", required=True)
    s = sub.add_parser("summary"); s.add_argument("logs", nargs="+")
    s = sub.add_parser("kept"); s.add_argument("logs", nargs="+"); s.add_argument("--out", required=True)
    s.add_argument("--yields-out", default=None, help="also each process's probe yield (returned over asked)")
    s.add_argument("--small-run", type=int, default=1000, help="the run size the yields are kept for")
    s.add_argument("--by-energy-out", default=None, help="a gridpack pool's yield and complete-run share per energy")
    s.add_argument("--rates-out", default=None, help="a gridpack pool's slowest probe run as core seconds per event "
                   "(lhe_parts plan --rates)")
    a = ap.parse_args()
    {"list": cmd_list, "parts": cmd_parts, "run": cmd_run, "node": cmd_node, "summary": cmd_summary, "kept": cmd_kept, "mixcheck": cmd_mixcheck, "runsize": cmd_runsize, "runlevel": cmd_runlevel, "runlevel-summary": cmd_runlevel_summary, "gridpack": cmd_gridpack, "gridpool": cmd_gridpool, "gridpool-plan": cmd_gridpool_plan}[a.cmd](a)


if __name__ == "__main__":
    main()
