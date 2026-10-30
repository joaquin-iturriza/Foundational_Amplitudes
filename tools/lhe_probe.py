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
    kept LOG... --out F.json [--yields-out Y.json]   the default-share runs' kept fractions per process and e+e-
                                            energy; with --yields-out each probed process's yield too
    mixcheck --queue Q...                   every assembled default-share pool read as the arms' mixture reads it
    runsize --queue Q... [--max-run N] [--min-events M]   default-share pp parts from runs of at most N events
                                            against the reuse pool's events (runs of 10k+), KS on sqrt(s_hat), max |eta|,
                                            max pT
    node                                    the node's disks (TMPDIR, /tmp, SCRATCH), cores and memory, one JSON line
"""
import argparse, glob, json, os, re, shutil, signal, socket, subprocess, sys, time

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
    d = os.path.join(keep, name)
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


def cmd_gridpool(a):
    """The generation rule's probe of lhe.gridpack pools (catalog_coverage step 7): for each --process, each of its
    gridpacks built as the queue builds it (mg.lhe_build_gridpack, at its path: the queue then reuses it), then --runs
    runs of the pool's run size drawn from it at probe seeds (lhe_iseed(process, 9999 + s), not the pool's), --cores at
    a time. One PROBE_JSON line per run, read by `summary` and `kept --yields-out` like a MadEvent probe run's (yield,
    kept fraction, seconds per kept event; "gridpack": true, the energy in its patches); one GRIDPACK_JSON line per
    gridpack (stage grid: seconds, bytes)."""
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
        run_ev = int(cfg["lhe"]["run_events"])
        try:
            for j, (tar, g) in enumerate(mg.lhe_gridpack_paths(name, items)):
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
                    seed = mg.lhe_iseed(name, 9999 + s)
                    r = {"process": name, "seed": seed, "events_asked": run_ev, "patches": g[2], "gridpack": True,
                         "grid": g[3], "host": socket.gethostname()}
                    rw = os.path.join(wd, f"run_{g[3]}_{s}")
                    t1 = time.time()
                    try:
                        lhe = mg.lhe_run_gridpack(tar, rw, run_ev, seed, os.path.join(wd, f"run_{g[3]}_{s}.log"),
                                                  timeout=a.timeout)
                        m, _, _ = mg.read_lhe_pp(lhe, cfg["pdg_ids"])
                        r["s_generate"] = round(time.time() - t1, 1)
                        r["events_returned"] = len(m)
                        r["yield"] = len(m) / run_ev
                        r.update(kept(m, accept_of(name)))
                        k = r.get("kept") or 0.0
                        r["s_per_kept_event"] = round(r["s_generate"] / (len(m) * k), 4) if len(m) and k else None
                    except Exception as e:
                        r["error"] = f"{type(e).__name__}: {e}"
                    finally:
                        shutil.rmtree(rw, ignore_errors=True)
                    print("PROBE_JSON " + json.dumps(r), flush=True)
                    return r
                with ThreadPoolExecutor(max(1, a.cores)) as ex:
                    list(ex.map(one, range(1, a.runs + 1)))
        finally:
            shutil.rmtree(wd, ignore_errors=True)


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
    acc, ys = {}, {}
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
    if a.yields_out:          # the events a probe run returned over those it asked, per process (mean over its runs)
        with open(a.yields_out, "w") as f:
            json.dump({n: round(sum(v) / len(v), 4) for n, v in sorted(ys.items())}, f, indent=1)
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
            feat = {"sqrt_shat": _sqrt_shat,
                    "max_abs_eta": lambda P: np.abs(np.arcsinh(P[:, 2:, 3] / np.clip(np.hypot(P[:, 2:, 1], P[:, 2:, 2]),
                                                                                    1e-9, None))).max(1),
                    "max_pt": lambda P: np.hypot(P[:, 2:, 1], P[:, 2:, 2]).max(1)}
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
    sub.add_parser("node")
    s = sub.add_parser("mixcheck"); s.add_argument("--queue", nargs="+", required=True)
    s = sub.add_parser("runsize"); s.add_argument("--queue", nargs="+", required=True)
    s.add_argument("--max-run", type=int, default=2000); s.add_argument("--min-events", type=int, default=5000)
    s = sub.add_parser("summary"); s.add_argument("logs", nargs="+")
    s = sub.add_parser("kept"); s.add_argument("logs", nargs="+"); s.add_argument("--out", required=True)
    s.add_argument("--yields-out", default=None, help="also each process's probe yield (returned over asked)")
    s.add_argument("--small-run", type=int, default=1000, help="the run size the yields are kept for")
    a = ap.parse_args()
    {"list": cmd_list, "parts": cmd_parts, "run": cmd_run, "node": cmd_node, "summary": cmd_summary, "kept": cmd_kept, "mixcheck": cmd_mixcheck, "runsize": cmd_runsize, "gridpack": cmd_gridpack, "gridpool": cmd_gridpool}[a.cmd](a)


if __name__ == "__main__":
    main()
