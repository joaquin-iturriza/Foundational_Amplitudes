#!/usr/bin/env python3
"""Probe of the lhe part work still to do (catalog_coverage step 7): what a MadEvent run of each remaining process
returns, how much of it its pool keeps, what it costs in time and disk, and what disk a CPU node has.

    list   --queue Q...                     the processes of the queues not assembled, one per line
    parts  --queue Q... --out F.json        from the part files already made: per process (assembled or not) the runs
                                            with a part, their events asked and returned, MadEvent seconds and cores;
                                            for a default-share pool (dsh_) the fraction of its events the mixture
                                            keeps (mg5_pipeline_final.default_share_momenta's accept: the class's
                                            sqrt(s_hat) window and fiducial cuts in the partonic frame) and what each
                                            part of that accept removes alone
    run    --process P [--events N] [--timeout S] [--work DIR] [--patch K=V ...]
                                            one MadEvent run of P (its first run's card and patches, a probe iseed) in
                                            DIR (default $TMPDIR): seconds of the MG5 output and of generate_events,
                                            events returned, the fraction kept (as above), the run directory's largest
                                            size while it ran; one JSON line. The directory is deleted afterwards.
    node                                    the node's disks (TMPDIR, /tmp, SCRATCH), cores and memory, one JSON line
"""
import argparse, glob, json, os, shutil, signal, socket, subprocess, sys, time

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
            asked, got, secs, cores, Ps = [], [], [], [], []
            for p, run in parts:
                try:
                    with np.load(p) as z:
                        m = np.asarray(z["mom"])
                        got.append(len(m)); secs.append(float(z["seconds"][1])); cores.append(int(z["mg_cores"]))
                        asked.append(int(run[1]))
                        if len(Ps) < 20:
                            Ps.append(m)
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
            out[name] = r
    with open(a.out, "w") as f:
        json.dump(out, f, indent=0, sort_keys=True)
    print(json.dumps({"processes": len(out), "out": a.out}))


def _du(path):
    try:
        return int(subprocess.run(["du", "-sb", path], capture_output=True, text=True, timeout=60).stdout.split()[0])
    except Exception:
        return None


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
            m, _, _ = mg.read_lhe_pp(lhe, cfg["pdg_ids"])
            res["events_returned"] = len(m)
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
    s = sub.add_parser("run"); s.add_argument("--process", required=True); s.add_argument("--events", type=int, default=1000)
    s.add_argument("--timeout", type=float, default=720.0); s.add_argument("--work", default=None)
    s.add_argument("--seed", type=int, default=0, help="probe seed offset (iseed of seed 9999 + this)")
    s.add_argument("--patch", nargs="*", default=[], help="k=v run-card values (a hidden one is added to the card)")
    sub.add_parser("node")
    a = ap.parse_args()
    {"list": cmd_list, "parts": cmd_parts, "run": cmd_run, "node": cmd_node}[a.cmd](a)


if __name__ == "__main__":
    main()
