"""Move sweeps that have not started from a site whose queue has stalled to one that is running jobs.

`site pick` places a batch once, at submit time. When a site then stops starting my jobs (its fair share
dropped, a submit cap filled, others' jobs took the GPUs), its share of the batch waits there however fast
the other sites drain theirs: on 2026-10-02 rung 9's fine-tune grid sat on CC-IN2P3 at zero starts for half a
day while lxplus and Jean Zay ran everything else. This re-plans the not-yet-started part.

The unit moved is a whole sweep none of whose trials has started: a sweep's DyHPO state lives in a file on one
site's filesystem, so a sweep with results stays where it is. A move never deletes anything: the source's
queued jobs of that sweep are cancelled and its directory gets a MOVED_TO marker (sweep_manager.py then skips
it), the destination generates the sweep from the same config in git and submits it. A fine-tune's parent
checkpoint is copied site to site (`site copy`) when the destination does not have it; recipe pools are
prebuilt there by generate_sweep's CPU job as for any new sweep.

Per site, from the sweeps matching the prefixes:
  backlog  trials not finished and not running (queued, held by a dependency, or not yet submitted)
  rate     trials finished per hour over the last --window hours (result-file times): what the site is
           actually giving me, fair share and caps included
  finish   backlog / rate (a site with backlog and no finished trial in the window: never)
Sweeps go, one at a time, from the site that finishes last to the one that finishes first, while that
brings the later of the two finish times at least --margin hours earlier.

    python3 sweep/rebalance.py tp3_r9fte_ tp3_uu64fte_ [--apply] [--allow-jeanzay] [--window 6]
On a site (through `site run`, by the planner):
    python sweep/rebalance.py --inventory PREFIX ...      one INVENTORY JSON line
    python sweep/rebalance.py --release DEST SWEEP ...    cancel the sweeps' queued jobs, mark them MOVED_TO DEST
"""
import argparse, glob, json, os, re, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES = ["ccin2p3", "jeanzay", "lxplus"]
PER_SWEEP_FALLBACK = 5        # trials in a sweep whose job scripts were never written (not generated yet)


# ------------------------------------------------------------------ on a site
def _queue():
    """{sweep name: (running, queued, ids to cancel)} for my jobs in this site's scheduler."""
    import siteconf
    q = {}
    if siteconf.CLUSTER.get("scheduler") == "htcondor":
        out = subprocess.run(["condor_q", "-af", "JobBatchName", "JobStatus", "ClusterId", "JobUniverse"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            p = line.split()
            if len(p) < 4:
                continue
            name = re.sub(r"_\d{4}$", "", p[0])             # DAG batch names are <sweep>_<first trial>
            run, qd, ids = q.get(name, (0, 0, []))
            if p[3] == "7":                                  # the DAG manager: removing it removes its nodes
                ids = ids + [p[2]]
            elif p[1] == "2":
                run += 1
            else:
                qd += 1
            q[name] = (run, qd, ids)
        return q
    sys.path.insert(0, os.path.join(REPO, "sweep"))
    from sweep_manager import load_registry, DEFAULT_REGISTRY
    states = {}
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i %T"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        jid, st = line.split()[:2]
        states[jid] = st
    for name, e in load_registry(DEFAULT_REGISTRY)["sweeps"].items():
        jobs = [j for j in e["jobs"] if j in states]
        run = sum(states[j] == "RUNNING" for j in jobs)
        q[name] = (run, len(jobs) - run, [j for j in jobs if states[j] != "RUNNING"])
    return q


def _ran(sd):
    """Trial indices of the sweep that have started on this site: both schedulers write each trial's log to
    <sweep>/output/trial_XXXX* when it starts. A result file is not enough: a trial whose result write failed (the
    EOS quota, 2026-10-03) ran all the same, and moving its sweep would run it twice."""
    return {m.group(1) for f in glob.glob(os.path.join(sd, "output", "trial_*"))
            for m in [re.match(r"trial_(\d+)", os.path.basename(f))] if m}


def inventory(prefixes, scope, window_h):
    """Backlog and rate over every sweep in `scope` (all the site's work of this kind competes for its slots);
    only sweeps matching `prefixes` may move."""
    import siteconf
    q = _queue()
    sched = siteconf.CLUSTER.get("scheduler")
    reg = {}
    if sched != "htcondor":
        sys.path.insert(0, os.path.join(REPO, "sweep"))
        from sweep_manager import load_registry, DEFAULT_REGISTRY
        reg = {n: {"scripts": [os.path.basename(i.get("script", "")) for i in e["jobs"].values()]}
               for n, e in load_registry(DEFAULT_REGISTRY)["sweeps"].items()}
    now, sweeps, recent = time.time(), {}, 0
    names = sorted({os.path.basename(d) for p in scope for d in glob.glob(os.path.join(siteconf.SWEEP_DIR, p + "*"))})
    for name in names:
        sd = os.path.join(siteconf.SWEEP_DIR, name)
        res = glob.glob(os.path.join(siteconf.RESULTS_DIR, name, "results", "hp*_t*.json"))
        recent += sum(os.path.getmtime(r) > now - window_h * 3600 for r in res)
        scripts = glob.glob(os.path.join(sd, "jobs", "trial_*.sh"))
        ran = _ran(sd)
        run, qd, ids = q.get(name, (0, 0, []))
        moved = os.path.exists(os.path.join(sd, "MOVED_TO"))
        # what will still run here: the queued trials, plus the never-submitted ones of a sweep that is being fed
        # (SLURM: scripts the registry has not seen, of a sweep it knows -- a sweep generated and never submitted
        # is a leftover, not backlog; on HTCondor the live DAG submits its own nodes). A failed trial of a
        # finished sweep is not backlog either.
        if moved:
            remaining = 0
        elif sched == "htcondor":
            remaining = max(qd, len(scripts) - len(ran)) if ids else qd
        else:
            names_ = {os.path.basename(x) for x in scripts}
            if name in reg:
                unsub = len(names_ - {os.path.basename(x) for x in reg[name]["scripts"]})
            else:                           # generated but never handed to sweep_manager: nothing will run it
                unsub = 0
            remaining = qd + unsub
        sweeps[name] = {"done": len(res), "running": run, "queued": qd, "cancel": ids, "moved": moved,
                        "remaining": remaining,
                        "started": bool(res) or bool(ran) or run > 0, "movable": any(name.startswith(p) for p in prefixes)}
    print("INVENTORY " + json.dumps({"site": siteconf.SITE, "scheduler": siteconf.CLUSTER.get("scheduler"),
                                     "recent": recent, "window_h": window_h, "sweeps": sweeps}))


def release(dest, names):
    import siteconf
    q = _queue()
    for name in names:
        sd = os.path.join(siteconf.SWEEP_DIR, name)
        if (glob.glob(os.path.join(siteconf.RESULTS_DIR, name, "results", "hp*_t*.json")) or _ran(sd)
                or q.get(name, (0,))[0]):
            print(f"KEEP {name}: it has started here")
            continue
        ids = q.get(name, (0, 0, []))[2]
        if ids:
            cmd = ["condor_rm"] if siteconf.CLUSTER.get("scheduler") == "htcondor" else ["scancel"]
            subprocess.run(cmd + ids, check=True)
        with open(os.path.join(sd, "MOVED_TO"), "w") as f:
            f.write(f"{dest} {time.strftime('%Y-%m-%d %H:%M')}\n")
        print(f"RELEASED {name} ({len(ids)} queued jobs cancelled)")


# ------------------------------------------------------------------ on the laptop
def _has(site, path):
    """Whether `path` (relative to the checkout) exists on `site`. The answer is a line of its own: site run
    echoes the command first, so a substring test would always find the word."""
    out = site_run(site, "bash", "-c", f"test -e {path} && echo __HAVE__ || echo __MISSING__")
    return any(l.strip() == "__HAVE__" for l in out.splitlines())


def site_run(site, *cmd, timeout=900):
    p = subprocess.run(["timeout", str(timeout + 10), "site", "--timeout", str(timeout), "run", "--quote", site,
                        "FA", "--"] + list(cmd), capture_output=True, text=True)
    return p.stdout + p.stderr


def _earlier(new, old):
    """new < old for finish times; a stalled site (no trial finished in the window) finishes at inf, and the
    planner treats that case on its own (inf < inf is False, which kept a stalled site from ever handing off)."""
    return new < old


def plan(inv, margin_h):
    """[(sweep, src, dst)] and the per-site (backlog, rate, finish) before and after."""
    st = {}
    for s, v in inv.items():
        backlog = sum(x["remaining"] for x in v["sweeps"].values())
        rate = v["recent"] / v["window_h"]
        st[s] = {"backlog": backlog, "rate": rate}
    fin = lambda s, extra=0: (st[s]["backlog"] + extra) / st[s]["rate"] if st[s]["rate"] > 0 else (
        float("inf") if st[s]["backlog"] + extra > 0 else 0.0)
    before = {s: (st[s]["backlog"], st[s]["rate"], fin(s)) for s in st}
    movable = {s: sorted((n for n, x in v["sweeps"].items()
                          if x["movable"] and not x["started"] and not x["moved"] and x["remaining"]),
                         reverse=True) for s, v in inv.items()}
    moves = []
    while True:
        # the latest-finishing site that still has a sweep to give: a site whose backlog is all started sweeps
        # (CC with rung 9's half-run cells) stays as it is, and the next one is balanced instead
        for src in sorted(st, key=fin, reverse=True):
            if not movable[src]:
                continue
            name = movable[src][0]
            k = inv[src]["sweeps"][name]["remaining"]
            dsts = [s for s in st if s != src and name not in inv[s]["sweeps"]]   # never onto a namesake
            if not dsts:
                movable[src].pop(0)
                continue
            dst = min(dsts, key=lambda s: fin(s, k))
            if (st[src]["rate"] == 0 and fin(dst, k) < float("inf")) \
                    or _earlier(max(fin(src, -k), fin(dst, k)), fin(src)):
                break         # a stalled source (rate 0) gains from any move to a site that will finish
        else:
            break
        movable[src].pop(0)
        st[src]["backlog"] -= k
        st[dst]["backlog"] += k
        moves.append((name, src, dst))
    # one sweep is a few trials: the margin is on what a site gains from all its moves together
    for src in list(st):
        was, now = before[src][2], fin(src)
        # from a stalled source every trial moved is a gain, even if what is left (started sweeps) still never finishes
        gain = float("inf") if st[src]["rate"] == 0 else was - now
        if src in {m[1] for m in moves} and not gain >= margin_h:
            for name, s_, dst in [m for m in moves if m[1] == src]:
                k = inv[src]["sweeps"][name]["remaining"]
                st[src]["backlog"] += k; st[dst]["backlog"] -= k
            moves = [m for m in moves if m[1] != src]
    after = {s: (st[s]["backlog"], st[s]["rate"], fin(s)) for s in st}
    return moves, before, after


def _parent(name):
    """The fine-tune's parent run directory relative to the project root (runs/<exp>/<trial>), or None."""
    m = re.search(r"fine_tune\.pretrained_path:\s*\S*?/(runs/\S+)/models/",
                  open(os.path.join(REPO, "sweep", f"sweep_config_{name}.yaml")).read())
    return m.group(1) if m else None


def apply(moves, sites):
    """Per (source, destination): make sure the destination has every parent checkpoint, generate and submit the
    sweeps there and check each one is in its queue, and only then release them at the source. A sweep the source
    has started in the meantime (KEEP) is released again at the destination instead, so it runs once. Nothing is
    cancelled at the source before its replacement is queued, so a failure on the way leaves it where it was."""
    by = {}
    for name, src, dst in moves:
        by.setdefault((src, dst), []).append(name)
    for (src, dst), names in by.items():
        ok = []
        for p in sorted({p for p in map(_parent, names) if p}):
            if not _has(dst, f"{p}/models"):
                holder = next((s for s in sites if s != dst and _has(s, f"{p}/models")), None)
                if holder is None or subprocess.run(["site", "copy", "FA", holder, dst, p]).returncode != 0 \
                        or not _has(dst, f"{p}/models"):
                    print(f"  parent {p} not on {dst}: its sweeps stay on {src}")
                    names = [n for n in names if _parent(n) != p]
        if not names:
            continue
        cfgs = [f"sweep/sweep_config_{n}.yaml" for n in names]
        if inv_sched[dst] == "htcondor":
            loop = " ".join(f"python sweep/generate_sweep.py --config {c} --submit;" for c in cfgs)
            site_run(dst, "bash", "-c", loop, timeout=3600)
        else:
            gen = " ".join(f"yes n | python sweep/generate_sweep.py --config {c} >/dev/null;" for c in cfgs)
            dirs = " ".join(f"$(python -c 'import siteconf;print(siteconf.SWEEP_DIR)')/{n}" for n in names)
            site_run(dst, "bash", "-c", f"{gen} python sweep/sweep_manager.py submit --chain --capacity 30 {dirs}",
                     timeout=3600)
        out = site_run(dst, "python", "sweep/rebalance.py", "--inventory", "--scope", *names, "--", *names)
        line = next((l for l in out.splitlines() if l.startswith("INVENTORY ")), None)
        got = json.loads(line[len("INVENTORY "):])["sweeps"] if line else {}
        ok = [n for n in names if n in got and got[n]["queued"] + got[n]["running"] + got[n]["remaining"] > 0]
        for n in sorted(set(names) - set(ok)):
            print(f"  {n}: not queued on {dst} after generating it there: it stays on {src}")
        if not ok:
            continue
        out = site_run(src, "python", "sweep/rebalance.py", "--release", dst, *ok, timeout=1200)
        released = re.findall(r"^RELEASED (\S+)", out, re.M)
        kept = sorted(set(ok) - set(released))
        if kept:            # started at the source meanwhile: the copy just queued at the destination goes instead
            site_run(dst, "python", "sweep/rebalance.py", "--release", f"{src} (started there first)", *kept, timeout=1200)
        print(f"  {src} -> {dst}: moved {len(released)}, kept {len(kept)} (started at {src}), "
              f"{len(names) - len(ok)} failed at {dst} and stayed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="*")
    ap.add_argument("--inventory", action="store_true", help="(on a site) print this site's INVENTORY line")
    ap.add_argument("--release", metavar="DEST", help="(on a site) cancel and mark the named sweeps as moved")
    ap.add_argument("--apply", action="store_true", help="carry the moves out (default: print the plan)")
    ap.add_argument("--allow-jeanzay", action="store_true")
    ap.add_argument("--scope", nargs="+", default=["tp3_"],
                    help="sweep prefixes whose backlog and finished trials set each site's load and rate")
    ap.add_argument("--window", type=float, default=6.0, help="hours of finished trials that set a site's rate")
    ap.add_argument("--margin", type=float, default=1.0, help="hours a move must bring the finish forward")
    a = ap.parse_args()
    sys.path.insert(0, REPO)
    if a.inventory:
        return inventory(a.prefixes, a.scope, a.window)
    if a.release:
        return release(a.release, a.prefixes)
    sites = [s for s in SITES if a.allow_jeanzay or s != "jeanzay"]
    inv = {}
    for s in sites:
        out = site_run(s, "python", "sweep/rebalance.py", "--inventory", "--window", str(a.window),
                       "--scope", *a.scope, "--", *a.prefixes)
        line = next((l for l in out.splitlines() if l.startswith("INVENTORY ")), None)
        if line is None:
            print(f"  {s}: no inventory (unreachable or not synced), left out")
            continue
        inv[s] = json.loads(line[len("INVENTORY "):])
    global inv_sched
    inv_sched = {s: v["scheduler"] for s, v in inv.items()}
    moves, before, after = plan(inv, a.margin)
    for s in inv:
        b, r, f = before[s]; b2, _, f2 = after[s]
        print(f"  {s:9s} backlog {b:5d} -> {b2:5d} trials   rate {r:6.1f}/h   finish {f:6.1f} h -> {f2:6.1f} h")
    for name, src, dst in moves:
        print(f"  move {name}: {src} -> {dst}")
    if not moves:
        print("  nothing to move")
    elif a.apply:
        apply(moves, list(inv))


if __name__ == "__main__":
    main()
