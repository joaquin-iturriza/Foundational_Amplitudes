"""Watchdog over a batch of sweeps on every site: finds what has gone wrong, so it is acted on within one check and not
hours later (2026-10-06: 75 chosen 32k points failed in seconds on every rerun, for hours, while every queue looked
quiet and the rebalancer counted them as done).

Per site (one `site run` each) it reads, for the sweeps matching PREFIX:
  running / queued   my scheduler jobs of those sweeps
  failed             trial logs written in the last --window minutes that end in "Trial FAILED", with the first error line
  results            result files of those sweeps
On the laptop it adds:
  missing            the points of --expect (JSON {sweep: [hp indices]}) with a result on no site
  stalled            a site with queued trials, none running and no new result for --stall minutes
and exits 1 with an ALERT line per problem (a waiter then wakes the agent), else prints one status line and exits 0.
A missing point is an alert only when nothing of its sweep is running or queued on any site.

    python3 sweep/watchdog.py tp3_ [--expect analysis/transfer/horizon32k_chosen.json] [--window 30]
On a site (through `site run`):
    python sweep/watchdog.py --site-report tp3_ --window 30      one WATCHDOG JSON line
"""
import argparse, glob, json, os, re, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES = ["ccin2p3", "jeanzay", "lxplus"]
STATE = os.path.join(REPO, ".claude", ".watchdog_state.json")


def site_report(prefixes, window_min):
    sys.path.insert(0, REPO)
    import siteconf
    sys.path.insert(0, os.path.join(REPO, "sweep"))
    from rebalance import _queue
    q = _queue()
    sweeps = [d for p in prefixes for d in glob.glob(os.path.join(siteconf.SWEEP_DIR, p + "*")) if os.path.isdir(d)]
    names = {os.path.basename(d) for d in sweeps}
    running = {n: q[n][0] for n in names if n in q and q[n][0]}
    queued = {n: q[n][1] for n in names if n in q and q[n][1]}
    cut = time.time() - 60 * window_min
    failed = []
    for d in sweeps:
        moved = os.path.exists(os.path.join(d, "MOVED_TO"))
        for f in glob.glob(os.path.join(d, "*", "*.err")) + glob.glob(os.path.join(d, "*", "*.out")):
            try:
                if os.path.getmtime(f) < cut:
                    continue
                txt = open(f, errors="replace").read()[-20000:]
            except OSError:
                continue
            if "Trial FAILED" not in txt:
                continue
            lines = txt.splitlines()
            err = next((l.strip() for l in reversed(lines) if re.match(r"\s*[\w.]*(Error|Exception)\b.*:", l)), "") or \
                next((l.strip() for l in reversed(lines) if re.search(r"(Killed|Aborting|CANCELLED|TIME LIMIT)", l)), "")
            hp = re.search(r"hp_(\d{4})", txt) or re.search(r"/trial_(\d{4})", txt)   # stdout or the run dir
            failed.append({"sweep": os.path.basename(d), "log": os.path.basename(f), "error": err[:200], "moved": moved,
                           "hp": int(hp.group(1)) if hp else None})
    results = {}
    rd = getattr(siteconf, "RESULTS_DIR", siteconf.SWEEP_DIR)
    for n in names:
        hp = sorted({int(m.group(1)) for f in glob.glob(os.path.join(rd, n, "results", "hp*.json"))
                     for m in [re.match(r"hp(\d+)", os.path.basename(f))] if m})
        if hp:
            results[n] = hp
    print("WATCHDOG " + json.dumps({"running": running, "queued": queued, "failed": failed, "results": results}),
          flush=True)


def ask(site, prefixes, window_min):
    cmd = ["site", "--timeout", "590", "run", site, "FA", "--", "python", "sweep/watchdog.py", "--site-report",
           *prefixes, "--window", str(window_min)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600).stdout
    except subprocess.TimeoutExpired:
        return None
    line = next((l for l in out.splitlines() if l.startswith("WATCHDOG ")), None)
    return json.loads(line[len("WATCHDOG "):]) if line else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--site-report", action="store_true")
    ap.add_argument("--window", type=int, default=30, help="minutes of trial logs read for failures")
    ap.add_argument("--expect", help="JSON {sweep: [hp indices]} of the points the batch must produce")
    ap.add_argument("--stall", type=int, default=60, help="minutes queued, nothing running, no new result: stalled")
    ap.add_argument("--sites", nargs="+", default=SITES)
    a = ap.parse_args()
    if a.site_report:
        return site_report(a.prefixes, a.window)

    rep, alerts = {}, []
    for s in a.sites:
        rep[s] = ask(s, a.prefixes, a.window)
        if rep[s] is None:
            alerts.append(f"{s}: no report (site down, or the checkout is not synced)")
    rep = {s: r for s, r in rep.items() if r}
    got = {}
    for r in rep.values():
        for n, hp in r["results"].items():
            got.setdefault(n, set()).update(hp)
    for s, r in rep.items():
        # a failed attempt at a point that has a result somewhere (a duplicate run) loses nothing
        live = [f for f in r["failed"] if not f["moved"] and f.get("hp") not in got.get(f["sweep"], set())]
        if live:
            by = {}
            for f in live:
                by.setdefault(f["error"], []).append(f["sweep"])
            for err, sw in by.items():
                alerts.append(f"{s}: {len(sw)} trial(s) FAILED in the last {a.window} min ({len(set(sw))} sweeps, e.g. "
                              f"{sw[0]}): {err or 'no error line'}")
    active = {n for r in rep.values() for n in list(r["running"]) + list(r["queued"])}
    n_missing = 0
    if a.expect:
        exp = json.load(open(os.path.join(REPO, a.expect) if not os.path.isabs(a.expect) else a.expect))
        idle = {}
        for sw, hps in exp.items():
            if not any(sw.startswith(p) for p in a.prefixes):
                continue
            miss = sorted(set(hps) - got.get(sw, set()))
            n_missing += len(miss)
            if miss and sw not in active:
                idle[sw] = miss
        if idle:
            alerts.append(f"{sum(map(len, idle.values()))} expected point(s) in {len(idle)} sweep(s) have no result and "
                          f"nothing running or queued anywhere, e.g. {next(iter(idle))} {next(iter(idle.values()))}")
    try:
        state = json.load(open(STATE))
    except (OSError, ValueError):
        state = {}
    key = " ".join(a.prefixes)
    hist = state.get(key, {})
    now = time.time()
    for s, r in rep.items():
        n_res, n_q, n_run = sum(len(v) for v in r["results"].values()), sum(r["queued"].values()), sum(r["running"].values())
        h = hist.get(s, {})
        if n_q == 0 or n_run > 0 or n_res != h.get("n_res"):
            h = {"since": now, "n_res": n_res}             # the stall clock restarts on any sign of life
        hist[s] = h
        if n_q > 0 and now - h["since"] >= 60 * a.stall:
            alerts.append(f"{s}: {n_q} trial(s) queued, none running and no new result for "
                          f"{(now - h['since']) / 60:.0f} min")
    state[key] = hist
    json.dump(state, open(STATE, "w"))
    status = " | ".join(f"{s} run {sum(r['running'].values())} q {sum(r['queued'].values())} "
                        f"res {sum(len(v) for v in r['results'].values())}" for s, r in rep.items())
    print(time.strftime("%H:%M"), status + (f" | missing {n_missing}" if a.expect else ""))
    for al in alerts:
        print("ALERT", al)
    return 1 if alerts else 0


if __name__ == "__main__":
    sys.exit(main())
