"""Free the space finished sweeps hold, keeping what they showed (the user's call of 2026-10-09: /sps/lpnhe at 98%,
half of runs/ was last models, most of the rest models of trials that lost their sweep).

Run on a site (through `site run`); without --apply it prints what it would delete. Per sweep with a run tree here
(runs/<sweep>/ and runs_from_lxplus/<sweep>/):
  1. a trial with a result at the sweep's top fidelity loses its last model (model_run<i>.pt[.gz]): it is read only to
     warm-start a higher fidelity; reported values and fine-tune parents are the best checkpoint
  2. once the sweep is finished here (nothing queued, running or unsubmitted, not HELD), every trial except its best
     (lowest val_loss at the top fidelity among this site's results) loses all its models
  4. ... and its plots (*.png, *.pdf) and prediction arrays (preds_*.npz)
A trial named as a fine-tune parent (fine_tune.pretrained_path) by any sweep config in the checkout or on this site is
never touched. config.yaml, logs, metrics, data_stats.json and the tokenizer are always kept, so a run can be
re-scored or rerun from its config. Before deleting anything in a sweep, runs/<sweep>/PRUNED.json records every trial
(hp, fidelity, val_loss, per-process losses, run dir, what was deleted and its size); one PRUNE JSON line per sweep
goes to stdout for the laptop to keep. A sweep that also ran on another site keeps its best here, so the global best
is kept wherever it ran.
  4b. --submit-logs: in a finished sweep's directory under SWEEP_DIR (on lxplus the 2 GB AFS home) the scheduler's
     bookkeeping (DAG manager output and logs, node logs, *.bak copies of the DyHPO state) is removed and trial
     stdout/stderr (*.out, *.err) gzipped in place
  5. --caches: pip's cache under $SCRATCH emptied; amp_data_cache entries untouched for --cache-days removed (both
     rebuilt on demand)

    python sweep/prune.py [PREFIX ...] [--apply] [--caches] [--submit-logs] [--cache-days 30] [--every HOURS]
--every: skip this run if the last applied one on this site was less than HOURS ago (for the supervisor's upkeep).
"""
import argparse, glob, json, os, re, shutil, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "sweep"))
import siteconf

PARENT_RE = re.compile(r"fine_tune\.pretrained_path:\s*\S*?/(runs\S*?/[^/\s]+/[^/\s]+)/models/")
RUN_ROOTS = ("runs", "runs_from_lxplus")


def _size(p):
    try:
        return os.lstat(p).st_size
    except OSError:
        return 0


def parents():
    """Run dirs (relative tails, runs/<sweep>/<trial>) any sweep config names as a fine-tune parent."""
    tails = set()
    cfgs = glob.glob(os.path.join(REPO, "sweep", "*.yaml")) + glob.glob(os.path.join(siteconf.SWEEP_DIR, "*",
                                                                                     "sweep_config*.yaml"))
    for f in cfgs:
        try:
            txt = open(f, errors="replace").read()
        except OSError:
            continue
        for m in PARENT_RE.finditer(txt):
            t = m.group(1)
            tails.add(t.split("/", 1)[1])            # <sweep>/<trial>, whichever run root it was under
    return tails


def results(sweep):
    """{hp: (t_steps, val_loss, proc_val_losses)} at each hp's highest fidelity, from this site's result files."""
    out = {}
    rd = getattr(siteconf, "RESULTS_DIR", siteconf.SWEEP_DIR)
    for f in glob.glob(os.path.join(rd, sweep, "results", "hp*.json")):
        m = re.match(r"hp(\d+)_t(\d+)_", os.path.basename(f))
        if not m:
            continue
        try:
            r = json.load(open(f))
            v = float(r["val_loss"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        hp, t = int(m.group(1)), int(m.group(2))
        if hp not in out or t > out[hp][0] or (t == out[hp][0] and v < out[hp][1]):
            out[hp] = (t, v, r.get("proc_val_losses"))
    return out


def top_fidelity(sweep):
    for f in (os.path.join(siteconf.SWEEP_DIR, sweep, "sweep_config.yaml"),
              os.path.join(REPO, "sweep", f"sweep_config_{sweep}.yaml")):
        try:
            import yaml
            c = yaml.safe_load(open(f))
            return max(c["fidelity_schedule"]["t_steps"])
        except Exception:
            continue
    return None


def finished(sweep, q):
    d = os.path.join(siteconf.SWEEP_DIR, sweep)
    if not os.path.isdir(d) or os.path.exists(os.path.join(d, "HELD")):
        return False
    run, qd = q.get(sweep, (0, 0, []))[:2]
    if run or qd:
        return False
    if os.path.exists(os.path.join(d, "MOVED_TO")):
        return True                                  # what ran here is over; the rest runs elsewhere
    try:
        from sweep_manager import DEFAULT_REGISTRY, load_registry
        e = load_registry(DEFAULT_REGISTRY)["sweeps"].get(sweep)
    except Exception:
        e = None
    if e is not None:
        sub = set(e.get("submitted_scripts", []))
        if any(f not in sub for f in glob.glob(os.path.join(d, "jobs", "trial_*.sh"))):
            return False                             # generated, not yet fed to the queue
    return True


def plan_sweep(sweep, q, protect, apply):
    tf = top_fidelity(sweep)
    res = results(sweep)
    done = {hp for hp, (t, v, _) in res.items() if tf is None or t >= tf}
    fin = finished(sweep, q)
    best = min(done, key=lambda h: res[h][1]) if done else None
    trials, freed = [], 0
    for root in RUN_ROOTS:
        for td in sorted(glob.glob(os.path.join(siteconf.PROJECT_DIR, root, sweep, "trial_*"))):
            m = re.match(r"trial_(\d+)", os.path.basename(td))
            hp = int(m.group(1)) if m else None
            rec = {"run_dir": os.path.relpath(td, siteconf.PROJECT_DIR), "hp": hp, "deleted": [], "bytes": 0}
            if hp in res:
                rec.update(t_steps=res[hp][0], val_loss=res[hp][1], proc_val_losses=res[hp][2])
            tail = "%s/%s" % (sweep, os.path.basename(td))
            if tail in protect:
                rec["kept"] = "fine-tune parent"
            else:
                kill = []
                models = glob.glob(os.path.join(td, "models", "*.pt")) + glob.glob(os.path.join(td, "models", "*.pt.gz"))
                if fin and done and hp != best:
                    kill += models                                                        # rule 2
                    kill += [os.path.join(dp, f) for dp, _, fs in os.walk(td) for f in fs
                             if f.endswith((".png", ".pdf")) or re.match(r"preds_\w+\.npz$", f)]   # rule 4
                    rec["kept"] = "config, logs, metrics, data_stats"
                elif hp in done:
                    has_best = any("_best.pt" in os.path.basename(f) for f in models)
                    if has_best:
                        kill += [f for f in models if re.match(r"model_run\d+\.pt(\.gz)?$", os.path.basename(f))]  # rule 1
                    rec["kept"] = "best checkpoint" + (" (the sweep's best)" if hp == best else "")
                else:
                    rec["kept"] = "unfinished trial: everything"
                for f in sorted(set(kill)):
                    rec["deleted"].append(os.path.relpath(f, td))
                    rec["bytes"] += _size(f)
            freed += rec["bytes"]
            trials.append(rec)
    if not trials:
        return None
    man = {"sweep": sweep, "site": siteconf.SITE, "when": time.strftime("%Y-%m-%d %H:%M"), "finished": fin,
           "top_fidelity": tf, "best_hp": best, "freed_bytes": freed, "trials": trials}
    if apply and freed:
        mdir = os.path.join(siteconf.PROJECT_DIR, "runs", sweep)
        os.makedirs(mdir, exist_ok=True)
        mf = os.path.join(mdir, "PRUNED.json")
        old = []
        if os.path.exists(mf):
            try:
                old = json.load(open(mf))
                old = old if isinstance(old, list) else [old]
            except ValueError:
                old = []
        json.dump(old + [man], open(mf, "w"), indent=1)       # written before anything is deleted
        for rec in trials:
            td = os.path.join(siteconf.PROJECT_DIR, rec["run_dir"])
            for f in rec["deleted"]:
                try:
                    os.remove(os.path.join(td, f))
                except FileNotFoundError:
                    pass
    return man


def submit_logs(sweep, apply):
    d = os.path.join(siteconf.SWEEP_DIR, sweep)
    freed = 0
    for dp, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(dp, f)
            if re.search(r"\.dag\.(dagman\.(out|log)|nodes\.log|metrics|lock)$|\.dag\.rescue\d+$|\.bak$", f) or \
                    (f.endswith(".log") and "/output" not in dp and f.startswith(("sweep_", "trial_"))):
                freed += _size(p)
                if apply:
                    os.remove(p)
            elif f.endswith((".out", ".err")) and _size(p) > 4096:
                import gzip
                n = _size(p)
                if apply:
                    with open(p, "rb") as fi, gzip.open(p + ".gz", "wb") as fo:
                        shutil.copyfileobj(fi, fo)
                    os.remove(p)
                    freed += n - _size(p + ".gz")
                else:
                    freed += int(0.85 * n)
    return freed


def caches(days, apply):
    freed = 0
    pip = os.path.join(siteconf.SCRATCH, "pip-cache")
    if os.path.isdir(pip):
        for dp, _, fs in os.walk(pip):
            freed += sum(_size(os.path.join(dp, f)) for f in fs)
        if apply:
            shutil.rmtree(pip, ignore_errors=True)
    cut = time.time() - 86400 * days
    for f in glob.glob(os.path.join(siteconf.SCRATCH, "amp_data_cache", "**", "*"), recursive=True):
        if os.path.isfile(f) and max(os.path.getmtime(f), os.path.getatime(f)) < cut:
            freed += _size(f)
            if apply:
                os.remove(f)
    return freed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="*", default=[""])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--caches", action="store_true")
    ap.add_argument("--submit-logs", action="store_true")
    ap.add_argument("--cache-days", type=float, default=30)
    ap.add_argument("--every", type=float, help="skip unless the last applied prune here is older than this (hours)")
    a = ap.parse_args()
    stamp = os.path.join(siteconf.SWEEP_DIR, ".last_prune")
    if a.every and a.apply and os.path.exists(stamp) and time.time() - os.path.getmtime(stamp) < 3600 * a.every:
        print("SUMMARY %s: last prune %.1f h ago, next after %.0f h" % (
            siteconf.SITE, (time.time() - os.path.getmtime(stamp)) / 3600, a.every))
        return
    from rebalance import _queue
    q = _queue()
    protect = parents()
    names = set()
    for root in RUN_ROOTS:
        for d in glob.glob(os.path.join(siteconf.PROJECT_DIR, root, "*")):
            n = os.path.basename(d)
            if os.path.isdir(d) and any(n.startswith(p) for p in a.prefixes):
                names.add(n)
    total, nsw = 0, 0
    for n in sorted(names):
        man = plan_sweep(n, q, protect, a.apply)
        if man and man["freed_bytes"]:
            total += man["freed_bytes"]; nsw += 1
            print("PRUNE " + json.dumps(man), flush=True)
    print("SUMMARY %s %s: %d sweeps, %.1f GB %s; %d parent trials protected" % (
        siteconf.SITE, "applied" if a.apply else "dry run", nsw, total / 1e9, "freed" if a.apply else "to free",
        len(protect)))
    if a.apply:
        open(stamp, "w").close()
    if a.submit_logs:
        sl = sum(submit_logs(n, a.apply) for n in sorted(names) if finished(n, q))
        print("SUMMARY submit logs: %.2f GB %s" % (sl / 1e9, "freed" if a.apply else "to free (estimate)"))
    if a.caches:
        c = caches(a.cache_days, a.apply)
        print("SUMMARY caches: %.1f GB %s" % (c / 1e9, "freed" if a.apply else "to free"))


if __name__ == "__main__":
    main()
