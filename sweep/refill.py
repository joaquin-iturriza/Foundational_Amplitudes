"""Give sweeps back the trials an infrastructure failure took: a full disk or quota, a failed checkpoint write, an
I/O error, a run directory left by an evicted attempt. Such a trial has no result and its failure says nothing
about its HPs, so its observation is retracted from the DyHPO state (extend_sweep.py --retract) and the sweep gets
new trials (generate_sweep.py --extend) up to its planned count. A trial that failed any other way is listed and
left alone: it may be its HPs. Only sweeps with nothing queued or running are touched, and none marked MOVED_TO.
Runs on the site that holds the sweeps; without --apply it only reports.
    python sweep/refill.py PREFIX [PREFIX ...] [--per-sweep 5] [--apply]
"""
import argparse, glob, os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "sweep"))
import siteconf  # noqa: E402
from rebalance import _queue  # noqa: E402

INFRA = [("quota/disk", r"Disk quota exceeded|Errno 122|No space left on device|Errno 28"),
         ("checkpoint write", r"PytorchStreamWriter failed|unexpected pos \d+ vs \d+"),
         ("I/O", r"Errno 5\b|Input/output error|Stale file handle"),
         ("leftover run dir", r"alredy exists|already exists\. Aborting")]


def failures(sd):
    """({hp_idx: reason} for the trials that ran and left no result, number of results)."""
    done = {int(m.group(1)) for r in glob.glob(os.path.join(siteconf.RESULTS_DIR, os.path.basename(sd), "results", "hp*_t*.json"))
            for m in [re.match(r"hp(\d+)_", os.path.basename(r))] if m}
    out = {}
    for log in glob.glob(os.path.join(sd, "output", "trial_*.out")):
        txt = open(log, errors="replace").read()
        for err_path in (re.sub(r"/output/(.*)\.out$", r"/error/\1.err", log), log[:-4] + ".err"):  # condor, slurm
            if os.path.exists(err_path):
                txt += open(err_path, errors="replace").read()
        m = re.search(r"hp_idx=(\d+)", txt)
        if not m or int(m.group(1)) in done:
            continue
        hp = int(m.group(1))
        reason = next((name for name, pat in INFRA if re.search(pat, txt)), "other")
        if out.get(hp) in (None, "other"):
            out[hp] = reason
    return out, len(done)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--per-sweep", type=int, default=5, help="trials each sweep is planned to have")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    q = _queue()
    tally, todo = {}, []
    for sd in sorted({d for p in a.prefixes for d in glob.glob(os.path.join(siteconf.SWEEP_DIR, p + "*"))}):
        name = os.path.basename(sd)
        if os.path.exists(os.path.join(sd, "MOVED_TO")) or sum(q.get(name, (0, 0, []))[:2]):
            continue
        fails, n_done = failures(sd)
        owed = a.per_sweep - n_done
        if owed <= 0:
            continue
        for r in fails.values():
            tally[r] = tally.get(r, 0) + 1
        print(f"{name}: {n_done} done, owes {owed}; failed {dict(sorted(fails.items()))}")
        todo.append((sd, sorted(hp for hp, r in fails.items() if r != "other"), owed))
    print(f"TALLY {tally}  owed {sum(t[2] for t in todo)} trials in {len(todo)} sweeps")
    if not a.apply:
        return
    for sd, infra, owed in todo:
        if infra:
            subprocess.run([sys.executable, os.path.join(REPO, "sweep", "extend_sweep.py"), sd, "--retract",
                            *map(str, infra)], check=True)
        subprocess.run([sys.executable, os.path.join(REPO, "sweep", "generate_sweep.py"), "--config",
                        os.path.join(sd, "sweep_config.yaml"), "--extend", "--n-trials", str(owed), "--submit"],
                       check=True)
        print(f"REFILLED {os.path.basename(sd)}: retracted {infra}, +{owed} trials")


if __name__ == "__main__":
    main()
