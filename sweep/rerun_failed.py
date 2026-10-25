"""Rerun the trials of single-fidelity sweeps that failed for a reason unrelated to their HPs (a full disk, a code bug
fixed since). A failed trial leaves an imputed penalty and a diverged mark in the DyHPO state and no result, so the sweep
ends a trial short and its surrogate believes the point is bad. Per sweep: find the failed trials whose error log
contains MATCH, take each one's candidate from its result path (hp<idx>_t<T>_), retract it from the state
(DyHPOSampler.retract) and add one fixed trial at that candidate (generate_sweep --extend --fixed-hp), which the feeder
(sweep/feed_capped.py) then submits. A candidate with a result, or with a fixed trial already written, is skipped, so a
second pass changes nothing. Runs on the site that holds the sweeps (through `site run`).

    python sweep/rerun_failed.py --match "TEXT" SWEEP_DIR... [--dry-run]
"""
import argparse, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep.dyhpo_sampler import DyHPOSampler  # noqa: E402
from sweep.generate_sweep import run_generate  # noqa: E402


def failed_points(d, match):
    """{hp_idx: error log} of the trials in sweep dir d whose error log contains match"""
    out = {}
    for err in sorted(glob.glob(os.path.join(d, "error", "*.err"))):
        if match not in open(err, errors="replace").read():
            continue
        log = os.path.join(d, "output", os.path.basename(err)[:-4] + ".out")
        m = re.search(r"results/hp(\d+)_t\d+_", open(log, errors="replace").read()) if os.path.exists(log) else None
        if m is None:
            print(f"  {err}: no candidate found in {log}, skipped")
            continue
        out[int(m.group(1))] = err
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", nargs="+")
    ap.add_argument("--match", required=True, help="text of the error that is unrelated to the HPs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    n = 0
    for d in a.sweeps:
        d = d.rstrip("/")
        done = {int(m.group(1)) for r in glob.glob(os.path.join(d, "results", "hp*.json"))
                for m in [re.match(r"hp(\d+)_", os.path.basename(r))] if m}
        fixed = {int(m.group(1)) for f in glob.glob(os.path.join(d, "jobs", "trial_*.sh"))
                 for m in [re.search(r"--fixed-hp-idx (\d+)", open(f).read())] if m}
        todo = sorted(h for h in failed_points(d, a.match) if h not in done and h not in fixed)
        if not todo:
            continue
        print(f"{os.path.basename(d)}: rerun hp {todo}")
        n += len(todo)
        if a.dry_run:
            continue
        with DyHPOSampler.locked(os.path.join(d, "dyhpo_state.pkl"), d) as s:
            for h in todo:
                print(f"  retract hp_{h:04d}: {'done' if s.retract(h) else 'nothing recorded'}")
        run_generate(os.path.join(d, "sweep_config.yaml"), extend=True, fixed_hp=todo, submit=False)
    print(f"RERUN {n} trials{' (dry run)' if a.dry_run else ''}")


if __name__ == "__main__":
    main()
