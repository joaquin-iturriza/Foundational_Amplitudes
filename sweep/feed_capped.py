"""Top up a capped SLURM queue (CC-IN2P3: 100 jobs per user) with the trials it refused.

sweep_manager.submit_sweeps keeps what the per-user cap let in and leaves the rest unsubmitted, to be picked up by
calling it again once the queue drains; nothing called it again, so on 2026-10-06 ~1500 generated trials sat on CC
behind a queue it kept at 100. Run on the site (through `site run`), every 20 minutes from the laptop:

    python sweep/feed_capped.py tp3_ [--dry-run]

Every registered sweep matching PREFIX that is not MOVED_TO and has trial scripts the registry has not submitted is
handed to submit_sweeps (chained, interleaved by round), which skips submitted scripts and stops at the cap.
"""
import argparse, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_manager import DEFAULT_REGISTRY, load_registry, save_registry, submit_sweeps  # noqa: E402


def _fixed(script):
    m = re.search(r"--fixed-hp-idx (\d+)", open(script).read())
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    reg = load_registry(DEFAULT_REGISTRY)["sweeps"]
    todo = []
    for name, e in reg.items():
        d = e.get("dir")
        if not d or not any(name.startswith(p) for p in a.prefixes) or os.path.exists(os.path.join(d, "MOVED_TO")) \
                or os.path.exists(os.path.join(d, "HELD")):
            continue
        sub = set(e.get("submitted_scripts", []))
        new = sorted(f for f in glob.glob(os.path.join(d, "jobs", "trial_*.sh")) if f not in sub)
        if not new:
            continue
        # a fixed point is run once: a script repeating a point already submitted or done (a sweep the rebalancer
        # moved back and forth carries one pair of scripts per visit) is marked submitted and never runs
        have = {_fixed(f) for f in sub if os.path.exists(f)} | {
            int(m.group(1)) for r in glob.glob(os.path.join(d, "results", "hp*.json"))
            for m in [re.match(r"hp(\d+)_", os.path.basename(r))] if m}
        keep = {}
        for f in new:
            h = _fixed(f)
            if h is None:
                keep[f] = f
            elif h not in have:
                keep[h] = f                    # the newest script of a point wins
        stale = [f for f in new if f not in keep.values()]
        if stale and not a.dry_run:
            e.setdefault("submitted_scripts", []).extend(stale)
        print(f"  {name}: {len(keep)} to submit, {len(stale)} stale duplicates skipped", flush=True)
        if keep:
            todo.append(d)
    print(f"FEED {len(todo)} sweeps with unsubmitted trials", flush=True)
    if not a.dry_run:
        r = load_registry(DEFAULT_REGISTRY)
        r["sweeps"] = reg
        save_registry(DEFAULT_REGISTRY, r)
    if todo and not a.dry_run:
        submit_sweeps(sorted(todo), chain=True)


if __name__ == "__main__":
    main()
