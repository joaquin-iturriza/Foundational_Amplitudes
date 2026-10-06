"""Top up a capped SLURM queue (CC-IN2P3: 100 jobs per user) with the trials it refused.

sweep_manager.submit_sweeps keeps what the per-user cap let in and leaves the rest unsubmitted, to be picked up by
calling it again once the queue drains; nothing called it again, so on 2026-10-06 ~1500 generated trials sat on CC
behind a queue it kept at 100. Run on the site (through `site run`), every 20 minutes from the laptop:

    python sweep/feed_capped.py tp3_ [--dry-run]

Every registered sweep matching PREFIX that is not MOVED_TO and has trial scripts the registry has not submitted is
handed to submit_sweeps (chained, interleaved by round), which skips submitted scripts and stops at the cap.
"""
import argparse, glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_manager import DEFAULT_REGISTRY, load_registry, submit_sweeps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    reg = load_registry(DEFAULT_REGISTRY)["sweeps"]
    todo = []
    for name, e in reg.items():
        d = e.get("dir")
        if not d or not any(name.startswith(p) for p in a.prefixes) or os.path.exists(os.path.join(d, "MOVED_TO")):
            continue
        scripts = {os.path.basename(f) for f in glob.glob(os.path.join(d, "jobs", "trial_*.sh"))}
        done = {os.path.basename(s) for s in e.get("submitted_scripts", [])}
        if scripts - done:
            todo.append(d)
    print(f"FEED {len(todo)} sweeps with unsubmitted trials", flush=True)
    if todo and not a.dry_run:
        submit_sweeps(sorted(todo), chain=True)


if __name__ == "__main__":
    main()
