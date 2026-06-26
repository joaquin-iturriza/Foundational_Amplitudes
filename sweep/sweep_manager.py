#!/usr/bin/env python3
"""
sweep_manager.py  —  Cross-sweep, priority-aware SLURM submitter for DyHPO sweeps.

PROBLEM THIS SOLVES
-------------------
Each sweep generates `jobs/trial_XXXX.sh` scripts (see generate_sweep.py). All
trials of one sweep share a DyHPO surrogate, but they only inform each other when
earlier trials `observe()` before later ones `suggest()`. If you submit a whole
sweep at once and many GPUs free up together, every trial calls `suggest()`
against an empty surrogate — so the feedback is lost.

When you run SEVERAL sweeps at once you can fix this for free: submit one trial of
each sweep first, then the second of each, and so on. Across sweeps the cluster
stays full; within a sweep, trial N+1 starts strictly after trial N has had a
chance to run — restoring the DyHPO feedback loop.

HOW IT WORKS
------------
This tool submits trials interleaved across sweeps and stamps each job with a
SLURM `nice` value encoding its *round*:

    for sweep s with weight w_s, its j-th still-pending trial (by trial index)
        round = j // w_s
        nice  = round * ROUND_GAP

Equal round  -> equal nice -> jobs run together (fill free GPUs).
Higher round -> higher nice -> strictly lower priority (runs later).
weight w_s>1 -> w_s trials per round -> that sweep advances w_s times faster.

`nice` is the right lever: non-privileged users may RAISE their own jobs' nice
(lower their own priority); we never need operator rights or negative nice.

USAGE (run this ON Jean Zay — it shells out to sbatch/squeue/scontrol)
--------------------------------------------------------------------
    # Submit one or more sweeps, interleaved (weight defaults to 1):
    python sweep/sweep_manager.py submit <sweeps_dir>/sweepA <sweeps_dir>/sweepB

    # Give a sweep more trials per round (finishes faster), then re-order queue:
    python sweep/sweep_manager.py boost sweepA --weight 3

    # Re-interleave everything still pending (e.g. after adding a new sweep):
    python sweep/sweep_manager.py rebalance

    # Show per-sweep running / pending / done:
    python sweep/sweep_manager.py status

    # Cancel all of a sweep's jobs:
    python sweep/sweep_manager.py cancel sweepA

Add `--dry-run` to any command to print the sbatch/scontrol calls without running
them. Registry defaults to ~/.sweep_manager/registry.json (override --registry).
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

DEFAULT_REGISTRY = os.path.expanduser("~/.sweep_manager/registry.json")
DEFAULT_ROUND_GAP = 100_000   # priority penalty per round; bump if fairshare swings dominate


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def load_registry(path):
    if not os.path.exists(path):
        return {"round_gap": DEFAULT_ROUND_GAP, "sweeps": {}}
    with open(path) as f:
        reg = json.load(f)
    reg.setdefault("round_gap", DEFAULT_ROUND_GAP)
    reg.setdefault("sweeps", {})
    return reg


def save_registry(path, reg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def sweep_name_from_dir(d):
    return os.path.basename(os.path.normpath(d))


# --------------------------------------------------------------------------- #
# SLURM helpers
# --------------------------------------------------------------------------- #
def _run(cmd, dry_run=False, capture=True):
    """Run a command. In dry-run, print and return ''. Returns stdout (stripped)."""
    if dry_run:
        print("  [dry-run] " + " ".join(cmd))
        return ""
    res = subprocess.run(cmd, capture_output=capture, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"command failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr.strip()}"
        )
    return (res.stdout or "").strip()


def write_prebuild_script(sweep_dir, cfg):
    """Write `<sweep_dir>/prebuild.sh` for a recipe-based sweep config dict.
    Returns (script, spec, seed) or None if the sweep isn't recipe-based. This is
    the single source of truth for the prebuild job, shared by generate_sweep
    (emit at generation) and ensure_prebuild_script (emit at submission)."""
    fp = cfg.get("fixed_params", {})
    if str(fp.get("data.source", "files")) != "recipes":
        return None
    spec = fp.get("data.processes_file")
    if not spec:
        return None
    seed    = int(fp.get("data.seed", 42))
    proj    = cfg["paths"]["project_dir"]
    account = cfg["cluster"].get("account", "itg@v100")
    setup   = "\n".join(cfg["paths"].get("setup_commands", []))
    name    = cfg.get("sweep_name", os.path.basename(sweep_dir.rstrip("/")))
    script  = os.path.join(sweep_dir, "prebuild.sh")
    with open(script, "w") as f:
        f.write(f"""#!/bin/bash
# SPEC: {spec}
# SEED: {seed}
# Auto-emitted CPU prebuild for this recipe sweep — materializes datasets on the
# prepost partition (CPU billed at weight 0; no GPU hours). Self-skips cached.
#SBATCH --job-name=prebuild_{name}
#SBATCH --partition=prepost
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --output={sweep_dir}/prebuild_%j.out
#SBATCH --error={sweep_dir}/prebuild_%j.err
set -euo pipefail
{setup}
cd {proj}
python prebuild_recipes.py {spec} --seed {seed} --workers "${{SLURM_CPUS_PER_TASK:-16}}"
""")
    os.chmod(script, 0o755)
    return script, spec, seed


def ensure_prebuild_script(sweep_dir):
    """Return the sweep's prebuild.sh, emitting it from sweep_config.yaml if a
    recipe-based sweep doesn't have one yet (covers every generator at submit
    time, regardless of how the cell was created). Returns path or None."""
    script = os.path.join(sweep_dir, "prebuild.sh")
    if os.path.exists(script):
        return script
    cfg_path = os.path.join(sweep_dir, "sweep_config.yaml")
    if not os.path.exists(cfg_path):
        return None
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    res = write_prebuild_script(sweep_dir, cfg)
    return res[0] if res else None


def _prebuild_info(sweep_dir):
    """Return (script, spec, seed) for a sweep's auto-emitted prebuild, or None."""
    script = ensure_prebuild_script(sweep_dir)
    if not script or not os.path.exists(script):
        return None
    spec, seed = None, 42
    with open(script) as f:
        for line in f:
            if line.startswith("# SPEC:"):
                spec = line.split(":", 1)[1].strip()
            elif line.startswith("# SEED:"):
                seed = int(line.split(":", 1)[1].strip())
            elif not line.startswith("#") and line.strip() and "SBATCH" not in line:
                break
    return (script, spec, seed) if spec else None


def _spec_fully_cached(spec, seed):
    """True if every (process, role) dataset for `spec` is already cached."""
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if proj not in sys.path:
        sys.path.insert(0, proj)
    import yaml
    import datagen
    import mg5_pipeline_final as mg
    with open(spec) as f:
        doc = yaml.safe_load(f)
    procs = doc["processes"] if isinstance(doc, dict) else doc
    ck = {"train": "n_train", "val": "n_val", "test": "n_test"}
    for role in ("train", "val", "test"):
        dest = datagen.dest_for_role(role)
        for p in procs:
            rec = mg.variable_energy_recipe(
                p["name"], float(p["sqrts"][0]), float(p["sqrts"][1]),
                int(p[ck[role]]), role=role, seed=seed)
            if not datagen._is_cached(mg.recipe_output_path(rec, dest), mg.recipe_id(rec)):
                return False
    return True


def submit_prebuilds(sweep_dirs, dry_run=False):
    """Submit the auto-emitted prebuild for each sweep whose recipe data isn't
    fully cached, deduped by spec. Returns {sweep_name: prebuild_jobid} so the
    trials can depend on their data being ready."""
    spec_job = {}          # spec -> jobid (dedup across sweeps sharing a spec)
    sweep_dep = {}         # sweep_name -> jobid or None
    for d in sweep_dirs:
        d = os.path.abspath(d)
        name = sweep_name_from_dir(d)
        info = _prebuild_info(d)
        if not info:
            sweep_dep[name] = None
            continue
        script, spec, seed = info
        if _spec_fully_cached(spec, seed):
            print(f"  {name}: datasets already cached — no prebuild needed")
            sweep_dep[name] = None
            continue
        if spec not in spec_job:
            out = _run(["sbatch", "--parsable", script], dry_run=dry_run)
            jid = out.split(";")[0].strip() if out else f"DRYPB{len(spec_job)}"
            spec_job[spec] = jid
            print(f"  prebuild submitted for {os.path.basename(spec)}  job={jid}")
        sweep_dep[name] = spec_job[spec]
    return sweep_dep


def squeue_states():
    """Return {job_id(str): state(str)} for the current user's jobs.

    Jobs absent from squeue are finished (completed/failed/cancelled).
    """
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    out = subprocess.run(
        ["squeue", "-h", "-u", user, "-o", "%i %T"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"squeue failed: {out.stderr.strip()}")
    states = {}
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        jid, _, state = line.partition(" ")
        states[jid.strip()] = state.strip()
    return states


def discover_trial_scripts(sweep_dir):
    """Return [(trial_idx:int, abspath:str), ...] sorted by trial_idx."""
    jobs_dir = os.path.join(sweep_dir, "jobs")
    if not os.path.isdir(jobs_dir):
        raise FileNotFoundError(f"no jobs/ directory in {sweep_dir}")
    out = []
    for fn in os.listdir(jobs_dir):
        if fn.startswith("trial_") and fn.endswith(".sh"):
            try:
                idx = int(fn[len("trial_"):-len(".sh")])
            except ValueError:
                continue
            out.append((idx, os.path.join(jobs_dir, fn)))
    return sorted(out, key=lambda x: x[0])


# --------------------------------------------------------------------------- #
# Core ordering
# --------------------------------------------------------------------------- #
def assign_rounds(items_by_sweep, weights):
    """Round-robin tagging.

    items_by_sweep : {sweep: [item, ...]}  items pre-sorted by trial index
    weights        : {sweep: int}
    Returns list of (round:int, sweep:str, item) sorted by (round, sweep).
    """
    tagged = []
    for s, items in items_by_sweep.items():
        w = max(1, int(weights.get(s, 1)))
        for j, it in enumerate(items):
            tagged.append((j // w, s, it))
    tagged.sort(key=lambda x: (x[0], x[1]))
    return tagged


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def submit_sweeps(sweep_dirs, weight=None, registry=DEFAULT_REGISTRY, dry_run=False):
    """Submit one or more sweeps, interleaved across them by round.

    Importable entry point used by the generate_*.py scripts. `sweep_dirs` are
    paths to sweep directories (each containing a `jobs/` folder). Already-submitted
    scripts (tracked in the registry) are skipped, so it is safe to call repeatedly
    (e.g. after `generate_sweep.py --extend`). Ends with a global rebalance so the
    new jobs interleave with everything already pending in the queue.
    """
    reg = load_registry(registry)
    gap = reg["round_gap"]

    # 1. Register sweeps + discover not-yet-submitted scripts.
    new_items = {}          # sweep -> [(trial_idx, script), ...]
    for d in sweep_dirs:
        d = os.path.abspath(d)
        name = sweep_name_from_dir(d)
        entry = reg["sweeps"].setdefault(
            name, {"dir": d, "weight": 1, "jobs": {}, "submitted_scripts": []}
        )
        entry["dir"] = d
        if weight is not None:
            entry["weight"] = weight
        already = set(entry["submitted_scripts"])
        pending = [(i, p) for (i, p) in discover_trial_scripts(d) if p not in already]
        if pending:
            new_items[name] = pending
        else:
            print(f"  {name}: no new trial scripts to submit")

    if not new_items:
        print("Nothing to submit.")
        return

    # 1b. Auto-emitted prebuilds: materialize any missing recipe datasets on
    #     prepost first, and make each sweep's trials depend on it (afterok), so
    #     GPUs are never allocated to wait for data generation.
    sweep_dep = submit_prebuilds(list(sweep_dirs), dry_run=dry_run)

    weights = {s: reg["sweeps"][s]["weight"] for s in new_items}
    ordered = assign_rounds(new_items, weights)
    print(f"Submitting {len(ordered)} trials across {len(new_items)} sweep(s), "
          f"interleaved by round (gap={gap}):")

    # 2. sbatch in round order, with provisional nice = round * gap.
    for rnd, sweep, (idx, script) in ordered:
        nice = rnd * gap
        cmd = ["sbatch", "--parsable", f"--nice={nice}"]
        dep = sweep_dep.get(sweep)
        if dep:
            cmd.append(f"--dependency=afterok:{dep}")
        cmd.append(script)
        out = _run(cmd, dry_run=dry_run)
        jid = out.split(";")[0].strip() if out else f"DRY{idx}"
        reg["sweeps"][sweep]["jobs"][jid] = {
            "trial_idx": idx, "script": script, "round": rnd, "nice": nice,
        }
        reg["sweeps"][sweep]["submitted_scripts"].append(script)
        print(f"  round {rnd:>3}  {sweep}  trial_{idx:04d}  nice={nice}  job={jid}")

    save_registry(registry, reg)

    # 3. Globally re-interleave everything still pending (fixes nice across sweeps).
    if not dry_run:
        _rebalance(reg, registry, gap, dry_run=dry_run)


def rebalance(registry=DEFAULT_REGISTRY, dry_run=False):
    """Re-interleave all currently-pending jobs across registered sweeps."""
    reg = load_registry(registry)
    _rebalance(reg, registry, reg["round_gap"], dry_run=dry_run)


def cmd_submit(args):
    submit_sweeps(args.sweep_dirs, weight=args.weight,
                  registry=args.registry, dry_run=args.dry_run)


def cmd_rebalance(args):
    rebalance(registry=args.registry, dry_run=args.dry_run)


def _rebalance(reg, registry_path, gap, dry_run=False):
    """Recompute nice for all PENDING jobs across sweeps and apply via scontrol."""
    states = squeue_states()

    pending_by_sweep = {}       # sweep -> [(trial_idx, job_id), ...]
    for sweep, entry in reg["sweeps"].items():
        pend = [
            (info["trial_idx"], jid)
            for jid, info in entry["jobs"].items()
            if states.get(jid) == "PENDING"
        ]
        if pend:
            pending_by_sweep[sweep] = sorted(pend, key=lambda x: x[0])

    if not pending_by_sweep:
        print("Rebalance: no pending jobs to reorder.")
        return

    weights = {s: reg["sweeps"][s]["weight"] for s in pending_by_sweep}
    ordered = assign_rounds(pending_by_sweep, weights)

    print(f"Rebalancing {len(ordered)} pending job(s) across "
          f"{len(pending_by_sweep)} sweep(s):")
    changes = 0
    for rnd, sweep, (idx, jid) in ordered:
        nice = rnd * gap
        info = reg["sweeps"][sweep]["jobs"][jid]
        if info.get("nice") == nice:
            continue
        _run(["scontrol", "update", f"jobid={jid}", f"nice={nice}"], dry_run=dry_run)
        info["nice"] = nice
        info["round"] = rnd
        changes += 1
        print(f"  round {rnd:>3}  {sweep}  trial_{idx:04d}  -> nice={nice}  job={jid}")
    if changes == 0:
        print("  (already balanced)")
    save_registry(registry_path, reg)


def cmd_boost(args):
    reg = load_registry(args.registry)
    if args.sweep not in reg["sweeps"]:
        sys.exit(f"unknown sweep '{args.sweep}'. Known: {list(reg['sweeps'])}")
    reg["sweeps"][args.sweep]["weight"] = args.weight
    save_registry(args.registry, reg)
    print(f"Set weight of '{args.sweep}' to {args.weight} (trials per round).")
    _rebalance(reg, args.registry, reg["round_gap"], dry_run=args.dry_run)


def cmd_cancel(args):
    reg = load_registry(args.registry)
    if args.sweep not in reg["sweeps"]:
        sys.exit(f"unknown sweep '{args.sweep}'. Known: {list(reg['sweeps'])}")
    jids = list(reg["sweeps"][args.sweep]["jobs"].keys())
    if not jids:
        print("No jobs recorded for that sweep.")
        return
    _run(["scancel"] + jids, dry_run=args.dry_run)
    print(f"Cancelled {len(jids)} job(s) for '{args.sweep}'.")


def cmd_status(args):
    reg = load_registry(args.registry)
    states = squeue_states()
    if not reg["sweeps"]:
        print("No sweeps registered.")
        return
    hdr = f"{'sweep':<32} {'w':>2} {'run':>4} {'pend':>4} {'done':>4} {'total':>5} {'next round':>10}"
    print(hdr)
    print("-" * len(hdr))
    for sweep in sorted(reg["sweeps"]):
        entry = reg["sweeps"][sweep]
        run = pend = done = 0
        pending_rounds = []
        for jid, info in entry["jobs"].items():
            st = states.get(jid)
            if st == "RUNNING":
                run += 1
            elif st == "PENDING":
                pend += 1
                pending_rounds.append(info.get("round", 0))
            elif st is None:
                done += 1
            else:
                pend += 1  # CONFIGURING/COMPLETING/etc. count as in-flight
        total = len(entry["jobs"])
        nxt = min(pending_rounds) if pending_rounds else "-"
        print(f"{sweep:<32} {entry['weight']:>2} {run:>4} {pend:>4} {done:>4} "
              f"{total:>5} {str(nxt):>10}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", default=DEFAULT_REGISTRY,
                   help=f"registry JSON path (default: {DEFAULT_REGISTRY})")
    p.add_argument("--dry-run", action="store_true",
                   help="print sbatch/scontrol calls without executing")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("submit", help="submit one or more sweeps, interleaved")
    s.add_argument("sweep_dirs", nargs="+", help="paths to sweep dirs (contain jobs/)")
    s.add_argument("--weight", type=int, default=None,
                   help="trials per round for these sweeps (default keeps existing/1)")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("rebalance", help="re-interleave all pending jobs")
    s.set_defaults(func=cmd_rebalance)

    s = sub.add_parser("boost", help="change a sweep's weight, then rebalance")
    s.add_argument("sweep", help="sweep name")
    s.add_argument("--weight", type=int, required=True, help="new trials-per-round")
    s.set_defaults(func=cmd_boost)

    s = sub.add_parser("cancel", help="scancel all jobs of a sweep")
    s.add_argument("sweep", help="sweep name")
    s.set_defaults(func=cmd_cancel)

    s = sub.add_parser("status", help="show per-sweep running/pending/done")
    s.set_defaults(func=cmd_status)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
