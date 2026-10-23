"""sweep/prune.py's rule (D19: a finished sweep keeps only its best trial's models; every trial keeps its config, logs,
metrics and data_stats, so any trial can be rerun from its config) applied to sweeps copied from another site
(runs_from_lxplus/), whose result files stayed there, so prune.py sees no finished trial and keeps everything (the user,
2026-10-23: "aren't we supposed to be pruning and deleting the models that don't win in a sweep?"). A sweep's best is
read from the trials' own logs instead: per trial, the lowest "Val loss (combined)" its out_*.log reports (the
validation the checkpoint selection reads), among the trials whose config ran the sweep's longest horizon
(training.iterations); a trial without one is unfinished and keeps everything. Kept whole: the best trial, every
trial named anywhere in the repo's job scripts, sweep configs, Delta lists or analysis files (runs/<sweep>/trial_N),
and every trial written to in the last 6 h. Without --apply it prints the plan; with it, a manifest
(<root>/PRUNED_COPIES_<time>.json: per trial what was deleted and its size) is written before anything is deleted.
    python tools/prune_copies.py runs_from_lxplus [--apply]
"""
import argparse, glob, json, os, re, sys, time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_RE = re.compile(r"Val loss \(combined\): (-?[0-9.]+(?:e[-+]?\d+)?)")


def referenced():
    refs = set()
    pats = ["scripts/*.sh", "sweep/*.py", "sweep/**/*.yaml", "analysis/**/*.txt", "analysis/**/*.json",
            "analysis/**/*.py", "recipes/**/*.yaml", "tools/*.py"]
    for p in pats:
        for f in glob.glob(os.path.join(REPO, p), recursive=True):
            try:
                refs |= set(re.findall(r"([A-Za-z0-9_.-]+/trial_\d+)", open(f, errors="replace").read()))
            except OSError:
                pass
    return refs


def best_val(td):
    best = None
    for log in glob.glob(os.path.join(td, "out_*.log")):
        for line in open(log, errors="replace"):
            m = VAL_RE.search(line)
            if m:
                v = float(m.group(1))
                best = v if best is None or v < best else best
    return best


def iterations(td):
    try:
        return int(yaml.safe_load(open(os.path.join(td, "config.yaml")))["training"]["iterations"])
    except Exception:
        return None


def model_files(td):
    md = os.path.join(td, "models")
    if not os.path.isdir(md) or os.path.islink(md):
        return []
    return [os.path.join(md, f) for f in sorted(os.listdir(md)) if f.startswith("model_run")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    refs, now = referenced(), time.time()
    plan, freed, kept_bytes = [], 0, 0
    for sd in sorted(glob.glob(os.path.join(a.root, "*"))):
        tds = sorted(glob.glob(os.path.join(sd, "trial_*")))
        if len(tds) < 2:
            continue
        sweep = os.path.basename(sd)
        info = {td: (iterations(td), best_val(td)) for td in tds}
        top = max((it for it, _ in info.values() if it is not None), default=None)
        done = {td: v for td, (it, v) in info.items() if it == top and v is not None}
        best = min(done, key=done.get) if done else None
        for td in tds:
            mf = model_files(td)
            if not mf:
                continue
            sz = sum(os.path.getsize(f) for f in mf)
            tail = "%s/%s" % (sweep, os.path.basename(td))
            newest = max(os.path.getmtime(os.path.join(dp, f)) for dp, _, fs in os.walk(td) for f in fs)
            if td == best:
                why = "the sweep's best (val %.4g)" % done[td]
            elif tail in refs:
                why = "referenced in the repo"
            elif newest > now - 6 * 3600:
                why = "written to in the last 6 h"
            elif td not in done:
                why = "unfinished (no validation at the sweep's horizon %s)" % top
            else:
                why = None
            if why:
                kept_bytes += sz
                plan.append({"run_dir": td, "kept": why, "bytes": sz})
            else:
                freed += sz
                plan.append({"run_dir": td, "val": done[td], "best": best, "deleted": mf, "bytes": sz})
    n_del = sum(1 for p in plan if "deleted" in p)
    print(f"{len(plan)} trials with models: {n_del} to prune ({freed / 1e9:.1f} GB), "
          f"{len(plan) - n_del} kept ({kept_bytes / 1e9:.1f} GB)")
    reasons = {}
    for p in plan:
        k = p.get("kept", "pruned").split(" (")[0]
        reasons[k] = reasons.get(k, 0) + p["bytes"]
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {k:40s} {v / 1e9:6.1f} GB")
    if not a.apply:
        return
    man = os.path.join(a.root, "PRUNED_COPIES_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    json.dump(plan, open(man, "w"), indent=1)
    print("manifest", man)
    for p in plan:
        for f in p.get("deleted", []):
            os.remove(f)
    print(f"PRUNED_COPIES {n_del} trials, {freed / 1e9:.1f} GB freed")


if __name__ == "__main__":
    main()
