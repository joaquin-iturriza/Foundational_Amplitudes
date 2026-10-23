"""sweep/prune.py's rule (D19: a finished sweep keeps only its best trial's models; every trial keeps its config, logs,
metrics and data_stats, so any trial can be rerun from its config) applied to sweeps copied from another site
(runs_from_lxplus/), whose result files stayed there, so prune.py sees no finished trial and keeps everything (the user,
2026-10-23: "aren't we supposed to be pruning and deleting the models that don't win in a sweep?"). A sweep's best is
read from the trials' own logs instead: per trial, the lowest val_loss_no_reg its out_*.log reports, among the trials whose config ran the sweep's longest horizon
(training.iterations). A copy is never resumed, so its sweep is finished: as prune.py does for a finished sweep, every
other trial loses its models (a lower-fidelity trial of a multi-fidelity sweep, a trial that crashed). Kept whole: the best trial, every
trial named anywhere in the repo's job scripts, sweep configs, Delta lists or analysis files (runs/<sweep>/trial_N),
every fine-tune parent sweep/prune.py protects (its parents(), this site's sweep configs included), and every trial written to in the last 6 h. Without --apply it prints the plan; with it, a manifest
(<root>/PRUNED_COPIES_<time>.json: per trial what was deleted and its size) is written before anything is deleted.
    python tools/prune_copies.py runs_from_lxplus [--apply]
"""
import argparse, glob, json, os, re, sys, time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# "Val loss (combined): x" (several processes, regularized) or "Val loss: x" (one process, without the regularization
# term); a multi-process log written before 2026-10-23 has no val_loss_no_reg field, and its regularized value cannot
# rank trials (lambda is swept), so such a sweep has no best here and keeps every model
VAL_RE = re.compile(r"Val loss(?: \(combined\))?: (-?[0-9.]+(?:e[-+]?\d+)?)")


def referenced():
    """Trials named in the repo, plus every fine-tune parent and job-script source sweep/prune.py protects (D19: fine-tune
    parents kept whole), which includes the sweep configs under this site's SWEEP_DIR"""
    sys.path.insert(0, os.path.join(REPO, "sweep"))
    import prune
    refs = set(prune.parents())
    pats = ["scripts/*.sh", "sweep/*.py", "sweep/**/*.yaml", "analysis/**/*.txt", "analysis/**/*.json",
            "analysis/**/*.py", "recipes/**/*.yaml", "tools/*.py"]
    for p in pats:
        for f in glob.glob(os.path.join(REPO, p), recursive=True):
            try:
                refs |= set(re.findall(r"([A-Za-z0-9_.-]+/trial_\d+)", open(f, errors="replace").read()))
            except OSError:
                pass
    return refs


NOREG_RE = re.compile(r"val_loss_no_reg: (-?[0-9.]+(?:e[-+]?\d+)?)")


def best_val(td):
    """The trial's lowest checkpoint metric, val_loss_no_reg: where the log prints it (multi-process runs since
    2026-10-23) or a single-process 'Val loss: x' line (already without the regularization term). A multi-process log
    with only the regularized 'Val loss (combined)' gives (None, True): its regularized value adds lambda |theta|^2, and
    lambda is a swept HP, so it cannot rank trials (CLAUDE.md "Reported values")."""
    best, reg_only = None, False
    for log in glob.glob(os.path.join(td, "out_*.log")):
        for line in open(log, errors="replace"):
            m = NOREG_RE.search(line)
            if m is None:
                m = VAL_RE.search(line)
                if m is not None and "(combined)" in line:
                    reg_only = True
                    continue
            if m:
                v = float(m.group(1))
                best = v if best is None or v < best else best
    return (None, True) if best is None and reg_only else (best, False)


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
        if len(tds) < 2 or not any(model_files(t) for t in tds):
            continue
        sweep = os.path.basename(sd)
        info = {td: (iterations(td),) + best_val(td) for td in tds}
        top = max((it for it, _, _ in info.values() if it is not None), default=None)
        done = {td: v for td, (it, v, _) in info.items() if it == top and v is not None}
        reg_only = any(r for it, _, r in info.values() if it == top)
        best = min(done, key=done.get) if done and not reg_only else None
        if best is None:
            # no trial reports val_loss_no_reg at the horizon (none validated there, or a multi-process log that prints
            # only the regularized value): no best to rank, so every model is kept
            for td in tds:
                mf = model_files(td)
                if mf:
                    sz = sum(os.path.getsize(f) for f in mf); kept_bytes += sz
                    plan.append({"run_dir": td, "bytes": sz, "kept": "no best found (%s)" % (
                        "only the regularized val loss in the logs" if reg_only else "no validation at the horizon")})
            continue
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
            else:
                why = None
            if why:
                kept_bytes += sz
                plan.append({"run_dir": td, "kept": why, "bytes": sz})
            else:
                freed += sz
                plan.append({"run_dir": td, "val": done.get(td), "iterations": info[td][0], "best": best,
                             "deleted": mf, "bytes": sz})
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
