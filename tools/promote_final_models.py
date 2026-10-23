"""Existing HETEROSC runs: their final model (after the sigma fit, the last part of training, D70) is moved into the
run's models/, replacing the pre-fit checkpoints, and <run>/sigma_stage/models becomes a link to it (run.py
promote_final_model does this at the end of every new run; the user, 2026-10-23: "There is no point in keeping the
model pre sigma fit"). Every <run>/sigma_stage/models that is a real directory with checkpoints, under the given roots.
Without --apply it only lists; with it, a manifest of the removed pre-fit files and their sizes is written first
(runs/PROMOTED_<time>.json). A run whose stage is still writing (a checkpoint modified in the last hour) is skipped.
    python tools/promote_final_models.py runs runs_from_lxplus [--apply]
"""
import argparse, glob, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import promote_final_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    todo, now = [], time.time()
    for root in a.roots:
        for sm in glob.glob(os.path.join(root, "**", "sigma_stage", "models"), recursive=True):
            if os.path.islink(sm) or not os.path.isdir(sm):
                continue
            ck = [os.path.join(sm, f) for f in os.listdir(sm) if f.startswith("model_run")]
            if not ck:
                continue
            if max(os.path.getmtime(f) for f in ck) > now - 3600:
                print(f"skip (stage still writing): {sm}")
                continue
            run = os.path.dirname(os.path.dirname(sm))
            pre = [os.path.join(run, "models", f) for f in sorted(os.listdir(os.path.join(run, "models")))
                   if f.startswith("model_run")] if os.path.isdir(os.path.join(run, "models")) else []
            todo.append((run, os.path.dirname(sm), pre))
    freed = sum(os.path.getsize(f) for _, _, pre in todo for f in pre)
    print(f"{len(todo)} runs, {freed / 1e9:.2f} GB of pre-fit checkpoints")
    if not a.apply:
        for run, _, pre in todo[:20]:
            print(f"  {run}: {len(pre)} pre-fit files")
        return
    man = os.path.join(a.roots[0], f"PROMOTED_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump([{"run": r, "removed": [(f, os.path.getsize(f)) for f in pre]} for r, _, pre in todo], open(man, "w"),
              indent=1)
    print("manifest", man)
    for run, stage, _ in todo:
        promote_final_model(run, stage)
    print(f"PROMOTED {len(todo)} runs")


if __name__ == "__main__":
    main()
