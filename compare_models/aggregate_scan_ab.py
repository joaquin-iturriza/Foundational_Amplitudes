#!/usr/bin/env python
"""Aggregate the 3-way coupling+mass A/B: for each run, read the best (minimum)
non-regularized val loss from per_process_metrics.json, then report best-vs-best
per arm. Run after compare_models/scan_ab_sweep.sh completes."""
import glob, json, os, collections

ROOT = os.path.join(os.path.dirname(__file__), "_scan_ab")
ARMS = ["off", "scalar", "diagram"]


def best_val_no_reg(run_dir):
    """Minimum combined val_loss_no_reg over the run (the checkpoint-selection
    metric). Falls back to val_loss if no-reg is absent."""
    hits = glob.glob(os.path.join(run_dir, "plots_*", "per_process_metrics.json"))
    if not hits:
        return None
    d = json.load(open(sorted(hits)[0]))
    series = d.get("val_loss_no_reg") or d.get("val_loss")
    vals = [v for v in (series or []) if v is not None]
    return min(vals) if vals else None


def main():
    by_arm = collections.defaultdict(dict)   # arm -> {lr: best_val}
    for run in sorted(glob.glob(os.path.join(ROOT, "*_lr*"))):
        name = os.path.basename(run)            # e.g. diagram_lr2e-3
        arm, lr = name.rsplit("_lr", 1)
        by_arm[arm][lr] = best_val_no_reg(run)

    print(f"{'arm':9s} {'best val_loss_no_reg':>22s}  {'best lr':>8s}   (per-lr)")
    summary = {}
    for arm in ARMS:
        runs = {lr: v for lr, v in by_arm.get(arm, {}).items() if v is not None}
        if not runs:
            print(f"{arm:9s} {'(no completed runs)':>22s}")
            continue
        blr = min(runs, key=runs.get)
        summary[arm] = (runs[blr], blr)
        per = "  ".join(f"{lr}:{v:.4f}" for lr, v in sorted(runs.items()))
        print(f"{arm:9s} {runs[blr]:>22.4f}  {blr:>8s}   {per}")

    if "off" in summary:
        base = summary["off"][0]
        print(f"\nBaseline (off) best val_loss_no_reg = {base:.4f}")
        for arm in ("scalar", "diagram"):
            if arm in summary:
                v = summary[arm][0]
                d = 100 * (base - v) / base
                verdict = "BETTER" if v < base else "worse"
                print(f"  {arm:8s}: {v:.4f}  ({d:+.2f}% vs baseline)  -> {verdict}")


if __name__ == "__main__":
    main()
