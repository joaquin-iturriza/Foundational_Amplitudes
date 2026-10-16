"""Zero-shot of the catalog pretraining's best trial (dancat_het_pre hp58, D59) on the 139 held-out processes of the
arXiv:2606.23791 split (D45, "Their split, report both"): per process, 1 - rho^2 of the prediction against the truth
on its validation split (the MSE after the best affine map over the variance; the parent standardizes per process, so
a process it has no data of has no prediction in log|M|^2 itself). The cumulative fraction of held-out processes below
a given 1 - rho^2, for the 96 that relabel a training process and the 43 that do not
(recipes/catalog_daniel_heldout_norelabel.yaml). Input: data/heldout_zero_shot.jsonl (ZERO_SHOT lines of
scripts/job_catalog_heldout.sh).
    python analysis/catalog_daniel/heldout_zero_shot.py   -> figs/heldout_zero_shot
"""
import json, os, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "heldout_zero_shot.jsonl")) if l.startswith("{")]
    for r in rows:
        if "error" in r:
            print("ERROR", r["recipe"], r["error"])          # listed, not dropped
    rows = [r for r in rows if "error" not in r]
    nr = {p["name"] for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_daniel_heldout_norelabel.yaml")))["processes"]}
    fig, ax = ps.figure()
    for sel, c, lab in ((lambda r: r["probe"] not in nr, ps.C.blue, "relabels"),
                        (lambda r: r["probe"] in nr, ps.C.vermillion, "no relabel")):
        v = np.sort([r["val_loss_affine"] for r in rows if sel(r)])
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, label=f"{lab} ({len(v)})")
        print(f"{lab}: {len(v)} processes, 1-rho^2 median {np.median(v):.3g}, 10-90% {np.quantile(v, .1):.3g} to "
              f"{np.quantile(v, .9):.3g}, max {v.max():.3g}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$1-\rho^2$, held-out process")
    ax.set_ylabel("fraction of processes")
    ps.legend(ax, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "heldout_zero_shot"))
    for r in sorted(rows, key=lambda r: -r["val_loss_affine"])[:8]:
        print(f"  {r['probe']:18s} {r['val_loss_affine']:.3f} {'no relabel' if r['probe'] in nr else 'relabel'}")


if __name__ == "__main__":
    main()
