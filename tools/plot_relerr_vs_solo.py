"""Per-process RELATIVE ERROR vs the solo scaling — the normalization-free analog of
plot_vs_solo.py.

The relative error  mean |A_pred - A_true| / |A_true|  is dimensionless, so there is
NO normalization mismatch between solo and pretrain and NO sigma-correction to apply:
the two are directly comparable. (This is the "actual physical error" view — unlike
raw MSE it isn't dominated by the few largest-|A| events.)

Relative error is only computed at the FINAL eval (one number per process per run) and
is logged but never saved to JSON, so BOTH sides are read from the run LOGS, and the
plot is an ENDPOINT (star) plot rather than a trajectory:
  * solo points = best (minimum) mean|rel err| at each compute level of the
    scaling_solo_full_<tag>_t* sweeps (over that cell's trial logs);
  * the solo power-law fit across the plotted compute range;
  * one star per --star run = its per-process final mean|rel err|, placed at that
    process's cumulative compute (from the run's per_process_metrics.json).

Run on Jean Zay (scans the solo sweep logs + the run logs; no data/*.npy needed).

Usage:
    python plot_relerr_vs_solo.py \
        --star runs/pretrain_full_nh4_sampler_002/trial_0071 \
        --star runs/pretrain_full_nh4_uniform_002/trial_0271
    options: --split val|test (default val)   --out PATH.pdf
A --star spec may be LABEL=run_dir to set the legend label.
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEPS = os.path.join(ROOT, "sweeps")
SOLO_BATCH = 16384
STAR_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
short = lambda s: s.replace("ee_", "").replace("-1000GeV_amplitudes", "")
tag = lambda n: n.replace("_amplitudes", "").replace("_", "")

_NUM = r"([0-9]*\.?[0-9]+(?:[eE][+\-]?[0-9]+)?)"


def fit(c, l):
    b, a = np.polyfit(np.log(c), np.log(l), 1)
    return a, b


def _cell_logs(d):
    """All plausible per-trial log files of a sweep cell (stdout/stderr/.log)."""
    pats = ["output/*.out", "error/*.err", "*.log", "*.out", "*.err"]
    return [f for p in pats for f in glob.glob(os.path.join(d, p))]


def _scan_relerr(path, line_filter, value_re):
    """Yield float values from lines that contain `line_filter` (and not the
    'largest amplitudes' variant), extracted with `value_re`."""
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                if "rel err" not in line or line_filter not in line:
                    continue
                if "largest amplitudes" in line:
                    continue
                m = value_re.search(line)
                if m:
                    yield float(m.group(1))
    except OSError:
        return


def solo_relerr_points(name, split):
    """[(compute, best_relerr)] over the solo cells of process `name`. compute =
    t_steps * SOLO_BATCH; best = min mean|rel err| across the cell's trial logs."""
    flt = f"Mean |rel err| {split} "          # solo is single-process: 'val <proc>:'
    vre = re.compile(_NUM + r"\s*$")          # value is the last token on the line
    pts = []
    for d in glob.glob(os.path.join(SWEEPS, f"scaling_solo_full_{tag(name)}_t*")):
        m = re.search(r"_t(\d+)$", d)
        if not m:
            continue
        best = None
        for lf in _cell_logs(d):
            for v in _scan_relerr(lf, flt, vre):
                if v > 0 and (best is None or v < best):
                    best = v
        if best is not None:
            pts.append((int(m.group(1)) * SOLO_BATCH, best))
    return sorted(pts)


def run_relerr(run_dir, split):
    """{proc: (compute, relerr)} for one run: per-process final mean|rel err| from
    the run log, placed at that process's cumulative compute from per_process_metrics."""
    # per-process rel err: multi-process eval logs 'val_<proc> <proc>: <val>'
    pat = re.compile(rf"Mean \|rel err\| {split}_(\S+)\s+\S+:\s*{_NUM}")
    rel = {}
    logs = glob.glob(os.path.join(run_dir, "*.log")) + glob.glob(os.path.join(run_dir, "*.out"))
    for lf in logs:
        try:
            with open(lf, errors="ignore") as f:
                for line in f:
                    if "rel err" not in line or "largest amplitudes" in line:
                        continue
                    m = pat.search(line)
                    if m:
                        rel[m.group(1)] = float(m.group(2))
        except OSError:
            continue
    # cumulative compute per process
    comp = {}
    mfiles = glob.glob(os.path.join(run_dir, "plots_*", "per_process_metrics.json"))
    if mfiles:
        C = json.load(open(sorted(mfiles)[0])).get("proc_compute", {})
        comp = {n: C[n][-1] for n in C if C.get(n)}
    return {n: (comp[n], rel[n]) for n in rel if comp.get(n)}


def _dataset_order(run_dir):
    mfiles = glob.glob(os.path.join(run_dir, "plots_*", "per_process_metrics.json"))
    if not mfiles:
        return None
    return json.load(open(sorted(mfiles)[0])).get("dataset_order")


def main():
    args = sys.argv[1:]
    split = args[args.index("--split") + 1] if "--split" in args else "val"
    out = args[args.index("--out") + 1] if "--out" in args else os.path.join(ROOT, "relerr_vs_solo.pdf")
    star_specs = [args[i + 1] for i, a in enumerate(args) if a == "--star" and i + 1 < len(args)]
    if not star_specs:
        print("Give at least one --star run_dir."); sys.exit(1)

    stars = []   # (label, {proc: (compute, relerr)}, dataset_order)
    for spec in star_specs:
        label, sep, path = spec.partition("=")
        if not sep:
            path = label
            par = os.path.basename(os.path.dirname(os.path.abspath(path)))
            label = (par or os.path.basename(os.path.abspath(path))).replace("pretrain_full_", "")
        stars.append((label, run_relerr(path, split), _dataset_order(path)))

    order = next((o for _, _, o in stars if o), None) or \
        sorted({p for _, d, _ in stars for p in d})
    if not order:
        print("No processes found in the --star runs."); sys.exit(1)

    ncol = 2; nrow = math.ceil(len(order) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.4 * nrow), squeeze=False)
    print(f"{'ds':10s} {'soloα':>6s} " + " ".join(f"{lab[:9]:>10s}/solo" for lab, _, _ in stars))
    for k, n in enumerate(order):
        ax = axes[k // ncol][k % ncol]
        sp = solo_relerr_points(n, split)
        computes = []
        a = b = None
        if len(sp) >= 2:
            sc = np.array([p[0] for p in sp], float)
            sl = np.array([p[1] for p in sp], float)
            a, b = fit(sc, sl)
            ax.plot(sc, sl, "s", color="0.25", ms=6, label="solo points")
            computes += [sc.min(), sc.max()]
        row = f"{short(n):10s} {(-b if b is not None else float('nan')):6.2f} "
        for si, (label, data, _) in enumerate(stars):
            pt = data.get(n)
            if not pt:
                row += f"{'-':>15s} "; continue
            c, v = pt
            ax.plot([c], [v], "*", color=STAR_COLORS[si % len(STAR_COLORS)], ms=15,
                    label=label, zorder=6)
            computes += [c]
            if a is not None:
                solo_at = math.exp(a + b * math.log(c))
                row += f"{v/solo_at:14.2f}x "
            else:
                row += f"{'-':>15s} "
        if a is not None and computes:
            cg = np.geomspace(min(computes), max(computes), 80)
            ax.plot(cg, np.exp(a + b * np.log(cg)), ":", color="0.25", lw=1.6,
                    label=f"solo fit (α={-b:.2f})")
        print(row)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(short(n), fontsize=12)
        ax.grid(True, which="both", lw=0.4, alpha=0.4)
        ax.legend(fontsize=8, frameon=False)
        if k // ncol == nrow - 1:
            ax.set_xlabel("compute (samples seen)")
        if k % ncol == 0:
            ax.set_ylabel(f"mean |rel err|  ({split})")
    for k in range(len(order), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Per-process relative error vs solo scaling  "
                 "(normalization-free; star = run endpoint)", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
