#!/usr/bin/env python3
"""Epochs of every point of catalog_coverage's width and length studies, and the cost of redoing them at >= 1 epoch
(plan catalog_coverage step 64, Joaquin 2026-11-06 D292/D293).

  count   every point of par:v4_width / par:v4_mult_width_length / par:v4_length (the 1x width cells, both seeds, and
          every trial of the 4x and 16x long sweeps): epochs = steps / batches per epoch (steps x 16384 events per step
          over the training events; the batches are each run's 'Starting to train ... with B batches' line, read
          into data/epochs.json by mixture_ab.py epochs and data/size_curves.json by size_figs.py pull)
          -> data/study_epochs.json
  grid    the width x length map at >= 1 epoch: widths 8, 16, 32 (LONG_HEADS) x lengths 4264 x {2, 4, 8, 16} steps,
          seeds 42 and 43, every cell at the long sweeps' best hyperparameter point (trial_1262, best of both the
          17056 and the 68224 sweep), its lr 7.82114e-3 at 17056 moved to the cell's steps by lr ~ t^-0.55. That
          exponent is not measured: it is the fixed slope with which tools/lr_surface_center.py centred both sweeps'
          windows (prin:invU, HPO rule 2; centres 2.633e-3 at 17056, 1.228e-3 at 68224), and trial_1262 sits at the
          same normalized position (0.9955) in both, so its two lrs are 4^0.55 apart by construction; that position is
          the window top (censored, best at the last validation): not converged, every cell's lr a lower bound on its
          optimum. Training hours from the measured V100 rates per width (data/size_mse.json 'Finished training' hours
          over iterations, main block, the 4264-step whole-set runs; tools/short_wall.py S_ITER_WIDTH); the rest of a
          job (start-up, default-share composition, sigma stage, the 2313 zero-shot probes) from tools/short_wall.py's
          rates; the job's time limit is short_wall.py --width's slurm_time (the job checks itself against it);
          GPU-h = 1 GPU x the job's work time; the cells already run (the sweeps' trial_1262 at 4x and 16x, width 8,
          seed 42) are reused -> data/scaling_grid_plan.json
  fig     figs/scaling_grid: (a) the existing points (size_figs.fit_points: MSE of log|M|^2 over every training pool at
          the best checkpoint, geometric mean, against C = 6 N S B), open where the run saw under one epoch, each
          labelled with its epochs; ticks at the bottom: the planned cells' compute (dark: the 2x row, light: 4x-16x);
          (b) the width x length map: per cell what exists and what the grid would run, with the new GPU-h of the cell
          (both seeds, data/scaling_grid_plan.json)
"""
import json, math, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
from size_figs import WIDTH_RUNS, N_PARAMS, BATCH   # noqa: E402

BASE = 4264
LR_1262 = {17056: 7.82114e-3, 68224: 3.649139e-3}
FACTORS = (2, 4, 8, 16)
WIDTHS = (8, 16, 32)
SEEDS = (42, 43)
REUSED = {(8, 4, 42): "runs/cov_v4_long_hpo8_cc_17056/trial_1262", (8, 16, 42): "runs/cov_v4_long_hpo8_cc_68224/trial_1262"}


def count():
    ep = json.load(open(os.path.join(DATA, "epochs.json")))
    tr = json.load(open(os.path.join(DATA, "size_curves.json")))["trials"]
    rows = []
    for w, runs in WIDTH_RUNS.items():
        for seed, r in zip(SEEDS, runs):
            e = ep[r]
            rows.append({"point": f"width {w}, 1x, seed {seed}", "run": r, "width": w, "steps": e["iterations"],
                         "batches_per_epoch": e["batches"], "epochs": round(e["iterations"] / e["batches"], 4)})
    for k, t in sorted(tr.items()):
        rows.append({"point": f"width 8, {t['steps'] // BASE}x sweep {t['trial']}", "run": f"runs/{t['sweep']}/{t['trial']}",
                     "width": 8, "steps": t["steps"], "batches_per_epoch": t["batches"],
                     "epochs": round(t["steps"] / t["batches"], 4)})
    out = {"source": __doc__.split("\n\n")[1].strip(), "rows": rows,
           "under_one_epoch": [r["point"] for r in rows if r["epochs"] < 1]}
    json.dump(out, open(os.path.join(DATA, "study_epochs.json"), "w"), indent=1)
    for r in rows:
        print(f"{r['point']:42s} {r['run']:48s} steps {r['steps']:6d} batches/epoch {r['batches_per_epoch']} epochs {r['epochs']:.3f}")
    print("under one epoch:", out["under_one_epoch"])


LR_SLOPE = -0.55    # tools/lr_surface_center.py SLOPE: the window law, not a measurement (see the module docstring)


def lr_at(steps):
    """trial_1262's lr at 17056 moved along the window law; its 68224 lr is reproduced by construction (checked)"""
    a = math.log(LR_1262[17056] / LR_1262[68224]) / math.log(68224 / 17056)
    assert abs(a + LR_SLOPE) < 1e-3, a         # the two sweeps' windows share the law, so the pair carries no slope
    return LR_1262[17056] * (steps / 17056) ** LR_SLOPE, -LR_SLOPE


def overhead_hours():
    """short_wall.py's hours for a het arm with the long cells' 12 axes, minus its training term, before the margin.
    An estimate at every width: short_wall's read rate (1.45 s per het probe), start-up and sigma stage (20 min) were
    measured on width-8 jobs only; the assumption is that they do not grow with width (a probe's forward pass and the
    250-step sigma stage do grow, so the width-16 and width-32 cells' overhead is a lower bound: at the training
    slowdown, x1.73 and x3.51 on the sigma stage's 250 steps, about 3 and 10 min more; the reads are not measured)."""
    axes = "H1 H2 H2p5 H3 H5 H6 H7E H7c PP4 EE4 HD4 HD5".split()
    o = json.loads(subprocess.run([sys.executable, os.path.join(ROOT, "tools", "short_wall.py"), "--train",
                                   os.path.join(ROOT, "recipes", "catalog_v4_default25_train.yaml"), "--cat", "v4",
                                   "--axes", *axes, "--iters", str(BASE), "--het"], capture_output=True, text=True,
                                  check=True).stdout.strip().splitlines()[-1])
    r = o["rates"]
    return o["hours"] / r["margin"] - BASE * r["s_per_step"] / 3600, r["margin"], o["n_probes"]


def grid():
    mse = json.load(open(os.path.join(DATA, "size_mse.json")))["runs"]
    rate = {}
    for w in WIDTHS:
        xs = [v["train_hours"] * 3600 / v["iterations"] for k, v in mse.items()
              if k.startswith("runs/catalog_v4_long/") and k.endswith(f"_w{w}_x1")]
        rate[w] = max(xs)            # the slower seed kept
    ep = json.load(open(os.path.join(DATA, "study_epochs.json")))
    batches = {r["batches_per_epoch"] for r in ep["rows"]}
    assert len(batches) == 1, batches
    batches = batches.pop()
    oh, margin, n_probes = overhead_hours()
    limits = {}
    for f in FACTORS:
        for w in WIDTHS:
            o = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "short_wall.py"), "--train",
                                os.path.join(ROOT, "recipes", "catalog_v4_default25_train.yaml"), "--cat", "v4", "--axes",
                                *"H1 H2 H2p5 H3 H5 H6 H7E H7c PP4 EE4 HD4 HD5".split(), "--iters", str(BASE * f), "--het",
                                "--width", str(w)], capture_output=True, text=True, check=True).stdout
            limits[(w, f)] = json.loads(o.strip().splitlines()[-1])["slurm_time"]
    cells, tot, tot_new = [], 0.0, 0.0
    for f in FACTORS:
        steps = BASE * f
        lr, a = lr_at(steps)
        for w in WIDTHS:
            for s in SEEDS:
                train_h = steps * rate[w] / 3600
                work_h = train_h + oh
                reused = REUSED.get((w, f, s))
                c = {"width": w, "factor": f, "steps": steps, "epochs": round(steps / batches, 3), "seed": s,
                     "lr": float(f"{lr:.6g}"), "train_hours": round(train_h, 2), "work_hours": round(work_h, 2),
                     "limit_hours": round(work_h * margin, 2), "slurm_time": limits[(w, f)], "C_6NSB": 6 * N_PARAMS[w] * steps * BATCH,
                     "reused": reused}
                cells.append(c)
                tot += work_h
                if not reused:
                    tot_new += work_h
    by_factor = {f: round(sum(c["work_hours"] for c in cells if c["factor"] == f and not c["reused"]), 1) for f in FACTORS}
    out = {"source": "analysis/catalog_v4/epoch_count.py grid", "rates_s_per_step_V100": rate,
           "batches_per_epoch": batches, "overhead_hours_per_job": round(oh, 3), "n_probes": n_probes,
           "margin": margin, "lr_exponent": round(a, 4), "cells": cells,
           "new_jobs": sum(1 for c in cells if not c["reused"]), "gpu_hours_new": round(tot_new, 1),
           "gpu_hours_new_by_factor": by_factor}
    json.dump(out, open(os.path.join(DATA, "scaling_grid_plan.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1))
    for c in cells:
        print(f"w{c['width']:2d} x{c['factor']:2d} s{c['seed']} steps {c['steps']:6d} ep {c['epochs']:6.2f} lr {c['lr']:.4g} "
              f"train {c['train_hours']:6.2f} h work {c['work_hours']:6.2f} h limit {c['slurm_time']} "
              f"{'REUSED ' + c['reused'] if c['reused'] else ''}")


def fig():
    sys.path.insert(0, ROOT)
    import plot_style as ps
    from matplotlib.lines import Line2D
    from size_figs import fit_points
    g = json.load(open(os.path.join(DATA, "scaling_grid_plan.json")))
    ep = 4264 / g["batches_per_epoch"]
    col = {8: ps.C.blue, 16: ps.C.green, 32: ps.C.vermillion}
    fig, (ax, axm) = ps.figure(ncols=2)
    for d, w, n, c, y in fit_points():
        e = n / g["batches_per_epoch"]
        ax.plot([c], [y], ls="", marker="o", ms=5, color=col[w], mfc="none" if e < 1 else col[w])
        ax.annotate(f"{e:.2g} ep", (c, y), textcoords="offset points", xytext=(5, 2), fontsize=6)
    lo = ax.get_ylim()[0]
    for cell in g["cells"]:
        if cell["reused"] is None:
            ax.plot([cell["C_6NSB"]], [lo], ls="", marker="|", ms=9, color=col[cell["width"]],
                    alpha=1.0 if cell["factor"] == 2 else 0.35)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training compute $6NSB$ [FLOP]")
    ax.set_ylabel(r"MSE of $\log|\mathcal{M}|^2$, all pools")
    hs = [Line2D([], [], ls="", marker="o", color=col[w], label=f"width {w}") for w in (8, 16, 32)]
    hs += [Line2D([], [], ls="", marker="o", color="k", mfc="none", label="run, under one epoch"),
           Line2D([], [], ls="", marker="|", ms=9, color="k", label="planned, 2x row"),
           Line2D([], [], ls="", marker="|", ms=9, color="k", alpha=0.35, label="planned, 4x to 16x")]
    ps.legend(ax, handles=hs, loc="upper right", fontsize=6)
    # (b) the map
    factors = (1, 2, 4, 8, 16); widths = (8, 16, 32)
    gh = {}
    for cell in g["cells"]:
        if cell["reused"] is None:
            gh[(cell["width"], cell["factor"])] = gh.get((cell["width"], cell["factor"]), 0) + cell["work_hours"]
    for i, f in enumerate(factors):
        for j, w in enumerate(widths):
            if f == 1:
                axm.plot(i, j, marker="s", ms=13, mfc="none", mec=ps.C.grey, ls="")
                axm.text(i, j - 0.33, "0.64 ep", ha="center", fontsize=5.5)
                continue
            reused = w == 8 and f in (4, 16)
            axm.plot(i, j, marker="s", ms=13, ls="", color=ps.C.vermillion if f == 2 else ps.C.orange,
                     alpha=1.0 if f == 2 else 0.55)
            if reused:
                axm.plot(i, j, marker="o", ms=5, ls="", color="k")
            axm.text(i, j - 0.33, f"{gh.get((w, f), 0):.0f} GPU-h", ha="center", fontsize=5.5)
    axm.set_xticks(range(len(factors)))
    axm.set_xticklabels([f"{f}x\n{f * ep:.2g} ep" for f in factors], fontsize=6)
    axm.set_yticks(range(len(widths))); axm.set_yticklabels([str(w) for w in widths])
    axm.set_xlim(-0.6, len(factors) - 0.4); axm.set_ylim(-0.7, len(widths) - 0.4)
    axm.set_xlabel("training length (x 4264 steps), epochs")
    axm.set_ylabel("width (attention heads)")
    hm = [Line2D([], [], ls="", marker="s", mfc="none", mec=ps.C.grey, label="run, under one epoch (replaced)"),
          Line2D([], [], ls="", marker="s", color=ps.C.vermillion, label="2x row: 6 jobs"),
          Line2D([], [], ls="", marker="s", color=ps.C.orange, alpha=0.55, label="rest of the grid: 16 jobs"),
          Line2D([], [], ls="", marker="o", color="k", ms=5, label="one seed exists (tuned trial)")]
    ps.legend(axm, handles=hm, loc="upper left", fontsize=5.5)
    ps.save(fig, os.path.join(HERE, "figs", "scaling_grid"))


if __name__ == "__main__":
    {"count": count, "grid": grid, "fig": fig}[sys.argv[1]]()
