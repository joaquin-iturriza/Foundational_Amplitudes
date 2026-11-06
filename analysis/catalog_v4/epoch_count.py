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
          17056 and the 68224 sweep, its lr at the two lengths 7.82114e-3 and 3.649139e-3, i.e. lr ~ t^-0.55, the
          HPO surface's decay; the lr at another length from that law). Training hours from the measured V100 rates per
          width (data/size_mse.json 'Finished training' hours over iterations, main block, the 4264-step whole-set
          runs); the rest of a job (start-up, default-share composition, sigma stage, the 2313 zero-shot probes) from
          tools/short_wall.py's rates; GPU-h = 1 GPU x the job's work time; the cells already run (the sweeps'
          trial_1262 at 4x and 16x, width 8, seed 42) are reused -> data/scaling_grid_plan.json
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


def lr_at(steps):
    # the two measured points fix the exponent: ln(7.82114/3.649139)/ln 4 = 0.550
    a = math.log(LR_1262[17056] / LR_1262[68224]) / math.log(68224 / 17056)
    return LR_1262[17056] * (steps / 17056) ** (-a), a


def overhead_hours():
    """short_wall.py's hours for a het arm with the long cells' 12 axes, minus its training term, before the margin."""
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
                     "limit_hours": round(work_h * margin, 2), "C_6NSB": 6 * N_PARAMS[w] * steps * BATCH,
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
              f"train {c['train_hours']:6.2f} h work {c['work_hours']:6.2f} h limit {c['limit_hours']:6.2f} h "
              f"{'REUSED ' + c['reused'] if c['reused'] else ''}")


if __name__ == "__main__":
    {"count": count, "grid": grid}[sys.argv[1]]()
