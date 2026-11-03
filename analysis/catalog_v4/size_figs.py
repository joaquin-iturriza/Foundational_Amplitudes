#!/usr/bin/env python3
"""Width and length figures of plan entry catalog_coverage (steps 16 and 33), for the group meeting of 2026-11-03.

  pull    the validation curves (val_loss_no_reg of every 'Val loss (combined)' line of out_0.log) of every trial
          of the long-run DyHPO sweeps cov_v4_long_hpo8_cc_17056 (17056 steps) and cov_v4_long_hpo8_cc_68224 (68224
          steps) that has validated (each runs/<sweep>/trial_* directory with such lines), each trial's learning rate
          (training.lr of its config.yaml) and its training set in batches (its 'Starting to train ... with B batches'
          line), read from CC-IN2P3 with one read-only `site run` -> data/size_curves.json (a trial without
          'Finished training' is marked running, at its last validation)
  best    each sweep's finished trials, its best, and the design's not-converged flags (lr at a window edge, best at
          the last validation) -> data/long_sweeps_best.json
  figs    figs/width_x1_loss (from data/width_x1.json, mixture_ab.py width), figs/length_w8_curves,
          figs/loss_vs_compute (png + pdf)

Best checkpoint: the lowest aggregate val_loss_no_reg (het NLL) of the main training block. The 4264-step curves are
the width runs' own (data/reads/<run>_val.json, mixture_ab.py val), their lr training.lr of the run's config.yaml
(LR_X1). Epochs = steps x batch / training events = steps / batches (data/epochs.json for the 4264-step runs, the
trials' own lines for the sweeps). Compute: C = 6 N S B, N the learnable parameter count of the run's model (its log:
width 8 1775829, 16 6516181, 32 25434069), S the steps, B = 16384 events per step: the 6ND estimate with D the events
seen (not the particles: every run here trains on the same mixture, so counting particles would shift every point
by the same factor); attention over particles and gradient checkpointing's recompute are not counted (an estimate).
Validations above OFF_SCALE (the warm-up spikes, up to ~1e11) are not drawn and are counted in the printed report.
"""
import argparse, json, math, os, re, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
SWEEPS = {"17056": "cov_v4_long_hpo8_cc_17056", "68224": "cov_v4_long_hpo8_cc_68224"}
BATCH = 16384
N_PARAMS = {8: 1775829, 16: 6516181, 32: 25434069}     # 'Instantiated model ... learnable parameters' of each log
WIDTH_RUNS = {8: ("FA-ccin2p3-804a65bc", "FA-ccin2p3-66e8c31f"), 16: ("FA-ccin2p3-493c98db", "FA-ccin2p3-7288b3fe"),
              32: ("FA-ccin2p3-e26059bf", "FA-ccin2p3-fd7fc137")}     # (seed 42, seed 43), mixture_ab.WIDTH
LR_X1 = 4.997423e-3      # training.lr of the 4264-step width runs' config.yaml (one lr at every width under muP)
OFF_SCALE = 1.0


def pull():
    cmd = ('for s in ' + " ".join(SWEEPS.values()) + '; do for d in runs/$s/trial_*; do '
           'grep -q "Val loss (combined)" $d/out_0.log 2>/dev/null || continue; echo "@@ $d"; '
           'grep -m1 -E "^  lr:" $d/config.yaml; '
           'grep -E "Starting to train for|Val loss \\(combined\\)|Finished training" $d/out_0.log; done; done')
    out = subprocess.run(["site", "--timeout", "600", "run", "ccin2p3", "FA", "--", cmd], capture_output=True,
                         text=True, timeout=700).stdout
    res, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"@@ runs/(\S+)/(trial_\S+)", line)
        if m:
            steps = next(k for k, v in SWEEPS.items() if v == m.group(1))
            cur = res.setdefault(f"{steps}_{m.group(2)}", {"sweep": m.group(1), "trial": m.group(2),
                                                           "steps": int(steps), "lr": None, "batches": None,
                                                           "every": None, "vals": [], "finished": False})
            continue
        if cur is None:
            continue
        if re.match(r"^  lr:", line):
            cur["lr"] = float(line.split(":", 1)[1])
        elif "Starting to train for" in line and cur["every"] is None:
            cur["every"] = int(re.search(r"validating every (\d+) steps", line).group(1))
            cur["batches"] = int(re.search(r"dataset with (\d+) batches", line).group(1))
        elif "Val loss (combined)" in line:
            x = float(re.search(r"val_loss_no_reg: ([0-9.eE+-]+|nan|inf|-inf)", line).group(1))
            cur["vals"].append([min(cur["every"] * (len(cur["vals"]) + 1), cur["steps"]), x])
        elif "Finished training" in line:
            cur["finished"] = True
    bad = {k: v for k, v in res.items() if not v["vals"] or v["lr"] is None or v["batches"] is None}
    if not res or bad:
        sys.exit("pull: incomplete trials: " + json.dumps({k: [len(v["vals"]), v["lr"], v["batches"]]
                                                          for k, v in bad.items()}))
    # each sweep's results/*.json (the trial's objective: the main run's best val_loss_no_reg, and its step)
    cmd2 = ('for s in ' + " ".join(SWEEPS.values()) + '; do for f in sweeps/$s/results/*.json; do '
            'echo "@@ $s $(basename $f)"; '
            'grep -oE "(val_loss|best_step|validate_every|traintime_hours).: [-0-9.eE+]+" $f; done; done')
    out2 = subprocess.run(["site", "--timeout", "600", "run", "ccin2p3", "FA", "--", cmd2], capture_output=True,
                          text=True, timeout=700).stdout
    results, key = {}, None
    for line in out2.splitlines():
        m = re.match(r"@@ (\S+) hp(\d+)_t\d+_\S+\.json", line)
        if m:
            key = f"{m.group(1)}/{m.group(2)}"
            results[key] = {}
            continue
        m = re.match(r'(val_loss|best_step|validate_every|traintime_hours).: (\S+)', line)
        if key and m and m.group(1) not in results[key]:
            results[key][m.group(1)] = float(m.group(2))
    if not results:
        sys.exit("pull: no sweep results read: " + out2[-500:])
    path = os.path.join(DATA, "size_curves.json")
    json.dump({"source": "out_0.log and config.yaml of each validated trial, results/*.json of each sweep, on "
                         "ccin2p3, read-only site run", "trials": res, "results": results}, open(path, "w"), indent=0)
    print(json.dumps({k: {"n": len(v["vals"]), "last_step": v["vals"][-1][0], "finished": v["finished"],
                          "lr": v["lr"]} for k, v in res.items()}))


SWEEP_CONFIGS = {"17056": "sweep/sweep_config_ccin2p3_cov_v4_long_hpo_x4.yaml",
                 "68224": "sweep/sweep_config_ccin2p3_cov_v4_long_hpo_x16.yaml"}


def sampler_edge_rule():
    """(top_k, boundary_frac, trigger_count) of sweep/dyhpo_sampler.py's check_and_extend_ranges: the sampler's own
    test that the best results cluster at a search-space edge (read from its defaults, not copied)"""
    import ast
    src = open(os.path.join(ROOT, "sweep", "dyhpo_sampler.py")).read()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "check_and_extend_ranges")
    names = [a.arg for a in fn.args.args][-len(fn.args.defaults):]
    d = {k: ast.literal_eval(v) for k, v in zip(names, fn.args.defaults)}
    return d["top_k"], d["boundary_frac"], d["trigger_count"]


def sweep_best():
    """Per sweep: every finished trial's objective (results/*.json val_loss: the main run's best val_loss_no_reg) with
    its lr, the best trial, where each lr sits in the search window (log position, 0 the low edge, 1 the high), and
    whether each trial's best checkpoint is its last validation -> data/long_sweeps_best.json. Not converged (the
    step-16 design: a best lr at a window edge, or top trials best at their last step): the best trial's lr in the
    window's outer boundary_frac, or the sampler's edge cluster (trigger_count of the top_k trials' lr in the same
    outer band; sweep/dyhpo_sampler.py check_and_extend_ranges, its defaults), or the best trial or a majority of the
    top_k best at their last validation. A trial without its lr or validate_every stops the tool."""
    import yaml
    top_k, frac, trig = sampler_edge_rule()
    d = json.load(open(os.path.join(DATA, "size_curves.json")))
    tr, results = d["trials"], d.get("results", {})
    out = {"edge_rule": {"top_k": top_k, "boundary_frac": frac, "trigger_count": trig,
                         "source": "sweep/dyhpo_sampler.py check_and_extend_ranges defaults"}}
    for steps, sweep in SWEEPS.items():
        space = {x["name"]: x for x in yaml.safe_load(open(os.path.join(ROOT, SWEEP_CONFIGS[steps])))["search_space"]}
        lo, hi = float(space["training.lr"]["low"]), float(space["training.lr"]["high"])
        rows = []
        for key, r in results.items():
            sw, hp = key.split("/")
            if sw != sweep or r.get("val_loss") is None:
                continue
            t = next((v for v in tr.values() if v["sweep"] == sweep and re.match(rf"trial_{hp}(_r\d+)?$", v["trial"])),
                     None)
            if t is None or t.get("lr") is None or not r.get("validate_every"):
                sys.exit(f"best: {sweep} hp{hp}: no lr (trial dir not pulled) or no validate_every in its result")
            pos = math.log(t["lr"] / lo) / math.log(hi / lo)
            rows.append({"hp": hp, "val_loss": r["val_loss"], "best_step": r["best_step"], "lr": t["lr"],
                         "lr_log_position": round(pos, 4),
                         "best_at_last_validation": r["best_step"] + r["validate_every"] > int(steps),
                         "traintime_hours": r.get("traintime_hours")})
        rows.sort(key=lambda x: x["val_loss"])
        b, top = (rows[0] if rows else None), rows[:top_k]
        edge = lambda x: x["lr_log_position"] >= 1 - frac or x["lr_log_position"] <= frac
        flags = []
        if b is not None:
            if edge(b):
                flags.append(f"best lr in the window's outer {frac:g} (log position {b['lr_log_position']})")
            n_edge = sum(edge(x) for x in top)
            if n_edge >= trig:
                flags.append(f"{n_edge} of the top {len(top)} lr in the outer {frac:g}")
            n_last = sum(x["best_at_last_validation"] for x in top)
            if b["best_at_last_validation"] or 2 * n_last > len(top):
                flags.append(f"{n_last} of the top {len(top)} best at their last validation, the best included"
                             if b["best_at_last_validation"] else f"{n_last} of the top {len(top)} best at their last validation")
        out[steps] = {"sweep": sweep, "lr_window": [lo, hi], "n_finished": len(rows), "trials": rows, "best": b,
                      "not_converged": flags if b is not None else None}
    json.dump(out, open(os.path.join(DATA, "long_sweeps_best.json"), "w"), indent=1)
    print(json.dumps({k: {"n": v["n_finished"], "best": v["best"], "not_converged": v["not_converged"]}
                      for k, v in out.items() if k != "edge_rule"}))


def best(vals):
    fin = [(s, x) for s, x in vals if math.isfinite(x)]
    return min(fin, key=lambda z: z[1])


def width_curve(run):
    return json.load(open(os.path.join(DATA, "reads", f"{run}_val.json")))["blocks"][0]["vals"]


def lr_tex(x):
    e = int(math.floor(math.log10(x)))
    return rf"lr ${x / 10 ** e:.1f}\times10^{{{e}}}$"


def figs():
    sys.path.insert(0, ROOT)
    import plot_style as ps
    from matplotlib.lines import Line2D
    w = json.load(open(os.path.join(DATA, "width_x1.json")))
    tr = json.load(open(os.path.join(DATA, "size_curves.json")))["trials"]
    ep = json.load(open(os.path.join(DATA, "epochs.json")))
    ep_x1 = {round(ep[r]["epochs"], 4) for runs in WIDTH_RUNS.values() for r in runs}
    assert len(ep_x1) == 1, ep_x1
    ep_x1 = ep_x1.pop()
    ep_of = {k: t["steps"] / t["batches"] for k, t in tr.items()}
    os.makedirs(FIGS, exist_ok=True)
    report = {}

    # 1: best val_loss_no_reg against width, one point per seed
    fig, ax = ps.figure()
    for i, (seed, c, mk) in enumerate(((42, ps.C.blue, "o"), (43, ps.C.vermillion, "s"))):
        ys = [w["best"][WIDTH_RUNS[k][i]]["val_loss_no_reg"] for k in (8, 16, 32)]
        ax.plot([8, 16, 32], ys, marker=mk, color=c, label=f"seed {seed}")
        report[f"width seed {seed}"] = [round(y, 4) for y in ys]
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32], ["8", "16", "32"])
    ax.minorticks_off()
    ax.set_xlabel(f"width (attention heads), 4264 steps ({ep_x1:.2g} epochs)")
    ax.set_ylabel("best validation NLL")
    ps.legend(ax, "upper right")
    ps.save(fig, os.path.join(FIGS, "width_x1_loss"))

    # 2: validation curves at width 8: the 4264-step run (seed 42) and every validated sweep trial (seed 42)
    groups = {"17056": sorted((k for k in tr if tr[k]["steps"] == 17056), key=lambda k: tr[k]["lr"]),
              "68224": sorted((k for k in tr if tr[k]["steps"] == 68224), key=lambda k: tr[k]["lr"])}
    pal = {g: ps.sequence(len(groups[g]), cm, lo=0.45, hi=0.95)
           for g, cm in (("17056", "Blues"), ("68224", "Oranges"))}
    colour = {k: pal[g][i] for g, ks in groups.items() for i, k in enumerate(ks)}
    style = {"17056": "-", "68224": "--"}
    fig, ax = ps.figure()
    off = {}

    def draw(vals, label, **kw):
        s = np.array([v[0] for v in vals], float); y = np.array([v[1] for v in vals], float)
        bad = ~(np.isfinite(y) & (y <= OFF_SCALE))
        off[label] = int(bad.sum())
        y[bad] = np.nan
        ax.plot(s, y, label=label, **kw)

    draw(width_curve(WIDTH_RUNS[8][0]), f"4264 steps ({ep_x1:.2g} epochs), {lr_tex(LR_X1)}, done", color="k",
         ls=":")
    for g in ("17056", "68224"):
        for k in groups[g]:
            t = tr[k]
            state = "done" if t["finished"] else f"running, at step {t['vals'][-1][0]}"
            draw(t["vals"], f"{g} steps ({ep_of[k]:.2g} epochs), {lr_tex(t['lr'])}, {state}", color=colour[k],
                 ls=style[g])
            report[f"curve {k}"] = {"lr": t["lr"], "last_step": t["vals"][-1][0], "finished": t["finished"],
                                    "best": round(best(t["vals"])[1], 4)}
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("validation NLL")
    ps.shared_legend(fig, ax, ncol=1)
    ps.save(fig, os.path.join(FIGS, "length_w8_curves"))
    report["validations above NLL 1 not drawn"] = off

    # 3: best validation NLL against the estimated compute C = 6 N S B
    fig, ax = ps.figure()
    C = lambda width, steps: 6.0 * N_PARAMS[width] * steps * BATCH
    wcol = {8: "k", 16: ps.C.green, 32: ps.C.vermillion}
    hs = []
    for k in (8, 16, 32):
        ys = [w["best"][r]["val_loss_no_reg"] for r in WIDTH_RUNS[k]]
        ax.plot([C(k, 4264)] * 2, ys, ls="", marker="D", color=wcol[k])
        hs.append(Line2D([], [], ls="", marker="D", color=wcol[k],
                         label=f"width {k}, 4264 steps ({ep_x1:.2g} epochs), seeds 42 and 43"))
        report[f"compute width {k}"] = {"C": C(k, 4264), "best": ys}
    for g, mk, col in (("17056", "o", ps.C.blue), ("68224", "s", ps.C.orange)):
        n_run = 0
        for k in groups[g]:
            t = tr[k]
            s_best, y = best(t["vals"])
            done = t["finished"]
            n_run += not done
            c = C(8, t["steps"] if done else t["vals"][-1][0])
            ax.plot([c], [y], ls="", marker=mk, color=col, mfc=col if done else "none")
            report[f"compute {k}"] = {"C": c, "best": round(y, 4), "at_step": s_best, "finished": done}
        hs.append(Line2D([], [], ls="", marker=mk, color=col,
                         label=f"width 8, {g} steps ({ep_of[groups[g][0]]:.2g} epochs), {len(groups[g])} trials"
                               f" of the learning-rate sweep, {n_run} running"))
    hs.append(Line2D([], [], ls="", marker="o", color="k", label="trial done"))
    hs.append(Line2D([], [], ls="", marker="o", color="k", mfc="none",
                     label="trial running: best so far, at the compute so far"))
    for h in hs:            # empty proxies, read by ps.shared_legend
        ax.add_line(h)
    ax.set_xscale("log")
    ax.set_xlabel("estimated compute $C = 6\\,N\\,S\\,B$ [FLOP]\n"
                  "($N$ parameters, $S$ steps, $B = 16384$ events per step)")
    ax.set_ylabel("best validation NLL")
    ps.shared_legend(fig, ax, ncol=1)
    ps.save(fig, os.path.join(FIGS, "loss_vs_compute"))
    print(json.dumps(report))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pull", "figs", "best"))
    a = ap.parse_args()
    {"pull": pull, "figs": figs, "best": sweep_best}[a.cmd]()


if __name__ == "__main__":
    main()
