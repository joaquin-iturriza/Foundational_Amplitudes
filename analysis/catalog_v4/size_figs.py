#!/usr/bin/env python3
"""Width and length figures of plan entry catalog_coverage (steps 16 and 33), for the group meeting of 2026-11-03.

  pull    the validation curves (val_loss_no_reg of every 'Val loss (combined)' line of out_0.log) of the long-run
          DyHPO trials cov_v4_long_hpo8_cc_17056 (4x, 17056 steps) and cov_v4_long_hpo8_cc_68224 (16x, 68224 steps),
          trials 0-2 (the three random start-up points hp1860, hp0353, hp1333, the same three in both sweeps), read
          from CC-IN2P3 with one read-only `site run` -> data/size_curves.json (a trial without 'Finished training'
          is marked running, at its last validation)
  figs    figs/width_x1_loss, figs/width_x1_ratio (from data/width_x1.json, mixture_ab.py width),
          figs/length_w8_curves, figs/loss_vs_compute_preliminary (png + pdf)

Best checkpoint: the lowest aggregate val_loss_no_reg (het NLL) of the main training block. The 1x curves are the
width runs' own (data/reads/<run>_val.json, mixture_ab.py val). Compute: C = steps x N/N_8, N the learnable parameter
count of the run's model (from its log: width 8 1775829, 16 6516181, 32 25434069), i.e. the 6 N D estimate with the
same events per step in every run (batch 16384), in units of width-8 steps; attention over particles and gradient
checkpointing's recompute are not counted (an estimate).
"""
import argparse, json, math, os, re, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
SWEEPS = {"4x": "cov_v4_long_hpo8_cc_17056", "16x": "cov_v4_long_hpo8_cc_68224"}
STEPS = {"1x": 4264, "4x": 17056, "16x": 68224}
HPS = ("1860", "0353", "1333")
N_PARAMS = {8: 1775829, 16: 6516181, 32: 25434069}     # 'Instantiated model ... learnable parameters' of each log
WIDTH_RUNS = {8: ("FA-ccin2p3-804a65bc", "FA-ccin2p3-66e8c31f"), 16: ("FA-ccin2p3-493c98db", "FA-ccin2p3-7288b3fe"),
              32: ("FA-ccin2p3-e26059bf", "FA-ccin2p3-fd7fc137")}     # (seed 42, seed 43), mixture_ab.WIDTH
OFF_SCALE = 1.0          # validations above this NLL (the warm-up spikes, up to ~1e11) are not drawn, and counted


def pull():
    dirs = " ".join(f"runs/{SWEEPS[l]}/trial_{h}_r2" for l in SWEEPS for h in HPS)
    cmd = (f'for d in {dirs}; do echo "@@ $d"; grep -E "Starting to train for|Val loss \\(combined\\)|'
           f'Finished training" $d/out_0.log; done')
    out = subprocess.run(["site", "--timeout", "600", "run", "ccin2p3", "FA", "--", cmd], capture_output=True,
                         text=True, timeout=700).stdout
    res, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"@@ runs/(\S+)/trial_(\d+)_r2", line)
        if m:
            length = next(l for l, s in SWEEPS.items() if s == m.group(1))
            cur = res.setdefault(f"{length}_hp{m.group(2)}", {"sweep": m.group(1), "hp": m.group(2), "length": length,
                                                            "every": None, "vals": [], "finished": False})
            continue
        if cur is None:
            continue
        if "Starting to train for" in line:
            cur["every"] = int(re.search(r"validating every (\d+) steps", line).group(1))
        elif "Val loss (combined)" in line:
            x = float(re.search(r"val_loss_no_reg: ([0-9.eE+-]+|nan|inf|-inf)", line).group(1))
            cur["vals"].append([min(cur["every"] * (len(cur["vals"]) + 1), STEPS[cur["length"]]), x])
        elif "Finished training" in line:
            cur["finished"] = True
    if len(res) != 6 or any(not r["vals"] for r in res.values()):
        sys.exit("pull: missing curves: " + json.dumps({k: len(v["vals"]) for k, v in res.items()}))
    path = os.path.join(DATA, "size_curves.json")
    json.dump({"source": "out_0.log of each trial on ccin2p3, read-only site run", "trials": res}, open(path, "w"),
              indent=0)
    print(json.dumps({k: {"n": len(v["vals"]), "last_step": v["vals"][-1][0], "finished": v["finished"]}
                      for k, v in res.items()}))


def best(vals):
    fin = [(s, x) for s, x in vals if math.isfinite(x)]
    return min(fin, key=lambda z: z[1])


def width_curve(run):
    return json.load(open(os.path.join(DATA, "reads", f"{run}_val.json")))["blocks"][0]["vals"]


def figs():
    sys.path.insert(0, ROOT)
    import plot_style as ps
    w = json.load(open(os.path.join(DATA, "width_x1.json")))
    tr = json.load(open(os.path.join(DATA, "size_curves.json")))["trials"]
    col = {"1860": ps.C.blue, "0353": ps.C.vermillion, "1333": ps.C.green}
    os.makedirs(FIGS, exist_ok=True)
    report = {}

    # 1a: best val_loss_no_reg against width, one point per seed
    fig, ax = ps.figure()
    for i, (seed, c, mk) in enumerate(((42, ps.C.blue, "o"), (43, ps.C.vermillion, "s"))):
        ys = [w["best"][WIDTH_RUNS[k][i]]["val_loss_no_reg"] for k in (8, 16, 32)]
        ax.plot([8, 16, 32], ys, marker=mk, color=c, label=f"seed {seed}")
        report[f"1a seed {seed}"] = [round(y, 4) for y in ys]
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32], ["8", "16", "32"])
    ax.minorticks_off()
    ax.set_xlabel("width (attention heads)")
    ax.set_ylabel("best validation NLL")
    ps.legend(ax, "upper right")
    ps.save(fig, os.path.join(FIGS, "width_x1_loss"))

    # 1b: zero-shot ratio R over width 8, per read, widened 95% class-bootstrap intervals
    reads = (("PP4", "all", r"$pp$ (PP4)"), ("EE4", "all", r"$e^+e^-$ (EE4)"),
             ("held_default", "pooled", "held out, default-sampled"),
             ("held", "pooled", "held out, catalog-sampled"), ("HD5", "all", "one-loop held out (HD5)"))
    cols = (ps.C.blue, ps.C.vermillion, ps.C.green, ps.C.orange, ps.C.purple)
    fig, ax = ps.figure()
    ax.axhline(1.0, color=ps.C.grey, ls=":", label="width 8")
    for j, ((r, part, lab), c) in enumerate(zip(reads, cols)):
        xs, ys, lo, hi = [], [], [], []
        for k in (16, 32):
            x = w["arms"][f"w{k}"][r][part]
            xs.append(k * 2 ** ((j - 2) * 0.06)); ys.append(x["R"])
            lo.append(x["R"] - x["widened"][0]); hi.append(x["widened"][1] - x["R"])
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", color=c, ls="-", capsize=2, label=lab)
        report[f"1b {r}"] = ys
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([16, 32], ["16", "32"])
    ax.set_xlim(16 / 1.5, 32 * 1.5)
    ax.minorticks_off()
    ax.set_yticks([0.3, 0.5, 1.0], ["0.3", "0.5", "1"])
    ax.set_xlabel("width (attention heads)")
    ax.set_ylabel(r"$R$ = MSE / MSE(width 8)")
    ps.legend(ax, "lower left")
    ps.save(fig, os.path.join(FIGS, "width_x1_ratio"))

    # 2: validation curves at width 8: the 1x run (seed 42, the base config's HPs) and the 4x/16x trials
    fig, ax = ps.figure()
    off = {}
    def draw(vals, **kw):
        s = np.array([v[0] for v in vals], float); y = np.array([v[1] for v in vals], float)
        bad = ~(np.isfinite(y) & (y <= OFF_SCALE))
        off[kw["label"]] = int(bad.sum())
        y[bad] = np.nan
        ax.plot(s, y, **kw)
    from matplotlib.lines import Line2D
    draw(width_curve(WIDTH_RUNS[8][0]), color=ps.C.grey, label="1x")
    for length, ls in (("4x", "-"), ("16x", "--")):
        for h in HPS:
            t = tr[f"{length}_hp{h}"]
            draw(t["vals"], color=col[h], ls=ls, label=f"{length} hp{h}")
            report[f"2 {length} hp{h} last step"] = [t["vals"][-1][0], t["finished"]]
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("validation NLL")
    hs = [Line2D([], [], color=ps.C.grey, label="1x")] + \
        [Line2D([], [], color=col[h], label=f"hp{h}") for h in HPS] + \
        [Line2D([], [], color="k", ls="-", label="4x"), Line2D([], [], color="k", ls="--", label="16x, running")]
    ps.legend(ax, "lower left", handles=hs, ncol=2)
    ps.save(fig, os.path.join(FIGS, "length_w8_curves"))
    report["2 off-scale validations not drawn"] = off

    # 3: best loss against compute C = steps x N/N_8 (preliminary: the 16x trials are running)
    fig, ax = ps.figure()
    wcol = {8: ps.C.grey, 16: ps.C.orange, 32: ps.C.purple}
    for k in (8, 16, 32):
        ys = [w["best"][r]["val_loss_no_reg"] for r in WIDTH_RUNS[k]]
        ax.plot([STEPS["1x"] * N_PARAMS[k] / N_PARAMS[8]] * 2, ys, ls="", marker="D", color=wcol[k])
    for length, filled in (("4x", True), ("16x", False)):
        for h in HPS:
            t = tr[f"{length}_hp{h}"]
            s, y = best(t["vals"])
            c = (t["vals"][-1][0] if not t["finished"] else STEPS[length])
            ax.plot([c], [y], ls="", marker="o", color=col[h], mfc=col[h] if filled else "none")
            report[f"3 {length} hp{h}"] = {"C": c, "best": round(y, 4), "at_step": s}
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([5e3, 1e4, 2e4, 5e4], ["5k", "10k", "20k", "50k"])
    ax.set_xlabel(r"$C$ = steps $\times N / N_8$")
    ax.set_ylabel("best validation NLL")
    hs = [Line2D([], [], ls="", marker="D", color=wcol[k], label=f"1x w{k}") for k in (8, 16, 32)] + \
        [Line2D([], [], ls="", marker="o", color=col[h], label=f"hp{h}") for h in HPS] + \
        [Line2D([], [], ls="", marker="o", color="k", label="4x w8"),
         Line2D([], [], ls="", marker="o", color="k", mfc="none", label="16x w8, running")]
    ps.legend(ax, "lower left", handles=hs, ncol=2)
    ps.save(fig, os.path.join(FIGS, "loss_vs_compute_preliminary"))
    print(json.dumps(report))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pull", "figs"))
    a = ap.parse_args()
    pull() if a.cmd == "pull" else figs()


if __name__ == "__main__":
    main()
