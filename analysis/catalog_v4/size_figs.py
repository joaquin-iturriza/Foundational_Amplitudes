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
  pull_mse  the validation MSE of log|M|^2 at the best checkpoint, training wall time and GPU of every finished
          run (pull_mse's docstring) -> data/size_mse.json
  eval_mse <eval run> <log> <trial dir>   a sweep trial's MSE from the eval job that re-read it (DyHPO trials
          have no end-of-run evaluation) -> data/size_mse.json
  figs    figs/width_x1_loss (from data/width_x1.json, mixture_ab.py width), figs/length_w8_curves,
          figs/mse_vs_compute (MSE of log|M|^2 against compute and wall time, from data/size_mse.json) (png + pdf)

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


MSE_RE = re.compile(r"MSE \(prepd\) val_(\S+) \S+: ([0-9.eE+-]+|nan|inf)")


def pull_mse():
    """The validation MSE of log|M|^2 at the best checkpoint of every finished run of the width and length studies
    (the 4264-step width runs runs/catalog_v4_long/v4_het_default25_*_x1 and every trial of the two long sweeps),
    with its training wall time and GPU, read from CC-IN2P3 with one read-only `site run` -> data/size_mse.json.

    What the runs log: after training, the run reloads models/model_run0_best.pt (the best checkpoint, lowest
    aggregate val_loss_no_reg) and logs per process 'MSE (prepd) val_<process>' on its validation events, in the
    preprocessed target (experiment.py, the 'MSE (prepd)' line of the end-of-run metrics). The preprocessing
    (preprocess_per_dataset, experiment.py) takes plain log of a positive pool and signedlog at the pool's own scale of
    a signed one, then one shared standardisation (the run's data_stats.json prepd_std: one sigma for every process);
    the emission-scales target multiplies |M|^2 by a factor that is the same for truth and prediction, so on a
    positive pool the error in ln|M|^2 is sigma times the error in the preprocessed target and MSE = sigma^2 MSE(prepd).
    Signed pools exist only among the one-loop pools (names ending _nlo; 92 of the 1153 trained processes): those
    are left out of the aggregate (mse_stats), which is over the tree processes only. The first block of these lines in a log is the main run's (the second, after the
    250-step sigma-only stage, has mu frozen and repeats it). Wall time: the main block's 'Finished training ... in X
    min' (training only: no data load, no end-of-run evaluation); GPU: the run's RUN_SUMMARY 'gpu'. A trial that was
    restarted (trial_<n>_r<k>) counts only its own block, flagged."""
    dirs = ("runs/catalog_v4_long/v4_het_default25_*x1 " +
            " ".join(f"runs/{s}/trial_*" for s in SWEEPS.values()))
    cmd = (f'for d in {dirs}; do f=$d/out_0.log; grep -q "Finished training" $f 2>/dev/null || continue; '
           'echo "@@ $d"; python3 -c "import json,sys; s=set(json.load(open(sys.argv[1]))[\\"prepd_std\\"]); '
           'print(\\"DATA_STATS_SIGMA\\", s.pop() if len(s) == 1 else \\"multi\\")" $d/data_stats.json; '
           'grep -E "shared_standardization|Starting to train for|Finished training|'
           'Loading model from|MSE \\(prepd\\) val_" $f; done; '
           'echo "@@ RUN_SUMMARY"; cat catalog_v3_short_*.out ' + " ".join(f"sweeps/{s}/output/*.out" for s in SWEEPS.values())
           + ' 2>/dev/null | grep -h "^RUN_SUMMARY" | grep -E "catalog_v4_long/v4_het_default25|cov_v4_long_hpo8_cc_"')
    out = subprocess.run(["site", "--timeout", "600", "run", "ccin2p3", "FA", "--", cmd], capture_output=True,
                         text=True, timeout=700).stdout
    runs, cur, gpu = {}, None, {}
    for line in out.splitlines():
        if line.startswith("@@ "):
            name = line[3:].strip()
            cur = None if name == "RUN_SUMMARY" else runs.setdefault(name, {"blocks": [], "mse_prepd": {},
                                                                              "sigma": None, "best_loaded": None})
            continue
        if line.startswith("RUN_SUMMARY "):
            d = json.loads(line.split(" ", 1)[1])
            rd = d.get("run_dir", "")
            if d.get("T_max", 0) > 1000:            # the main run's summary, not the sigma stage's
                gpu[rd.split("Foundational_Amplitudes/", 1)[-1]] = d.get("gpu")
            continue
        if cur is None:
            continue
        if line.startswith("DATA_STATS_SIGMA "):        # the run's data_stats.json: one prepd_std for every process
            v = line.split()[1]
            cur["sigma"] = None if v == "multi" else float(v)
        m = re.search(r"Starting to train for (\d+) iterations", line)
        if m:
            cur["blocks"].append({"iterations": int(m.group(1)), "minutes": None})
        m = re.search(r"Finished training: (\d+) iterations .* in ([0-9.]+)min", line)
        if m and cur["blocks"]:
            cur["blocks"][-1]["minutes"] = float(m.group(2))
        if "Loading model from" in line and cur["best_loaded"] is None:
            cur["best_loaded"] = line.split("Loading model from", 1)[1].strip()
        m = MSE_RE.search(line)
        if m and m.group(1) not in cur["mse_prepd"]:     # the first evaluation: the main run's
            cur["mse_prepd"][m.group(1)] = float(m.group(2))
    res = {}
    for d, r in runs.items():
        main = [b for b in r["blocks"] if b["iterations"] > 1000 and b["minutes"] is not None]
        if not main:
            continue
        res[d] = {"sigma": r["sigma"], "iterations": main[0]["iterations"],
                  "train_hours": round(sum(b["minutes"] for b in main) / 60, 4), "n_main_blocks": len(main),
                  "gpu": gpu.get(d), "best_loaded": r["best_loaded"], "restarted": bool(re.search(r"_r\d+$", d)),
                  **mse_stats(r["mse_prepd"], r["sigma"])}
    if not res:
        sys.exit("pull_mse: nothing read: " + out[-800:])
    json.dump({"source": "out_0.log 'MSE (prepd) val_<process>' (main run, best checkpoint) times sigma^2 of its "
                         "shared standardisation (data_stats.json prepd_std), tree processes only (one-loop _nlo pools, "
                         "which hold every signed pool, left out); 'Finished training' minutes; RUN_SUMMARY gpu; CC-IN2P3, read-only "
                         "site run", "runs": res}, open(os.path.join(DATA, "size_mse.json"), "w"), indent=1)
    print(json.dumps({d: {k: v[k] for k in ("iterations", "train_hours", "gpu", "n_processes", "mse_logm2_gmean")}
                      for d, v in res.items()}))


def mse_stats(mse_prepd, sigma):
    """n_processes, geometric and arithmetic mean over the TREE processes (one-loop pools, which hold every signed
    pool, left out) of MSE(log|M|^2) = sigma^2 MSE(prepd); None when the
    log has no end-of-run evaluation (a DyHPO trial: no 'MSE (prepd)' lines)"""
    mse = [sigma ** 2 * x for p, x in mse_prepd.items() if math.isfinite(x) and not p.endswith("_nlo")] if sigma else []
    if not mse:
        return {"n_processes": 0, "mse_logm2_gmean": None, "mse_logm2_mean": None}
    return {"n_processes": len(mse), "n_one_loop_left_out": sum(p.endswith("_nlo") for p in mse_prepd), "mse_logm2_gmean": float(np.exp(np.mean(np.log(mse)))),
            "mse_logm2_mean": float(np.mean(mse))}


def eval_mse(run, log, trial_dir):
    """A sweep trial's validation MSE of log|M|^2 at its best checkpoint, from the eval job that re-read it (its
    LONG_BEST arm: the trial's config and data load, mu frozen at the trial's best, so its first end-of-run
    'MSE (prepd) val_' block is the trial's) -> added to data/size_mse.json under the trial's directory, with the
    eval run named"""
    sigma, mp = None, {}
    for line in open(log, errors="replace"):
        m = re.search(r"one \(mu, sigma\) = \(([-0-9.eE+]+), ([-0-9.eE+]+)\)", line)
        if m and sigma is None:
            sigma = float(m.group(2))
        m = MSE_RE.search(line)
        if m and m.group(1) not in mp:
            mp[m.group(1)] = float(m.group(2))
    path = os.path.join(DATA, "size_mse.json")
    d = json.load(open(path))
    t = d["runs"][trial_dir]
    if t.get("sigma") is None:
        sys.exit(f"eval_mse: no sigma for {trial_dir} (its data_stats.json was not read)")
    if sigma is not None and abs(t["sigma"] - sigma) > 1e-3:
        sys.exit(f"eval_mse: sigma {sigma} of {run} differs from the trial's {t['sigma']}")
    t.update(mse_stats(mp, t["sigma"]), mse_from_eval_run=run)   # the trial's sigma (its data_stats.json, equal
    # to the eval run's: the eval job checks 'data_stats.json ... equal the trial's')
    json.dump(d, open(path, "w"), indent=1)
    print(json.dumps({trial_dir: t}))


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
    ax.set_ylabel("best validation het. loss")
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

    draw(width_curve(WIDTH_RUNS[8][0]), f"4264 steps, {lr_tex(LR_X1)}", color="k",
         ls=":")
    for g in ("17056", "68224"):
        for k in groups[g]:
            t = tr[k]
            state = "" if t["finished"] else ", running"
            draw(t["vals"], f"{g} steps, {lr_tex(t['lr'])}{state}", color=colour[k],
                 ls=style[g])
            report[f"curve {k}"] = {"lr": t["lr"], "last_step": t["vals"][-1][0], "finished": t["finished"],
                                    "best": round(best(t["vals"])[1], 4)}
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("validation het. loss")
    ps.shared_legend(fig, ax, ncol=2)
    ps.save(fig, os.path.join(FIGS, "length_w8_curves"))
    report["validations above NLL 1 not drawn"] = off

    # 3: validation MSE of log|M|^2 at the best checkpoint (data/size_mse.json, pull_mse / eval_mse) against the
    # estimated compute C = 6 N S B (a) and the training wall time on one GPU (b); per category the best run on that
    # MSE (the width cells: the better of seeds 42 and 43; the 17056-step sweep: its best trial by the sweep's
    # objective, the only trial with an end-of-run evaluation, from its eval re-read)
    sm = json.load(open(os.path.join(DATA, "size_mse.json")))["runs"]
    C = lambda width, steps: 6.0 * N_PARAMS[width] * steps * BATCH
    cats = [(f"width {k}, 4264 steps ({ep_x1:.2g} epochs)", k,
             [f"runs/catalog_v4_long/v4_het_default25{sfx}_w{k}_x1" for sfx in ("", "_seed2")], wc, "D")
            for k, wc in ((8, "k"), (16, ps.C.green), (32, ps.C.vermillion))]
    b17 = json.load(open(os.path.join(DATA, "long_sweeps_best.json")))["17056"]["best"]["hp"]
    cats.append((f"width 8, 17056 steps ({ep_of['17056_trial_' + b17]:.2g} epochs), best of the 8-trial sweep", 8,
                 [f"runs/{SWEEPS['17056']}/trial_{b17}"], ps.C.blue, "o"))
    gpu_fill = {"Tesla V100-SXM2-32GB": True, "Tesla V100-PCIE-32GB": False}
    fig, (axc, axw) = ps.figure(ncols=2)
    hs = []
    for label, width, dirs, col, mk in cats:
        have = [(sm[d]["mse_logm2_gmean"], d) for d in dirs if sm.get(d, {}).get("mse_logm2_gmean") is not None]
        y, d = min(have)
        r = sm[d]
        fill = gpu_fill[r["gpu"]]
        for ax, x in ((axc, C(width, r["iterations"])), (axw, r["train_hours"])):
            ax.plot([x], [y], ls="", marker=mk, color=col, mfc=col if fill else "none")
        hs.append(Line2D([], [], ls="", marker=mk, color=col, label=label))
        report[f"mse {label}"] = {"run_dir": d, "mse_logm2_gmean": round(y, 5), "C": C(width, r["iterations"]),
                                  "train_hours": r["train_hours"], "gpu": r["gpu"],
                                  "of": {x: sm.get(x, {}).get("mse_logm2_gmean") for x in dirs}}
    hs += [Line2D([], [], ls="", marker="s", color="k", label="V100 SXM2"),
           Line2D([], [], ls="", marker="s", color="k", mfc="none", label="V100 PCIe")]
    for ax in (axc, axw):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"validation MSE of $\log|\mathcal{M}|^2$, trees")
    axc.set_xlabel("compute [FLOP]")
    axw.set_xlabel("training wall time [GPU h]")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    for ax, ticks in ((axw, (1, 2, 4)), (axc, None)):
        if ticks:
            ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
            ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1]))
        ax.yaxis.set_major_formatter(FixedFormatter(["0.02", "0.05", "0.1"]))
        ax.yaxis.set_minor_formatter(NullFormatter())
    for h in hs:            # empty proxies, read by ps.shared_legend
        axc.add_line(h)
    ps.shared_legend(fig, axc, ncol=2)
    ps.save(fig, os.path.join(FIGS, "mse_vs_compute"))
    print(json.dumps(report))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pull", "pull_mse", "eval_mse", "figs", "best"))
    ap.add_argument("args", nargs="*", help="eval_mse: <eval run> <its log> <trial dir>")
    a = ap.parse_args()
    if a.cmd == "eval_mse":
        return eval_mse(*a.args)
    {"pull": pull, "pull_mse": pull_mse, "figs": figs, "best": sweep_best}[a.cmd]()


if __name__ == "__main__":
    main()
