"""critical_batch: the batch-size scan's readout (McCandlish et al. 2018, arXiv:1812.06162).

    python analysis/critical_batch/scan_state.py --collect   # reads the cbs_b<B> sweeps on CC (a CPU job: call twice)
    python analysis/critical_batch/scan_state.py             # reads data/cbs_state.json, draws figs/cbs_readout

Each trial's validation curve (its result.json, its log's where it left none: timed out or diverged) and lr go to
cbs_read.read (data/cbs_trials.json -> data/cbs_readout.json): per target, the best trial's steps S_B, the lr-edge
rule, S = S_min (1 + B_crit / B) with its bootstrap. Figure: (a) S_B against B per target with the fits; (b) seconds
per training step against B per GPU model, from the validation timestamps of each trial's log (validation included),
beside the design's cost model 0.12 s + 0.86 s x B / 16384; (c) B_crit per target against the noise scale B_simple
of the noise-scale run (all windows, noise_read.py's rule) and the canonical batch."""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sweep"))
DATA = f"{HERE}/data/cbs_state.json"
BATCHES = (2048, 4096, 8192, 16384, 32768)
TARGET = -2.08          # the design's objective target
EVERY = 85              # validate_every_n_steps of every cbs sweep config

REMOTE = r'''
import glob, gzip, json, os, re, sys
from datetime import datetime
out = {}
for B in sys.argv[1:]:
    d = f"sweeps/cbs_b{B}"
    if not os.path.isdir(d):
        continue
    res = {}
    for f in glob.glob(f"{d}/results/*.json"):
        r = json.load(open(f))
        res[int(re.match(r"hp(\d+)_", os.path.basename(f)).group(1))] = {"val_curve": r["val_curve"],
                                                                        "validate_every": r["validate_every"]}
    trials = []
    rd = lambda p: (gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")).read()
    for f in sorted(glob.glob(f"{d}/error/*.err*")):   # finished sweeps keep their logs gzipped
        o = next((x for x in (f.replace("/error/", "/output/").replace(".err", ".out"),
                              f.replace("/error/", "/output/").replace(".err.gz", ".out")) if os.path.exists(x)), None)
        txt = rd(f)
        summ = next((json.loads(l.split(" ", 1)[1]) for l in rd(o).splitlines() if l.startswith("RUN_SUMMARY ")),
                    {}) if o else {}
        dv = re.search(r"non-finite-gradient steps \(patience \d+\) at step (\d+)", txt)
        m = re.search(r"trial_(\d+)$", summ.get("run_dir", ""))
        vals = re.findall(r"^\[(\S+ \S+) I\] Val loss \(combined\): \S+ \| val_loss_no_reg: (\S+)", txt, re.M)
        ts = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp() for t, _ in vals]
        trials.append({"log": os.path.basename(f), "hp": int(m.group(1)) if m else None, "gpu": summ.get("gpu"),
                       "lr": summ.get("lr"), "log_curve": [float(v) for _, v in vals],
                       "sec_per_step": (ts[-1] - ts[0]) / ((len(ts) - 1) * {EVERY}) if len(ts) > 2 else None,
                       "stopped": "STOP_AT_VAL_LOSS" in txt, "diverged": int(dv.group(1)) if dv else None,
                       "timeout": "DUE TO TIME LIMIT" in txt})
    out[B] = {"results": res, "trials": trials}
print("CBSSTATE " + json.dumps(out))
'''.replace("{EVERY}", str(EVERY))


def collect():
    from analysis.common.readers import site_python
    d = site_python(REMOTE, [str(b) for b in BATCHES], "CBSSTATE")
    json.dump(d, open(DATA, "w"))
    print(f"wrote {DATA}: " + ", ".join(f"{b}: {len(v['trials'])} logs, {len(v['results'])} results"
                                        for b, v in d.items()))


def trials_for_read(d):
    """cbs_read's input {B: [trial]}: each logged trial's result curve, its log's curve where it left no result
    (timed out or diverged), its lr from RUN_SUMMARY, where it diverged and whether it timed out."""
    out = {}
    for b, v in d.items():
        ts = []
        for t in v["trials"]:
            if t["hp"] is None:
                continue
            r = v["results"].get(str(t["hp"]))
            ts.append({"hp_idx": t["hp"], "lr": t["lr"], "validate_every": r["validate_every"] if r else EVERY,
                       "val_curve": r["val_curve"] if r else t["log_curve"], "diverged_at": t["diverged"],
                       "timeout": t["timeout"], "from_log": r is None})
        out[b] = ts
    return out


def b_simple():
    """B_simple over all windows of the noise-scale run (noise_read.py's summary rule): median and quartiles."""
    w = json.load(open(f"{HERE}/data/v4_het_noise_scale.json"))["windows"]
    Bs = [x["B_simple"] for x in w if x["B_simple"] is not None]
    return [float(np.percentile(Bs, q)) for q in (25, 50, 75)]


def draw():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import plot_style as ps
    from cbs_read import read, TARGETS
    d = json.load(open(DATA))
    tr = trials_for_read(d)
    json.dump(tr, open(f"{HERE}/data/cbs_trials.json", "w"))
    res = read(tr)
    json.dump(res, open(f"{HERE}/data/cbs_readout.json", "w"), indent=1)
    for b, ts in sorted(tr.items(), key=lambda x: int(x[0])):
        print(f"B {b}: {len(ts)} trials; diverged " + ", ".join(f"hp {t['hp_idx']} at step {t['diverged_at']}"
              for t in ts if t["diverged_at"]) + "; timed out " + ", ".join(
              f"hp {t['hp_idx']} at step {len(t['val_curve']) * t['validate_every']}" for t in ts if t["timeout"]))
    # seconds per step, median per batch and GPU model
    sps = {}
    for b, v in d.items():
        for t in v["trials"]:
            if t["sec_per_step"] and t["gpu"]:
                sps.setdefault("H100" if "H100" in t["gpu"] else "V100", {}).setdefault(int(b), []).append(t["sec_per_step"])
    med = {g: {b: float(np.median(x)) for b, x in v.items()} for g, v in sps.items()}
    for g, v in sorted(med.items()):
        print(f"{g} s/step median: " + ", ".join(f"{b} {x:.3f} ({len(sps[g][b])})" for b, x in sorted(v.items())))
    q25, q50, q75 = b_simple()
    print(f"B_simple quartiles {q25:.1f} / median {q50:.1f} / {q75:.1f}")
    for L in TARGETS:
        r = res[str(L)]
        print(f"L {L}: " + "; ".join(f"{x['B']} S {x.get('S') and round(x['S'])} lr {x.get('lr')} edge {x.get('lr_edge')}"
                                     f" reached {x['reached']}/{x['trials']}" + (f" wall(V100) {x['S'] * med['V100'][x['B']] / 3600:.2f} h"
                                     if x.get("S") and x["B"] in med.get("V100", {}) else "") for x in r["batches"]))
        if "B_crit" in r:
            print(f"  fit S_min {r['S_min']:.0f} B_crit {r['B_crit']:.0f} q16/50/84 {[round(q) for q in r['B_crit_q16_50_84']]}"
                  f" B_crit/B_simple_median {r['B_crit'] / q50:.0f}; S over fit: " + ", ".join(
                  f"{x['B']} {x['S'] / (r['S_min'] * (1 + r['B_crit'] / x['B'])):.3f}" for x in r["batches"]
                  if x.get("lr_edge") is False) + f"; S2048/S32768 {r['S_2048_over_S_32768']}; unreached_above_fit "
                  f"{r['unreached_above_fit']}")
    fig, axs = ps.figure(ncols=2, nrows=2)
    axS, axT, axC, cell = axs[0][0], axs[0][1], axs[1][0], axs[1][1]
    cols = (ps.C.blue, ps.C.vermillion, ps.C.green)
    for L, c in zip(TARGETS, cols):
        r = res[str(L)]
        fit = [(x["B"], x["S"]) for x in r["batches"] if x.get("lr_edge") is False]
        edge = [(x["B"], x["S"]) for x in r["batches"] if x.get("lr_edge")]
        lb = [(x["B"], x["S_lower_bound"]) for x in r["batches"] if x.get("S_lower_bound")]
        axS.plot(*zip(*fit), "o", color=c, label=rf"$\mathcal{{L}}_{{\rm val}}={L}$")
        if edge:
            axS.plot(*zip(*edge), "o", mfc="none", color=c, label="best lr at the window edge")
        if lb:
            axS.plot(*zip(*lb), "^", mfc="none", color=c, label="target not reached")
        if "B_crit" in r:
            x = np.geomspace(min(fit)[0], max(fit)[0], 100)
            axS.plot(x, r["S_min"] * (1 + r["B_crit"] / x), "-", color=c)
            lo, mid, hi = r["B_crit_q16_50_84"]
            axC.errorbar([L], [r["B_crit"]], yerr=[[r["B_crit"] - lo], [hi - r["B_crit"]]], fmt="o", color=c)
    axS.plot([], [], "-", color=ps.C.grey, label=r"fit $S_{\min}(1+B_{\rm crit}/B)$")
    axS.set_yscale("log")
    axS.set_xlabel(r"batch size $B$")
    axS.set_ylabel(r"steps $S$ to $\mathcal{L}_{\rm val}$")
    for (g, v), c in zip(sorted(sps.items(), key=lambda x: x[0] != "V100"), (ps.C.blue, ps.C.green)):
        xs = [(b, y) for b, ys in v.items() for y in ys]
        axT.plot(*zip(*xs), "o", color=c, alpha=0.6, label=f"measured, {g}")
    x = np.geomspace(1500, 40000, 100)
    axT.plot(x, 0.12 + 0.86 * x / 16384, "--", color=ps.C.vermillion, label="design's cost model")
    axT.set_yscale("log")
    axT.set_xlabel(r"batch size $B$")
    axT.set_ylabel("seconds per training step")
    ps.legend(axT, "upper left")
    for ax in (axS, axT):   # the scanned batches as powers of two: one-line ticks that do not collide
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCHES, [rf"$2^{{{int(np.log2(b))}}}$" for b in BATCHES])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    axC.axhspan(q25, q75, color=ps.C.grey, alpha=0.3, label=r"noise scale $B_{\rm simple}$, quartiles")
    axC.axhline(q50, color=ps.C.grey, label=r"noise scale $B_{\rm simple}$, median")
    axC.axhline(16384, color=ps.C.purple, ls=":", label="canonical batch 16384")
    axC.errorbar([], [], yerr=[], fmt="o", color="k", label=r"$B_{\rm crit}$, 16-84% bootstrap")
    axC.set_yscale("log")
    axC.set_ylim(q25 / 2, 2 * 16384)   # by rule: the noise scale's band and the canonical batch both on the frame
    axC.set_xlabel(r"target $\mathcal{L}_{\rm val}$")
    axC.set_ylabel(r"batch size $B$")
    axC.set_xticks(TARGETS, [str(L) for L in TARGETS])
    ps.legend_cell(cell, [axS, axC])
    ps.save(fig, f"{HERE}/figs/cbs_readout")
    # the summary the notes' claims key into: per target (t166, t208, t230) the fit, its bootstrap, the design's ratio
    # S(2048)/S(32768), B_crit over the noise scale's median, each batch's best S and V100 hours to the target
    r3 = lambda v: float(f"{v:.3g}")
    summ = {"Bsimple": {"q25": r3(q25), "median": r3(q50), "q75": r3(q75)},
            "v100": {str(b): r3(x) for b, x in med.get("V100", {}).items()},
            "n_diverged": sum(bool(t["diverged_at"]) for ts in tr.values() for t in ts),
            "n_timeout": sum(t["timeout"] for ts in tr.values() for t in ts),
            "diverged_at": {b: sorted(t["diverged_at"] for t in ts if t["diverged_at"]) for b, ts in tr.items()},
            "timeout_at": {b: sorted(len(t["val_curve"]) * t["validate_every"] for t in ts if t["timeout"])
                           for b, ts in tr.items()}}
    for L in TARGETS:
        r, k = res[str(L)], "t%d" % round(-100 * L)
        summ[k] = {"B_crit": r3(r["B_crit"]), "q16": r3(r["B_crit_q16_50_84"][0]), "q84": r3(r["B_crit_q16_50_84"][2]),
                   "S_min": r3(r["S_min"]), "ratio": r3(r["S_2048_over_S_32768"]), "over_Bsimple": r3(r["B_crit"] / q50),
                   "S": {str(x["B"]): r3(x["S"]) for x in r["batches"] if x.get("S")},
                   "hours": {str(x["B"]): r3(x["S"] * med["V100"][x["B"]] / 3600) for x in r["batches"]
                             if x.get("S") and x["B"] in med.get("V100", {})}}
    print(json.dumps(summ))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    collect() if a.collect else draw()
