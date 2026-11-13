"""Plan optimizations_recap step 3, design passed 2026-11-12 00:25: Muon against AdamW on the longest catalog_v4
pretraining (width 8, 16x = 68224 steps, the whole training set, trial 1262's HPs, seed 42; only training.optimizer
differs).

  curves   both runs' validation curves (result.json on CC: val_mse_curve = the validation mu-MSE, 50 points every
           1364 steps) -> data/muon_long_curves.json
  read     -> data/muon_long.json:
           curve: AdamW/Muon at every validation; steps-to-level (the first validation at which AdamW's running best
             reaches Muon's running best at 8528, 17056, 34112, 68224 steps, None if never); the plateau rule per run
             (best within 10% of its value at the validation nearest 0.75 T, or the best before 0.9 T)
           reads: R = median over classes of the class geometric mean of MSE_AdamW/MSE_Muon of ln|M|^2 (each run's
             final model), interval widened by the width-8 seed pair's |ln R_seed| (mixture_ab.compare), per read
             PP4, EE4, held out pooled and per axis, HD4, HD5; the share of classes where Muon is lower
           level: each run's median over classes of the class gmean MSE of ln|M|^2 on each read (against 1e-4)
  hists <run> <log>   the PP4 and EE4 ZERO_SHOT_HIST records of a run's log (edges, truth and predicted counts of
           ln|M|^2 on the probe's test pool) -> data/reads/<run>_hists.json
  plot     -> figs/muon_long_curves.{png,pdf} (the two validation curves), figs/muon_long_levels.{png,pdf} (each run's
           per-class MSE per read: median and quartiles over classes, reads ordered by Muon's median),
           figs/muon_long_examples.{png,pdf} (true and predicted ln|M|^2 of the PP4 probes at Muon's 10th, 50th and
           90th percentile of MSE, ratio predicted/true below)
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import mixture_ab as m

RUNS = {"adamw": "FA-ccin2p3-f08e90e8", "muon": "FA-ccin2p3-1a46f3e0"}     # muon: the reads of trial 1262 (par:v4_length)
RESULTS = {"adamw": "runs/catalog_v4_long/v4_het_default25_w8_x16_hp1262_adamw/result.json",
           "muon": "sweeps/cov_v4_long_hpo8_cc_68224/results/hp1262_t68224_1793686584.json"}
T, LEVELS = 68224, (8528, 17056, 34112, 68224)
READS = ("PP4", "EE4", "held", "HD4", "HD5")
CURVES = os.path.join(m.DATA, "muon_long_curves.json")
OUT = os.path.join(m.DATA, "muon_long.json")

REMOTE = """
import json, os, sys
P = os.environ.get("PROJECT_DIR", ".")
out = {}
for k, f in json.loads(sys.argv[1]).items():
    d = json.load(open(os.path.join(P, f)))
    out[k] = {x: d[x] for x in ("val_mse_curve", "val_curve", "validate_every", "best_step", "traintime_hours")}
print("MUONLONG " + json.dumps(out))
"""


def curves():
    from analysis.common.readers import site_python
    d = site_python(REMOTE, ["'" + json.dumps(RESULTS) + "'"], "MUONLONG")
    d["paths"] = RESULTS
    json.dump(d, open(CURVES, "w"), indent=1)
    print({k: len(v["val_mse_curve"]) for k, v in d.items() if k != "paths"})


def _steps(c):
    return np.arange(1, len(c["val_mse_curve"]) + 1) * c["validate_every"]


def plateau(c):
    s, v = _steps(c), np.array(c["val_mse_curve"])
    i75 = int(np.argmin(abs(s - 0.75 * T)))
    best, ib = float(v.min()), int(v.argmin())
    return {"best": best, "best_step": int(s[ib]), "at_0p75T": float(v[i75]), "step_0p75T": int(s[i75]),
            "best_over_0p75T": round(best / float(v[i75]), 4),
            "plateaued": bool(best >= 0.9 * float(v[i75]) or s[ib] < 0.9 * T)}


def read():
    cv = json.load(open(CURVES))
    a, mu = cv["adamw"], cv["muon"]
    s = _steps(mu)
    va, vm = np.array(a["val_mse_curve"]), np.array(mu["val_mse_curve"])
    ra, rm = np.minimum.accumulate(va), np.minimum.accumulate(vm)
    to_level = {}
    for L in LEVELS:
        i = int(np.argmin(abs(s - L)))
        hit = np.nonzero(ra <= rm[i])[0]
        to_level[L] = {"muon_step": int(s[i]), "muon_best_so_far": float(rm[i]),
                       "adamw_step": int(s[hit[0]]) if len(hit) else None}
    out = {"design": "plan optimizations_recap step 3, design passed 2026-11-12 00:25", "runs": RUNS,
           "curve": {"steps": s.tolist(), "ratio_adamw_over_muon": (va / vm).round(4).tolist(),
                     "ratio_running_best": (ra / rm).round(4).tolist(),
                     "median_ratio": round(float(np.median(va / vm)), 4), "steps_to_level": to_level,
                     "plateau": {"adamw": plateau(a), "muon": plateau(mu)},
                     "last_ratio": round(float(va[-1] / vm[-1]), 4),
                     "runs": {k: {"first": c["val_mse_curve"][0], "last": c["val_mse_curve"][-1],
                                  "max_after_best": max(c["val_mse_curve"][int(np.argmin(c["val_mse_curve"])):]),
                                  "traintime_hours": round(c["traintime_hours"], 2)} for k, c in cv.items() if k != "paths"}},
           "seed_pair": m.WIDTH["w8"], "reads": {}, "level": {}}
    cls = m.class_of()
    for r in READS:
        x = {k: m.read_of(run, r) for k, run in {**RUNS, **dict(zip(("c42", "c43"), m.WIDTH["w8"]))}.items()}
        miss = [k for k, v in x.items() if v is None]
        if miss:
            sys.exit(f"muon_long: {r} reads missing for {miss} (mixture_ab.py extract first)")
        keys = sorted(set(x["adamw"]) & set(x["muon"]) & set(x["c42"]) & set(x["c43"]), key=str)
        parts = {"all": keys}
        if r == "held":
            parts.update({ax: [k for k in keys if k[0] == ax] for ax in m.HELD})
        res = {n: m.compare(x["adamw"], x["muon"], x["c42"], x["c43"], ks, cls) for n, ks in parts.items() if ks}
        for v in res.values():
            v["frac_classes_muon_lower"] = round(1 - v["frac_classes_below_1"], 4)
        res["unmatched"] = len(set(x["adamw"]) ^ set(x["muon"]))
        out["reads"][r] = res
        out["level"][r] = {}
        for k in RUNS:
            pc = per_class(x[k], keys, cls)
            out["level"][r][k] = {"median_class_mse": round(float(np.median(list(pc.values()))), 6),
                                  "frac_classes_below_1e-4": round(float(np.mean(np.array(list(pc.values())) < 1e-4)), 4)}
    out["blowups"] = m.blowups([RUNS["adamw"]])
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    print(json.dumps(out, default=str))


def per_class(x, keys, cls):
    by = {}
    for k in keys:
        by.setdefault(cls(m.probe_name(k)), []).append(x[k])
    return {c: m.gmean(v) for c, v in by.items()}


def hists(run, log):
    out = {}
    for line in open(log, errors="replace"):
        for ax in ("PP4", "EE4"):
            tag = f"{ax} ZERO_SHOT_HIST "
            if line.startswith(tag):
                d = json.loads(line[len(tag):])
                out[f"{ax}:{d['probe']}"] = {k: d[k] for k in ("edges", "truth", "pred")}
    json.dump(out, open(os.path.join(m.READS, f"{run}_hists.json"), "w"))
    print(run, len(out))


def plot():
    import plot_style as ps
    cv, d = json.load(open(CURVES)), json.load(open(OUT))
    cls = m.class_of()
    fig, a1 = ps.figure(width="half")
    col = {"muon": ps.C.blue, "adamw": ps.C.vermillion}
    lab = {"muon": "Muon", "adamw": "AdamW"}
    for k in ("muon", "adamw"):
        a1.plot(_steps(cv[k]), cv[k]["val_mse_curve"], color=col[k], label=lab[k])
    a1.set_yscale("log"); a1.set_xlabel("step"); a1.set_ylabel(r"validation MSE of $\mu$")
    ps.legend(a1, "upper right")
    ps.save(fig, os.path.join(HERE, "figs", "muon_long_curves"))
    fig, a2 = ps.figure(width="half", figsize=ps.figsize(1, 2, "half"))
    # (b) per-class MSE of ln|M|^2 per read, reads ordered by Muon's median (best to worst)
    names = {"PP4": "PP4", "EE4": "EE4"}
    groups = [("PP4",), ("EE4",), ("HD4",), ("HD5",)] + [("held", ax) for ax in m.HELD]
    rows = []
    for g in groups:
        vals = {}
        for k, run in RUNS.items():
            x = m.read_of(run, g[0])
            if len(g) == 2:
                x = {kk: v for kk, v in x.items() if kk[0] == g[1]}
            vals[k] = per_class(x, list(x), cls)
        rows.append((names.get(g[0], g[-1]), vals))
    rows.sort(key=lambda r: np.median(list(r[1]["muon"].values())))
    for i, (_, vals) in enumerate(rows):
        for j, k in enumerate(("muon", "adamw")):
            v = np.array(list(vals[k].values()))
            q = np.percentile(v, [25, 50, 75])
            y = i + (j - 0.5) * 0.3
            a2.plot([q[0], q[2]], [y, y], color=col[k], lw=3, alpha=0.5)
            a2.plot(q[1], y, "o", color=col[k], label=lab[k] if i == 0 else None)
    a2.axvline(1e-4, color=ps.C.grey, ls="--", label=r"$10^{-4}$")
    a2.set_yticks(range(len(rows))); a2.set_yticklabels([r[0] for r in rows])
    a2.set_ylim(len(rows) - 0.5, -4.3); a2.set_xscale("log"); a2.set_xlabel(r"per-class MSE of $\ln|\mathcal{M}|^2$")
    ps.legend(a2, "upper right")
    ps.save(fig, os.path.join(HERE, "figs", "muon_long_levels"))
    examples(ps, col, lab)


def examples(ps, col, lab):
    h = {k: json.load(open(os.path.join(m.READS, f"{run}_hists.json"))) for k, run in RUNS.items()}
    mu = m.read_of(RUNS["muon"], "PP4")
    keys = sorted((p for p in mu if f"PP4:{p}" in h["muon"] and f"PP4:{p}" in h["adamw"]), key=lambda p: mu[p])
    pick = [keys[int(q * (len(keys) - 1))] for q in (0.1, 0.5, 0.9)]
    fig, axes = ps.figure(ncols=3, nrows=2, sharex="col", gridspec_kw={"height_ratios": [3, 1]})
    for j, p in enumerate(pick):
        # each run's record bins truth and prediction on its own edges (the range of both): each ratio on its own
        for k in ("muon", "adamw"):
            r = h[k][f"PP4:{p}"]
            e, t, q = np.array(r["edges"]), np.array(r["truth"], float), np.array(r["pred"], float)
            if k == "muon":
                axes[0, j].stairs(t, e, color="k", label="true")
            axes[0, j].stairs(q, e, color=col[k], label=lab[k])
            with np.errstate(divide="ignore", invalid="ignore"):
                axes[1, j].stairs(np.where(t > 0, q / t, np.nan), e, color=col[k])
        axes[1, j].axhline(1, color=ps.C.grey, ls="--")
        axes[1, j].set_ylim(0, 2)
        axes[0, j].set_yscale("log")
        axes[1, j].set_xlabel(r"$\ln|\mathcal{M}|^2$")
        ps.process_label(axes[0, j], p.replace("_", " "))
        print("example", p, {k: m.read_of(run, "PP4")[p] for k, run in RUNS.items()})
    axes[0, 0].set_ylabel("events"); axes[1, 0].set_ylabel("pred/true")
    ps.legend(axes[0, 0], "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "muon_long_examples"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("curves", "read", "hists", "plot"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    {"curves": curves, "read": read, "hists": hists, "plot": plot}[a.cmd](*a.args)
