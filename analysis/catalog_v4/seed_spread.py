#!/usr/bin/env python3
"""The seed-to-seed spread of the test error, as one relative number (plan catalog_coverage; Joaquin 2026-11-06 22:31,
D315: 'I see no reason to keep doing multiple seeds, we have plenty of multiple seeded runs and the variation seems
to be pretty consistent').

Selection: every pair of catalog_v4 runs already read in this repo whose two runs differ only in the seed (42 vs 43,
and 42..45 where four seeds exist, every pair of them), taken from the run tables the earlier comparisons used
(mixture_ab.py MIX, MIX_HET, HD4_*/HD5_*, BOUND*, WIDTH, LOADER, WARM; ee_gain.py BEAM_RUNS; het_oneloop.py SIGNHEAD);
no pair is chosen by its result. Reads: the per-probe zero-shot MSE of log|M|^2 at the best checkpoint
(data/reads/<run>_<axis>.json, mse_logm2_real) on PP4 (pp test pools), EE4 (e+e- test pools), the catalog-sampled
held-out axes H1-H7c pooled ('held'), HD4 and HD5 (default-sampled held-out tree and one-loop pools).

Measure, per pair and read: per class (census_classes.json) the geometric mean over its probes of MSE_A / MSE_B,
then over classes
  class spread    median and 90th percentile of |ln(MSE_A / MSE_B)|        (what one class's error moves with seed)
  aggregate       |ln R_seed|, R_seed = the median over classes of the class ratio (what a class-median R moves)
A pair already taken under another set (the warm-start controls shared by several changes) is counted once.
A pair whose training run blew up (mixture_ab.blowups) is listed apart and left out of the summary.
Summary: per read and over everything, the median over pairs of each pair's value (a four-seed set's six pairs
count as one pair, by their median), by loss (MSE / het) and data setup; relative = exp(x) - 1.

  python3 analysis/catalog_v4/seed_spread.py run    -> data/seed_spread.json
  python3 analysis/catalog_v4/seed_spread.py fig    -> figs/seed_spread.{png,pdf} (from the json)
"""
import itertools, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import mixture_ab as mx  # noqa: E402

READS = ("PP4", "EE4", "held", "HD4", "HD5")
OUT = os.path.join(HERE, "data", "seed_spread.json")


def seed_sets():
    """[(name, loss, setup, {read: (run per seed, ...)})]: the runs of one arm at different seeds, per read."""
    import ee_gain, het_oneloop
    S = []
    short = "1e4 events/process, 4264 steps"
    for arm in ("control", "default50", "default25"):
        S.append((f"mse_{arm}", "MSE", short, dict(mx.MIX[arm], HD4=mx.HD4_MSE[arm], HD5=mx.HD5_MSE[arm])))
        S.append((f"het_{arm}", "het", short, dict(mx.MIX_HET[arm], HD4=mx.HD4_HET[arm], HD5=mx.HD5_HET[arm])))
    S.append(("mse_bound_pure_default", "MSE", short,
              {"PP4": mx.BOUND_DEFAULT["PP4"], "EE4": mx.BOUND_DEFAULT["EE4"], "HD4": mx.BOUND_DEFAULT["HD"],
               "HD5": mx.BOUND_DEFAULT["HD"]}))
    S.append(("mse_control_seeds42-45", "MSE", short, {"HD5": het_oneloop.SIGNHEAD["mse_hd5"]}))
    S.append(("het_signhead_seeds42-45", "het", short, {r: het_oneloop.SIGNHEAD["arm"] for r in READS}))
    S.append(("het_magnitude_seeds42-45", "het", short, {r: het_oneloop.SIGNHEAD["ref"] for r in READS}))
    for arm, runs in ee_gain.BEAM_RUNS.items():
        S.append((f"het_beams_{arm}", "het", short, {r: runs for r in READS}))
    for change, v in mx.WARM.items():
        for part in ("W", "C"):
            for k, runs in v.get(part, {}).items():
                if isinstance(runs, tuple) and len(runs) >= 2:
                    S.append((f"het_warm_{change}_{part}{k}", "het", "warm start, k steps from the 4264-step parent",
                              {r: runs for r in READS}))
    for w, runs in mx.WIDTH.items():
        S.append((f"het_whole_{w}", "het", f"whole set (1e5/process), width {w[1:]}, 4264 steps, memmap",
                  {r: runs for r in READS}))
    S.append(("het_whole_w8_inmemory_rerun", "het", "whole set (1e5/process), width 8, 4264 steps, in memory",
              {r: mx.LOADER["rerun"] for r in READS}))
    S.append(("het_whole_w8_memmap_1105", "het", "whole set (1e5/process), width 8, 4264 steps, memmap",
              {r: mx.LOADER["memmap"] for r in READS}))
    return S


def pair_stats(a, b, cls):
    keys = sorted(set(a) & set(b), key=str)
    cl = lambda k: cls(mx.probe_name(k))
    r = mx.class_ratios(a, b, keys, cl)
    ln = np.abs(np.log(np.array(list(r.values()))))
    rs = float(np.median(list(r.values())))
    return {"n_probes": len(keys), "n_classes": len(r), "median_abs_ln": float(np.median(ln)),
            "p90_abs_ln": float(np.percentile(ln, 90)), "R_seed": rs, "abs_ln_R_seed": abs(math.log(rs)),
            "class_ratios": {k: round(v, 5) for k, v in r.items()}}


def run():
    cls = mx.class_of()
    out = {"tool": "analysis/catalog_v4/seed_spread.py", "doc": __doc__.split("\n\n")[1:3], "sets": {}, "missing": [],
           "blew_up": []}
    seen = set()
    for name, loss, setup, reads in seed_sets():
        # a pair already taken under another set name (the warm-start controls C_k are shared by several changes)
        reads = {r: v for r, v in reads.items() if (r, tuple(sorted(v))) not in seen}
        seen |= {(r, tuple(sorted(v))) for r, v in reads.items()}
        if not reads:
            continue
        runs = sorted({x for v in reads.values() for x in v})
        bl = mx.blowups(runs)
        ent = {"loss": loss, "setup": setup, "runs": {r: list(v) for r, v in reads.items()}, "reads": {},
               "blowup_flagged": bl["flagged"], "spiked_recovered": bl["spiked_recovered"]}
        blown = [f for f in bl["flagged"] if "blew up" in f]
        if blown:
            out["blew_up"].append({"set": name, "flagged": blown})
        for r in READS:
            if r not in reads:
                continue
            data = [mx.read_of(x, r) for x in reads[r]]
            for x, d in zip(reads[r], data):
                if d is None:
                    out["missing"].append(f"{name} {r} {x}")
            pairs = [(i, j) for i, j in itertools.combinations(range(len(data)), 2)
                     if data[i] is not None and data[j] is not None]
            if not pairs:
                continue
            ps = {f"{reads[r][i]}|{reads[r][j]}": pair_stats(data[i], data[j], cls) for i, j in pairs}
            ent["reads"][r] = {"pairs": ps, **{k: float(np.median([p[k] for p in ps.values()]))
                                               for k in ("median_abs_ln", "p90_abs_ln", "abs_ln_R_seed")}}
        out["sets"][name] = ent
    # summary: one value per (set, read), the set's median over its pairs; blown-up sets left out
    bad = {b["set"] for b in out["blew_up"]}
    rows = [(n, e["loss"], e["setup"], r, v) for n, e in out["sets"].items() if n not in bad
            for r, v in e["reads"].items()]

    def summ(sel):
        if not sel:
            return None
        res = {"n_set_reads": len(sel)}
        for k in ("median_abs_ln", "p90_abs_ln", "abs_ln_R_seed"):
            x = np.array([v[k] for *_, v in sel])
            res[k] = {"median": round(float(np.median(x)), 4), "q10": round(float(np.percentile(x, 10)), 4),
                      "q90": round(float(np.percentile(x, 90)), 4),
                      "relative_median": round(math.expm1(float(np.median(x))), 4)}
        return res

    by_setup = {s: summ([x for x in rows if x[2] == s]) for s in sorted({x[2] for x in rows})}
    out["summary"] = {"all": summ(rows),
                      "setups_equal_weight": {k: round(float(np.median([v[k]["median"] for v in by_setup.values()])), 4)
                                              for k in ("median_abs_ln", "p90_abs_ln", "abs_ln_R_seed")},
                      "by_read": {r: summ([x for x in rows if x[3] == r]) for r in READS},
                      "by_loss": {l: summ([x for x in rows if x[1] == l]) for l in ("MSE", "het")},
                      "by_setup": by_setup,
                      "by_loss_read": {f"{l} {r}": summ([x for x in rows if x[1] == l and x[3] == r])
                                       for l in ("MSE", "het") for r in READS}}
    a = out["summary"]["all"]
    out["headline"] = (f"seed 42 vs 43 (and 42-45) on {a['n_set_reads']} (arm, read) cells: one class's test MSE "
                       f"moves by a median factor exp({a['median_abs_ln']['median']}) = "
                       f"{1 + a['median_abs_ln']['relative_median']:.3f} (90th percentile over classes "
                       f"{math.exp(a['p90_abs_ln']['median']):.3f}); the class-median ratio R moves by "
                       f"{1 + a['abs_ln_R_seed']['relative_median']:.3f}")
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: out[k] for k in ("headline", "missing", "blew_up")}, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in (v or {}).items()} for k, v in out["summary"]["by_read"].items()},
                     indent=0)[:3000])


def fig():
    import matplotlib
    matplotlib.use("Agg")
    import plot_style as ps
    ps.use()
    d = json.load(open(OUT))
    bad = {b["set"] for b in d["blew_up"]}
    fig, (a, b) = ps.figure(ncols=2)
    # (a) per class, seed B's MSE over seed A's, PP4, every pair: the histogram of ln(MSE_B / MSE_A)
    bins = np.linspace(-1.5, 1.5, 61)
    for loss, col in (("MSE", ps.C.blue), ("het", ps.C.vermillion)):
        x = [math.log(v) for n, e in d["sets"].items() if e["loss"] == loss and n not in bad
             for p in e["reads"].get("PP4", {}).get("pairs", {}).values() for v in p["class_ratios"].values()]
        if x:
            a.hist(x, bins=bins, density=True, histtype="step", color=col, label=f"{loss} loss, pp test classes")
    a.set_xlabel(r"$\ln(\mathrm{MSE}_{\rm seed\,B}/\mathrm{MSE}_{\rm seed\,A})$ per class")
    a.set_ylabel("density")
    ps.legend(a, "upper left")
    # (b) per (arm, read): the class-median |ln| (x) and the 90th percentile (y)
    mk = {"PP4": "o", "EE4": "s", "held": "^", "HD4": "D", "HD5": "v"}
    name = {"PP4": "pp test", "EE4": "e+e- test", "held": "held-out, catalog", "HD4": "held-out, default",
            "HD5": "held-out one-loop"}
    from matplotlib.lines import Line2D
    for r, m in mk.items():
        for loss, col in (("MSE", ps.C.blue), ("het", ps.C.vermillion)):
            pts = [(v["median_abs_ln"], v["p90_abs_ln"]) for n, e in d["sets"].items() if e["loss"] == loss
                   and n not in bad for rr, v in e["reads"].items() if rr == r]
            if pts:
                x, y = zip(*pts)
                b.scatter(x, y, marker=m, facecolor="none", edgecolor=col)
    s = d["summary"]["all"]
    b.axvline(s["median_abs_ln"]["median"], color=ps.C.grey, ls="--")
    b.axhline(s["p90_abs_ln"]["median"], color=ps.C.grey, ls="--")
    h = [Line2D([], [], ls="none", marker=m, markerfacecolor="none", markeredgecolor=ps.C.grey, label=name[r])
         for r, m in mk.items()]
    h += [Line2D([], [], ls="none", marker="o", color=c, label=f"{l} loss") for l, c in (("MSE", ps.C.blue),
                                                                                        ("het", ps.C.vermillion))]
    h += [Line2D([], [], color=ps.C.grey, ls="--", label="median over cells")]
    b.set_xlabel(r"median over classes of $|\ln$ ratio$|$")
    b.set_ylabel(r"90th percentile of $|\ln$ ratio$|$")
    b.set_ylim(0, 4.2)
    ps.legend(b, "upper left", handles=h, ncol=2)
    ps.save(fig, os.path.join(HERE, "figs", "seed_spread"))


if __name__ == "__main__":
    {"run": run, "fig": fig}[sys.argv[1]]()
