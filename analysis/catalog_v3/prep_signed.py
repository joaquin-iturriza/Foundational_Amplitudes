"""Reads the signed-data check on catalog_v3 (prep_signed_pools.py, run on CC; its SIGNED_DATA lines copied home into
data/prep_signed_raw.txt) against the short pretrainings' signs on the same pools (data/short_reads/<arm>/H5_signed.txt,
tools/signed_ab_score.py: each arm's wrong-sign share on the held-out one-loop pools it resolves as sign-changing).
Per set (the one-loop pools of the training recipe, and of the one-loop held-out axis H5), over the sign-changing pools:
the medians of the negative share, the shares within a factor 10 and 100 of zero, the wrong-sign share of one sign per
pool (err_major), of a classifier on the invariants (err_mlp, and within a factor 10 of zero) and on log s_hat alone
(err_shat). The classifier's statistics (err_mlp, err_mlp_near, err_shat, their counts, and err_major, err_major_near
quoted against them) are over the pools it was fitted on only: a pool with fewer than 20 fit events of one sign (rare)
has no classifier, its stored err_mlp and err_shat are one sign per pool's, so it is listed apart (rare_pools) with
its counts and left out of them. Comparisons are strict (no margin); and the magnitude's shape per target (raw, + (N - 4) log s_hat, + the emission scales) with the per-pool
ratio of each to the raw one. On the held-out pools, per arm: the median wrong-sign share, how many pools it gets worse
than one sign per pool would, how many it gets more than half wrong, and the median of its wrong-sign share less the
classifier's (over the pools the classifier was fitted on); the held-out pools of one sign in every event that the scorer still reads as
sign-changing count there with no error for one sign per pool.
Figure: per sign-changing pool, (a) the classifier's wrong-sign share against one sign per pool's, (b) on the held-out
pools, each arm's wrong-sign share against one sign per pool's.
    python3 analysis/catalog_v3/prep_signed.py   -> data/prep_signed.json, figs/prep_signed.{png,pdf}
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
SETS = ("train", "H5")
TARGETS = ("raw", "dless", "qperp")
PCT = ("neg", "near10", "near100", "err_major", "err_mlp", "err_mlp_near", "err_major_near", "err_shat")
ARMS = {"baseline": "base", "dimensionless": "qperp", "dimensionless_shat": "shat", "dimensionless_hybrid": "hyb",
        "dimensionless_blend025": "b25", "dimensionless_blend050": "b50", "dimensionless_context": "ctx",
        "dimensionless_twin": "twin"}
# the two sign arms, each the baseline with one change at its HPs (scripts/job_catalog_v3_short.sh): a pair term in the
# sign logit, and the near-zero events' sign loss weighted 10 times. Scored like ARMS; the ranges over arms quoted in
# the notes before them (e+e- -> c cbar g, the all-negative pools) stay over ARMS, the sign arms' values are apart.
# The third, the baseline at its HPs and step count with 95 more sign-changing one-loop training pools
# (recipes/catalog_v3_train_signext.yaml, 1492 processes against 1397), changes the data, not the model
SIGN_ARMS = {"baseline_sign_pairs": "pair", "baseline_sign_near": "near", "baseline_more_signed": "more",
             # the pair sign head again with its pair channels bounded by tanh (model.sign_pair_bound), one seed (42)
             "baseline_sign_pairs_bounded": "bpair"}
# second seeds (43; the arms above are seed 42) of the baseline and of the more-signed-pools arm, everything else
# unchanged, to size that arm's sign gain and zero-shot cost against the seed-to-seed spread of each
SEED2 = {"baseline_seed2": "base2", "baseline_more_signed_seed2": "more2",
         # the more-signed-pools arm's seed 43 run again, config unchanged (it writes the per-step record): it did not
         # blow up, so the seed-43 comparisons are read on it too (the r pairs below)
         "baseline_more_signed_seed2_steplog": "more2r"}
# the comparisons, each y against x on the same pools and probes: one setting's two seeds (the spread), the arm
# against the baseline at each seed, and the two-seed means (per pool the mean wrong-sign share, per probe the
# geometric mean MSE)
PAIRS = {"sd_base": (("baseline",), ("baseline_seed2",)),
         "sd_more": (("baseline_more_signed",), ("baseline_more_signed_seed2",)),
         "s42": (("baseline",), ("baseline_more_signed",)),
         "s43": (("baseline_seed2",), ("baseline_more_signed_seed2",)),
         "m2": (("baseline", "baseline_seed2"), ("baseline_more_signed", "baseline_more_signed_seed2")),
         # with the seed-43 rerun that did not blow up in place of the one that did
         "sd_more_r": (("baseline_more_signed",), ("baseline_more_signed_seed2_steplog",)),
         "s43r": (("baseline_seed2",), ("baseline_more_signed_seed2_steplog",)),
         "m2r": (("baseline", "baseline_seed2"), ("baseline_more_signed", "baseline_more_signed_seed2_steplog")),
         # the two seed-43 runs of one config against each other: the run-to-run spread at a fixed seed
         "rerun": (("baseline_more_signed_seed2",), ("baseline_more_signed_seed2_steplog",))}


# The pair sign head test at the sign head search's best HPs (plan v3_prep_explore, the design passed 2026-10-26; the
# user's "Yes, run that test."): the more-signed-pools arm with Muon (decoupled weight decay 0.0217), lr 1.246e-2,
# warmup 0.081, pair head on (width 16, hidden 16, unbounded) against off, seeds 42 and 43
# (scripts/job_catalog_v3_short.sh arms sign_pair_best[_seed2], sign_off_best[_seed2]); beside, the Muon baseline at its
# own HPs (lr 5e-3, warmup 0.158, seed 42) and the search's best trial (trial_0177, the pair head at the same HPs and
# seed 42). Decision statistic: per pool the seed mean of each arm, the count of pools where the pair arm is lower
# (ties excluded), one-sided sign test; the median secondary; seed 43 alone stated.
PB_ARMS = {"pair": ("sign_pair_best", "sign_pair_best_seed2"), "off": ("sign_off_best", "sign_off_best_seed2")}
PB_REFS = {"muon": "baseline_more_signed_muon", "search": "sign_search_best"}
# the design's outcome rule: yes needs the pair arm lower on at least PB_YES of the pools, the seed-mean median lower,
# and the same direction at seed 43 alone
PB_YES = 13
# a blow-up: a validation over BLOWUP_FACTOR times its running minimum (pair_head_blowup.py's rule, the same factor)
BLOWUP_FACTOR = 10


def load_raw():
    rows = []
    for l in open(os.path.join(HERE, "data", "prep_signed_raw.txt")):
        if l.startswith("SIGNED_DATA "):
            rows.append(json.loads(l[len("SIGNED_DATA "):]))
    return rows


def key(r):
    f = r["files"][0]
    return f[: f.index("GeV_") + 3]                    # v3_<name>_<lo>-<hi>GeV, the scorer's pool name


def arm_signs(arm, field="wrong_sign"):
    f = os.path.join(HERE, "data", "short_reads", arm, "H5_signed.txt")
    out = {}
    for l in open(f):
        if "SIGNED_AB " in l:
            d = json.loads(l.split("SIGNED_AB ", 1)[1])
            if "skipped" not in d and "error" not in d and d["pool"].startswith("v3_"):   # catalog_v4 pools a shared folder let in
                out[d["pool"]] = d[field]
    return out


def sign_arms(h5all, n_axis, flat, out):
    """The two sign arms against the baseline on the held-out one-loop pools: the median wrong-sign share over the
    pools the scorer reads as sign-changing, and over their events within a factor 10 of zero (the scorer's near
    events), the mean over the whole one-loop axis (n_axis pools of 2000 validation events each; a pool the scorer
    reads as one sign is a plain log, so no arm makes a sign error there, nor does the data's one sign), the median
    magnitude MSE on the scored pools, and the zero-shot MSE on log|M|^2 per probe over every held-out axis
    (data/short_reads.json): the median of its ratio to the baseline's on the same probes, how many probes it is
    above, and the median per axis."""
    sr = json.load(open(os.path.join(HERE, "data", "short_reads.json")))
    b = arm_signs("baseline")
    flat["axis_n_pools"] = n_axis
    flat["axis_major"] = round(100 * sum(h5all[p]["err_major"] for p in b if p in h5all) / n_axis, 1)
    flat["axis_mlp"] = round(100 * sum(h5all[p]["err_mlp"] for p in b if p in h5all) / n_axis, 1)
    out["sign_arms"] = {}
    for arm, sh in (("baseline", "base"),) + tuple(SIGN_ARMS.items()):
        w = arm_signs(arm); wn = arm_signs(arm, "wrong_sign_near"); mm = arm_signs(arm, "mse_mag")
        ps_ = sorted(set(w) & set(h5all))
        o = {"n": len(ps_), "wrong": med([w[p] for p in ps_]), "wrong_near": med([wn[p] for p in ps_]),
             "axis_wrong": sum(w[p] for p in ps_) / n_axis, "mse_mag": med([mm[p] for p in ps_])}
        if arm != "baseline":
            o["better_than_base"] = int(sum(w[p] < b[p] for p in ps_))
            o["worse_than_base"] = int(sum(w[p] > b[p] for p in ps_))
            rat, ax = [], {}
            for a, ra in sr[arm].items():
                pb = sr["baseline"].get(a, {}).get("probes", {})
                rr = [v["mse"] / pb[p]["mse"] for p, v in ra.get("probes", {}).items() if p in pb and pb[p]["mse"] > 0]
                rat += rr
                ax[a] = {"mse": ra.get("mse"), "base_mse": sr["baseline"][a].get("mse"), "ratio": med(rr)}
            o.update({"mse_ratio": med(rat), "n_probes": len(rat), "n_probes_worse": int(sum(r > 1 for r in rat)),
                      "axes": ax})
            flat[f"{sh}_mse_ratio"] = round(o["mse_ratio"], 2)
            flat[f"{sh}_n_probes"] = o["n_probes"]
            flat[f"{sh}_n_probes_worse"] = o["n_probes_worse"]
            flat[f"{sh}_better_base"] = o["better_than_base"]
            flat[f"{sh}_worse_base"] = o["worse_than_base"]
            flat[f"{sh}_H5_mse"] = round(ax["H5"]["mse"], 2)
            flat["base_H5_mse"] = round(ax["H5"]["base_mse"], 2)
            for a in ax:
                flat[f"{sh}_{a}_ratio"] = round(ax[a]["ratio"], 2)
        flat[f"{sh}_wrong1"] = round(100 * o["wrong"], 1)
        flat[f"{sh}_wrong_near"] = round(100 * o["wrong_near"], 1)
        flat[f"{sh}_axis_wrong"] = round(100 * o["axis_wrong"], 1)
        flat[f"{sh}_mse_mag"] = round(o["mse_mag"], 2)
        out["sign_arms"][arm] = o
        print(f"{arm:24s} n={o['n']} wrong {100 * o['wrong']:.1f}% near {100 * o['wrong_near']:.1f}% "
              f"axis {100 * o['axis_wrong']:.2f}% mag {o['mse_mag']:.3g}"
              + (f" mse-ratio {o['mse_ratio']:.3g} over {o['n_probes']} (above on {o['n_probes_worse']}) "
                 f"better/worse than baseline {o['better_than_base']}/{o['worse_than_base']}" if arm != "baseline" else ""))
    return sr


def seed_spread(h5all, n_axis, flat, out, sr):
    """PAIRS: per comparison, on the held-out pools the scorer reads as sign-changing in every run of both settings, how
    many y gets less wrong than x and more, the median wrong-sign share of each, within a factor 10 of zero, and over
    the whole one-loop axis (n_axis pools); and per held-out probe of every axis read by all its runs, the median of
    y's zero-shot MSE over x's, how many probes it is above 1, and the median per axis."""
    def signs(arms, field):
        d = [arm_signs(a, field) for a in arms]
        # a pool with no event within a factor 10 of zero has no near share (None): None for the mean too
        return {p: (None if any(x[p] is None for x in d) else float(np.mean([x[p] for x in d])))
                for p in set.intersection(*[set(x) for x in d]) if p in h5all}

    def mses(arms):
        out_ = {}
        for a in set.intersection(*[set(sr[x]) for x in arms]):
            pr = [sr[x][a].get("probes", {}) for x in arms]
            out_[a] = {p: float(np.exp(np.mean([np.log(q[p]["mse"]) for q in pr])))
                       for p in set.intersection(*[set(q) for q in pr]) if all(q[p]["mse"] > 0 for q in pr)}
        return out_
    out["seeds"] = {}
    for k, (xa, ya) in PAIRS.items():
        wx, wy = signs(xa, "wrong_sign"), signs(ya, "wrong_sign")
        nx, ny = signs(xa, "wrong_sign_near"), signs(ya, "wrong_sign_near")
        ps_ = sorted(set(wx) & set(wy))
        pn = [p for p in ps_ if nx.get(p) is not None and ny.get(p) is not None]
        mx, my = mses(xa), mses(ya)
        rat, ax = [], {}
        for a in sorted(set(mx) & set(my)):
            rr = [my[a][p] / mx[a][p] for p in sorted(set(mx[a]) & set(my[a]))]
            rat += rr
            ax[a] = med(rr)
        o = {"x": xa, "y": ya, "n": len(ps_), "better": int(sum(wy[p] < wx[p] for p in ps_)),
             "worse": int(sum(wy[p] > wx[p] for p in ps_)),
             "wrong_x": med([wx[p] for p in ps_]), "wrong_y": med([wy[p] for p in ps_]),
             "n_near": len(pn), "near_x": med([nx[p] for p in pn]), "near_y": med([ny[p] for p in pn]),
             "axis_x": sum(wx[p] for p in ps_) / n_axis, "axis_y": sum(wy[p] for p in ps_) / n_axis,
             "mse_ratio": med(rat), "n_probes": len(rat), "n_probes_worse": int(sum(r > 1 for r in rat)), "axes": ax}
        out["seeds"][k] = o
        flat.update({f"{k}_n": o["n"], f"{k}_n_near": o["n_near"], f"{k}_better": o["better"], f"{k}_worse": o["worse"],
                     f"{k}_n_probes": o["n_probes"], f"{k}_n_probes_worse": o["n_probes_worse"],
                     f"{k}_mse_ratio": round(o["mse_ratio"], 2)})
        for f_ in ("wrong", "near", "axis"):
            flat[f"{k}_{f_}_x"] = round(100 * o[f"{f_}_x"], 1)
            flat[f"{k}_{f_}_y"] = round(100 * o[f"{f_}_y"], 1)
        for a, r in ax.items():
            flat[f"{k}_{a}_ratio"] = round(r, 2)
        print(f"{k:8s} n={o['n']} y better/worse {o['better']}/{o['worse']} wrong {100 * o['wrong_x']:.1f}->"
              f"{100 * o['wrong_y']:.1f}% near {100 * o['near_x']:.1f}->{100 * o['near_y']:.1f}% axis "
              f"{100 * o['axis_x']:.1f}->{100 * o['axis_y']:.1f}% mse-ratio {o['mse_ratio']:.3g} over {o['n_probes']} "
              f"(above on {o['n_probes_worse']}) " + " ".join(f"{a}:{r:.2f}" for a, r in ax.items()))


def train_curve(arm):
    """from the run's training log (data/short_reads/<arm>/train.log, the out_0.log lines copied home): the validation
    curve (val_loss_no_reg every `every` steps, the last at the last step), the best and its step, the first
    validation over BLOWUP_FACTOR times the running minimum (the blow-up) with the one before, the steps skipped for a
    non-finite gradient norm, the final validation"""
    import re
    t = open(os.path.join(HERE, "data", "short_reads", arm, "train.log")).read()
    every = int(re.search(r"validating every (\d+) steps", t).group(1))
    iters = int(re.search(r"Starting to train for (\d+) iterations", t).group(1))
    v = [float(x) for x in re.findall(r"Val loss \(combined\).*?val_loss_no_reg: ([0-9.eE+-]+)", t)]
    steps = [min(every * (i + 1), iters) for i in range(len(v))]
    i = int(np.argmin(v))
    j = next((k for k in range(1, len(v)) if v[k] > BLOWUP_FACTOR * min(v[:k])), None)
    o = {"best": v[i], "best_step": steps[i], "final": v[-1], "n_skipped": t.count("Skipping update, non-finite"),
         "steps": steps, "vals": v, "iters": iters}
    if j is not None:
        o.update({"blow_step": steps[j], "blow_step_before": steps[j - 1], "blow_val": v[j], "blow_val_before": v[j - 1]})
    return o


def sign_test(k, n):
    """one-sided sign test: P(at least k of n lower) under no effect"""
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else None


def pair_best(h5all, n_axis, flat, out):
    """The pair sign head test (PB_ARMS) scored under its passed design, on the held-out one-loop pools every run reads
    as sign-changing: per run the median wrong-sign share; per arm the seed mean per pool and its median; the count of
    pools where the pair arm is lower and higher (ties excluded) with the one-sided sign test, on the seed means, at
    seed 43 alone and at seed 42 alone; each arm's seed means against the Muon baseline at its own HPs; the zero-shot
    MSE per probe on every held-out axis (pair over off on the two seeds' geometric mean per probe, and each arm over
    the Muon baseline); the runs' training curves and blow-ups; and the outcome by the design's rule."""
    from short_reads import READ, read
    runs = PB_ARMS["pair"] + PB_ARMS["off"] + tuple(PB_REFS.values())
    w = {a: arm_signs(a) for a in runs}
    pools = sorted(set.intersection(*[set(x) for x in w.values()]) & set(h5all))
    o = {"n_pools": len(pools), "runs": {}, "curves": {a: train_curve(a) for a in runs if a != PB_REFS["search"]}}
    for a in runs:
        o["runs"][a] = {"wrong": med([w[a][p] for p in pools]), "axis": sum(w[a][p] for p in pools) / n_axis,
                        "per_pool": {p: w[a][p] for p in pools}}
    mean = {k: {p: float(np.mean([w[a][p] for a in arms])) for p in pools} for k, arms in PB_ARMS.items()}
    mean["muon"] = {p: w[PB_REFS["muon"]][p] for p in pools}

    def count(y, x, tag):
        lo = int(sum(y[p] < x[p] for p in pools)); hi = int(sum(y[p] > x[p] for p in pools))
        r = {"lower": lo, "higher": hi, "ties": len(pools) - lo - hi, "p": sign_test(lo, lo + hi),
             "med_y": med([y[p] for p in pools]), "med_x": med([x[p] for p in pools])}
        o[tag] = r
        flat.update({f"pb_{tag}_lower": lo, f"pb_{tag}_higher": hi, f"pb_{tag}_ties": r["ties"],
                     f"pb_{tag}_p": round(r["p"], 3), f"pb_{tag}_med_y": round(100 * r["med_y"], 1),
                     f"pb_{tag}_med_x": round(100 * r["med_x"], 1)})
        return r
    m = count(mean["pair"], mean["off"], "mean")
    s43 = count(w[PB_ARMS["pair"][1]], w[PB_ARMS["off"][1]], "s43")
    count(w[PB_ARMS["pair"][0]], w[PB_ARMS["off"][0]], "s42")
    count(mean["pair"], mean["muon"], "pair_muon")
    count(mean["off"], mean["muon"], "off_muon")
    count(w[PB_ARMS["pair"][0]], w[PB_REFS["search"]], "rerun42")      # seed 42's pair run against the search's trial
    # the only full-length pair: the search's trial (pair head on, clean) against the control at seed 42 (clean)
    count(w[PB_REFS["search"]], w[PB_ARMS["off"][0]], "clean42")
    # the outcome by the design's rule
    blew = {a: "blow_step" in o["curves"][a] for a in PB_ARMS["pair"] + PB_ARMS["off"]}
    off_only = any(blew[a] and not blew[b] for a, b in zip(PB_ARMS["off"], PB_ARMS["pair"]))
    if m["lower"] >= PB_YES and m["p"] < 0.05 and m["med_y"] < m["med_x"] and s43["lower"] > s43["higher"] and not off_only:
        outcome = "yes"
    elif m["lower"] <= m["higher"]:
        outcome = "no"
    else:
        outcome = "inconclusive"
    o["outcome"] = outcome
    o["blew_up"] = blew
    flat["pb_outcome"] = outcome
    flat["pb_n_pools"] = len(pools)
    for a in runs:
        flat[f"pb_{a}_wrong"] = round(100 * o["runs"][a]["wrong"], 1)
        flat[f"pb_{a}_axis"] = round(100 * o["runs"][a]["axis"], 1)
    for a, c in o["curves"].items():
        flat.update({f"pb_{a}_best": float(f"{c['best']:.3g}"), f"pb_{a}_best_step": c["best_step"],
                     f"pb_{a}_final": float(f"{c['final']:.3g}"), f"pb_{a}_skipped": c["n_skipped"]})
        if "blow_step" in c:
            flat.update({f"pb_{a}_blow_step": c["blow_step"], f"pb_{a}_blow_step_before": c["blow_step_before"],
                         f"pb_{a}_blow_val": float(f"{c['blow_val']:.3g}"),
                         f"pb_{a}_blow_val_before": float(f"{c['blow_val_before']:.3g}")})
    # the zero-shot MSE per probe on every held-out axis
    mse = {a: {ax: read(a, ax) for ax in READ} for a in runs}

    def gm(arms):
        return {ax: {p: float(np.exp(np.mean([np.log(mse[a][ax][p]["mse"]) for a in arms])))
                     for p in set.intersection(*[set(mse[a][ax]) for a in arms])
                     if all(mse[a][ax][p]["mse"] > 0 for a in arms)} for ax in READ}
    g = {k: gm(arms) for k, arms in PB_ARMS.items()}
    g["muon"] = gm((PB_REFS["muon"],))
    g["s43_pair"] = gm((PB_ARMS["pair"][1],)); g["s43_off"] = gm((PB_ARMS["off"][1],))
    for tag, (y, x) in {"mse_mean": ("pair", "off"), "mse_s43": ("s43_pair", "s43_off"),
                        "mse_pair_muon": ("pair", "muon"), "mse_off_muon": ("off", "muon")}.items():
        rat, ax_ = [], {}
        for ax in READ:
            rr = [g[y][ax][p] / g[x][ax][p] for p in sorted(set(g[y][ax]) & set(g[x][ax]))]
            rat += rr
            ax_[ax] = med(rr)
        o[tag] = {"ratio": med(rat), "n": len(rat), "above": int(sum(r > 1 for r in rat)), "axes": ax_}
        flat.update({f"pb_{tag}_ratio": round(med(rat), 2), f"pb_{tag}_n": len(rat),
                     f"pb_{tag}_above": o[tag]["above"]})
    out["pair_best"] = o
    print("pair_best", json.dumps({k: v for k, v in o.items() if k not in ("runs", "curves")}))
    for a in runs:
        c = o["curves"].get(a, {})
        print(f"  {a:28s} median {100 * o['runs'][a]['wrong']:.2f}% axis {100 * o['runs'][a]['axis']:.2f}% "
              + (f"best {c['best']:.3g}@{c['best_step']} blow {c.get('blow_step')} skipped {c['n_skipped']} "
                 f"final {c['final']:.3g}" if c else ""))
    return o


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def main():
    rows = load_raw()
    out = {"errors": [(r["set"], r["name"], r["error"]) for r in rows if "error" in r]}
    flat = {"n_errors": len(out["errors"])}
    sig = {s: [r for r in rows if r["set"] == s and r.get("signed") and "error" not in r] for s in SETS}
    for s in SETS:
        allp = [r for r in rows if r["set"] == s and "error" not in r]
        d = sig[s]
        c = [r for r in d if not r.get("rare")]      # the pools a classifier was fitted on
        o = {"n_pools": len(allp), "n_signed": len(d), "n_clf": len(c)}
        for k in ("neg", "near10", "near100", "err_major", "err_major_near"):
            o[k + "_all"] = med([r[k] for r in d])     # the data alone, every sign-changing pool
        for k in ("neg", "near10", "near100"):
            o[k] = o[k + "_all"]
        for k in ("err_major", "err_mlp", "err_mlp_near", "err_major_near", "err_shat"):
            o[k] = med([r[k] for r in c])
        o["mlp_over_major"] = med([r["err_mlp"] / r["err_major"] for r in c if r["err_major"] > 0])
        o["n_rare"] = len(d) - len(c)
        o["n_shat_beats_major"] = int(sum(r["err_shat"] < r["err_major"] for r in c))
        o["rare_pools"] = {r["name"]: {"n_minor_fit": r["n_minor_fit"], "neg": r["neg"], "err_major": r["err_major"]}
                           for r in d if r.get("rare")}
        o["rare_max_err_major"] = max([r["err_major"] for r in d if r.get("rare")], default=None)
        o["rare_max_minor_fit"] = max([r["n_minor_fit"] for r in d if r.get("rare")], default=None)
        for t in TARGETS:
            o[f"shape_{t}"] = med([r[f"shape_{t}"] for r in d])
            o[f"frac_{t}"] = med([r[f"frac_{t}"] for r in d])
            if t != "raw":
                rr = np.array([r[f"shape_{t}"] / r["shape_raw"] for r in d])
                o[f"{t}_over_raw"] = float(np.median(rr))
                o[f"{t}_above_raw"] = int((rr > 1).sum())
        out[s] = o
        print(s, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k != "rare_pools"}))
        for n_, v_ in o["rare_pools"].items():
            print(f"  no classifier (rare) {n_:24s} {v_['n_minor_fit']:3d} of one sign in the fit events, "
                  f"negative {100 * v_['neg']:.2f}%, one sign per pool wrong {100 * v_['err_major']:.2f}%")
        sh = {"train": "tr", "H5": "H5"}[s]
        for k in ("n_pools", "n_signed", "n_clf", "neg", "near10", "near100", "err_major", "err_mlp", "err_mlp_near",
                  "err_major_near", "err_shat", "mlp_over_major", "n_rare", "n_shat_beats_major", "shape_raw", "shape_dless", "shape_qperp",
                  "dless_over_raw", "qperp_over_raw", "qperp_above_raw", "err_major_all", "err_major_near_all",
                  "rare_max_err_major", "rare_max_minor_fit"):
            v = o[k]
            if k in PCT or k in ("err_major_all", "err_major_near_all", "rare_max_err_major"):                              # shares of events, in percent as the notes write them
                v = round(100 * v, 1)
            flat[f"{sh}_{k}"] = round(v, 3) if isinstance(v, float) else v
        for n_, v_ in o["rare_pools"].items():            # the rare pools, by their minority count in the fit events
            flat[f"{sh}_rare_{n_[3:]}"] = v_["n_minor_fit"]
    # the arms' signs on the held-out pools, against the data's floors on the same pools
    h5 = {key(r): r for r in sig["H5"]}
    # a held-out pool of one sign in every validation and test event that the scorer still reads as sign-changing:
    # the transform is resolved on every role of the pool (experiment._resolve_amp_preprocessing), the train split
    # included, which this check does not read, and any non-positive value there makes it signed (an all-negative
    # pool too, which has no plain log). The truth's sign is the raw amplitude's, so the arms' shares there are real
    # errors; one sign per pool and the classifier make none. Per role, the negative share from the pool audit.
    one_sign = {key(r): {**r, "err_major": 0.0, "err_mlp": 0.0} for r in rows
                if r["set"] == "H5" and "error" not in r and not r.get("signed")}
    h5all = {**one_sign, **h5}
    out["one_sign_scored"] = {}
    out["arms"] = {}
    for arm, sh in {**ARMS, **SIGN_ARMS, **SEED2}.items():
        ws = arm_signs(arm)
        ps = sorted(set(ws) & set(h5all))
        w = np.array([ws[p] for p in ps]); mj = np.array([h5all[p]["err_major"] for p in ps])
        ml = np.array([h5all[p]["err_mlp"] for p in ps])
        clf = np.array([not h5all[p].get("rare") for p in ps])     # the classifier's pools (and the one-sign ones)
        for p in sorted(set(ws) & set(one_sign)):
            out["one_sign_scored"].setdefault(p, {"neg": one_sign[p]["neg"]})[arm] = ws[p]
        o = {"n": len(ps), "n_one_sign": len(set(ws) & set(one_sign)), "unmatched": sorted(set(ws) - set(h5all)),
             "wrong_sign": float(np.median(w)),
             "worse_than_major": int((w > mj).sum()), "minus_mlp": float(np.median((w - ml)[clf])),
             "per_pool": {p: ws[p] for p in ps}}
        out["arms"][arm] = o
        print(f"{arm:24s} n={o['n']:2d} wrong {o['wrong_sign']:.3f} worse-than-one-sign {o['worse_than_major']} "
              f"minus-classifier {o['minus_mlp']:.3f} unmatched {o['unmatched']}")
        flat[f"{sh}_n"] = o["n"]
        flat[f"{sh}_n_one_sign"] = o["n_one_sign"]
        flat[f"{sh}_inverted"] = int((w > 0.5).sum())
        flat[f"{sh}_wrong"] = round(100 * o["wrong_sign"])
        flat[f"{sh}_worse_major"] = o["worse_than_major"]
        flat[f"{sh}_minus_mlp"] = round(100 * o["minus_mlp"])
    out["pools"] = {s: {key(r): {k: r.get(k) for k in ("name", "neg", "near10", "err_major", "err_mlp", "err_shat",
                                                         "shape_raw", "shape_dless", "shape_qperp")} for r in sig[s]}
                    for s in SETS}
    n_axis = len([r for r in rows if r["set"] == "H5" and "error" not in r])
    sr = sign_arms(h5all, n_axis, flat, out)
    seed_spread(h5all, n_axis, flat, out, sr)
    pair_best(h5all, n_axis, flat, out)
    json.dump(out, open(os.path.join(HERE, "data", "prep_signed.json"), "w"), indent=1)
    plot(sig, h5, out)
    plot_sign_arms(out, sr)
    plot_seeds(out, sr)
    plot_pair_best(out)
    # e+e- -> c cbar g at one loop: the pool every arm gets most wrong
    cg = "v3_epem_ccxg_nlo_25-1000GeV"
    wc = [out["arms"][a]["per_pool"][cg] for a in ARMS]
    flat.update({"ccg_neg": round(100 * h5[cg]["neg"]), "ccg_mlp": round(100 * h5[cg]["err_mlp"], 1),
                 "ccg_wrong_min": round(100 * min(wc)), "ccg_wrong_max": round(100 * max(wc))})
    for a, sh in SIGN_ARMS.items():
        flat[f"{sh}_ccg_wrong"] = round(100 * out["arms"][a]["per_pool"][cg])
    flat["n_one_sign_scored"] = len(out["one_sign_scored"])
    audit = json.load(open(os.path.join(HERE, "pool_audit.json")))["pools"]
    n_allneg = 0
    for p, o in out["one_sign_scored"].items():
        au = audit[one_sign[p]["name"]]
        o["audit"] = {r: {"rows": au[r]["rows"], "negative": au[r]["negative"]} for r in ("train", "val", "test")}
        if all(au[r]["negative"] == 1.0 for r in ("train", "val", "test")):
            n_allneg += 1
            wa = [o[a] for a in ARMS]                 # every all-negative pool, keyed by its name
            nm = one_sign[p]["name"][3:]
            flat[f"allneg_{nm}_wrong_min"] = round(100 * min(wa), 1)
            flat[f"allneg_{nm}_wrong_max"] = round(100 * max(wa), 1)
        if one_sign[p]["neg"] == 0.0:                 # positive in val and test: the negatives are in train
            tr = au["train"]
            flat["pos_one_sign_pool"] = one_sign[p]["name"]
            flat["pos_one_sign_train_neg"] = int(round(tr["negative"] * tr["rows"]))
            flat["pos_one_sign_train_rows"] = tr["rows"]
            flat["pos_one_sign_base_wrong"] = round(100 * o["baseline"])
            flat["pos_one_sign_more_wrong"] = round(100 * o["baseline_more_signed"], 1)
    flat["n_one_sign_allneg"] = n_allneg
    print("one sign in every event, scored as sign-changing:", json.dumps(out["one_sign_scored"]))
    print(json.dumps(flat))


def plot(sig, h5, out):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    ax = axs[0]
    for s, col, mk, lab in (("train", ps.C.blue, "o", "training pools"), ("H5", ps.C.vermillion, "^", "held-out pools")):
        c = [r for r in sig[s] if not r.get("rare")]  # no classifier on the rare pools
        ax.plot([r["err_major"] for r in c], [r["err_mlp"] for r in c], mk, mfc="none", color=col, label=lab)
    ax.plot([0, 0.5], [0, 0.5], ls="--", color=ps.C.grey, label="equal")
    ax.set_xlabel("wrong sign, one sign per pool"); ax.set_ylabel("wrong sign, classifier")
    ps.legend(ax, "upper left")
    ax = axs[1]
    ps_ = sorted(h5)
    for arm, col, mk, lab in (("baseline", ps.C.grey, "s", "baseline"), ("dimensionless", ps.C.blue, "o", "emission scales"),
                              ("dimensionless_hybrid", ps.C.vermillion, "^", "hybrid")):
        pp = out["arms"][arm]["per_pool"]
        xs = [h5[p]["err_major"] for p in ps_ if p in pp]; ys = [pp[p] for p in ps_ if p in pp]
        # the pools of one sign in every event: at zero on one sign per pool's axis
        xs += [0.0 for p in sorted(out["one_sign_scored"]) if p in pp]
        ys += [pp[p] for p in sorted(out["one_sign_scored"]) if p in pp]
        ax.plot(xs, ys, mk, mfc="none", color=col, label=lab)
    ax.plot([0, 0.5], [0, 0.5], ls="--", color=ps.C.grey, label="one sign per pool")
    ax.set_xlabel("wrong sign, one sign per pool"); ax.set_ylabel("wrong sign, zero-shot")
    ps.legend(ax, "upper left")
    for ax in axs:
        ps.make_room(ax, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "prep_signed"))


def plot_sign_arms(out, sr):
    """(a) per held-out pool the scorer reads as sign-changing, each sign arm's zero-shot wrong-sign share against the
    baseline's; (b) per held-out probe on every axis, each sign arm's zero-shot MSE on log|M|^2 against the baseline's."""
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    sty = {"baseline_sign_pairs": (ps.C.blue, "o", "pair sign head"),
           "baseline_sign_near": (ps.C.vermillion, "^", "near-zero weighting"),
           "baseline_more_signed": (ps.C.green, "s", "more signed pools"),
           "baseline_sign_pairs_bounded": (ps.C.purple, "D", "bounded pair sign head")}
    b = out["arms"]["baseline"]["per_pool"]
    for arm, (col, mk, lab) in sty.items():
        pp = out["arms"][arm]["per_pool"]
        q = sorted(set(pp) & set(b))
        axs[0].plot([b[p] for p in q], [pp[p] for p in q], mk, mfc="none", color=col, label=lab)
    axs[0].plot([0, 1], [0, 1], ls="--", color=ps.C.grey, label="equal")
    axs[0].set_xlabel("wrong sign, baseline"); axs[0].set_ylabel("wrong sign, sign arm")
    ps.legend(axs[0], "upper left")
    lo, hi = np.inf, 0
    for arm, (col, mk, lab) in sty.items():
        xs, ys = [], []
        for a, ra in sr[arm].items():
            pb = sr["baseline"].get(a, {}).get("probes", {})
            for p, v in ra.get("probes", {}).items():
                if p in pb:
                    xs.append(pb[p]["mse"]); ys.append(v["mse"])
        axs[1].plot(xs, ys, mk, mfc="none", color=col, ms=3, label=lab)
        lo, hi = min(lo, min(xs), min(ys)), max(hi, max(xs), max(ys))
    axs[1].plot([lo, hi], [lo, hi], ls="--", color=ps.C.grey, label="equal")
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set_xlabel("zero-shot MSE, baseline"); axs[1].set_ylabel("zero-shot MSE, sign arm")
    ps.legend(axs[1], "upper left")
    for ax in axs:
        ps.make_room(ax, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "prep_signed_arms"))


def plot_seeds(out, sr):
    """The more-signed-pools arm against the seed-to-seed spread, as empirical distributions of one comparison's
    per-unit differences (PAIRS, the single-seed ones): (a) over the held-out pools the scorer reads as sign-changing,
    the wrong-sign share's difference y - x in percentage points; (b) over the held-out probes of every axis, log10 of
    y's zero-shot MSE over x's."""
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    sty = {"sd_base": (ps.C.grey, "--", "baseline, seed 43 against 42"),
           "sd_more": (ps.C.green, "--", "more signed, seed 43 against 42"),
           "s42": (ps.C.blue, "-", "more signed against baseline, seed 42"),
           "s43": (ps.C.vermillion, "-", "more signed against baseline, seed 43")}
    for k, (col, ls, lab) in sty.items():
        (xa,), (ya,) = PAIRS[k]
        wx, wy = arm_signs(xa), arm_signs(ya)
        d = np.sort([100 * (wy[p] - wx[p]) for p in sorted(set(wx) & set(wy)) if p in out["arms"][xa]["per_pool"]])
        axs[0].step(d, np.arange(1, len(d) + 1) / len(d), where="post", color=col, ls=ls, label=lab)
        r = []
        for a in set(sr[xa]) & set(sr[ya]):
            px, py = sr[xa][a].get("probes", {}), sr[ya][a].get("probes", {})
            r += [np.log10(py[p]["mse"] / px[p]["mse"]) for p in set(px) & set(py) if px[p]["mse"] > 0 and py[p]["mse"] > 0]
        r = np.sort(r)
        axs[1].step(r, np.arange(1, len(r) + 1) / len(r), where="post", color=col, ls=ls, label=lab)
    axs[0].set_xlabel("wrong sign, y - x (points)"); axs[0].set_ylabel("fraction of pools")
    axs[1].set_xlabel(r"$\log_{10}$ MSE$_y$ / MSE$_x$"); axs[1].set_ylabel("fraction of probes")
    ps.legend(axs[0], "upper left")
    for ax in axs:
        ps.make_room(ax, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "prep_signed_seeds"))


def plot_pair_best(out):
    """The pair sign head test: (a) per held-out sign-changing one-loop pool, the pair arm's zero-shot wrong-sign share
    against the control's (pair head off), on the two seeds' mean and at seed 43 alone; (b) the four runs' and the Muon
    baseline's validation loss without regularisation against step."""
    import plot_style as ps
    o = out["pair_best"]
    fig, axs = ps.figure(ncols=2)
    R = o["runs"]
    pools = sorted(R[PB_ARMS["pair"][0]]["per_pool"])
    mp = [100 * np.mean([R[a]["per_pool"][p] for a in PB_ARMS["pair"]]) for p in pools]
    mo = [100 * np.mean([R[a]["per_pool"][p] for a in PB_ARMS["off"]]) for p in pools]
    axs[0].plot(mo, mp, "o", mfc="none", color=ps.C.blue, label="mean of seeds 42, 43")
    axs[0].plot([100 * R[PB_ARMS["off"][1]]["per_pool"][p] for p in pools],
                [100 * R[PB_ARMS["pair"][1]]["per_pool"][p] for p in pools], "^", mfc="none", color=ps.C.vermillion,
                label="seed 43")
    hi = max(mp + mo) * 1.05
    axs[0].plot([0, hi], [0, hi], ls="--", color=ps.C.grey, label="equal")
    axs[0].set_xlabel("wrong sign, pair head off (%)"); axs[0].set_ylabel("wrong sign, pair head on (%)")
    ps.legend(axs[0], "upper left")
    sty = ((PB_ARMS["pair"][0], ps.C.vermillion, "-", "on, seed 42"), (PB_ARMS["pair"][1], ps.C.vermillion, "--", "on, seed 43"),
           (PB_ARMS["off"][0], ps.C.blue, "-", "off, seed 42"), (PB_ARMS["off"][1], ps.C.blue, "--", "off, seed 43"),
           (PB_REFS["muon"], ps.C.grey, ":", "Muon baseline"))
    for a, col, ls, lab in sty:
        c = o["curves"][a]
        axs[1].plot(c["steps"], c["vals"], color=col, ls=ls, label=lab)
    axs[1].set_yscale("log")
    axs[1].set_xlabel("step"); axs[1].set_ylabel("validation loss, no regularisation")
    ps.legend(axs[1], "upper right", ncol=2)
    for ax in axs:
        ps.make_room(ax, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "prep_signed_pair_best"))


if __name__ == "__main__":
    main()
