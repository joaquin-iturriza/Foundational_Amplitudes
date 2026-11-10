#!/usr/bin/env python
"""The ee->uu online-generation study (branch l2-gen, 2026-07-18 and 2026-09-07): read its eval summaries, print the
numbers the write-up quotes, draw its figure.

Every arm finetunes mu in a warm chain over R rounds of 1500 steps, each round on a fresh pool of N=4e5 events drawn
without replacement from one frozen, labelled 3e6-event candidate pool of ee->uu (sqrt(s) 91-1000 GeV, half of it on
the Z pole); round 0 is the same coverage draw for every arm. The arms differ only in how rounds >= 1 draw their pool:
  static   : the coverage base p_cov = f flat-in-log|M|^2 + (1-f) uniform-in-sqrt(s)
  l2       : p_cov * sigma^alpha, sigma from a sigma-only fit on the previous round's mu, loss IS-corrected (c_b=Q_b/pi_b
             per log|M|^2 bin) back to the objective Q
  oracle   : as l2 with |pred - true| in place of sigma
  reweight : the static draw, its loss IS-corrected to the objective Q (no sigma): separates the objective from sigma
  adapt    : p ~ sigma^gamma over the candidate pool, sigma EMA-damped, no base density, no correction (coordinate-free);
             adapt_oracle the same with |pred - true|
Before commit 35230574 (2026-07-18 23:08) the static and reweight arms re-drew the SAME pool every round (fixed seed),
so they saw 4e5 distinct events against the sigma arms' fresh pool each round; the 'fair' summaries (2026-09-07) are
those two arms rerun after the fix.

Metric: MSE of ln|M|^2 on a fixed 1e5-event uniform-sqrt(s) test set, per sqrt(s) bin (8 bins, dense on the Z peak),
'logflat' = the plain mean over the 8 bins (eval_eeuu_resonance.py, model_run0_best.pt of each round). Seeds are
combined by geometric mean. CPU only; reads analysis/divergences/l2_*_summary.json.

    python3 analysis/divergences/l2_eeuu_online.py [--plot]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def load(name):
    """{tag: record} of one summary file."""
    with open(os.path.join(HERE, name)) as f:
        return {r["tag"]: r for r in json.load(f)}


def curve(summ, tag, arm, rounds, key="logflat"):
    return [summ[f"{tag}_{arm}_mu_r{r}"][key] for r in range(rounds)]


def seeds_curve(summ, tags, arm, rounds, key="logflat"):
    """(per-seed curves (S, R), their geometric mean (R,))."""
    c = np.array([curve(summ, t, arm, rounds, key) for t in tags])
    return c, np.exp(np.log(c).mean(axis=0))


def sig(x):
    return float(f"{x:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    corr = load("l2_online_summary.json")             # f=0.25 base, alpha=1, IS-corrected, 1 seed (pre-fix static)
    alpha = load("l2_alpha_summary.json")             # f=0.25 base, sigma arm at alpha 0.5 and 0.25
    starv = load("l2_starv_summary.json")             # starved base f=0, objective q_f=0.25, alpha=0.5 (pre-fix static)
    rw = load("l2_rw_summary.json")                   # starved base, reweight arm (pre-fix)
    fair_st = load("l2_fair_starv_summary.json")      # starved base, static + reweight after the fix
    adapt5 = load("l2_adapt_summary.json")            # coordinate-free, seed 0, rounds 0-4
    adapt8b = load("l2_adapt8b_summary.json")         # the same seed-0 runs continued to rounds 5-7
    conf = load("l2_conf_summary.json")               # coordinate-free seeds 1, 2 (rounds 0-5)
    fair_cf = load("l2_fair_conf_summary.json")       # f=0.25 static, 3 seeds, 6 rounds, after the fix

    out = {}
    R = 3
    # (a) the good coverage base f=0.25, one seed
    base = {"static": curve(corr, "corr", "static", R), "sigma_a1": curve(corr, "corr", "l2", R),
            "oracle_a1": curve(corr, "corr", "oracle", R), "sigma_a05": curve(alpha, "a050", "l2", R),
            "sigma_a025": curve(alpha, "a025", "l2", R)}
    out["f025"] = {k: sig(v[-1]) for k, v in base.items()}
    out["f025"]["sigma_a1_over_static"] = sig(base["sigma_a1"][-1] / base["static"][-1])
    out["f025"]["oracle_a1_over_static"] = sig(base["oracle_a1"][-1] / base["static"][-1])

    # (b) the starved base, three seeds
    st = ["starv0", "starv1", "starv2"]
    fs = ["fstv0", "fstv1", "fstv2"]
    arms_b = {"static_prefix": seeds_curve(starv, st, "static", R), "sigma_a05": seeds_curve(starv, st, "l2", R),
              "reweight_prefix": seeds_curve(rw, st, "reweight", R),
              "static_fair": seeds_curve(fair_st, fs, "static", R),
              "reweight_fair": seeds_curve(fair_st, fs, "reweight", R)}
    oracle_b = curve(starv, "starv0", "oracle", R)
    out["starved"] = {k: {"gmean": sig(g[-1]), "seeds": [sig(x) for x in c[:, -1]]} for k, (c, g) in arms_b.items()}
    out["starved"]["oracle_seed0"] = sig(oracle_b[-1])
    g = {k: v[1][-1] for k, v in arms_b.items()}
    out["starved"]["static_fair_over_sigma"] = sig(g["static_fair"] / g["sigma_a05"])
    out["starved"]["static_fair_over_reweight_fair"] = sig(g["static_fair"] / g["reweight_fair"])
    out["starved"]["reweight_fair_over_sigma"] = sig(g["reweight_fair"] / g["sigma_a05"])
    out["starved"]["static_prefix_over_fair"] = sig(g["static_prefix"] / g["static_fair"])

    # (c) coordinate-free sigma^gamma against the fair f=0.25 static, three seeds, six rounds
    R6 = 6
    # seed 0: rounds 0-4 from the 5-round job, round 5 from the same chain continued (adapt8b)
    ad0 = curve(adapt5, "adapt", "adapt", 5) + [adapt8b["adapt_adapt_mu_r5"]["logflat"]]
    ad = np.array([ad0, curve(conf, "adaptS1", "adapt", R6), curve(conf, "adaptS2", "adapt", R6)])
    ad_g = np.exp(np.log(ad).mean(axis=0))
    orc0 = curve(adapt5, "adapt", "adapt_oracle", 5) + [adapt8b["adapt_adapt_oracle_mu_r5"]["logflat"]]
    sf_c, sf_g = seeds_curve(fair_cf, ["ff25s0", "ff25s1", "ff25s2"], "static", R6)
    out["coordfree"] = {"adapt": {"gmean": sig(ad_g[-1]), "seeds": [sig(x) for x in ad[:, -1]]},
                        "static_fair": {"gmean": sig(sf_g[-1]), "seeds": [sig(x) for x in sf_c[:, -1]]},
                        "adapt_oracle_seed0": sig(orc0[-1]),
                        "adapt_over_static": sig(ad_g[-1] / sf_g[-1])}

    r0 = [v[0] for v in base.values()]
    out["f025"]["round0_max_over_min"] = sig(max(r0) / min(r0))
    # the plotted curves, round by round (what each panel draws)
    out["curves"] = {"f025": {k: [sig(x) for x in v] for k, v in base.items()},
                     "starved": {**{k: [sig(x) for x in v[1]] for k, v in arms_b.items()},
                                 "oracle_seed0": [sig(x) for x in oracle_b]},
                     "coordfree": {"adapt": [sig(x) for x in ad_g], "static_fair": [sig(x) for x in sf_g],
                                   "adapt_oracle_seed0": [sig(x) for x in orc0]}}

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.ticker  # noqa: F401
        sys.path.insert(0, REPO)
        import plot_style as ps
        ylab = r"$\langle{\rm MSE}(\Delta\ln|\mathcal{M}|^2)\rangle_{\sqrt{s}\,{\rm bins}}$"
        figs = ps.panels(3)
        (f1, ax1), (f2, ax2), (f3, ax3) = figs
        r3 = np.arange(R)
        for k, lab, c, ls in [("static", "static, reused", ps.C.blue, "-"),
                              ("sigma_a1", r"$\sigma^{1}$ + IS", ps.C.vermillion, "-"),
                              ("sigma_a05", r"$\sigma^{0.5}$ + IS", ps.C.vermillion, "--"),
                              ("sigma_a025", r"$\sigma^{0.25}$ + IS", ps.C.vermillion, ":"),
                              ("oracle_a1", r"$|r|^{1}$ + IS", ps.C.grey, "-")]:
            ax1.plot(r3, base[k], ls, marker="o", color=c, label=lab)
        for k, lab, c, ls in [("static_prefix", "static, reused", ps.C.blue, "--"),
                              ("static_fair", "static", ps.C.blue, "-"),
                              ("reweight_prefix", "IS only, reused", ps.C.green, "--"),
                              ("reweight_fair", "IS only", ps.C.green, "-"),
                              ("sigma_a05", r"$\sigma^{0.5}$ + IS", ps.C.vermillion, "-")]:
            ax2.plot(r3, arms_b[k][1], ls, marker="o", color=c, label=lab)
        ax2.plot(r3, oracle_b, "-", marker="o", color=ps.C.grey, label=r"$|r|^{0.5}$ + IS")
        r6 = np.arange(R6)
        ax3.plot(r6, sf_g, "-", marker="o", color=ps.C.blue, label="static")
        ax3.plot(r6, ad_g, "-", marker="o", color=ps.C.vermillion, label=r"$\sigma^{\gamma}$")
        ax3.plot(r6, orc0, "-", marker="o", color=ps.C.grey, label=r"$|r|^{\gamma}$")
        for ax, nc, loc in ((ax1, 1, "lower left"), (ax2, 1, "lower left"), (ax3, 1, "upper right")):
            ax.set_yscale("log")
            ax.set_xlabel("round")
            ax.set_ylabel(ylab)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ps.legend(ax, loc, ncol=nc)
        ps.save_panels(figs, os.path.join(HERE, "figs", "l2_eeuu_online"), repo=REPO)

    print(json.dumps(out))


if __name__ == "__main__":
    main()
