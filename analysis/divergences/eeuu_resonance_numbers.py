#!/usr/bin/env python
"""The numbers of the ee->uu Z-resonance studies of branch heterosc (2026-07-17/18): the generated fraction sweep,
generation against resampling, and sigma-reweighting (L1) on the f=0.25 base and on the starved RAMBO base, 3 seeds.

Reads the corrected (one tag per subprocess) evaluations eeuu_reson_clean_summary.json, eeuu_reson_l1seeds_summary.json
and eeuu_reson_l1raw_summary.json (written by eval_eeuu_resonance.py: MSE of ln|M|^2 on the common 1e5-event
uniform-sqrt(s) ee->uu test set, per sqrt(s) bin, each run's model_run0_best.pt). Metrics:
  logflat   the plain mean of the 8 per-bin MSEs (every sqrt(s) bin counts once)
  zpeak     the MSE of the [91, 93) GeV bin
  resonance the mean of the per-bin MSEs of the bins below 100 GeV
Seeds: mean and sample standard deviation over the 3 seeds; 'gain' = base over arm (above 1: the arm is better);
'recovered_pct' = 100 (base - sigma) / (base - oracle) on the seed means. CPU only.

    python3 analysis/divergences/eeuu_resonance_numbers.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return {r["tag"]: r for r in json.load(f)}


def resonance(r):
    return float(np.mean([b[3] for b in r["binmse"] if b[1] <= 100]))


def sig(x):
    return float(f"{x:.3g}")


def seeds(summ, tags, key):
    v = np.array([resonance(summ[t]) if key == "resonance" else summ[t][key] for t in tags])
    return v


def main():
    clean = load("eeuu_reson_clean_summary.json")
    out = {"sweep": {t: sig(clean[t]["logflat"]) for t in ("raw", "mix025", "mix050", "mix075", "genflat")}}
    lf = {t: clean[t]["logflat"] for t in clean}
    out["sweep"]["raw_over_mix025"] = sig(lf["raw"] / lf["mix025"])
    out["sweep"]["raw_over_mix050"] = sig(lf["raw"] / lf["mix050"])
    out["sweep"]["genflat_over_mix025"] = sig(lf["genflat"] / lf["mix025"])
    out["sweep"]["genflat_over_mix050"] = sig(lf["genflat"] / lf["mix050"])
    out["sweep"]["overall_best"] = min(("raw", "mix025", "mix050", "mix075", "genflat"),
                                       key=lambda t: clean[t]["overall"])
    out["generation"] = {"resampled_flat": sig(lf["flatlogm"]), "generated_flat": sig(lf["genflat"]),
                         "resampled_over_generated": sig(lf["flatlogm"] / lf["genflat"]),
                         "resampled_over_mix025": sig(lf["flatlogm"] / lf["mix025"]),
                         "raw_over_generated": sig(lf["raw"] / lf["genflat"]),
                         "raw_over_generated_zpeak": sig(clean["raw"]["zpeak"] / clean["genflat"]["zpeak"])}

    for name, fname, tags in [
            ("l1_mix025", "eeuu_reson_l1seeds_summary.json",
             {"base": ["l1base", "l1base_s1", "l1base_s2"], "sigma": ["l1sigma", "l1sigma_s1", "l1sigma_s2"],
              "oracle": ["l1oracle", "l1oracle_s1", "l1oracle_s2"]}),
            ("l1_starved", "eeuu_reson_l1raw_summary.json",
             {a: [f"l1{a}_raw_s{s}" for s in range(3)] for a in ("base", "sigma", "oracle")})]:
        summ = load(fname)
        d = {}
        for key in ("logflat", "resonance"):
            v = {a: seeds(summ, t, key) for a, t in tags.items()}
            d[key] = {a: {"mean": sig(x.mean()), "std": sig(x.std(ddof=1))} for a, x in v.items()}
            gain = v["base"] / v["sigma"]
            d[key]["gain_sigma"] = [sig(g) for g in gain]
            d[key]["gain_sigma_mean"] = sig(gain.mean())
            d[key]["sigma_over_base_worst"] = sig((v["sigma"] / v["base"]).max())
            m = {a: x.mean() for a, x in v.items()}
            d[key]["recovered_pct"] = round(100 * (m["base"] - m["sigma"]) / (m["base"] - m["oracle"]))
        out[name] = d
    print(json.dumps(out))


if __name__ == "__main__":
    main()
