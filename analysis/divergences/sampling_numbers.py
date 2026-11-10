#!/usr/bin/env python
"""The numbers of the July ee->uug sampling and heteroscedastic-head studies (branch heterosc, 2026-07), from their
committed summaries in analysis/divergences/:
  deep_sampling_summary.json     uniform (flat RAMBO) vs antenna fine-tune of base22, 400k events each, on the common
                                 deep-IR test set: overall MSE/MAE of ln|M|^2 and, per y_min decade [1e-6, 1), the
                                 MSE, the median relative error |M_pred/M - 1| and the training events in the decade
  fraction_sweep_summary.json    the antenna-fraction sweep f in {0, .25, .5, .75, 1}: log-flat (mean of the six
                                 per-decade MSEs) and per-event MSE, and the per-decade MSE
  figs/heterosc_sigma_summary.json  the float32 label floor per decade (relative |M|^2 change from float32-rounded
                                 momenta, median)
  heterosc_rank.json             the HETEROSC foundation grid lr x beta: in-distribution validation MSE of mu
  q2rw_eval_summary.json         the sigma-reweighting (Q2) arms on the deep-IR test: event-weighted MSE; gain =
                                 (M_base - M_arm)/(M_base - M_oracle), base = no reweighting, oracle = true |residual|
  evtest_A.json, leaktest.json, leakfix.json  the multi-tag eval leak check (eval_eeuu_resonance.py): ft_mix025 as the
                                 only tag, as the second tag after raw, and the same with a per-tag run_dir
Keys: deep.*, fraction.*, float32.*, foundation.*, q2.*, coverage.*, leak.* (fraction of an arm's 400k training events per
decade). CPU only.

    python3 analysis/divergences/sampling_numbers.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def sig(x, n=3):
    return float(f"{x:.{n}g}")


def main():
    out = {}
    deep = load("deep_sampling_summary.json")
    u, a = deep["uniform"], deep["antenna"]
    out["deep"] = {"uniform_mse": sig(u["mse_all"]), "antenna_mse": sig(a["mse_all"]),
                   "uniform_over_antenna": sig(u["mse_all"] / a["mse_all"]),
                   "uniform_medrel_deepest_pct": sig(100 * u["per_decade"][0]["median_rel_err"]),
                   "antenna_medrel_deepest_pct": sig(100 * a["per_decade"][0]["median_rel_err"]),
                   "uniform_train_below_1e-5_per_1e6": sig(1e6 * u["per_decade"][0]["n_train"]
                                                           / sum(d["n_train"] for d in u["per_decade"])),
                   "antenna_bulk_mse": sig(a["per_decade"][-1]["mse_log"]),
                   "antenna_deepest_mse": sig(a["per_decade"][0]["mse_log"])}
    cov = {}
    for k, r in deep.items():
        n = sum(d["n_train"] for d in r["per_decade"])
        cov[k] = {"pct": [sig(100 * d["n_train"] / n, 2) for d in r["per_decade"]],
                  "bulk_events_k": sig(r["per_decade"][-1]["n_train"] / 1e3, 2)}
    out["coverage"] = cov

    fr = load("fraction_sweep_summary.json")
    lf = np.array(fr["logflat_mse"])
    best = lf.min()
    out["fraction"] = {"f": fr["f"], "logflat": [sig(x) for x in lf],
                       "f0_over_best": sig(lf[0] / best), "f1_over_best": sig(lf[-1] / best),
                       "bulk_mse": {k: sig(v[-1]) for k, v in fr["per_decade"].items()}}

    fl = load("figs/heterosc_sigma_summary.json")["float32_floor"]
    out["float32"] = {"deepest_pct": sig(100 * fl[0], 2),
                      "per_decade_ratio": [sig(fl[i] / fl[i + 1], 2) for i in range(2)]}

    rk = {r["tag"]: r["mse"] for r in load("heterosc_rank.json")}
    out["foundation"] = {"best_tag": min(rk, key=rk.get), "best_mse": sig(min(rk.values()), 2),
                         "b0_over_b1_lr0.004": sig(rk["lr0.004_b0.0"] / rk["lr0.004_b1.0"], 2),
                         "mse": {k: sig(v) for k, v in rk.items()}}

    q = {r["tag"]: r["mse"] for r in load("q2rw_eval_summary.json")}
    q.update({r["tag"]: r["mse"] for r in load("q2rank_eval_summary.json")})
    b, o = q["baseQ"], q["oracle"]
    out["q2"] = {"event_gain_pct": {t: sig(100 * (b - m) / (b - o), 2) for t, m in q.items()}}
    qb = load("q2rw_eval_summary.json")[0]["binmse"]
    out["q2"]["test_pct_below_1e-3"] = sig(100 * qb[0][2] / sum(x[2] for x in qb), 2)

    # the eval leak: ft_mix025 scored as the first tag of its own call (evtest_A) and as the second after raw
    # (leaktest; leakfix is the same with a per-tag run_dir, which did not fix it)
    first = load("evtest_A.json")[0]
    second = {r["tag"]: r for r in load("leaktest.json")}["mix025"]
    fixd = {r["tag"]: r for r in load("leakfix.json")}["mix025"]
    out["leak"] = {"mix025_tag1_logflat": sig(first["logflat"], 2), "mix025_tag2_logflat": sig(second["logflat"], 2),
                   "mix025_tag2_rundir_logflat": sig(fixd["logflat"], 2)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
