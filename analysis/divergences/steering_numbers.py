#!/usr/bin/env python
"""The numbers of the sigma-steering paragraphs of docs/results.tex sec:div-sampling (ee -> u ubar gg loop, its transfer
to u ubar g and u ubar ggg, the massive b bbar g test, sigma against the divergence coordinate and the pre-flight
contrast, the Bayes-by-backprop sweep, the polynomial keep rule, the saturation floor), from
analysis/divergences/steering_eval_summary.json.

That summary is computed on Jean Zay (`--fetch`, read-only, through analysis.common.site_python), the only place its
inputs exist (main checkout, listed 2026-11-10): the held-out evaluations analysis/divergences/heldout_eval_<tag>.npz
(per event: pred_logamp, true_logamp, y_min, sigma in preprocessed units, sqrt_s where the process has a resonance),
the sweeps sweeps/l2_bbb_uugg/results/*.json and sweeps/l2_poly_uugg/results/*.json, and the saturation logs
analysis/divergences/satlogs/uugg_<arm>_n<N>.json. Per evaluation file it keeps the event count and the MSE of
log|M|^2 over all events, per y_min decade (DEC), per y_min band (BANDS) and per sqrt(s) region (REG), the mean
sigma over the same selections, the Spearman correlation of sigma with |error|, the span of the true log|M|^2, and
the R^2 of degree-1 and degree-2 fits of log E[err^2 | sigma] against log sigma over 20 equal-count sigma bins.

Definitions, as the plotting scripts of these paragraphs use them (plot_l2_uugg_gamma.py, plot_gamma_response.py,
plot_l2_compare.py, plot_l2_bbbarg.py, plot_sigma_vs_divergence.py, plot_poly_keep.py, plot_bbb_sweep.py,
plot_saturation.py): log-flat = mean over the populated y_min decades of the per-decade MSE; an arm over base = the
same statistic of the sigma arm over the base (uniform-keep) arm; the sigma-contrast = mean sigma in the singular
region over mean sigma in the bulk, on the base arm; sigma in log|M|^2 units = sigma times the round-0 pool's
std(log|M|^2) (AMP_STD).
    python3 analysis/divergences/steering_numbers.py            # the claims, a final JSON line
    python3 analysis/divergences/steering_numbers.py --fetch    # recompute the summary on Jean Zay
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "steering_eval_summary.json")
# std(log|M|^2) of each process's round-0 pool (plot_sigma_vs_divergence.py)
AMP_STD = {"uug": 3.3545, "uugg": 6.1200, "uuggg": 8.9193}

REMOTE = r'''
import glob, json, os
import numpy as np
D = "analysis/divergences"
MZ = 91.1876
DEC = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]
BANDS = {"lt1e-6": lambda y: y < 1e-6, "lt1e-5": lambda y: y < 1e-5, "lt1e-3": lambda y: y < 1e-3,
         "gt1e-2": lambda y: y > 1e-2, "gt1e-1": lambda y: y > 1e-1}
REG = {"peak": lambda s, y: np.abs(s - MZ) < 3.0, "shoulder": lambda s, y: (np.abs(s - MZ) >= 3.0) & (np.abs(s - MZ) < 15.0),
       "continuum": lambda s, y: np.abs(s - MZ) >= 15.0, "peak_ir": lambda s, y: (np.abs(s - MZ) < 3.0) & (y < 1e-3),
       "continuum_ir": lambda s, y: (np.abs(s - MZ) >= 15.0) & (y < 1e-3)}
def sel(m, e2, sg):
    n = int(m.sum())
    return [n, float(e2[m].mean()) if n else None, float(sg[m].mean()) if (n and sg is not None) else None]
def rank(a):
    return np.argsort(np.argsort(a))
out = {"files": {}}
for f in sorted(glob.glob(os.path.join(D, "heldout_eval_*.npz"))):
    d = np.load(f)
    e2 = (d["pred_logamp"].astype(np.float64) - d["true_logamp"]) ** 2
    y = d["y_min"]
    sg = d["sigma"].astype(np.float64) if "sigma" in d.files and d["sigma"].size == e2.size else None
    ss = d["sqrt_s"] if "sqrt_s" in d.files and d["sqrt_s"].size == e2.size else None
    r = dict(n=int(e2.size), mse=float(e2.mean()), dec=[sel((y >= lo) & (y < hi), e2, sg) for lo, hi in DEC],
             bands={k: sel(fn(y), e2, sg) for k, fn in BANDS.items()},
             true_span=float(d["true_logamp"].max() - d["true_logamp"].min()))
    if ss is not None:
        r["reg"] = {k: sel(fn(ss, y), e2, sg) for k, fn in REG.items()}
    if sg is not None:
        r["spearman_sigma_abserr"] = float(np.corrcoef(rank(sg), rank(np.sqrt(e2)))[0, 1])
        q = np.quantile(sg, np.linspace(0, 1, 21))
        b = np.clip(np.searchsorted(q, sg, side="right") - 1, 0, 19)
        ls = np.array([np.log(sg[b == i].mean()) for i in range(20)])
        le = np.array([np.log(e2[b == i].mean()) for i in range(20)])
        r2 = {}
        for deg in (1, 2):
            c = np.polyfit(ls, le, deg)
            res = le - np.polyval(c, ls)
            r2[str(deg)] = float(1 - (res ** 2).sum() / ((le - le.mean()) ** 2).sum())
            if deg == 2:
                r2["c2"] = float(c[0])
        r["r2_logerr_logsigma"] = r2
    out["files"][os.path.basename(f)[len("heldout_eval_"):-len(".npz")]] = r
for name in ("l2_bbb_uugg", "l2_poly_uugg"):
    out[name] = [json.load(open(f)) for f in sorted(glob.glob(os.path.join("sweeps", name, "results", "*.json")))]
out["satlogs"] = {}
for f in sorted(glob.glob(os.path.join(D, "satlogs", "uugg_*_n*.json"))):
    rr = json.load(open(f))["rounds"]
    out["satlogs"][os.path.basename(f)[:-5]] = dict(p99=[x["sigma_p99"] for x in rr], pool=[x["pool"] for x in rr],
                                                    drop=[x["rel_drop_p99"] for x in rr])
print("STEER " + json.dumps(out))
'''


def claims(S):
    F = S["files"]
    dec = lambda t: np.array([np.nan if d[1] is None else d[1] for d in F[t]["dec"]])
    logflat = lambda t: float(np.nanmean(dec(t)))                       # the populated decades
    band = lambda t, k, i=1: F[t]["bands"][k][i]
    reg = lambda t, k, i=1: F[t]["reg"][k][i]
    out = {}
    # ee -> u ubar gg, base against sigma^gamma, 3 seeds: per-decade MSE averaged over seeds, then the ratio
    seeds = lambda tag: np.array([dec(f"{tag}_s{s}") for s in (0, 1, 2)])
    b = seeds("base")
    g3s = {}
    for g, tag in ((1, "sigma"), (2, "g2"), (3, "g3"), (5, "g5"), (10, "g10")):
        a = seeds(tag)
        g3s[str(g)] = dict(logflat=float(a.mean() / b.mean()), deepest=float(a[:, 0].mean() / b[:, 0].mean()),
                           shallowest=float(a[:, -1].mean() / b[:, -1].mean()))
    out["uugg_3seed"] = g3s
    # seed 0, gamma up to 30: deepest decade, bulk (y_min > 0.1) and all-event MSE over base
    out["uugg_s0"] = {str(g): dict(deepest=band(t, "lt1e-6") / band("base_s0", "lt1e-6"),
                                   bulk=band(t, "gt1e-1") / band("base_s0", "gt1e-1"),
                                   overall=F[t]["mse"] / F["base_s0"]["mse"])
                      for g, t in ((1, "sigma_s0"), (3, "g3_s0"), (10, "g10_s0"), (20, "uugg_g20_s0"), (30, "uugg_g30_s0"))}
    # u ubar g: MSE per sqrt(s) region (peak |sqrt s - M_Z| < 3 GeV, continuum >= 15, peak and y_min < 1e-3), overall
    out["uug"] = {g: dict(mse=F[t]["mse"], peak=reg(t, "peak"), peak_ir=reg(t, "peak_ir"), continuum=reg(t, "continuum"))
                  for g, t in (("0", "uug_base_s0"), ("1", "uug_sigma_s0"), ("3", "uug_g3_s0"))}
    out["uug_gain"] = dict(peak=reg("uug_base_s0", "peak") / reg("uug_g3_s0", "peak"),
                           peak_ir=reg("uug_base_s0", "peak_ir") / reg("uug_g3_s0", "peak_ir"))
    # u ubar ggg: overall MSE, deep-IR (y_min < 1e-3) ratio to base, the base's per-decade range
    out["uuggg"] = {g: dict(mse=F[t]["mse"], deep=band(t, "lt1e-3") / band("uuggg_base_s0", "lt1e-3"))
                    for g, t in (("0", "uuggg_base_s0"), ("1", "uuggg_sigma_s0"), ("3", "uuggg_g3_s0"),
                                 ("10", "uuggg_g10_s0"), ("20", "uuggg_g20_s0"))}
    d = dec("uuggg_base_s0")
    out["uuggg_dec_min"], out["uuggg_dec_max"] = float(np.nanmin(d)), float(np.nanmax(d))
    # b bbar g, gamma = 3 over base, per seed: regions, deep IR y_min < 1e-5, overall, log-flat (decades with >= 20
    # events, plot_l2_bbbarg.py), and the base arm's sigma-contrasts (peak over continuum; y_min < 1e-5 over > 0.1)
    out["bbg"] = []
    for s in (0, 1, 2):
        bt, gt = f"bbbarg_base_s{s}", f"bbbarg_g3_s{s}"
        lfb = np.mean([x[1] for x in F[bt]["dec"] if x[0] >= 20])
        lfg = np.mean([x[1] for x in F[gt]["dec"] if x[0] >= 20])
        out["bbg"].append(dict(peak=reg(gt, "peak") / reg(bt, "peak"), shoulder=reg(gt, "shoulder") / reg(bt, "shoulder"),
                               continuum=reg(gt, "continuum") / reg(bt, "continuum"),
                               deep=band(gt, "lt1e-5") / band(bt, "lt1e-5"), mse_base=F[bt]["mse"], mse_g3=F[gt]["mse"],
                               overall=F[gt]["mse"] / F[bt]["mse"], logflat=float(lfg / lfb),
                               contrast_peak=reg(bt, "peak", 2) / reg(bt, "continuum", 2),
                               contrast_ir=band(bt, "lt1e-5", 2) / band(bt, "gt1e-1", 2),
                               span=float(np.exp(F[bt]["true_span"]))))
    # sigma against the divergence (base arms; sigma times AMP_STD over the RMSE) and the pre-flight contrasts
    ub, A = F["uug_base_s0"], AMP_STD["uug"]
    out["preflight"] = dict(
        uug_sigma_rise=ub["reg"]["peak"][2] / ub["reg"]["continuum"][2],
        uug_rmse_rise=float(np.sqrt(ub["reg"]["peak"][1] / ub["reg"]["continuum"][1])),
        uug_calib={k: ub["reg"][k][2] * A / np.sqrt(ub["reg"][k][1]) for k in ("peak", "shoulder", "continuum")},
        uugg_calib=[x[2] * AMP_STD["uugg"] / np.sqrt(x[1]) for x in F["base_s0"]["dec"]],
        uug_contrast=ub["reg"]["peak"][2] / ub["reg"]["continuum"][2],
        uug_gain=reg("uug_g3_s0", "peak") / reg("uug_base_s0", "peak"),
        uugg_contrast=band("base_s0", "lt1e-6", 2) / band("base_s0", "gt1e-1", 2),
        uugg_gain=band("g10_s0", "lt1e-6") / band("base_s0", "lt1e-6"),
        uuggg_contrast=band("uuggg_base_s0", "lt1e-6", 2) / band("uuggg_base_s0", "gt1e-2", 2),
        uuggg_gain=band("uuggg_g10_s0", "lt1e-6") / band("uuggg_base_s0", "lt1e-6"))
    out["preflight"]["uugg_calib_rest_min"] = min(out["preflight"]["uugg_calib"][1:])
    out["preflight"]["uugg_calib_rest_max"] = max(out["preflight"]["uugg_calib"][1:])
    # Bayes-by-backprop sweep (objective: deep-IR MSE, y_min < 1e-3) against uniform and the het head sigma^3, seed 0
    rows = S["l2_bbb_uugg"]
    obj = np.array([r["val_loss"] for r in rows])
    rank = lambda a: np.argsort(np.argsort(a))
    best = rows[int(obj.argmin())]
    uni, het = band("base_s0", "lt1e-3"), band("g3_s0", "lt1e-3")
    out["bbb"] = dict(trials=len(rows), best=float(obj.min()), uniform=uni, het=het, over_uniform=obj.min() / uni,
                      over_het=obj.min() / het, best_at_srel_min=best["bbb_sigma_rel"] == min(r["bbb_sigma_rel"] for r in rows),
                      **{f"spearman_{k}": float(np.corrcoef(rank(obj), rank(np.array([r[k] for r in rows])))[0, 1])
                         for k in ("bbb_sigma_rel", "bbb_beta", "gamma")},
                      het_rank_min=min(F[t]["spearman_sigma_abserr"] for t in ("sigma_s0", "g3_s0")),
                      het_rank_max=max(F[t]["spearman_sigma_abserr"] for t in ("sigma_s0", "g3_s0")))
    # polynomial keep rule: R^2 of the error-vs-sigma fits on the base arms, the sweep, and the 3-seed confirmation
    poly = sorted(S["l2_poly_uugg"], key=lambda r: r["logflat_mse"])
    pl = [logflat(f"uugg_polyopt_s{s}") for s in (0, 1, 2)]
    pw = [logflat(f"g10_s{s}") for s in (0, 1, 2)]
    out["poly"] = dict(r2_uugg=F["base_s0"]["r2_logerr_logsigma"], r2_uuggg=F["uuggg_base_s0"]["r2_logerr_logsigma"],
                       trials=len(poly), best_c2=[r["keep_c2"] for r in poly[:3]], best_logflat=poly[0]["logflat_mse"],
                       power_s0=pw[0], best_over_power_s0=poly[0]["logflat_mse"] / pw[0],
                       neg_c2_ranks=[i + 1 for i, r in enumerate(poly) if r["keep_c2"] < 0],
                       poly_mean=float(np.mean(pl)), poly_std=float(np.std(pl)), power_mean=float(np.mean(pw)),
                       power_std=float(np.std(pw)), edge=float(1 - np.mean(pl) / np.mean(pw)))
    # saturation: the final round's sigma p99 over the fresh proposal batch, per arm and data budget
    sat = {k: v["p99"][-1] for k, v in S["satlogs"].items()}
    out["sat"] = dict(p99=sat, base_rise=sat["uugg_base_n600000"] / sat["uugg_base_n75000"] - 1,
                      budget_factor=600000 / 75000,
                      sigma_below_base=[1 - sat[f"uugg_sigma_n{n}"] / sat[f"uugg_base_n{n}"] for n in (75000, 150000, 600000)])
    return json.loads(json.dumps(out, default=float))


def fetch():
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    from analysis.common import site_python
    out = site_python(REMOTE, [], "STEER", site="jeanzay", timeout=1800)
    with open(SUMMARY, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {SUMMARY}: {len(out['files'])} evaluation files")


def main():
    if "--fetch" in sys.argv:
        return fetch()
    print(json.dumps(claims(json.load(open(SUMMARY)))))


if __name__ == "__main__":
    main()
