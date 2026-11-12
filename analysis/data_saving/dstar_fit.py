"""pretrain_data_saving readout: the data a pretraining saves at a precision level, from L(D) of a scratch and a
fine-tune arm. Joint fit, weighted in log10 L, of L_arm(D) = A_arm D^-alpha + L_inf,arm with one exponent alpha
shared by the arms (D373: similar exponents, so the saving is the same at any level; draft Fig. 22: the transfer
gain is in the prefactor and the floor, not the slope). Per arm D*(level) solves the fit = level; the saving is
S = D*_scr / D*_ft, beside it S_A = (A_scr / A_ft)^(1/alpha) (the saving where both floors are negligible).
Band: bootstrap over cells (each arm's cells resampled with replacement, refit), the 16-84% range of log10 S.

    --calib: the scratch calibration alone (horizons.py --ladder outputs): per target the fit of the single arm and the
             bootstrap band of log10 D*(1e-4), the width a saving's band starts from (the stage-2 design's check 3).
    python analysis/data_saving/dstar_fit.py --calib
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
LEVEL = 1e-4


def fit(arms):
    """arms: {arm: (D array, L array)}. Returns {"alpha", arm: (A, Linf)} of the joint fit (one shared alpha)."""
    names = sorted(arms)

    def unpack(p):
        return p[0], {a: (10 ** p[1 + 2 * i], 10 ** p[2 + 2 * i]) for i, a in enumerate(names)}

    def res(p):
        al, pa = unpack(p)
        return np.concatenate([np.log10(pa[a][0] * arms[a][0] ** -al + pa[a][1]) - np.log10(arms[a][1])
                               for a in names])
    p0 = [1.5]
    for a in names:
        D, L = arms[a]
        p0 += [np.log10(L.max() * D.min() ** 1.5), np.log10(max(L.min() / 10, 1e-12))]
    lo = [0.1] + [-30, -14] * len(names)
    hi = [6.0] + [40, 2] * len(names)
    r = least_squares(res, p0, bounds=(lo, hi))
    al, pa = unpack(r.x)
    return {"alpha": al, **pa, "rms": float(np.sqrt(np.mean(r.fun ** 2)))}


def dstar(A, Linf, al, level=LEVEL):
    return (A / (level - Linf)) ** (1 / al) if level > Linf else np.inf


def boot(arms, stat, n=500, seed=0):
    """16/50/84% of stat over cell resamples. A replicate whose refit floor reaches the level (D* infinite, S unbounded)
    is kept as +-inf in the percentiles, so an unbounded edge shows as inf; counts: kept, of them infinite, and the
    resamples rejected for under 3 distinct D (which cannot fix a power law and a floor)."""
    rng = np.random.default_rng(seed)
    out, rej = [], 0
    for _ in range(n):
        b = {}
        for a, (D, L) in arms.items():
            i = rng.integers(0, len(D), len(D))
            if len(set(D[i])) < 3:
                break
            b[a] = (D[i], L[i])
        else:
            out.append(stat(fit(b)))
            continue
        rej += 1
    out = np.array(out, float)
    out = out[~np.isnan(out)]                      # inf - inf: both arms unbounded, no saving defined; counted below
    pct = np.percentile(out, [16, 50, 84], method="nearest") if len(out) else np.full(3, np.nan)
    return pct, {"kept": int(len(out)), "infinite": int(np.isinf(out).sum()), "rejected": rej,
                 "undefined": int(n - rej - len(out))}


def saving(arms, level=LEVEL):
    """{S, S_A, band of log10 S, fit} for arms {'scr': (D, L), 'ft': (D, L)}."""
    def logS(f):
        return np.log10(dstar(*f["scr"], f["alpha"], level) / dstar(*f["ft"], f["alpha"], level))
    f = fit(arms)
    band, nb = boot(arms, logS)
    return {"log10_S": logS(f), "log10_S_A": float(np.log10(f["scr"][0] / f["ft"][0]) / f["alpha"]),
            "band": list(map(float, band)), "boot": nb, "fit": {k: (list(v) if isinstance(v, tuple) else v)
                                                                   for k, v in f.items()}}


def calib():
    files = ["horizons_cal_f22_cal_f22_lrhalf.json", "horizons_cal2_ladder.json", "horizons_cal_vb_ladder.json"]
    allr = [r for f in files for r in json.load(open(os.path.join(HERE, "data", f)))]
    # the calibration's fixed-lr protocol only: a rerun at another lr (tag *_lrhalf) is listed apart, as are cells
    # without a result (blowup, missing)
    other_lr = lambda r: str(r.get("dir", "")).endswith("_lrhalf")
    rows = [r for r in allr if r.get("best") and not other_lr(r)]
    out = {}
    for t in ["ee_uu", "epem_ttx", "ee_uu_nlo", "ee_ttbar_nlo", "bbx_zz_nlo"]:
        for r in allr:
            if r["target"] == t and not (r.get("best") and not other_lr(r)):
                why = ("other lr (" + os.path.basename(r["dir"]) + ")" if other_lr(r) and r.get("best") else
                       f"blowup at step {r['blowup_step']}" if "blowup_step" in r else "no result yet")
                print(f"  not fitted: {t} k={r['k']}: {why}")
        rr = sorted((r for r in rows if r["target"] == t), key=lambda r: r["k"])
        arms = {"scr": (np.array([r["D"] for r in rr], float), np.array([r["best"] for r in rr]))}
        f = fit(arms)
        ld = lambda g: np.log10(dstar(*g["scr"], g["alpha"]))
        band, nb = boot(arms, ld)
        out[t] = {"n_cells": len(rr), "alpha": f["alpha"], "rms_decades": f["rms"], "log10_Dstar": float(ld(f)),
                  "band_16_84": list(map(float, band)), "half_width_decades": float((band[2] - band[0]) / 2),
                  "boot": nb, "cells_k": [r["k"] for r in rr]}
        print(f"{t:13s} cells {len(rr)}  alpha {f['alpha']:.2f}  rms {f['rms']:.2f} dec  log10 D*(1e-4) "
              f"{ld(f):.2f}  band [{band[0]:.2f}, {band[2]:.2f}]  half-width {(band[2] - band[0]) / 2:.2f} dec"
              f"  (x sqrt2 for S: {np.sqrt(2) * (band[2] - band[0]) / 2:.2f})")
    p = os.path.join(HERE, "data", "dstar_calib.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calib", action="store_true")
    a = ap.parse_args()
    if a.calib:
        calib()
