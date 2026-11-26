#!/usr/bin/env python3
"""kit_benchmark step 23 (D571): the Monte Carlo test of a KIT surrogate on the benchmark's flat bulk_test events.

    python tools/kit_mc_test.py <run_dir> [<run_dir> ...] [--boot 1000] [--q 0.999] [--out <json>]

Each run is a single-process KIT run scored by tools/kit_predict.py (<run_dir>/kit_benchmark/predictions.h5). The
bulk_test events are drawn flat in the benchmark's bulk (README metadata `bulk`): e+e- -> u u~ g (g): flat massless
RAMBO at sqrt s = 1 TeV with s_ij/s > 0.01; e+e- -> t t~: sqrt s flat in [353.1, 1738.7] GeV, cos theta, phi flat.
So each event's cross-section weight is, up to one constant per file,
    w = f                       (RAMBO: flat in phase space at fixed s)
    w = f beta / s              (t t~: two-body phase space beta/(32 pi^2) dOmega, flux 1/(2s), flat in sqrt s)
with f the exact target (truth) or the surrogate's (10^prediction, the truth's sign: the model predicts log10|f|).
Per run:
  bias      sigma_surr / sigma_exact - 1 = sum w_surr / sum w_exact - 1, its 16-84% bootstrap band over events, and the
            MC statistical error of sigma_exact on the same events (std(w)/mean(w)/sqrt N) for scale
  eff_exact <w_exact> / max w_exact: one-stage unweighting efficiency with the exact |M|^2
  eff_surr  <w_surr> / max w_surr: first stage, unweighting with the surrogate
  eff_x     second stage (unbiased surrogate unweighting, arXiv:2301.13562): the surrogate-unweighted events accepted
            with x / x_max, x = w_exact / w_surr, efficiency E_surr[x] / x_max = (sum w_exact / sum w_surr) / max x;
            also with x_max at the q-quantile of x under the surrogate measure and the share of sigma_exact in
            events above it (overweights, the bias that cut leaves)
  eff_two   eff_surr * eff_x: exact evaluations needed per unweighted event fall from 1/eff_exact to 1/eff_x
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import kit_hardps as kh  # noqa: E402
from tools.kit_predict import run_target  # noqa: E402


def weights(stem, target, run_dir):
    import h5py
    lname = kh.log10_name(stem, target, kh.h5_path(stem))
    with h5py.File(kh.h5_path(stem), "r") as t, h5py.File(os.path.join(run_dir, "kit_benchmark", "predictions.h5")) as p:
        g = t["bulk_test/bulk"]
        f = g[target][:].astype(np.float64)
        lp = p["bulk_test/bulk"][lname][:].astype(np.float64)
        mom = g["momenta"][:]
    assert len(lp) == len(f) and np.isfinite(lp).all()
    fs = np.sign(f) * 10.0 ** lp
    if stem == "ee_tt":
        P = mom[:, 0] + mom[:, 1]
        s = P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(1)
        pt = mom[:, 2]
        beta = np.sqrt((pt[:, 1:] ** 2).sum(1)) / pt[:, 0]   # CM frame: |p_t| / E_t
        jac = beta / s
    else:
        jac = np.ones_like(f)
    return f * jac, fs * jac, float((f < 0).mean())


def mc(we, ws, boot, q, seed=0):
    n = len(we)
    rng = np.random.default_rng(seed)
    bias = ws.sum() / we.sum() - 1
    bb = np.empty(boot)
    for b in range(boot):
        i = rng.integers(0, n, n)
        bb[b] = ws[i].sum() / we[i].sum() - 1
    x = we / ws
    # q-quantile of x under the surrogate measure (events weighted by w_surr)
    o = np.argsort(x)
    cw = np.cumsum(ws[o]) / ws.sum()
    xq = x[o][min(np.searchsorted(cw, q), n - 1)]
    ex = we.sum() / ws.sum()
    over = float(we[x > xq].sum() - (ws[x > xq] * xq).sum()) / we.sum()
    return {"n": n, "bias": float(bias), "bias_lo": float(np.quantile(bb, 0.16)), "bias_hi": float(np.quantile(bb, 0.84)),
            "mc_err": float(we.std() / we.mean() / np.sqrt(n)),
            "eff_exact": float(we.mean() / we.max()), "eff_surr": float(ws.mean() / ws.max()),
            "eff_x": float(ex / x.max()), "eff_x_q": float(min(ex / xq, 1.0)), "x_q": float(xq), "overweight_share": over,
            "eff_two": float(ws.mean() / ws.max() * ex / x.max())}


def _selfcheck():
    rng = np.random.default_rng(1)
    we = rng.lognormal(0, 1, 20000)
    r = mc(we, we.copy(), 50, 0.999)
    assert abs(r["bias"]) < 1e-12 and abs(r["eff_x"] - 1) < 1e-12 and r["overweight_share"] == 0
    r = mc(we, 2 * we, 50, 0.999)
    assert abs(r["bias"] - 1) < 1e-12 and abs(r["eff_x"] - 1) < 1e-12


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--q", type=float, default=0.999)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    _selfcheck()
    if a.selfcheck:
        print("SELFCHECK OK")
        return
    res = {}
    for r in a.runs:
        rd = r if os.path.isabs(r) else os.path.join(ROOT, r)
        stem, target, _ = kh.parse(run_target(rd))
        we, ws, neg = weights(stem, target, rd)
        res[os.path.basename(rd.rstrip("/"))] = dict(mc(we, ws, a.boot, a.q), file=stem, target=target, neg_share=neg)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
    print("KITMC " + json.dumps(res))


if __name__ == "__main__":
    main()
