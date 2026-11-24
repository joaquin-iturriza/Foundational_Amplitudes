"""Plan selective_rescale (D552): which event-dependent form of the momentum log-rescale leaves the bulk inputs of
q q~ -> Z+4g and e+e- -> u u~ g as they are, while lifting the soft momenta of the kit's soft bins as the D539 rescale
does. Input-level check on the val pools (first N events, CC), no training.

Schemes, each a factor f_i on p_i from x_i = 2 p_i.Q/s alone (equivariant), F(x) = log1p(x/eps)/(x log1p(1/eps)) the
D539 rescale (wrappers._soft_rescale):
  rescale eps         f = F(x)                            (D539; eps 1e-8, 1e-4, 1e-2, 1: the ladder of par:ftm_d539)
  massless eps=1e-8   f = F(x), massive particles f = 1   (D539's massless-only arm)
  soft-only x0        f = max(1, F(x)/F(x0))              (per particle: unchanged for x >= x0; x0 = 1 is the rescale)
  event gate t        f = F(x) on every particle of an event with min_i x_i < t, else 1
Per scheme: the share of final-state particles and events whose input changes, and the compression
c = mean over final-state particles of -ln k, k = dln x'/dln x (0 when unchanged, ~ln ln(x/eps) under the rescale),
the measure calibrated on D539's eps ladder against its bulk cost r (Z+4g, 1e4). Soft lift: x' at the kit's deepest
soft bin (x = 1e-7, 1e-6) and the trained range of x' (its 0.1% quantile).

    python3 analysis/selective_rescale/input_check.py [--cached]   -> data/input_check.json, figs/input_check
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from analysis.common.readers import site_python  # noqa: E402

POOLS = {"zgggg": "kit_uq_zgggg__me2_1000-1000GeV_val_n100000_kit_amplitudes.npy",
         "uug": "kit_ee_uug__me2_1000-1000GeV_val_n100000_kit_amplitudes.npy"}   # as finetune_mechanism/soft_rescale_inputs.py
N = 20000
EPS = 1e-8
LADDER = {1e-8: 2.41, 1e-4: 1.61, 1e-2: 1.33, 1.0: 1.10}   # par:ftm_d539, cells.zng_1e4.eps*.over_off.r @c3c6f9516
X0 = [1.0, 0.3, 0.1, 0.05, 0.03]
GATES = [0.1, 0.05, 0.03]

REMOTE = r'''
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd()); import siteconf
N = int(sys.argv[1]); eta = np.array([1., -1., -1., -1.]); out = {}
for a in sys.argv[2:]:
    lab, f = a.split("#")
    rows = np.load(os.path.join(siteconf.DATA_DIR, f), mmap_mode="r")[:N].astype(np.float64)
    n = (rows.shape[1] - 1) // 5
    p = rows[:, :4 * n].reshape(-1, n, 4)
    Q = p[:, 0] + p[:, 1]; s = (Q * Q * eta).sum(-1)
    x = np.abs(2 * (p * Q[:, None] * eta).sum(-1) / s[:, None])[:, 2:]
    mass = (np.abs((p * p * eta).sum(-1))[:, 2:] / s[:, None]) > 1e-6
    out[lab] = {"x": np.round(x, 7).tolist(), "massive": mass[0].tolist()}
print("SRCHK " + json.dumps(out))
'''


def F(x, eps):
    return np.log1p(x / eps) / (x * np.log1p(1 / eps))


def kappa(x, eps):   # dln x'/dln x of x' = x F(x)
    return (x / (x + eps)) / np.log1p(x / eps)


def schemes(x, massive):
    """{name: (f, k)} per final-state particle, x (N, n)."""
    out = {}
    for e in LADDER:
        out[f"rescale eps={e:g}"] = (F(x, e), kappa(x, e))
    f, k = F(x, EPS), kappa(x, EPS)
    m = np.broadcast_to(np.array(massive), x.shape)
    if m.any():
        out["massless eps=1e-08"] = (np.where(m, 1.0, f), np.where(m, 1.0, k))
    for x0 in X0:
        on = x < x0
        out[f"soft-only x0={x0:g}"] = (np.where(on, f / F(x0, EPS), 1.0), np.where(on, k, 1.0))
    for t in GATES:
        on = np.broadcast_to((x.min(1) < t)[:, None], x.shape)
        out[f"event gate t={t:g}"] = (np.where(on, f, 1.0), np.where(on, k, 1.0))
    return out


def main():
    cache = os.path.join(HERE, "data", "input_check_raw.json")
    raw = json.load(open(cache)) if "--cached" in sys.argv else site_python(
        REMOTE, [str(N)] + [f"{k}#{v}" for k, v in POOLS.items()], "SRCHK")
    json.dump(raw, open(cache, "w"))
    res = {"N": N, "pools": {}}
    for lab, r in raw.items():
        x = np.array(r["x"]).clip(1e-30)
        cell = {"x_q001": float(np.quantile(x, 0.001)), "x_min": float(x.min()), "schemes": {}}
        for name, (f, k) in schemes(x, r["massive"]).items():
            ch = np.abs(f - 1) > 1e-9
            xp = x * f
            cell["schemes"][name] = {"share_particles_changed": float(ch.mean()), "share_events_changed": float(ch.any(1).mean()),
                                     "compression": float(np.mean(-np.log(k))), "xp_q001": float(np.quantile(xp, 0.001))}
        res["pools"][lab] = cell
    # calibration: ln r against the compression over D539's eps ladder on Z+4g; slope through the origin (r = 1 at c = 0)
    z = res["pools"]["zgggg"]["schemes"]
    c = np.array([z[f"rescale eps={e:g}"]["compression"] for e in LADDER]); lr = np.log(list(LADDER.values()))
    b = float(c @ lr / (c @ c))
    res["calib"] = {"slope": b, "points": dict(zip(map(repr, LADDER), c.tolist())),
                    "resid_max": float(np.max(np.abs(lr - b * c)))}
    for lab, cell in res["pools"].items():
        for name, v in cell["schemes"].items():
            v["r_pred"] = float(np.exp(b * v["compression"]))
    # soft lift: x' of a gluon at the kit's deepest soft bin, x = 1e-7 and 1e-6
    res["soft_xp"] = {}
    for x in (1e-7, 1e-6):
        d = {f"rescale eps={EPS:g}": x * F(x, EPS)} | {f"soft-only x0={x0:g}": x * F(x, EPS) / F(x0, EPS) for x0 in X0}
        res["soft_xp"][repr(x)] = {"off": x} | d
    json.dump(res, open(os.path.join(HERE, "data", "input_check.json"), "w"), indent=1)
    print(json.dumps(res))   # one line: verify_claims reads the last {...} line


if __name__ == "__main__":
    main()
