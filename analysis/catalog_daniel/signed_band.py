"""D83, step 3 of the 2026-10-23 plan: how a signed target T = asinh(V / (F r0)) would sit on the 55 sign-changing one-loop
pools (recipes/catalog_loop_signed.yaml), before any training. The user, 2026-10-23: the linear part of asinh costs
against its log part, so it should hold as few events as possible. Per pool, on its val events: V the stored value,
F = s_hat^(4-N) (the mass dimension; the coupling orders are one constant per pool and only shift r0), u = log|V F|
(the log-magnitude a log target would see). Reported per pool: the negative fraction, the quantiles of u, and the share
of events with |V F| below r0 (lin_shift: for an r0 off its own quantile by a level error e, in nat) for (a) one shared r0, the median over pools of their 0.5% quantile, and (b) a per-pool r0
at its own 0.5% quantile (0.5% by construction); and how thin the zero crossing is, the share of events within a factor
10 and 100 of zero relative to the pool's median |V F|.
Login node, CPU only; prints JSON after a __B64__ line.
    python analysis/catalog_daniel/signed_band.py
"""
import base64, glob, gzip, json, os

import numpy as np
import yaml


def main():
    rec = yaml.safe_load(open("recipes/catalog_loop_signed.yaml"))["processes"]
    pools = {}
    for p in rec:
        lo, hi = p["sqrts"]
        g = sorted(glob.glob(os.path.join(os.environ["DATA_DIR"], f"{p['name']}_{lo}-{hi}GeV_val_*amplitudes.npy")))
        if not g:
            continue
        x = np.asarray(np.load(g[0]), dtype=np.float64)
        N = (x.shape[1] - 1) // 5
        P = x[:, : 4 * N].reshape(len(x), N, 4); q = P[:, 0] + P[:, 1]
        s = q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1)
        v = x[:, -1] * s ** (N - 4)
        if np.all(v > 0):
            continue
        pools[f"{p['name']}_{lo}-{hi}GeV"] = v
    q05 = {k: np.quantile(np.abs(v), 0.005) for k, v in pools.items()}
    r0 = float(np.median(list(q05.values())))
    out = {"shared_r0": r0, "pools": {}}
    for k, v in pools.items():
        a = np.abs(v); u = np.log(np.clip(a, 1e-300, None)); med = np.median(a)
        out["pools"][k] = {"n": int(len(v)), "neg_frac": float(np.mean(v < 0)),
                           "u_q": [float(x) for x in np.quantile(u, [0.005, 0.05, 0.5, 0.95, 0.995])],
                           "lin_shared": float(np.mean(a < r0)), "lin_own": float(np.mean(a < q05[k])),
                           "near_zero_10": float(np.mean(a < med / 10)), "near_zero_100": float(np.mean(a < med / 100)),
                           # r0 set from a level known only up to an error e (nat), r0 = q_0.5% * exp(e): the share in the
                           # linear band for that r0 (Level item (e): r0 following a predicted level)
                           "lin_shift": {f"{e:+g}": float(np.mean(a < q05[k] * np.exp(e))) for e in (-3, -2, -1, -0.5, 0.5, 1, 2, 3)}}
    print("__B64__"); print(base64.b64encode(gzip.compress(json.dumps(out).encode())).decode())


if __name__ == "__main__":
    main()
