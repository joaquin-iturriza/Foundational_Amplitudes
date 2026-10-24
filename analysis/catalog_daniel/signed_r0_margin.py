"""D83/D99, the signed target's r0 margin before the GPU A/B: r0 = q_hat e^-m, the pool's 0.5% quantile of |V F| as a
level model would predict it, m nats below. With the prediction off by the level error o (nat, prediction minus truth),
the r0 used sits e = o - m off the pool's own quantile, and signed_band.py's lin_shift gives the share of events in the
linear band at that e. Instead of the median |offset| alone, this reads the whole spread: o from the zero-shot one-loop
errors (catalog_loop_short_<arm>.npz, the one-loop pools: per process the mean of q - t, as qperp_ab.py), crossed with
the 53 signed pools' lin_shift curves (data/signed_band.json; linear interpolation in e, the curve's end values held
beyond -3 and +3, so e > 3 is a lower bound and counted separately). Per margin m in 0..5: the median share over the
pairs (signed pool, offset), the fraction of pairs over 10%, and the fraction with e > 3.
    .venv-driver/bin/python analysis/catalog_daniel/signed_r0_margin.py   -> data/signed_r0_margin.json
"""
import json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("DlessInOrd", "DlessInOrdQ")
LOOP = ("_nlo_", "_loop_")


def offsets(arm):
    z = np.load(os.path.join(HERE, "data", f"catalog_loop_short_{arm}.npz"))
    ps = sorted({k.split("__")[0] for k in z.files if k.endswith("__q")})
    return np.array([float(np.mean(z[f"{p}__q"].astype(float) - z[f"{p}__t"].astype(float)))
                     for p in ps if any(s in p for s in LOOP)])


def main():
    sb = json.load(open(os.path.join(HERE, "data", "signed_band.json")))["pools"]
    curves = []
    for v in sb.values():
        e = sorted((float(k), x) for k, x in v["lin_shift"].items())
        e = [(0.0, v["lin_own"])] + e
        e.sort()
        curves.append((np.array([a for a, _ in e]), np.array([b for _, b in e])))
    out = {"pools": len(curves)}
    for arm in ARMS:
        o = offsets(arm)
        row = {"n_offsets": int(len(o)), "offset_q": [float(x) for x in np.quantile(o, [0.1, 0.5, 0.9])],
               "abs_offset_median": float(np.median(np.abs(o)))}
        for m in range(6):
            e = o - m
            sh = np.array([np.interp(e, ex, ey) for ex, ey in curves])     # (pools, offsets)
            row[f"m{m}"] = {"median_share": float(np.median(sh)), "frac_over_10pct": float(np.mean(sh > 0.1)),
                            "frac_e_over_3": float(np.mean(e > 3))}
        out[arm] = row
        print(arm, json.dumps(row))
    json.dump(out, open(os.path.join(HERE, "data", "signed_r0_margin.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
