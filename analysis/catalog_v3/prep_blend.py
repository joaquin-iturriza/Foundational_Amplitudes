"""The blend of the hybrid and the emission scales in a network: catalog_v3's short pretraining with the hybrid target
(s_hat alone plus the per-process constant c_p) plus lambda times the emission-scale term less its process mean
(physics_scales.hybrid_blend; arms dimensionless_blend025 and dimensionless_blend050 of scripts/job_catalog_v3_short.sh,
runs FA-ccin2p3-6d35589c and FA-ccin2p3-0bddbf48; lambda 0 is the hybrid arm, lambda 1 the emission scales'
(dimensionless) arm), read zero-shot on every held-out axis like the other short arms (data/short_reads.json,
analysis/catalog_v3/short_reads.py). Per axis and lambda in (0, 0.25, 0.5, 1): the medians of MSE, level and shape; per
blend the median per-probe ratio of its MSE, level and shape to the hybrid's and to the emission scales', the probes
whose MSE beats the hybrid's, and the arm with the lowest median MSE among the eight short arms read on the same axis;
on the sign-changing one-loop pools, the median wrong-sign fraction (<axis>_signed.txt).
Figure: the median level (a) and shape (b) per axis against lambda, each over the hybrid's.
    python3 analysis/catalog_v3/prep_blend.py   -> data/prep_blend.json, figs/prep_blend.{png,pdf}; one flat JSON line last
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from analysis.catalog_v3.prep_shat import signed  # noqa: E402
from analysis.catalog_v3.short_reads import AXES as AXIS_NAMES  # noqa: E402

AXES = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c")
LAMBDA = {"dimensionless_hybrid": 0.0, "dimensionless_blend025": 0.25, "dimensionless_blend050": 0.5,
          "dimensionless": 1.0}
BLENDS = ("dimensionless_blend025", "dimensionless_blend050")
SHORT = {"dimensionless_hybrid": "hyb", "dimensionless_blend025": "b25", "dimensionless_blend050": "b50",
         "dimensionless": "qperp", "dimensionless_shat": "shat", "baseline": "base", "dimensionless_context": "ctx",
         "dimensionless_twin": "twin"}


def ratios(r, ax, num, den):
    a, b = r[den][ax]["probes"], r[num][ax]["probes"]
    ps = sorted(set(a) & set(b))
    rat = {k: np.array([b[p][k] / a[p][k] for p in ps]) for k in ("mse", "level", "shape")}
    return {"n": len(ps), **{f"{k}_ratio": float(np.median(v)) for k, v in rat.items()},
            "mse_better": int((rat["mse"] < 1).sum())}


def main():
    r = json.load(open(os.path.join(HERE, "data", "short_reads.json")))
    out, flat = {}, {}
    for ax in AXES:
        o = {"medians": {arm: {k: r[arm][ax][k] for k in ("n", "mse", "level", "shape")} for arm in SHORT}}
        o["best_mse_arm"] = min(SHORT, key=lambda a: r[a][ax]["mse"])
        for b in BLENDS:
            o[f"{b}_vs_hybrid"] = ratios(r, ax, b, "dimensionless_hybrid")
            o[f"{b}_vs_qperp"] = ratios(r, ax, b, "dimensionless")
        out[ax] = o
        print(f"{ax:5s} n={r['dimensionless_blend050'][ax]['n']:3d}  best {o['best_mse_arm']}")
        for arm in LAMBDA:
            m = o["medians"][arm]
            print(f"   lambda {LAMBDA[arm]:4.2f} mse {m['mse']:.3g} level {m['level']:.3g} shape {m['shape']:.3g}")
            flat.update({f"{ax}_{SHORT[arm]}_{k}": float(f"{m[k]:.3g}") for k in ("mse", "level", "shape")})
        for b in BLENDS:
            for ref, rs in (("hybrid", "hyb"), ("qperp", "qperp")):
                v = o[f"{b}_vs_{ref}"]
                print(f"   {b} vs {ref:6s} per probe mse {v['mse_ratio']:.2f} level {v['level_ratio']:.2f} "
                      f"shape {v['shape_ratio']:.2f}  better {v['mse_better']}/{v['n']}")
                flat.update({f"{ax}_{SHORT[b]}_{rs}_{k}": round(v[f"{k}_ratio"], 2) for k in ("mse", "level", "shape")})
                flat[f"{ax}_{SHORT[b]}_{rs}_better"] = v["mse_better"]
        flat[f"{ax}_n"] = r["dimensionless_blend050"][ax]["n"]
        flat[f"{ax}_best"] = SHORT[o["best_mse_arm"]]
        flat[f"{ax}_best_mse"] = float(f"{r[o['best_mse_arm']][ax]['mse']:.3g}")
    out["signed_H5"] = {arm: signed(arm) for arm in LAMBDA}
    print("signed", out["signed_H5"])
    for arm in LAMBDA:
        flat[f"wrong_sign_{SHORT[arm]}"] = round(100 * out["signed_H5"][arm]["wrong_sign"])
    flat["n_probes"] = sum(r["dimensionless_blend050"][ax]["n"] for ax in AXES)
    json.dump(out, open(os.path.join(HERE, "data", "prep_blend.json"), "w"), indent=1)
    plot(r)
    print(json.dumps(flat))


def plot(r):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    lams = list(LAMBDA.values())
    cols = (ps.C.blue, ps.C.vermillion, ps.C.green, ps.C.orange, ps.C.purple, ps.C.sky, ps.C.yellow, ps.C.grey)
    mks = ("o", "s", "^", "v", "D", "<", ">", "p")
    for ax, key in zip(axs, ("level", "shape")):
        for a, c, m in zip(AXES, cols, mks):
            y = [r[arm][a][key] / r["dimensionless_hybrid"][a][key] for arm in LAMBDA]
            ax.plot(lams, y, marker=m, color=c, mfc="none", label=AXIS_NAMES[a])
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(f"{key}" + r", over $\lambda=0$")
    ps.shared_legend(fig, axs[0], ncol=4)                  # eight axes fit in no panel: one strip above
    ps.save(fig, os.path.join(HERE, "figs", "prep_blend"))


if __name__ == "__main__":
    main()
