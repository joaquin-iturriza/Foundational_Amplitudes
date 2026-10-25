"""The hybrid reference scale in a network: catalog_v3's short pretraining with the s_hat target plus a per-process
constant c_p (the mean over the process's own training events of the emission scales' factor less the s_hat one; arm
dimensionless_hybrid, scripts/job_catalog_v3_short.sh, data.target_physics_hybrid), read zero-shot on every held-out
axis like the other short arms (data/short_reads.json, analysis/catalog_v3/short_reads.py). Per axis: the medians of
its MSE, level and shape; against each of the other five arms (baseline, dimensionless = the emission scales, s_hat
alone, with the context vector, the twin) the median per-probe ratio of its MSE, its level and its shape, and how many
probes its MSE beats; on the sign-changing one-loop pools, the median wrong-sign fraction (<axis>_signed.txt).
Figure: per probe, (a) the hybrid's shape against s_hat alone's, (b) its level against the emission scales', the
probes of the two extrapolation axes (2->5, 1-10 TeV) apart: the design says (a) follows s_hat and (b) the emission
scales.
    python3 analysis/catalog_v3/prep_hybrid.py   -> data/prep_hybrid.json, figs/prep_hybrid.{png,pdf}
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from analysis.catalog_v3.prep_shat import signed  # noqa: E402

AXES = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c")
EXTRA = ("H2p5", "H7E")
ARM = "dimensionless_hybrid"
REFS = ("baseline", "dimensionless", "dimensionless_shat", "dimensionless_context", "dimensionless_twin")


def main():
    r = json.load(open(os.path.join(HERE, "data", "short_reads.json")))
    out = {}
    for ax in AXES:
        me = r[ARM][ax]
        o = {k: me[k] for k in ("n", "n_expected", "mse", "level", "shape")}
        o["missing"] = me["missing"]
        o["rank_mse"] = 1 + sum(r[ref][ax]["mse"] < me["mse"] for ref in REFS)   # 1 = the lowest median of the six
        for ref in REFS:
            a, b = r[ref][ax]["probes"], me["probes"]
            ps = sorted(set(a) & set(b))
            ratio = {k: np.array([b[p][k] / a[p][k] for p in ps]) for k in ("mse", "level", "shape")}
            o[f"vs_{ref}"] = {"n": len(ps), "median_mse_ratio": me["mse"] / r[ref][ax]["mse"],
                              **{f"{k}_ratio": float(np.median(v)) for k, v in ratio.items()},
                              "mse_better": int((ratio["mse"] < 1).sum()),
                              "shape_better": int((ratio["shape"] < 1).sum())}
        out[ax] = o
        print(ax, f"n={o['n']}/{o['n_expected']} mse {o['mse']:.3g} level {o['level']:.3g} shape {o['shape']:.3g} "
              f"rank {o['rank_mse']}")
        for ref in REFS:
            v = o[f"vs_{ref}"]
            print(f"   vs {ref:22s} medians {v['median_mse_ratio']:.2f}  per probe mse {v['mse_ratio']:.2f} "
                  f"level {v['level_ratio']:.2f} shape {v['shape_ratio']:.2f}  better {v['mse_better']}/{v['n']}")
    out["signed_H5"] = {arm: signed(arm) for arm in (ARM,) + REFS}
    print("signed", out["signed_H5"])
    json.dump(out, open(os.path.join(HERE, "data", "prep_hybrid.json"), "w"), indent=1)
    flat = {}                                            # the numbers the notes cite, one flat JSON line last
    short = dict(zip(REFS, ("base", "qperp", "shat", "ctx", "twin")))
    for ax in AXES:
        o = out[ax]
        flat.update({f"{ax}_{k}": float(f"{o[k]:.3g}") for k in ("mse", "level", "shape")})
        flat[f"{ax}_rank"] = o["rank_mse"]
        for ref, sh in short.items():
            v = o[f"vs_{ref}"]
            flat[f"{ax}_{sh}_ratio"] = round(v["median_mse_ratio"], 2)
            flat[f"{ax}_{sh}_level"] = round(v["level_ratio"], 2)
            flat[f"{ax}_{sh}_shape"] = round(v["shape_ratio"], 2)
            flat[f"{ax}_{sh}_better"] = v["mse_better"]
    for arm, sh in (("dimensionless_hybrid", "hyb"),) + tuple(short.items()):
        flat[f"wrong_sign_{sh}"] = round(100 * out["signed_H5"][arm]["wrong_sign"])
    plot(r)
    print(json.dumps(flat))


def plot(r):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    for ax, key, ref, lab, rl in zip(axs, ("shape", "level"), ("dimensionless_shat", "dimensionless"),
                                     ("shape", "level"), (r"$\hat s$ alone", "emission scales")):
        lo, hi = np.inf, 0
        for grp, col, mk, gl in ((False, ps.C.blue, "o", "within the catalog's range"),
                                 (True, ps.C.vermillion, "^", r"$2\to5$ and $1$-$10$ TeV")):
            xs, ys = [], []
            for a in AXES:
                if (a in EXTRA) != grp:
                    continue
                pa, pb = r[ref][a]["probes"], r[ARM][a]["probes"]
                for p in sorted(set(pa) & set(pb)):
                    xs.append(pa[p][key]); ys.append(pb[p][key])
            ax.plot(xs, ys, mk, mfc="none", color=col, label=gl)
            lo, hi = min(lo, min(xs), min(ys)), max(hi, max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], ls="--", color=ps.C.grey, label="equal")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"{lab}, {rl}"); ax.set_ylabel(f"{lab}, hybrid")
    ps.legend(axs[0], "upper left")
    for ax in axs:
        ps.make_room(ax, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "prep_hybrid"))


if __name__ == "__main__":
    main()
