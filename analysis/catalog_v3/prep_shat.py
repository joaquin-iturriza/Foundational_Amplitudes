"""The shape proxy's candidate in a network: catalog_v3's short pretraining with s_hat alone as the target's reference
scale (arm dimensionless_shat, scripts/job_catalog_v3_short.sh; the same dimensionless inputs and coupling orders as the
dimensionless arm, whose reference is the emission scales), read zero-shot on every held-out axis like the other short
arms (data/short_reads.json, analysis/catalog_v3/short_reads.py). Per axis: the medians of its MSE, level and shape,
the median per-probe ratio of its MSE and of its shape to the dimensionless arm's and the baseline's, and how many probes
it beats them on; on the sign-changing one-loop pools, the median wrong-sign fraction of each arm (<axis>_signed.txt).
Figure: per probe, (a) the shape and (b) the MSE of the s_hat arm against the dimensionless arm's, the probes of the
two extrapolation axes (2->5, 1-10 TeV) apart.
    python3 analysis/catalog_v3/prep_shat.py   -> data/prep_shat.json, figs/prep_shat.{png,pdf}
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
AXES = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c")
EXTRA = ("H2p5", "H7E")
ARM, REFS = "dimensionless_shat", ("dimensionless", "baseline")


def signed(arm):
    f = os.path.join(HERE, "data", "short_reads", arm, "H5_signed.txt")
    rows = [json.loads(l.split("SIGNED_AB ", 1)[1]) for l in open(f) if "SIGNED_AB" in l]
    s = [r for r in rows if "skipped" not in r and r["pool"].startswith("v3_")]   # catalog_v4 pools a shared folder let in
    return {"pools": len(s), "wrong_sign": float(np.median([r["wrong_sign"] for r in s]))}


def main():
    r = json.load(open(os.path.join(HERE, "data", "short_reads.json")))
    out = {}
    for ax in AXES:
        me = r[ARM][ax]
        o = {k: me[k] for k in ("n", "n_expected", "mse", "level", "shape")}
        o["missing"] = me["missing"]
        for ref in REFS:
            a, b = r[ref][ax]["probes"], me["probes"]
            ps = sorted(set(a) & set(b))
            rm = np.array([b[p]["mse"] / a[p]["mse"] for p in ps])
            rs = np.array([b[p]["shape"] / a[p]["shape"] for p in ps])
            o[f"vs_{ref}"] = {"n": len(ps), "mse_ratio": float(np.median(rm)), "mse_better": int((rm < 1).sum()),
                              "shape_ratio": float(np.median(rs)), "shape_better": int((rs < 1).sum())}
        out[ax] = o
        print(ax, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if k != "missing"}))
    out["signed_H5"] = {arm: signed(arm) for arm in (ARM,) + REFS}
    print("signed", out["signed_H5"])
    json.dump(out, open(os.path.join(HERE, "data", "prep_shat.json"), "w"), indent=1)
    plot(r)


def plot(r):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    for ax, key, lab in zip(axs, ("shape", "mse"), ("shape", "MSE")):
        lo, hi = np.inf, 0
        for grp, col, mk, gl in ((False, ps.C.blue, "o", "within the catalog's range"),
                                 (True, ps.C.vermillion, "^", r"$2\to5$ and $1$-$10$ TeV")):
            xs, ys = [], []
            for a in AXES:
                if (a in EXTRA) != grp:
                    continue
                pa, pb = r["dimensionless"][a]["probes"], r[ARM][a]["probes"]
                for p in sorted(set(pa) & set(pb)):
                    xs.append(pa[p][key]); ys.append(pb[p][key])
            ax.plot(xs, ys, mk, mfc="none", color=col, label=gl)
            lo, hi = min(lo, min(xs), min(ys)), max(hi, max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], ls="--", color=ps.C.grey, label="equal")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"{lab}, emission scales"); ax.set_ylabel(rf"{lab}, $\hat s$ alone")
    ps.legend(axs[0], "upper left")
    for ax in axs:
        ps.make_room(ax, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "prep_shat"))


if __name__ == "__main__":
    main()
