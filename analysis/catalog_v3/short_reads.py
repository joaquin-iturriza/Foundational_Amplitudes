"""catalog_v3's short pretrainings read zero-shot on every held-out axis (scripts/job_catalog_v3_short.sh; predictions
copied home from $SCRATCH/zz_v3_short_<arm>/<axis>/<probe>.npz into data/short_reads/<arm>/<axis>/). Per probe, from
its saved truth t and the run's prediction q, both log|M|^2 on 2000 validation events (tools/zero_shot_parent.py):
  mse     mean (q - t)^2, the zero-shot error
  level   the mean of q - t squared, the part one number per process removes
  shape   the variance of q - t, what is left after that number
Per arm and axis: the medians over its probes, the 10% and 90% quantiles of the MSE, the probe count against the
axis recipe's (recipes/catalog_v3_heldout_<axis>.yaml; the pp-trained arm's and the twin's own axes likewise), and every
expected probe with no prediction, listed by name (a probe that errored on the site writes no .npz). A sign-changing one-loop pool (H5, the twin's H3p) is read on
log max(|M|^2, floor) (--signed-magnitude), the sign head's magnitude; its sign is scored apart (<axis>_signed.txt). Figure: the median zero-shot MSE per axis, one marker per arm.
    python3 analysis/catalog_v3/short_reads.py   -> data/short_reads.json, figs/short_reads.{png,pdf}
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
ARMS = ("baseline", "dimensionless", "dimensionless_shat", "dimensionless_context", "dimensionless_twin", "dimensionless_pp")
# the axes each arm's job reads (scripts/job_catalog_v3_short.sh): every arm the eight, the twin also its two
READ = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c")
READS = {arm: READ + (("H3p", "H4") if arm == "dimensionless_twin" else ()) for arm in
         ("baseline", "dimensionless", "dimensionless_shat", "dimensionless_context", "dimensionless_twin",
          "dimensionless_pp")}
AXES = {"H1": "new combinations", "H2": r"$2\to4$ families", "H2p5": r"$2\to5$", "H3": "the top",
        "H5": "one loop", "H6": r"$\alpha_s=0.13$", "H7E": r"$1$-$10$ TeV", "H7c": r"$p_T\geq30$ GeV",
        "H3p": r"$\tau$, $\nu_\tau$ never seen", "H4": "quartic vertex never seen"}


def expected(axis):
    import yaml
    f = os.path.join(os.path.dirname(os.path.dirname(HERE)), "recipes", f"catalog_v3_heldout_{axis}.yaml")
    return [e["name"] for e in yaml.safe_load(open(f))["processes"]]


def read(arm, axis):
    rows = {}
    for f in sorted(glob.glob(os.path.join(HERE, "data", "short_reads", arm, axis, "*.npz"))):
        z = np.load(f)
        if "q" not in z.files:
            continue
        d = z["q"].astype(float) - z["t"].astype(float)
        rows[os.path.basename(f)[:-4]] = {"mse": float(np.mean(d ** 2)), "level": float(np.mean(d) ** 2),
                                          "shape": float(np.var(d))}
    return rows


def main():
    res = {}
    for arm in ARMS:
        if not os.path.isdir(os.path.join(HERE, "data", "short_reads", arm)):
            print(f"{arm}: no reads copied home")         # the arm's reads not copied home yet
            continue
        for axis in READS[arm]:
            r = read(arm, axis)
            exp = expected(axis)
            missing = sorted(set(exp) - set(r))
            if not r:                                      # every probe of the axis failed: recorded, never dropped
                res.setdefault(arm, {})[axis] = {"n": 0, "n_expected": len(exp), "missing": missing}
                print(f"{arm:24s} {axis:5s} n=   0/{len(exp):4d}  NO READS")
                continue
            m = np.array([v["mse"] for v in r.values()])
            res.setdefault(arm, {})[axis] = {
                "n": len(r), "n_expected": len(exp), "missing": missing, "mse": float(np.median(m)), "mse_q10_q90": [float(x) for x in np.quantile(m, [0.1, 0.9])],
                "level": float(np.median([v["level"] for v in r.values()])),
                "shape": float(np.median([v["shape"] for v in r.values()])), "probes": r}
            print(f"{arm:24s} {axis:5s} n={len(r):4d}/{len(exp):4d} mse {res[arm][axis]['mse']:.3g} "
                  f"level {res[arm][axis]['level']:.3g} shape {res[arm][axis]['shape']:.3g}")
    json.dump(res, open(os.path.join(HERE, "data", "short_reads.json"), "w"), indent=1)
    plot(res)


def plot(res):
    """per probe, one arm's zero-shot MSE against another's on the same probes: (a) the dimensionless target against
    the baseline, (b) the context vector against the dimensionless target without it; the probes of the two
    extrapolation axes (2->5, 1-10 TeV) apart from the rest. The table in docs/results.tex carries the counts."""
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    extra = ("H2p5", "H7E")
    for ax, (x_arm, y_arm, xl, yl) in zip(axs, (("baseline", "dimensionless", "baseline", "dimensionless target"),
                                               ("dimensionless", "dimensionless_context", "dimensionless target",
                                                "with the context vector"))):
        lo, hi = np.inf, 0
        for grp, col, mk, lab in ((False, ps.C.blue, "o", "within the catalog's range"),
                                  (True, ps.C.vermillion, "^", r"$2\to5$ and $1$-$10$ TeV")):
            xs, ys = [], []
            for a in AXES:
                if (a in extra) != grp or a not in res.get(x_arm, {}) or a not in res.get(y_arm, {}):
                    continue
                px, py = res[x_arm][a].get("probes", {}), res[y_arm][a].get("probes", {})
                for p in sorted(set(px) & set(py)):
                    xs.append(px[p]["mse"]); ys.append(py[p]["mse"])
            if xs:
                ax.plot(xs, ys, mk, mfc="none", color=col, label=lab)
                lo, hi = min(lo, min(xs), min(ys)), max(hi, max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], ls="--", color=ps.C.grey, label="equal")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"MSE, {xl}"); ax.set_ylabel(f"MSE, {yl}")
    ps.legend(axs[0], "upper left")
    for ax in axs:
        ps.make_room(ax, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "short_reads"))


if __name__ == "__main__":
    main()
