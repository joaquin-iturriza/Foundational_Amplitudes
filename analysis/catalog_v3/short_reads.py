"""catalog_v3's short pretrainings read zero-shot on every held-out axis (scripts/job_catalog_v3_short.sh; predictions
copied home from $SCRATCH/zz_v3_short_<arm>/<axis>/<probe>.npz into data/short_reads/<arm>/<axis>/). Per probe, from
its saved truth t and the run's prediction q, both log|M|^2 on 2000 validation events (tools/zero_shot_parent.py):
  mse     mean (q - t)^2, the zero-shot error
  level   the mean of q - t squared, the part one number per process removes
  shape   the variance of q - t, what is left after that number
Per arm and axis: the medians over its probes, the 10% and 90% quantiles of the MSE, the probe count against the
axis recipe's (recipes/catalog_v3_heldout_<axis>.yaml; the pp-trained arm's and the twin's own axes likewise), and every
expected probe with no prediction, listed by name (a probe that errored on the site writes no .npz). The one-loop H5 probes are read on log|M| (--signed-magnitude), so their
numbers are on half the scale of the trees'. Figure: the median zero-shot MSE per axis, one marker per arm.
    python3 analysis/catalog_v3/short_reads.py   -> data/short_reads.json, figs/short_reads.{png,pdf}
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
ARMS = ("baseline", "dimensionless", "dimensionless_context", "dimensionless_twin", "dimensionless_pp")
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
        for axis in AXES:
            r = read(arm, axis)
            if not r:
                continue
            exp = expected(axis)
            missing = sorted(set(exp) - set(r))
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
    import plot_style as ps
    fig, ax = ps.figure()
    axes = [a for a in AXES if any(a in res.get(arm, {}) for arm in ARMS)]
    cols = {"baseline": ps.C.grey, "dimensionless": ps.C.blue, "dimensionless_context": ps.C.green,
            "dimensionless_twin": ps.C.vermillion, "dimensionless_pp": ps.C.orange}
    marks = {"baseline": "o", "dimensionless": "s", "dimensionless_context": "^", "dimensionless_twin": "D",
             "dimensionless_pp": "v"}
    for j, arm in enumerate(ARMS):
        xs = [i + 0.15 * (j - 2) for i, a in enumerate(axes) if a in res.get(arm, {})]
        ys = [res[arm][a]["mse"] for a in axes if a in res.get(arm, {})]
        if xs:
            ax.plot(xs, ys, marks[arm], color=cols[arm], label=arm.replace("_", " "))
    ax.set_yscale("log")
    ax.set_xticks(range(len(axes)))
    # each axis's read count, n of the recipe's (the smallest over the arms, so a short arm shows)
    ax.set_xticklabels([a + "\n" + str(min(res[arm][a]["n"] for arm in ARMS if a in res.get(arm, {}))) + "/"
                        + str(next(res[arm][a]["n_expected"] for arm in ARMS if a in res.get(arm, {}))) for a in axes])
    ax.set_xlabel("held-out axis"); ax.set_ylabel(r"median MSE$(\log|\mathcal{M}|^2)$")
    ps.legend(ax, "upper left"); ps.make_room(ax)
    ps.save(fig, os.path.join(HERE, "figs", "short_reads"))


if __name__ == "__main__":
    main()
