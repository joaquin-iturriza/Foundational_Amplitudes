"""Step 43(1) loader control as a figure (plan catalog_coverage, design passed 2026-11-05T05:20).

Arm v4_het_default25[_seed2] at one commit on V100: memmap (FA-ccin2p3-5b5215b6, e184bbc5) against the same-commit
in-memory rerun (ae39ee5a, 75ad7b20), seeds 42 and 43; reads PP4 (trained classes), EE4 (trained) and the
held-out default-sampled probes, exactly as mixture_ab.loader() selects them.

(a) per read: R_L = median over classes of the class's geometric mean of MSE(memmap)/MSE(rerun), seeds pooled by the
    per-class geometric mean over seeds, with the bootstrap 95% interval over classes; beside it the rerun spread
    R_0 = rerun / the 2026-10-31 in-memory arm, same construction, and the gap the control was sent to split
    (PP4 MSE of the full-set width-8 cell over the 1e4-per-process arm, 1.39).
(b) per class, PP4 trained: MSE of log|M|^2 memmap against in-memory rerun (seed geometric mean), with y = x.

Writes figs/loader_control.{png,pdf} and data/loader_control_fig.json (the plotted numbers).
"""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import mixture_ab as mx  # noqa: E402
import plot_style as ps  # noqa: E402

READS3 = (("PP4_trained", "PP4"), ("EE4_trained", "EE4"), ("held_default", "held-out\ndefault"))


def reads(run, ref_src, cls, ax_of):
    """mixture_ab.loader()'s selection, for one run (or the 10-31 arm via ref_src)."""
    out = {}
    for r in ("PP4", "EE4"):
        d = mx.read_of(ref_src(r), r) if ref_src else mx.read_of(run, r)
        out[r + "_trained"] = {p: x for p, x in d.items() if cls(p) not in ax_of}
        out.setdefault("held_default", {}).update({(r, p): x for p, x in d.items() if cls(p) in ax_of})
    return out


def class_gm(d, keys, cl):
    return {c: mx.gmean(v) for c, v in mx._by_class(d, keys, cl).items()}


def median_ci(v, n_boot=10000, seed=0):
    v = np.asarray(v)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(np.median(v)), float(lo), float(hi)


def main():
    cls = mx.class_of()
    ax_of = mx._held_axes(cls)
    cl = lambda k: cls(mx.probe_name(k))
    L = mx.LOADER
    per = {"memmap": [], "rerun": [], "arm": []}
    for i in range(2):
        per["memmap"].append(reads(L["memmap"][i], None, cls, ax_of))
        per["rerun"].append(reads(L["rerun"][i], None, cls, ax_of))
        per["arm"].append(reads(None, lambda r, i=i: L["arm_1031"][r][i], cls, ax_of))
    res = {"runs": L, "reads": {}}
    scatter = None
    for q, _ in READS3:
        cm = {}
        for w in per:
            ks = sorted(set.intersection(*(set(per[x][i][q]) for x in per for i in range(2))), key=str)
            cm[w] = [class_gm(per[w][i][q], ks, cl) for i in range(2)]
        classes = sorted(cm["memmap"][0])
        sgm = {w: {c: math.sqrt(cm[w][0][c] * cm[w][1][c]) for c in classes} for w in cm}
        rl = [sgm["memmap"][c] / sgm["rerun"][c] for c in classes]
        r0 = [sgm["rerun"][c] / sgm["arm"][c] for c in classes]
        res["reads"][q] = {"n_classes": len(classes), "R_L": median_ci(rl), "R_0": median_ci(r0),
                           "frac_RL_below_1": float(np.mean(np.array(rl) < 1)),
                           "per_seed_R_L": [float(np.median([cm["memmap"][i][c] / cm["rerun"][i][c] for c in classes]))
                                            for i in range(2)]}
        if q == "PP4_trained":
            scatter = (np.array([sgm["rerun"][c] for c in classes]), np.array([sgm["memmap"][c] for c in classes]))
            res["reads"][q]["spearman_memmap_rerun"] = float(np.corrcoef(np.argsort(np.argsort(scatter[0])),
                                                                           np.argsort(np.argsort(scatter[1])))[0, 1])
            res["reads"][q]["median_abs_log10_ratio"] = float(np.median(np.abs(np.log10(scatter[1] / scatter[0]))))
    res["gap"] = math.exp(mx.LOADER_GAP)

    fig, (a, b) = ps.figure(ncols=2)
    x = np.arange(len(READS3))
    for off, key, col, lab in ((-0.12, "R_L", ps.C.vermillion, r"$R_L$ = memmap / in-memory rerun"),
                               (0.12, "R_0", ps.C.blue, r"$R_0$ = rerun / in-memory arm of 10-31")):
        m = np.array([res["reads"][q][key] for q, _ in READS3])
        a.errorbar(x + off, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt="o", color=col, label=lab)
    a.axhline(1.0, color=ps.C.grey, ls="--", label=r"$R=1$")
    a.axhline(res["gap"], color=ps.C.green, ls=":",
              label=r"full set / $10^4$ per process, PP4")
    a.set_ylim(0.9, res["gap"] + 0.05)     # the gap line is part of the comparison, so it stays in range
    a.set_xticks(x)
    a.set_xticklabels([lab for _, lab in READS3])
    a.set_ylabel(r"median over classes of MSE ratio")
    ps.legend(a, "upper left")
    rr, mm = scatter
    b.scatter(rr, mm, color=ps.C.vermillion, label="PP4 trained class")
    lo, hi = min(rr.min(), mm.min()), max(rr.max(), mm.max())
    b.plot([lo, hi], [lo, hi], color="0.5", ls="--", label=r"$y=x$")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel(r"MSE$(\log|\mathcal{M}|^2)$, in-memory")
    b.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$, memmap")
    ps.legend(b, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "loader_control"))
    json.dump(res, open(os.path.join(HERE, "data", "loader_control_fig.json"), "w"), indent=1)
    print(json.dumps({q: {k: v for k, v in r.items()} for q, r in res["reads"].items()}, indent=1))


if __name__ == "__main__":
    main()
