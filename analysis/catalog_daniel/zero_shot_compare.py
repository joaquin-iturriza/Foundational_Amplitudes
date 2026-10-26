"""Zero-shot reads on the 139 held-out processes of arXiv:2606.23791's split, drawn for comparison with the draft's
sec:cond (per-process MSE of log|M|^2, natural log, no affine map; interpolation = the 96 exact relabels of a training
process, extrapolation = the 43 others).

  zs_vs_draft      fraction of held-out processes below a per-process MSE(log|M|^2), (a) the 96, (b) the 43, for four
                   of our pretrainings read on their own; the draft's medians (its figure mse_grid_by_type: the pID
                   embedding alone, and the in-out text prompt without it, mean of three seeds) as markers at 0.5.
  diagram_ab       per process, the dimensionless-input short run with the diagram encoder of arXiv:2606.23791 (a) and
                   with its virtuality-weighted pooling (b) against the same run without diagrams, on the 96 relabels,
                   the 43 new, the 40 2->3 processes at 100 TeV and the 40 2->4 probes (+g and +photon). One seed each.

Inputs: data/heldout_preds.npz + data/stats_net.json (hp58 read with the predicted statistics, as
catalog_short_heldout.py), data/catalog_norm_A.npz, data/catalog_short_*_sigstage.npz, data/catalog_2to4*_short_*.npz.
    python3 analysis/catalog_daniel/zero_shot_compare.py
"""
import json, os, sys

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402
from heldout_delta import G  # noqa: E402

FIG = os.path.join(HERE, "figs")
# the draft's medians (paper.tex sec:cond, fig. mse_grid_by_type): (interpolation, extrapolation)
DRAFT = {"draft, pID emb. only": (0.079, 0.649), "draft, in-out prompt": (0.072, 1.332)}


def load(f):
    return dict(np.load(os.path.join(HERE, "data", f + ".npz")))


def mse(z):
    ps_ = sorted({k.split("__")[0] for k in z if k.endswith("__t")})
    return {p: float(np.mean((z[p + "__q"].astype(float) - z[p + "__t"].astype(float)) ** 2))
            for p in ps_ if p + "__q" in z}


def hp58():
    z = load("heldout_preds")
    sn = json.load(open(os.path.join(HERE, "data", "stats_net.json")))
    for n, (mu, lsd) in zip(sn["heldout_names"], sn["heldout_pred"][G]):
        if f"{n}__h" in z:
            z[f"{n}__q"] = mu + np.exp(lsd) * z[f"{n}__h"].astype(float)
    return z


def ecdf(ax, v, **kw):
    v = np.sort(v)
    ax.step(np.r_[v[0], v], np.r_[0, np.arange(1, len(v) + 1) / len(v)], where="post", **kw)


def zs_vs_draft(twins):
    arms = [("per-process stats, predicted", hp58(), ps.C.grey),
            ("shared standardization", load("catalog_short_B_sigstage"), ps.C.blue),
            (r"$\hat s$ + coupling factor", load("catalog_norm_A"), ps.C.orange),
            ("dimensionless + coupling orders", load("catalog_short_DlessInOrd_sigstage"), ps.C.vermillion)]
    fig, axes = ps.figure(ncols=2)
    out = {}
    for j, (ax, rel) in enumerate(zip(axes, (True, False))):
        for lab, z, c in arms:
            m = mse(z)
            v = np.array([m[p] for p in m if (p in twins) == rel])
            ecdf(ax, v, color=c, label=lab)
            out[f"{lab}|{'96' if rel else '43'}"] = dict(n=len(v), median=float(np.median(v)))
        for (lab, med), mk in zip(DRAFT.items(), ("*", "D")):
            ax.plot([med[0 if rel else 1]], [0.5], mk, color="k", mfc="none" if mk == "D" else "k", label=lab)
        ax.set_xscale("log")
        ax.set_xlabel(r"per-process MSE$(\log|\mathcal{M}|^2)$")
        ax.set_ylabel("fraction of processes")
        ps.process_label(ax, "96 interpolation" if rel else "43 extrapolation", loc="lower right")
    ps.shared_legend(fig, axes[0], ncol=3)
    ps.save(fig, os.path.join(FIG, "zs_vs_draft"))
    return out


def diagram_ab(twins):
    sets = {"96 relabels": ("catalog_short", lambda p: p in twins),
            "43 new": ("catalog_short", lambda p: p not in twins),
            r"$2\to3$ at 100 TeV": ("catalog_2to4E100_short", None),
            r"$2\to4$": ("catalog_2to4_short", None)}
    colors = [ps.C.blue, ps.C.vermillion, ps.C.green, ps.C.purple]
    sfx = lambda pre, arm: f"{pre}_{arm}" + ("_sigstage" if pre == "catalog_short" else "")
    fig, axes = ps.figure(ncols=2)
    out = {}
    for ax, arm, name in zip(axes, ("DlessIn", "DlessInVirt"), ("encoder", "virtuality pooling")):
        lo, hi = np.inf, -np.inf
        for (lab, (pre, sel)), c in zip(sets.items(), colors):
            a, b = mse(load(sfx(pre, "DlessInNoDiag"))), mse(load(sfx(pre, arm)))
            pp = [p for p in a if p in b and (sel is None or sel(p))]
            x, y = np.array([a[p] for p in pp]), np.array([b[p] for p in pp])
            ax.plot(x, y, "o", color=c, label=lab, alpha=0.75)
            r = np.log10(x / y)
            out[f"{name}|{lab}"] = dict(n=len(pp), median_off=float(np.median(x)), median_on=float(np.median(y)),
                                        better=int((r > 0).sum()), median_dex=float(np.median(r)),
                                        p=float(wilcoxon(r).pvalue))
            lo, hi = min(lo, x.min(), y.min()), max(hi, x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--", color=ps.C.grey, label="equal")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo / 2, hi * 2); ax.set_ylim(lo / 2, hi * 2)
        ax.set_xlabel(r"MSE$(\log|\mathcal{M}|^2)$, no diagrams")
        ax.set_ylabel("MSE, " + name)
    ps.shared_legend(fig, axes[0], ncol=5)
    ps.save(fig, os.path.join(FIG, "diagram_ab"))
    return out


def main():
    twins = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    res = dict(zs_vs_draft=zs_vs_draft(twins), diagram_ab=diagram_ab(twins))
    for k, v in res.items():
        for kk, vv in v.items():
            print(k, kk, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in vv.items()})
    json.dump(res, open(os.path.join(HERE, "data", "zero_shot_compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
