"""The two last held-out axes of the dimensionless round (D93) as per-process ECDFs of the MSE of log|M|^2.
Left, particle content: the coupling-order design run DlessInOrd and its twin trained without any process with a top
quark, on all 162 catalog processes with a top (catalog_top_heldout.py). Right, perturbative order: DlessIn and
DlessInOrd on the trees and the one-loop pools of recipes/catalog_loop_probes.yaml, on the 79 probes both can score
(catalog_loop_heldout.py; mixed-order entries are refused by the coupling-order factor).
Inputs: data/catalog_top_short_DlessInOrd{,_notop}.npz, data/catalog_loop_short_{DlessIn,DlessInOrd}.npz.
Output: figs/dimless_axes.{png,pdf}.
    .venv-driver/bin/python analysis/catalog_daniel/dimless_axes.py
"""
import os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402


def per_mse(name):
    z = np.load(os.path.join(HERE, "data", name + ".npz"))
    ps_ = sorted({k.split("__")[0] for k in z.files})
    return {p: float(np.mean((z[f"{p}__q"].astype(float) - z[f"{p}__t"].astype(float)) ** 2)) for p in ps_}


def ecdf(ax, v, **kw):
    x = np.sort(np.asarray(v)); y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", **kw)


def main():
    fig, (axL, axR) = ps.figure(ncols=2)
    for name, col, lab in (("catalog_top_short_DlessInOrd", ps.C.blue, "trained with top"),
                           ("catalog_top_short_DlessInOrd_notop", ps.C.vermillion, "trained without top")):
        ecdf(axL, list(per_mse(name).values()), color=col, label=lab)
    a, b = per_mse("catalog_loop_short_DlessIn"), per_mse("catalog_loop_short_DlessInOrd")
    common = sorted(set(a) & set(b))
    loop = lambda p: "_nlo_" in p or "_loop_" in p
    for m, col, lab in ((a, ps.C.blue, "DlessIn"), (b, ps.C.vermillion, "+ coupling orders")):
        ecdf(axR, [m[p] for p in common if not loop(p)], color=col, ls="--", label=f"{lab}, trees")
        ecdf(axR, [m[p] for p in common if loop(p)], color=col, ls="-", label=f"{lab}, one loop")
    for ax in (axL, axR):
        ax.set_xscale("log"); ax.set_xlabel(r"MSE$(\log|\mathcal{M}|^2)$ per process"); ax.set_yticks([0, 0.5, 1])
    axL.set_ylabel("fraction of processes"); axR.set_ylabel("fraction of processes")
    ps.legend(axL, "upper left"); ps.legend(axR, "upper left")
    ps.make_room(axL); ps.make_room(axR)
    ps.save(fig, os.path.join(HERE, "figs", "dimless_axes"))


if __name__ == "__main__":
    main()
