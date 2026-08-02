#!/usr/bin/env python
"""3D surfaces of <log|M|^2> over 2D cross sections of the 2->3 / 2->4 gluon phase
space, with the divergences visible as ridges/corners. Truth is a solid colour-mapped
surface, the model prediction a red wireframe on top: where they agree the wireframe
lies on the surface, and it peels off exactly where the model degrades.

Cross sections:
  ee->uug  : the Dalitz plane (x_q, x_qbar)  -> collinear ridges at x->1, soft corner
             (also the IR plane for comparison)
  ee->uugg : the IR-resolution plane (log10 x_gmin, log10 y_min) -> deep-IR corner
CPU only (reads the extract_ir npz)."""
import argparse
import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import binned_statistic_2d

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402


def surface(x, y, tl, pl, xlabel, ylabel, process, out_base,
            nb=42, mincount=10, views=((26, -58), (26, 128))):
    xb = np.linspace(np.percentile(x, 0.1), np.percentile(x, 99.9), nb + 1)
    yb = np.linspace(np.percentile(y, 0.1), np.percentile(y, 99.9), nb + 1)
    T, xe, ye, _ = binned_statistic_2d(x, y, tl, "mean", bins=[xb, yb])
    P, _, _, _ = binned_statistic_2d(x, y, pl, "mean", bins=[xb, yb])
    C, _, _, _ = binned_statistic_2d(x, y, np.ones_like(tl), "sum", bins=[xb, yb])
    sparse = C < mincount
    T = np.where(sparse, np.nan, T)          # keep the physical support's shape
    P = np.where(sparse, np.nan, P)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    norm = plt.Normalize(np.nanmin(T), np.nanmax(T))
    fc = cm.viridis(norm(np.nan_to_num(T, nan=np.nanmin(T))))
    fc[..., 3] = np.where(np.isnan(T), 0.0, 0.55)   # transparent where unpopulated

    fig = plt.figure(figsize=ps.figsize(ncols=2))
    rms = float(np.sqrt(np.nanmean((pl - tl) ** 2)))
    for k, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(X, Y, T, facecolors=fc, rstride=1, cstride=1, linewidth=0,
                        antialiased=True, shade=False)
        ax.plot_wireframe(X, Y, P, color=ps.C.vermillion, linewidth=0.55, rstride=2,
                          cstride=2, alpha=0.9)
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.set_zlabel(r"$\langle\log|\mathcal{M}|^2\rangle$", labelpad=6)
        ax.view_init(elev=elev, azim=azim)
    m = cm.ScalarMappable(norm=norm, cmap=cm.viridis); m.set_array([])
    cb = fig.colorbar(m, ax=fig.axes, fraction=0.02, pad=0.02)
    cb.set_label(r"truth $\langle\log|\mathcal{M}|^2\rangle$")
    from matplotlib.lines import Line2D
    # The process label goes at figure level here: a 3-D axes has no sane in-axes corner.
    fig.legend([Line2D([0], [0], color=ps.C.vermillion, lw=1.5)],
               ["model prediction"], loc="upper right", frameon=False)
    fig.text(0.02, 0.96, process, ha="left", va="top")
    ps.save(fig, out_base)
    plt.close(fig)
    print(f"  RMSD={rms:.3g}")


def main():
    D = "analysis/divergences"
    uug = np.load(f"{D}/preds_ir_pretrain8_ee_uug_91-1000GeV_amplitudes.npz")
    uugg = np.load(f"{D}/preds_ir_pretrain8_ee_uugg_91-1000GeV_amplitudes.npz")

    # ee->uug : Dalitz plane (collinear ridges at x->1, soft gluon at the (1,1) corner)
    surface(uug["x_q"], uug["x_qbar"], uug["true_logamp"], uug["pred_logamp"],
            r"$x_q$", r"$x_{\bar q}$",
            r"$e^+e^-\to u\bar u g$",
            f"{D}/figs/ir3d_uug_dalitz")

    # ee->uug and ee->uugg : IR-resolution plane
    for d, tag, ttl in [
        (uug, "uug", r"$e^+e^-\to u\bar u g$"),
        (uugg, "uugg", r"$e^+e^-\to u\bar u gg$")]:
        surface(np.log10(np.clip(d["y_min"], 1e-12, None)),
                np.log10(np.clip(d["x_gmin"], 1e-12, None)),
                d["true_logamp"], d["pred_logamp"],
                r"$\log_{10} y_{\min}$", r"$\log_{10} x_{g,\min}$",
                ttl,
                f"{D}/figs/ir3d_{tag}_irplane")


if __name__ == "__main__":
    main()
