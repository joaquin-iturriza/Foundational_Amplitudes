#!/usr/bin/env python
"""3D surface of ⟨log|M|²⟩ over the (√s, cosθ*) phase space, with the model
prediction overlaid on the truth. Truth is a solid colour-mapped surface; the
prediction is a red wireframe drawn on top — where they agree the wireframe lies
exactly on the surface. Two viewing azimuths per process for legibility.
CPU only (reads the npz produced by extract_preds.py)."""
import argparse
import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

REPO = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402
from scipy.stats import binned_statistic_2d


def grid_means(x, y, z, xbins, ybins):
    m, xe, ye, _ = binned_statistic_2d(x, y, z, statistic="mean", bins=[xbins, ybins])
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    return X, Y, m   # m shape (nx, ny)


def make(npz, label, out_base, nb=40):
    d = np.load(npz, allow_pickle=True)
    s, c = d["sqrt_s"], d["cos_theta"]
    tl, pl = d["true_logamp"], d["pred_logamp"]
    xbins = np.linspace(s.min(), s.max(), nb + 1)
    ybins = np.linspace(-1.0, 1.0, nb + 1)
    X, Y, T = grid_means(s, c, tl, xbins, ybins)
    _, _, P = grid_means(s, c, pl, xbins, ybins)

    # fill sparse/empty bins by nearest finite so surfaces stay continuous
    def fill(a):
        a = a.copy()
        if np.isnan(a).any():
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(np.isnan(a), return_distances=False,
                                         return_indices=True)
            a = a[tuple(idx)]
        return a
    T, Pf = fill(T), fill(P)

    resid = pl - tl
    rms = float(np.sqrt(np.mean(resid ** 2)))

    fig = plt.figure(figsize=ps.figsize(ncols=2))
    norm = plt.Normalize(np.nanmin(T), np.nanmax(T))
    for k, (elev, azim) in enumerate([(28, -60), (28, 130)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(X, Y, T, facecolors=cm.viridis(norm(T)),
                        rstride=1, cstride=1, linewidth=0, antialiased=True,
                        alpha=0.55, shade=False)
        ax.plot_wireframe(X, Y, Pf, color=ps.C.vermillion, linewidth=0.6,
                          rstride=2, cstride=2, alpha=0.95)
        ax.set_xlabel(r"$\sqrt{s}$ [GeV]", labelpad=8)
        ax.set_ylabel(r"$\cos\theta^{*}$", labelpad=8)
        ax.set_zlabel(r"$\langle\log|\mathcal{M}|^2\rangle$", labelpad=6)
        ax.view_init(elev=elev, azim=azim)

    # shared legend / colorbar
    m = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    m.set_array([])
    # ps.layout() skips any figure with a 3-D axes (a projection's extent is its bounding
    # square, not a plot box), so the right margin has to be reserved here or the colourbar
    # label runs off the canvas and savefig (bbox=None) cuts it.
    fig.subplots_adjust(left=0.01, right=0.84)
    cb = fig.colorbar(m, ax=fig.axes, fraction=0.02, pad=0.02)
    cb.set_label(r"truth $\langle\log|\mathcal{M}|^2\rangle$")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], color=ps.C.vermillion, lw=1.5)], ["model prediction"],
               loc="upper right", frameon=False)
    fig.text(0.02, 0.96, label, ha="left", va="top")
    ps.save(fig, out_base)
    plt.close(fig)
    print(f"wrote {out_base}.png / .pdf  (RMS Δ={rms:.3g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--nbins", type=int, default=40)
    args = ap.parse_args()
    make(args.npz, args.label, args.out_base, args.nbins)


if __name__ == "__main__":
    main()
