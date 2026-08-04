#!/usr/bin/env python
"""Plot true vs predicted log-amplitude across the 2->2 phase space (sqrt_s,
cos_theta*) for a finetuned NLO run, to expose how the model behaves near the
amplitude divergences (low-sqrt_s / resonant growth, forward-backward peaks,
threshold) and whether accuracy degrades there.

Produces, per process, THREE figures (png + pdf each), because one canvas holding all of
this came out wider than the paper and had to shrink its panels to fit:
  <out>_{a,b,c}          : <log|M|^2> true, model, mean |Δlog|M|^2| (error) 2D maps
  <out>_proj_{a,b,c}     : projection vs sqrt_s, vs cos_theta, pred-vs-true hexbin
  <out>_collinear        : collinear-resolved log|M|^2 vs eta (mean & max), error vs eta
CPU only.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, binned_statistic_2d

import sys
REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402


def load(npz):
    d = np.load(npz, allow_pickle=True)
    return d


def collinear_coord(cos):
    """Signed collinear coordinate that stretches both beam directions so the
    collinear divergence (crammed against cos->+-1, washed out by mean-per-bin in
    linear cos) becomes a visible linear ramp:
        eta = sign(cos) * log10( 1/(1-|cos|) );  eta=0 central, eta->+-6 collinear."""
    a = np.clip(1.0 - np.abs(cos), 1e-7, None)
    return -np.sign(cos) * np.log10(a)


def _map(x, y, z, xbins, ybins, stat="mean"):
    s, xe, ye, _ = binned_statistic_2d(x, y, z, statistic=stat, bins=[xbins, ybins])
    return s.T, xe, ye   # transpose so rows=y, cols=x for pcolormesh(x,y)


def make_figure(npz, process_label, out_base, split="all"):
    d = load(npz)
    sqrt_s = d["sqrt_s"]
    cos_t = d["cos_theta"]
    true_l = d["true_logamp"]
    pred_l = d["pred_logamp"]
    resid = pred_l - true_l                    # error in log|M|^2

    if split != "all":
        code = {"train": 0, "val": 1, "test": 2}[split]
        m = d["split"] == code
        sqrt_s, cos_t, true_l, pred_l, resid = (
            sqrt_s[m], cos_t[m], true_l[m], pred_l[m], resid[m])

    n = sqrt_s.shape[0]
    s_lo, s_hi = np.percentile(sqrt_s, [0.0, 100.0])
    xbins = np.linspace(s_lo, s_hi, 45)
    ybins = np.linspace(-1.0, 1.0, 45)

    true_map, xe, ye = _map(sqrt_s, cos_t, true_l, xbins, ybins)
    pred_map, _, _ = _map(sqrt_s, cos_t, pred_l, xbins, ybins)
    err_map, _, _ = _map(sqrt_s, cos_t, np.abs(resid), xbins, ybins)
    cnt_map, _, _ = _map(sqrt_s, cos_t, np.ones_like(true_l), xbins, ybins, stat="sum")

    Xc = 0.5 * (xe[:-1] + xe[1:])
    Yc = 0.5 * (ye[:-1] + ye[1:])

    vmin = np.nanpercentile(true_map, 1)
    vmax = np.nanpercentile(true_map, 99)

    # THREE figures, not one 3x3. As a single canvas this was 8.97in wide -- wider than the
    # paper, so \includegraphics at natural size CLIPPED it -- and it needed a three-column
    # grid, shared y-axes, horizontal colourbars and a panel spanning two cells to get there.
    # Every one of those is a concession the plot box should never have had to make. Split by
    # what the panels actually say: the maps, the projections, the collinear ramp. Within each,
    # the panels are separate files that results.tex packs two per line, so a set of three has
    # no hole where a fourth would be.
    #
    # COLOURBAR RULE, applied to every 2-D map in this document without exception: the bar is
    # HORIZONTAL, directly under its own panel. Everything that is not a map gets the vertical
    # bar immediately right of its plot. The split is not taste, it is width: a vertical bar
    # costs ~0.8in of column, which puts a map panel at 3.9in and means two of them can never
    # share a line, while a horizontal bar costs height and leaves the panel at 3.08in. What
    # the document must not do is mix the two ACROSS maps, or share one bar between some panels
    # and not others -- that is what made the colourbars look arbitrary.
    CB = ps.CBAR_KW

    # ---- figure 1: 2D maps ----
    def draw_map(ax, M, name, cmap, vmn, vmx, label):
        pm = ax.pcolormesh(xe, ye, M, cmap=cmap, vmin=vmn, vmax=vmx,
                           shading="flat")
        ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
        ax.set_ylabel(r"$\cos\theta^{*}$")
        # The white box stays: without it the panel name is dark text on the dark end of the
        # colour map. The corner is the document default.
        ps.process_label(ax, name,
                         bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        ax.figure.colorbar(pm, ax=ax, **CB).set_label(label)
        return pm

    maps = ps.panels(3)
    ax0, ax1, ax2 = (f[1] for f in maps)
    AMP = r"$\langle \log|\mathcal{M}|^2\rangle$"
    draw_map(ax0, true_map, "truth", "viridis", vmin, vmax, AMP)
    draw_map(ax1, pred_map, "model", "viridis", vmin, vmax, AMP)
    emax = np.nanpercentile(err_map, 99)
    draw_map(ax2, err_map, r"$|$error$|$", "inferno", 0.0, emax,
             r"$\langle |\Delta\log|\mathcal{M}|^2|\rangle$")
    # Sparse-statistics contour marks where there is too little data to trust the error map.
    # It gets a legend entry: an unlabelled cyan squiggle is exactly the kind of unexplained
    # overlay the figure style forbids.
    with np.errstate(invalid="ignore"):
        sparse = np.where(cnt_map < max(3, 0.02 * n / (44 * 44)), 1.0, 0.0)
    ax2.contour(Xc, Yc, sparse, levels=[0.5], colors="cyan", linewidths=0.6,
                alpha=0.6)
    ax2.plot([], [], color="cyan", lw=0.6, alpha=0.6, label="sparse statistics")
    # Needs a solid frame: the default frameless legend puts dark text straight onto the dark
    # end of the inferno map, where it is unreadable.
    ps.legend(ax2, "lower left", frameon=True, facecolor="white", framealpha=0.85,
              edgecolor="none")

    # ---- figure 2: projections + scatter ----
    # The mean-|error| twin axis is gone: a second y-scale with its own coloured label per
    # panel is what drove these columns to a fraction of an inch, and row 1's error map plus
    # row 3's error panel already carry that information.
    # The line weights here ENCODE the pairing: a thick opaque truth curve under a thin dashed
    # model curve, so agreement reads as the dashed line sitting on the solid one.
    def proj(ax, coord, xlabel, bins):
        tm, e, _ = binned_statistic(coord, true_l, statistic="mean", bins=bins)
        pmn, _, _ = binned_statistic(coord, pred_l, statistic="mean", bins=bins)
        c = 0.5 * (e[:-1] + e[1:])
        ax.plot(c, tm, color="k", lw=1.8, label="truth")
        ax.plot(c, pmn, color=ps.C.vermillion, lw=1.3, ls="--", label="model")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\langle \log|\mathcal{M}|^2\rangle$")
        return c

    projs = ps.panels(3)
    axp1, axp2, axsc = (f[1] for f in projs)
    proj(axp1, sqrt_s, r"$\sqrt{s}$ [GeV]", np.linspace(s_lo, s_hi, 60))
    ps.legend(axp1, "lower right")

    proj(axp2, cos_t, r"$\cos\theta^{*}$", np.linspace(-1, 1, 60))

    hb = axsc.hexbin(true_l, pred_l, gridsize=55, bins="log", cmap="magma",
                     mincnt=1)
    lo = min(true_l.min(), pred_l.min())
    hi = max(true_l.max(), pred_l.max())
    axsc.plot([lo, hi], [lo, hi], color="cyan", lw=1.0, ls=":", label="ideal")
    axsc.set_xlabel(r"truth $\log|\mathcal{M}|^2$")
    axsc.set_ylabel(r"model $\log|\mathcal{M}|^2$")
    ps.legend(axsc, "upper left")
    axsc.figure.colorbar(hb, ax=axsc, **CB).set_label("count")

    # Error summary goes to stdout and the caption, not into the axes as a text box.
    mse = float(np.mean(resid ** 2))
    mae = float(np.mean(np.abs(resid)))

    # ---- row 3: collinear-resolved (stretch cos->+-1 so the divergence shows) ----
    eta = collinear_coord(cos_t)
    ebins = np.linspace(-6, 6, 49)
    ec = 0.5 * (ebins[:-1] + ebins[1:])

    def ebin(v, how):
        r, _, _ = binned_statistic(eta, v, statistic=how, bins=ebins)
        return r
    t_mean, p_mean = ebin(true_l, "mean"), ebin(pred_l, "mean")
    t_max, p_max = ebin(true_l, "max"), ebin(pred_l, "max")
    e_err = ebin(np.abs(resid), "mean")

    # Two panels, so a plain 1x2 -- no spanning cell. A panel spanning two columns of a grid
    # is a panel of a different size, which is the one thing these figures may not have.
    fig3, (axcol, axce) = ps.figure(ncols=2)
    axcol.plot(ec, t_mean, color="k", lw=1.9, label="truth (mean)")
    axcol.plot(ec, p_mean, color=ps.C.vermillion, lw=1.3, ls="--", label="model (mean)")
    axcol.plot(ec, t_max, color="k", lw=1.0, ls=":", alpha=0.7, label="truth (max)")
    axcol.plot(ec, p_max, color=ps.C.vermillion, lw=1.0, ls=":", alpha=0.7,
               label="model (max)")
    # The formula alone says how to read the axis; the "backward . central . forward" gloss and
    # the unlabelled centre line were reading instructions inside the figure.
    axcol.set_xlabel(r"$\eta=\mathrm{sign}(\cos\theta^*)\,"
                     r"\log_{10}\frac{1}{1-|\cos\theta^*|}$")
    axcol.set_ylabel(r"$\log|\mathcal{M}|^2$")
    # The ramp is a V with its minimum at eta=0, so both bottom corners are clear; "upper
    # center" put the legend on the two peaks it is meant to let you read.
    ps.legend(axcol, "lower left")
    # The secondary |cos theta*| axis keeps only the ticks that stay legible across a 2.40in
    # box: seven of them collided, and shrinking the font is not the fix.
    axt = axcol.secondary_xaxis("top")
    et = [-6, -3, 0, 3, 6]
    axt.set_xticks(et)
    axt.set_xticklabels(["0" if e == 0 else f"{1.0 - 10.0**(-abs(e)):.6g}" for e in et])
    axt.set_xlabel(r"$|\cos\theta^*|$")

    axce.plot(ec, e_err, color=ps.C.blue)
    axce.set_xlabel(r"$\eta$")
    axce.set_ylabel(r"$\langle|\Delta\log|\mathcal{M}|^2|\rangle$")
    axce.set_ylim(bottom=0)

    ps.save_panels(maps, out_base)
    ps.save_panels(projs, out_base + "_proj")
    ps.save(fig3, out_base + "_collinear")
    plt.close("all")
    print(f"wrote {out_base}*.png / .pdf  (N={n}, MSEΔ={mse:.3g}, MAEΔ={mae:.3g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--split", default="all",
                    choices=["all", "train", "val", "test"])
    args = ap.parse_args()
    make_figure(args.npz, args.label, args.out_base, args.split)


if __name__ == "__main__":
    main()
