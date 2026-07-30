#!/usr/bin/env python
"""Plot true vs predicted log-amplitude across the 2->2 phase space (sqrt_s,
cos_theta*) for a finetuned NLO run, to expose how the model behaves near the
amplitude divergences (low-sqrt_s / resonant growth, forward-backward peaks,
threshold) and whether accuracy degrades there.

Produces, per process, ONE figure (png + pdf):
  row 1 : <log|M|^2> true | predicted | mean |Δlog|M|^2| (error) 2D maps
  row 2 : projection vs sqrt_s | projection vs cos_theta | pred-vs-true hexbin
  row 3 : collinear-resolved log|M|^2 vs eta (mean & max, truth vs model) | error vs eta
CPU only.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

    # Same width arithmetic as make_ir.py: three columns of y-labels and VERTICAL colourbars
    # do not fit across \textwidth at 11pt and collapse the panels. Shared y per row,
    # horizontal colourbars, no twin axes. No suptitle: N, split and the error summary belong
    # in the caption and are printed to stdout at the end.
    fig = plt.figure(figsize=(ps.TEXTWIDTH_IN, 8.8), layout="constrained")
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 0.95, 0.95])
    CBH = dict(orientation="horizontal", location="bottom")

    # ---- row 1: 2D maps ----
    def draw_map(ax, M, name, cmap, vmn, vmx, first):
        pm = ax.pcolormesh(xe, ye, M, cmap=cmap, vmin=vmn, vmax=vmx,
                           shading="flat")
        ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
        if first:
            ax.set_ylabel(r"$\cos\theta^{*}$")
        else:
            ax.tick_params(labelleft=False)
        ps.process_label(ax, name, loc="upper left",
                         bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        return pm

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)
    pm_t = draw_map(ax0, true_map, "truth", "viridis", vmin, vmax, True)
    draw_map(ax1, pred_map, "model", "viridis", vmin, vmax, False)
    emax = np.nanpercentile(err_map, 99)
    pm_e = draw_map(ax2, err_map, r"$|$error$|$", "inferno", 0.0, emax, False)
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
    ax2.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.85,
               edgecolor="none")
    fig.colorbar(pm_t, ax=[ax0, ax1], **CBH).set_label(
        r"$\langle \log|\mathcal{M}|^2\rangle$")
    fig.colorbar(pm_e, ax=ax2, **CBH).set_label(
        r"$\langle |\Delta\log|\mathcal{M}|^2|\rangle$")

    # ---- row 2: projections + scatter ----
    # The mean-|error| twin axis is gone: a second y-scale with its own coloured label per
    # panel is what drove these columns to a fraction of an inch, and row 1's error map plus
    # row 3's error panel already carry that information.
    def proj(ax, coord, xlabel, bins, first):
        tm, e, _ = binned_statistic(coord, true_l, statistic="mean", bins=bins)
        pmn, _, _ = binned_statistic(coord, pred_l, statistic="mean", bins=bins)
        c = 0.5 * (e[:-1] + e[1:])
        ax.plot(c, tm, color="k", lw=1.8, label="truth")
        ax.plot(c, pmn, color=ps.C.vermillion, lw=1.3, ls="--", label="model")
        ax.set_xlabel(xlabel)
        if first:
            ax.set_ylabel(r"$\langle \log|\mathcal{M}|^2\rangle$")
        else:
            ax.tick_params(labelleft=False)
        return c

    axp1 = fig.add_subplot(gs[1, 0])
    proj(axp1, sqrt_s, r"$\sqrt{s}$ [GeV]", np.linspace(s_lo, s_hi, 60), True)
    axp1.legend(loc="lower right")

    axp2 = fig.add_subplot(gs[1, 1], sharey=axp1)
    proj(axp2, cos_t, r"$\cos\theta^{*}$", np.linspace(-1, 1, 60), False)

    axsc = fig.add_subplot(gs[1, 2])
    hb = axsc.hexbin(true_l, pred_l, gridsize=55, bins="log", cmap="magma",
                     mincnt=1)
    lo = min(true_l.min(), pred_l.min())
    hi = max(true_l.max(), pred_l.max())
    axsc.plot([lo, hi], [lo, hi], color="cyan", lw=1.0, ls=":", label="ideal")
    axsc.set_xlabel(r"truth $\log|\mathcal{M}|^2$")
    axsc.set_ylabel(r"model $\log|\mathcal{M}|^2$")
    axsc.legend(loc="upper left")
    ps.process_label(axsc, "predicted vs true", loc="lower right")
    fig.colorbar(hb, ax=axsc, **CBH).set_label("count")

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

    axcol = fig.add_subplot(gs[2, 0:2])
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
    axcol.legend(ncol=2, loc="upper center")
    axt = axcol.secondary_xaxis("top")
    et = [-6, -4, -2, 0, 2, 4, 6]
    axt.set_xticks(et)
    axt.set_xticklabels(["0" if e == 0 else f"{1.0 - 10.0**(-abs(e)):.6g}" for e in et])
    axt.set_xlabel(r"$|\cos\theta^*|$")

    axce = fig.add_subplot(gs[2, 2])
    axce.plot(ec, e_err, color=ps.C.blue, lw=1.4)
    axce.set_xlabel(r"$\eta$")
    axce.set_ylabel(r"$\langle|\Delta\log|\mathcal{M}|^2|\rangle$")
    axce.set_ylim(bottom=0)

    fig._ps_layout_done = True      # GridSpec + colourbars own the layout
    ps.save(fig, out_base)
    plt.close(fig)
    print(f"wrote {out_base}.png / .pdf  (N={n}, MSEΔ={mse:.3g}, MAEΔ={mae:.3g})")


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
