#!/usr/bin/env python
"""IR-resolved plots for the gluon channels (ee->uug, ee->uugg): how the model
behaves at the genuine SOFT and COLLINEAR singularities.

Main figure (per process), reading the extract_ir npz:
  row 1 : <log|M|^2> truth | model | error, over (log10 y_min, log10 x_gmin)
          [y_min = IR resolution var ->0 soft&collinear; x_gmin = softest gluon frac]
  row 2 : log|M|^2 vs log10 y_min (IR ramp, mean&max) | vs log10 x_gmin (soft)
          | pred-vs-true hexbin
Plus, for ee->q qbar g, a Dalitz figure: <log|M|^2> over (x_q, x_qbar) truth/model/error.
CPU only.
"""
import argparse
import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402
from matplotlib.gridspec import GridSpec
from scipy.stats import binned_statistic, binned_statistic_2d


def _map(x, y, z, xb, yb, stat="mean"):
    s, xe, ye, _ = binned_statistic_2d(x, y, z, statistic=stat, bins=[xb, yb])
    return s.T, xe, ye


def analytic_dalitz_logM(xq, xqb):
    """LO QCD antenna for ee -> q qbar g (up to an additive constant):
    |M|^2 ~ (x_q^2 + x_qbar^2)/((1-x_q)(1-x_qbar)), so log|M|^2 = ln[...] + c.
    The C_F antenna factor is universal (photon/Z production only shifts c and a
    smooth non-singular orientation modulation), so this captures the collinear
    edges (x->1) and the soft-gluon (1,1) corner exactly."""
    eps = 1e-9
    num = xq ** 2 + xqb ** 2
    den = np.clip((1.0 - xq) * (1.0 - xqb), eps, None)
    return np.log(np.clip(num / den, eps, None))


def _anchor_offset(coord, val, slope, deep_frac=40.0):
    """Additive constant c for the line val ~= slope*coord + c, fit (robust median)
    over the deep-IR portion (leftmost `deep_frac`% of coord) where the asymptotics
    dominate. Returns c."""
    thr = np.nanpercentile(coord, deep_frac)
    sel = (coord <= thr) & np.isfinite(val)
    return float(np.nanmedian(val[sel] - slope * coord[sel]))


def make_collinear_ramp(d, label, out_base,
                        xg_bands=((0.25, 0.45), (0.45, 0.65), (0.65, 0.85))):
    """PURE COLLINEAR ramp for ee->q qbar g, complementary to the soft (x_g) ramp.
    Uses 1-x_q = (Q-p_q)^2/s = (p_qbar+p_g)^2/s, the normalized qbar-g invariant,
    which ->0 iff qbar || g (collinear) with the gluon staying HARD; symmetrized to
    the nearest collinear edge via min(1-x_q, 1-x_qbar). Restricting to bands of
    fixed, O(1) x_g switches the SOFT pole off, so only the collinear pole survives:
    |M|^2 ~ 1/(1-x_q) => slope -ln10 vs log10, and the ramp is x_g-INDEPENDENT
    (collinear factorization -> parallel lines offset only by the splitting kernel)."""
    tl, pl = d["true_logamp"], d["pred_logamp"]
    resid = pl - tl
    xq, xqb, xg = d["x_q"], d["x_qbar"], d["x_gmin"]
    tcol = np.clip(np.minimum(1.0 - xq, 1.0 - xqb), 1e-6, None)   # nearest collinear invariant
    lt = np.log10(tcol)
    ln10 = np.log(10.0)
    # DGLAP momentum fraction z of the nearest collinear split (qbar||g -> parent qbar,
    # else q||g), from energy fractions; P_qq(z) ~ (1+z^2)/(1-z) sets the pole RESIDUE,
    # i.e. the x_g-dependent vertical offset between otherwise-parallel ramps.
    zfrac = np.where((1.0 - xq) <= (1.0 - xqb), xqb / (xqb + xg), xq / (xq + xg))
    zfrac = np.clip(zfrac, 1e-4, 1.0 - 1e-4)
    logP = np.log((1.0 + zfrac ** 2) / (1.0 - zfrac))
    C0 = -2.0                                       # reference log10(1-x_q) for offset compare

    fig, (axL, axR) = ps.figure(ncols=2, width="full")
    fig.set_size_inches(ps.TEXTWIDTH_IN, 3.7)   # +0.8in for the legend strip below
    colors = plt.cm.plasma(np.linspace(0.12, 0.78, len(xg_bands)))
    anchor_off, span, fits = None, [], []
    for (lo, hi), col in zip(xg_bands, colors):
        m = (xg >= lo) & (xg < hi)
        if int(m.sum()) < 500:
            continue
        bins = np.linspace(np.percentile(lt[m], 1.0), np.percentile(lt[m], 99.0), 40)
        c = 0.5 * (bins[:-1] + bins[1:]); span.append(c)
        tm, _, _ = binned_statistic(lt[m], tl[m], "mean", bins=bins)
        pm, _, _ = binned_statistic(lt[m], pl[m], "mean", bins=bins)
        am, _, _ = binned_statistic(lt[m], np.abs(resid[m]), "mean", bins=bins)
        # slope fit over the deep-collinear (linear) portion where the pole dominates
        fitsel = np.isfinite(tm) & (c < -1.0)
        slope_fit = ybar = logPbar = np.nan
        if int(fitsel.sum()) >= 4:
            slope_fit, b_fit = np.polyfit(c[fitsel], tm[fitsel], 1)
            ybar = slope_fit * C0 + b_fit          # fit value at the reference point
            logPbar = float(np.nanmean(logP[m]))   # <log P(z)> over the band
            fits.append((0.5 * (lo + hi), slope_fit, ybar, logPbar))
        # The fitted slope belongs in the text, not the legend: with it, each of the seven
        # entries ran to ~0.8in and the legend reached 5.71in -- wider than the 6.5in figure.
        # tight_layout counts in-axes legends, so it shrank both panels to 0.85in to fit it.
        band = rf"$x_g\in[{lo:.2f},{hi:.2f}]$"
        axL.plot(c, tm, color=col, lw=4.2, alpha=0.30, solid_capstyle="round",
                 label=band + " truth", zorder=2)
        axL.plot(c, pm, color=col, lw=1.3, ls=(0, (4, 2)), marker="o", ms=2.6,
                 markevery=3, label=band + " model", zorder=6)
        axR.plot(c, am, color=col, lw=1.6, label=rf"$x_g\in[{lo:.2f},{hi:.2f}]$")
        if anchor_off is None:              # anchor the analytic slope to the first valid band
            anchor_off = _anchor_offset(c, tm, -ln10)
    # report: measured slopes vs -ln10, and offset spacing vs DGLAP <log P(z)>
    if fits:
        print(f"  [collinear fit] {label}:  analytic slope -ln10 = {-ln10:.3f}")
        x0, s0, y0, lp0 = fits[0]
        for xgc, s, y, lp in fits:
            print(f"    x_g~{xgc:.2f}: slope={s:+.3f}  "
                  f"offset@(1-x_q=1e{C0:.0f}) meas Δ={y - y0:+.2f}  "
                  f"DGLAP Δ<logP>={lp - lp0:+.2f}")
    if anchor_off is not None:
        allc = np.concatenate(span)
        xline = np.linspace(np.nanmin(allc), np.nanmax(allc), 50)
        axL.plot(xline, -ln10 * xline + anchor_off, color="darkgreen", lw=1.8, ls="-.",
                 zorder=5, label=r"analytic $\propto 1/(1-x_q)$ (slope $-\ln 10$)")
    axL.set_xlabel(r"$\log_{10}(1-x_q)$")
    axL.set_ylabel(r"$\log|\mathcal{M}|^2$")
    axR.set_xlabel(r"$\log_{10}(1-x_q)$")
    axR.set_ylabel(r"$\langle|\Delta\log|\mathcal{M}|^2|\rangle$")
    axR.set_ylim(bottom=0)
    # Both panels are the same three x_g bands, so one legend below the figure serves both and
    # costs no panel width. axL's own 7-entry legend is what collapsed this figure.
    h, l = axL.get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", frameon=False)
    fig.tight_layout(rect=[0, 0.22, 1, 1])
    fig._ps_layout_done = True
    ps.save(fig, f"{out_base}_collinear")
    plt.close(fig)
    print(f"wrote {out_base}_collinear.png/.pdf")


def make_ir(npz, label, out_base):
    d = np.load(npz)
    tl, pl = d["true_logamp"], d["pred_logamp"]
    resid = pl - tl
    ly = np.log10(np.clip(d["y_min"], 1e-12, None))     # IR / collinear master var
    lx = np.log10(np.clip(d["x_gmin"], 1e-12, None))    # softest gluon energy fraction
    n = len(tl)

    xb = np.linspace(np.percentile(ly, 0.2), np.percentile(ly, 99.8), 45)
    yb = np.linspace(np.percentile(lx, 0.2), np.percentile(lx, 99.8), 45)
    tmap, xe, ye = _map(ly, lx, tl, xb, yb)
    pmap, _, _ = _map(ly, lx, pl, xb, yb)
    emap, _, _ = _map(ly, lx, np.abs(resid), xb, yb)
    vmin, vmax = np.nanpercentile(tmap, 1), np.nanpercentile(tmap, 99)

    # Three columns of decorations do not fit across \textwidth at 11pt. With a VERTICAL
    # colourbar and a y-label per panel, plus a twinx label on the bottom row, constrained
    # layout gave up entirely -- "axes sizes collapsed to zero" -- and every panel rendered
    # as a 0.3in vertical sliver. Measured panel width for this 2x3 grid:
    #
    #   vertical colourbars, per-panel y-labels, twinx   0.31 in   (collapsed)
    #   vertical colourbars, shared y, no twinx          0.99 in
    #   HORIZONTAL colourbars, shared y, no twinx        1.50 in   <- this layout
    #
    # Everything that costs COLUMN width is therefore shared or turned sideways: one y-axis
    # per row, one colourbar for the truth/prediction pair (they already share vmin/vmax),
    # colourbars laid horizontally so they cost height instead of width, no twinx, and the
    # ramp legend hung below the figure. 1.5in is the ceiling for a 3-column \textwidth
    # figure at 11pt; the panels are square rather than roomy.
    fig = plt.figure(figsize=(ps.TEXTWIDTH_IN, 6.4), layout="constrained")
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.95])
    CBH = dict(orientation="horizontal", location="bottom")
    mse = float(np.mean(resid ** 2))

    def draw(ax, M, name, cmap, vmn, vmx, first):
        pm = ax.pcolormesh(xe, ye, M, cmap=cmap, vmin=vmn, vmax=vmx, shading="flat")
        ax.set_xlabel(r"$\log_{10} y_{\min}$")
        if first:
            ax.set_ylabel(r"$\log_{10} x_{g,\min}$")
        else:
            ax.tick_params(labelleft=False)
        # process_label, not set_title: the style forbids titles, and an in-axes tag also
        # costs no vertical space. White bbox so it stays legible over a dense map.
        ps.process_label(ax, name, loc="upper left",
                         bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        return pm

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)
    draw(ax0, tmap, "truth", "viridis", vmin, vmax, True)
    pm1 = draw(ax1, pmap, "model", "viridis", vmin, vmax, False)
    pm2 = draw(ax2, emap, r"$|$error$|$", "inferno",
               0.0, np.nanpercentile(emap, 99), False)
    # One bar for truth+model (same scale by construction), one for the error panel.
    fig.colorbar(pm1, ax=[ax0, ax1], **CBH).set_label(r"$\langle\log|\mathcal{M}|^2\rangle$")
    fig.colorbar(pm2, ax=ax2, **CBH).set_label(r"$\langle|\Delta\log|\mathcal{M}|^2|\rangle$")

    def ramp(ax, coord, xlabel, name, slope, first):
        bins = np.linspace(np.percentile(coord, 0.2), np.percentile(coord, 99.8), 55)
        c = 0.5 * (bins[:-1] + bins[1:])
        tm, _, _ = binned_statistic(coord, tl, "mean", bins=bins)
        pm, _, _ = binned_statistic(coord, pl, "mean", bins=bins)
        tx, _, _ = binned_statistic(coord, tl, "max", bins=bins)
        px, _, _ = binned_statistic(coord, pl, "max", bins=bins)
        # Analytic leading-power IR slope, offset anchored to the mean ramp (the clean
        # asymptote). Labelled generically: the two panels carry DIFFERENT slopes
        # (1/y_min and 1/x_g^2), so one shared legend cannot name both -- the caption does.
        off = _anchor_offset(c, tm, slope)
        ax.plot(c, slope * c + off, color="darkgreen", lw=1.6, ls="-.",
                label="analytic leading power", zorder=5)
        ax.plot(c, tm, "k", lw=1.9, label="truth (mean)")
        ax.plot(c, pm, color=ps.C.vermillion, lw=1.3, ls="--", label="model (mean)")
        ax.plot(c, tx, "k", lw=1.0, ls=":", alpha=0.7, label="truth (max)")
        ax.plot(c, px, color=ps.C.vermillion, lw=1.0, ls=":", alpha=0.7, label="model (max)")
        ax.set_xlabel(xlabel)
        if first:
            ax.set_ylabel(r"$\log|\mathcal{M}|^2$")
        else:
            ax.tick_params(labelleft=False)
        ps.process_label(ax, name, loc="upper right")
        return ax

    ln10 = np.log(10.0)
    axr0 = fig.add_subplot(gs[1, 0])
    ramp(axr0, ly, r"$\log_{10} y_{\min}$", "soft $+$ collinear", -ln10, True)
    axr1 = fig.add_subplot(gs[1, 1], sharey=axr0)
    ramp(axr1, lx, r"$\log_{10} x_{g,\min}$", "soft gluon", -2.0 * ln10, False)

    axsc = fig.add_subplot(gs[1, 2])
    hb = axsc.hexbin(tl, pl, gridsize=55, bins="log", cmap="magma", mincnt=1)
    lo, hi = min(tl.min(), pl.min()), max(tl.max(), pl.max())
    axsc.plot([lo, hi], [lo, hi], color="cyan", lw=1.0, ls=":", label="ideal")
    axsc.set_xlabel(r"truth $\log|\mathcal{M}|^2$"); axsc.set_ylabel(r"model $\log|\mathcal{M}|^2$")
    axsc.legend(loc="upper left")
    ps.process_label(axsc, "predicted vs true", loc="lower right")
    fig.colorbar(hb, ax=axsc, **CBH).set_label("count")

    # The ramp panels' five series need a legend wider than a 1.5in panel. Hung outside the
    # grid ("outside lower center") it costs figure height, whereas a legend axes spanning
    # the columns constrains their width and squeezed the panels from 0.99in to 0.30in.
    h, l = axr0.get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="outside lower center", frameon=False)

    fig._ps_layout_done = True      # GridSpec + colourbars own the layout
    ps.save(fig, out_base)
    plt.close(fig)
    print(f"wrote {out_base}.png/.pdf (N={n}, MSEΔ={mse:.3g})")

    # Dalitz (ee -> q qbar g only)
    if "x_q" in d.files:
        xq, xqb = d["x_q"], d["x_qbar"]
        b = np.linspace(0, 1, 60)
        cb = 0.5 * (b[:-1] + b[1:])
        tm, xe2, ye2 = _map(xq, xqb, tl, b, b)
        pm, _, _ = _map(xq, xqb, pl, b, b)
        em, _, _ = _map(xq, xqb, np.abs(resid), b, b)
        # analytic LO QCD-antenna map on the same grid; kinematic region is x_q+x_qbar>1
        XX, YY = np.meshgrid(cb, cb)                       # XX=x_q, YY=x_qbar (row=x_qbar)
        amap = analytic_dalitz_logM(XX, YY)
        phys = (XX + YY) > 1.0                             # 3-body Dalitz boundary
        amap = np.where(phys, amap, np.nan)
        # anchor the analytic additive constant to truth over populated, physical bins
        good = np.isfinite(tm) & np.isfinite(amap)
        off = float(np.nanmedian(tm[good] - amap[good]))
        amap_a = amap + off
        corr = float(np.corrcoef(tm[good], amap[good])[0, 1])
        vmn, vmx = np.nanpercentile(tm, 1), np.nanpercentile(tm, 99)
        clev = np.linspace(vmn, vmx, 7)                    # shared analytic contour levels
        # Four columns, so the width budget is even tighter than the 3-column figure above:
        # a y-label plus a vertical colourbar per panel collapsed these to 0.21in. Same
        # remedy -- shared y-axis, horizontal colourbars, in-axes tags instead of titles.
        figd = plt.figure(figsize=(ps.TEXTWIDTH_IN, 2.9), layout="constrained")
        gsd = GridSpec(1, 4, figure=figd)
        a0 = figd.add_subplot(gsd[0, 0])
        axs = [a0] + [figd.add_subplot(gsd[0, i], sharey=a0) for i in (1, 2, 3)]
        panels = [
            (axs[0], amap_a, r"analytic LO", "viridis", vmn, vmx),
            (axs[1], tm, "truth", "viridis", vmn, vmx),
            (axs[2], pm, "model", "viridis", vmn, vmx),
            (axs[3], em, r"$|$error$|$", "inferno", 0.0, np.nanpercentile(em, 99))]
        for ax, M, ti, cm, a, bb in panels:
            pmesh = ax.pcolormesh(xe2, ye2, M, cmap=cm, vmin=a, vmax=bb, shading="flat")
            # overlay analytic iso-|M|^2 contours on truth & model to show they track
            if ti in ("truth", "model"):
                cs = ax.contour(cb, cb, amap_a, levels=clev, colors="white",
                                linewidths=0.7, alpha=0.75)
                ax.clabel(cs, fmt="%.0f")
            ax.set_xlabel(r"$x_q=2E_q/\sqrt{s}$")
            if ax is axs[0]:
                ax.set_ylabel(r"$x_{\bar q}=2E_{\bar q}/\sqrt{s}$")
            else:
                ax.tick_params(labelleft=False)
            ps.process_label(ax, ti, loc="upper right",
                             bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
            if ax is axs[3]:
                figd.colorbar(pmesh, ax=ax, orientation="horizontal", location="bottom"
                              ).set_label(r"$\langle|\Delta\log|\mathcal{M}|^2|\rangle$")
        # The first three panels share one scale by construction, so they share one bar.
        # Take the mappable from panel 0, NOT the loop's last `pmesh` (the error panel, on a
        # different scale) -- that would label the shared bar with the error normalisation.
        figd.colorbar(axs[0].collections[0], ax=axs[:3],
                      orientation="horizontal", location="bottom"
                      ).set_label(r"$\langle\log|\mathcal{M}|^2\rangle$")
        figd._ps_layout_done = True
        ps.save(figd, f"{out_base}_dalitz")
        plt.close(figd)
        print(f"wrote {out_base}_dalitz.png/.pdf  (analytic-vs-truth corr={corr:.3f})")
        make_collinear_ramp(d, label, out_base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out_base", required=True)
    args = ap.parse_args()
    make_ir(args.npz, args.label, args.out_base)


if __name__ == "__main__":
    main()
