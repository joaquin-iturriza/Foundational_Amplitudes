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

    fig, (axL, axR) = ps.figure(ncols=2)
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
        # Line weight ENCODES the pairing here: a thick translucent truth band with the thin
        # dashed model curve riding inside it, so "the model tracks the ramp" is visible
        # without reading the legend. One of the two sanctioned lw/ms overrides.
        band = rf"$x_g\in[{lo:.2f},{hi:.2f}]$"
        axL.plot(c, tm, color=col, lw=4.2, alpha=0.30, solid_capstyle="round", zorder=2)
        axL.plot(c, pm, color=col, lw=1.3, ls=(0, (4, 2)), marker="o", ms=2.6,
                 markevery=3, zorder=6)
        axR.plot(c, am, color=col, label=band)
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
        axL.plot(xline, -ln10 * xline + anchor_off, color="darkgreen", ls="-.", zorder=5)
        axR.plot([], [], color="darkgreen", ls="-.", label=r"analytic slope $-\ln 10$")
    axL.set_xlabel(r"$\log_{10}(1-x_q)$")
    axL.set_ylabel(r"$\log|\mathcal{M}|^2$")
    axR.set_xlabel(r"$\log_{10}(1-x_q)$")
    axR.set_ylabel(r"$\langle|\Delta\log|\mathcal{M}|^2|\rangle$")
    axR.set_ylim(bottom=0)
    # ONE legend, INSIDE the right panel. Spelling out band x {truth, model} gave seven entries
    # of "$x_g\in[0.25,0.45]$ truth", which fits in no panel and is why this figure used to
    # carry a legend strip below it. Factorised, it is three bands plus the analytic slope: the
    # truth/model distinction is carried by the thick-band-and-dashed-line convention, which is
    # what the eye uses anyway. The right panel is the one with room -- its error curves all sit
    # near the bottom, while the left panel's ramps run corner to corner.
    ps.legend(axR, "upper left")
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
    # THREE SEPARATE PANEL FILES, not a 2x2 with a hole in it. A 2x2 holding three maps leaves
    # an empty cell where the fourth would go, which is the first thing anyone notices about
    # the figure; results.tex includes the three files and lets them fall on the page. Each map
    # carries its own y-label, its own tick labels and its own VERTICAL colourbar on its right
    # -- the standard placement. The old layout shared a y-axis per row and laid the colourbars
    # horizontally under the panels to buy column width, which made these the only maps in the
    # document with a bar underneath them.
    maps = ps.panels(3)
    ax0, ax1, ax2 = (f[1] for f in maps)
    # COLOURBAR RULE, applied to every 2-D map in this document without exception: the bar is
    # HORIZONTAL, directly under its own panel. Everything that is not a map gets the vertical
    # bar immediately right of its plot. The split is not taste, it is width: a vertical bar
    # costs ~0.8in of column, which puts a map panel at 3.9in and means two of them can never
    # share a line, while a horizontal bar costs height and leaves the panel at 3.08in. What
    # the document must not do is mix the two ACROSS maps, or share one bar between some panels
    # and not others -- that is what made the colourbars look arbitrary.
    CB = ps.CBAR_KW
    mse = float(np.mean(resid ** 2))

    def draw(ax, M, name, cmap, vmn, vmx, label):
        pm = ax.pcolormesh(xe, ye, M, cmap=cmap, vmin=vmn, vmax=vmx, shading="flat")
        ax.set_xlabel(r"$\log_{10} y_{\min}$")
        ax.set_ylabel(r"$\log_{10} x_{g,\min}$")
        # process_label, not set_title: the style forbids titles, and an in-axes tag also
        # costs no vertical space. White bbox so it stays legible over a dense map.
        ps.process_label(ax, name,
                         bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        ax.figure.colorbar(pm, ax=ax, **CB).set_label(label)
        return pm

    AMP = r"$\langle\log|\mathcal{M}|^2\rangle$"
    draw(ax0, tmap, "truth", "viridis", vmin, vmax, AMP)
    draw(ax1, pmap, "model", "viridis", vmin, vmax, AMP)
    draw(ax2, emap, r"$|$error$|$", "inferno", 0.0, np.nanpercentile(emap, 99),
         r"$\langle|\Delta\log|\mathcal{M}|^2|\rangle$")

    def ramp(ax, coord, xlabel, name, slope):
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
        ax.plot(c, slope * c + off, color="darkgreen", ls="-.",
                label="analytic leading power", zorder=5)
        # Line weight ENCODES which curve is which: the mean ramps are the thick pair and the
        # max ramps the dotted pair, so truth-vs-model reads as one overlay per weight.
        ax.plot(c, tm, "k", lw=1.9, label="truth (mean)")
        ax.plot(c, pm, color=ps.C.vermillion, lw=1.3, ls="--", label="model (mean)")
        ax.plot(c, tx, "k", lw=1.0, ls=":", alpha=0.7, label="truth (max)")
        ax.plot(c, px, color=ps.C.vermillion, lw=1.0, ls=":", alpha=0.7, label="model (max)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\log|\mathcal{M}|^2$")
        ps.process_label(ax, name)
        return ax

    ps.save_panels(maps, out_base)
    print(f"wrote {out_base}_[abc].png/.pdf (N={n}, MSE\u0394={mse:.3g})")

    # --- second set: the IR ramps and the prediction scatter -----------------------------
    # Split out of the maps: six panels of three different kinds was never one figure. Three
    # panels again, so three files rather than a 2x2 with an empty cell. No shared y between
    # the ramps -- the two carry different slopes and the shared axis only ever cost the right
    # panel its tick labels.
    ramps = ps.panels(3)
    axr0, axr1, axsc = (f[1] for f in ramps)
    ln10 = np.log(10.0)
    ramp(axr0, ly, r"$\log_{10} y_{\min}$", "soft $+$ collinear", -ln10)
    ramp(axr1, lx, r"$\log_{10} x_{g,\min}$", "soft gluon", -2.0 * ln10)

    hb = axsc.hexbin(tl, pl, gridsize=55, bins="log", cmap="magma", mincnt=1)
    lo, hi = min(tl.min(), pl.min()), max(tl.max(), pl.max())
    axsc.plot([lo, hi], [lo, hi], color="cyan", lw=1.0, ls=":", label="ideal")
    axsc.set_xlabel(r"truth $\log|\mathcal{M}|^2$"); axsc.set_ylabel(r"model $\log|\mathcal{M}|^2$")
    ps.legend(axsc, "upper left")
    axsc.figure.colorbar(hb, ax=axsc, **CB).set_label("count")

    # Inside the first ramp panel, not a strip under the grid. Lower RIGHT: the mean ramp falls
    # left-to-right and the max ramp sits well above it, so the free area is under the right-hand
    # end of the mean curve. Lower left is where the mean ramp starts, and the legend printed
    # across it.
    ps.legend(axr0, "lower right")

    ps.save_panels(ramps, f"{out_base}_ramps")
    plt.close("all")
    print(f"wrote {out_base}_ramps_[abc].png/.pdf")

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
        # FOUR separate panel files. As one 2x2 this needed a hand-rolled GridSpec with a
        # dedicated colourbar row, constrained_layout with a measured w_pad, shared x and y,
        # and axis labels on the outside edges only -- a stack of workarounds for putting four
        # maps on one canvas. Four files carry the standard plot box each, and results.tex
        # packs them two per line.
        dal = ps.panels(4)
        axs = [f[1] for f in dal]
        panels = [
            (axs[0], amap_a, r"analytic LO", "viridis", vmn, vmx, AMP),
            (axs[1], tm, "truth", "viridis", vmn, vmx, AMP),
            (axs[2], pm, "model", "viridis", vmn, vmx, AMP),
            (axs[3], em, r"$|$error$|$", "inferno", 0.0, np.nanpercentile(em, 99),
             r"$\langle|\Delta\log|\mathcal{M}|^2|\rangle$")]
        for ax, M, ti, cm, a, bb, clab in panels:
            pmesh = ax.pcolormesh(xe2, ye2, M, cmap=cm, vmin=a, vmax=bb, shading="flat")
            # overlay analytic iso-|M|^2 contours on truth & model to show they track
            if ti in ("truth", "model"):
                cs = ax.contour(cb, cb, amap_a, levels=clev, colors="white",
                                linewidths=0.7, alpha=0.75)
                ax.clabel(cs, fmt="%.0f")
            ax.set_xlabel(r"$x_q=2E_q/\sqrt{s}$")
            ax.set_ylabel(r"$x_{\bar q}=2E_{\bar q}/\sqrt{s}$")
            ps.process_label(ax, ti,
                             bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
            ax.figure.colorbar(pmesh, ax=ax, **CB).set_label(clab)
        ps.save_panels(dal, f"{out_base}_dalitz")
        plt.close("all")
        print(f"wrote {out_base}_dalitz_[abcd].png/.pdf  (analytic-vs-truth corr={corr:.3f})")
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
