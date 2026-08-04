"""Shared figure style for Foundational_Amplitudes.

Every plotting script in this repo imports this module and uses `figure()` + `save()`.
Importing it applies the style; you do not need to call `use()` yourself.

    import sys; sys.path.insert(0, "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes")
    import plot_style as ps

    fig, (axL, axR) = ps.figure(ncols=2)
    axL.plot(x, y, color=ps.C.blue, label="base")
    axL.legend()
    ps.process_label(axL, r"$e^+e^-\to u\bar u gg$")
    ps.save(fig, "analysis/divergences/figs/l2_uugg_perdecade")

WHY THESE CHOICES

Size. THE PLOT BOX IS THE FIXED QUANTITY. Every axes is drawn at exactly PLOT_W_IN x
PLOT_H_IN, in every figure, whatever grid it sits in; the CANVAS is then computed from the
measured decorations. So a colourbar, a long y-label, rotated ticks, an outside legend or
scientific-notation tick labels all grow the canvas and none of them can shrink a plot. A
1-column figure is a ~3.4in-wide canvas centred on the page, not a stretched 6.5in one.

Do not use tight_layout or constrained_layout for sizing. They distribute a FIXED canvas
among the axes, which makes the plot box a residual: add a colourbar and the plot shrinks.
`layout()` (called by `save()`) does the opposite and is what keeps the set coherent.

Because the saved width varies, figures are included in LaTeX at their NATURAL size
(`\includegraphics{f.pdf}`, no `width=`). That keeps the scale factor exactly 1.000, so 11pt
in the figure prints as 11pt on the page. Forcing `width=\textwidth` on a 3.4in figure would
blow it up 2x and print its labels at 21pt.

Font. cmr10 is Computer Modern Roman, the document's own body font, so figure text is
typographically identical to the surrounding prose. cmr10 has no glyph for U+2212 MINUS,
hence `axes.unicode_minus = False`.

One size everywhere. Titles, labels, ticks, legend and annotations are all BASE_PT. Crowded
tick labels are fixed by using fewer ticks, not by shrinking the font.

THE STYLE RULES THESE FIGURES FOLLOW (see .claude/skills/figure-style/SKILL.md)

The figures are minimal and self-contained. No titles or suptitles: the explanation lives in
the LaTeX caption and body text, never in the figure. The only text allowed inside the axes
is the process label (`process_label`), and, very rarely, a label for a line that cannot go
in the legend. Everything drawn must be identifiable from the legend, which means no
unlabeled guide lines, no unlabeled shaded bands, no arrows or circles pointing at features,
and no parenthetical reading instructions like "(higher = better)" or "(IR<->bulk)" in axis
labels. If a metric needs explaining, put its formula on the axis instead of prose.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- geometry ---------------------------------------------------------------

#: \textwidth of docs/results.tex, in inches (11pt article, margin=1in -> 469.755pt).
TEXTWIDTH_IN = 6.5

#: Body font size of docs/results.tex. Figures use this everywhere.
BASE_PT = 11

#: Width:height every DATA PANEL is drawn at, whatever grid it sits in. This is the number
#: that makes the figures look like a set: one cell of a 3x2 has the same shape as a lone 1x1.
PANEL_ASPECT = 1.25

# --- THE INVARIANT ----------------------------------------------------------
#
# THE PLOT BOX IS A CONSTANT. Every axes in every figure is drawn at exactly
# PLOT_W_IN x PLOT_H_IN, and the CANVAS is computed from it: canvas = plot boxes + whatever
# the decorations measure. Decorations are therefore free -- a colourbar, a long y-label,
# rotated ticks, an outside legend, scientific-notation tick labels all grow the canvas and
# NONE of them can change the size of a plot.
#
# This is the third attempt at this and the first one that can actually hold, so it is worth
# being explicit about why the other two could not:
#
#   v1: figure pinned at \textwidth, panel = whatever was left after decorations.
#   v2: panel "enforced" by measuring and solving for the figure size -- but the solve was
#       CAPPED at \textwidth, so the instant the decorations did not fit, the cap bound and
#       the plot box silently absorbed the difference again. Same bug, one level down: fig 2
#       (no colourbars) got 2.35in plots and fig 3 (a colourbar per panel) got 1.95in plots,
#       on the same page.
#
# Both failed the same way: the plot box was a RESIDUAL. Anything competing for width won,
# and the plot paid. Worse, the workaround for a cramped figure was always to move a
# decoration somewhere else (legend outside, colourbar to the bottom, colourbar to an inset on
# top), which traded a size inconsistency for a PLACEMENT inconsistency -- and made both
# visible in the same document.
#
# Here the plot box is an input, not an output. Nothing negotiates with it.
#
# The value is set by the worst case that must still fit \textwidth: a 2-column figure whose
# columns each carry their own y-label and scientific-notation tick labels. Measured, that
# leaves ~2.25-2.4in per plot; see _fits_textwidth() which asserts it rather than trusting
# this comment. A 1-column figure is then a ~3.4in-wide canvas centred on the page, NOT a
# stretched 6.5in one -- that is the price of a plot being the same size wherever it appears.
PLOT_W_IN = 2.40
PLOT_H_IN = PLOT_W_IN / PANEL_ASPECT

#: A colourbar is allocated its own strip beside the plot; it never comes out of the plot box.
CBAR_W_IN = 0.13
CBAR_GAP_IN = 0.12

#: Gap between adjacent plot boxes, on top of whatever tick labels each one needs.
COL_GAP_IN = 0.10
ROW_GAP_IN = 0.10

#: Minimum outer margin, so the axes frame is never clipped by the canvas edge.
SPINE_PAD_IN = 0.03

def figsize(ncols: int = 1, nrows: int = 1, width: float | str = "full",
            shared_y: bool = False) -> tuple[float, float]:
    """Provisional canvas for an `ncols` x `nrows` grid of standard plot boxes.

    Only a STARTING size: `layout()` measures the real decorations and sets the final canvas.
    Nothing downstream depends on this being right, which is the point -- the previous design
    had a fitted decoration model here and every figure the model did not anticipate came out
    at the wrong plot size.
    """
    frac = {"full": 1.0, "half": 0.49}.get(width, width)
    w = ncols * PLOT_W_IN + (0.35 if shared_y else 0.85) * ncols + 0.30
    h = nrows * PLOT_H_IN + 0.60 * nrows + 0.15
    return (round(min(w, TEXTWIDTH_IN * float(frac)), 2), round(h, 2))


def _is_3d(ax) -> bool:
    """A 3-D axes. Its window extent is the projection's bounding square, not a plot box."""
    return hasattr(ax, "get_proj")


def _is_cbar(ax) -> bool:
    return ax.get_label() == "<colorbar>" or getattr(ax, "_colorbar", None) is not None


def _data_axes(fig):
    """The axes that hold data: everything except colourbars, spacers and empty cells."""
    out = []
    for ax in fig.axes:
        if _is_cbar(ax) or not ax.get_visible():
            continue
        if not (ax.lines or ax.collections or ax.images or ax.patches):
            continue
        out.append(ax)
    return out


def _decor_in(ax, renderer, dpi):
    """Inches of decoration (ticks, labels) on each side of the plot box: L, R, B, T."""
    ab = ax.get_window_extent()
    try:
        tb = ax.get_tightbbox(renderer)
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)
    if tb is None:
        return (0.0, 0.0, 0.0, 0.0)
    return (max(0.0, (ab.x0 - tb.x0)) / dpi, max(0.0, (tb.x1 - ab.x1)) / dpi,
            max(0.0, (ab.y0 - tb.y0)) / dpi, max(0.0, (tb.y1 - ab.y1)) / dpi)


def _cbar_owner_map(fig, data_axes, dpi):
    """Map each colourbar axes to the data axes it sits beside, and its width in inches."""
    out = {}
    for cb in fig.axes:
        if not _is_cbar(cb) or not cb.get_visible():
            continue
        cbb = cb.get_window_extent()
        bar = getattr(cb, "_colorbar", None)
        horiz = getattr(bar, "orientation", "vertical") == "horizontal"
        best, bestd = None, None
        for ax in data_axes:
            ab = ax.get_window_extent()
            if horiz:
                # A horizontal bar belongs to the axes ABOVE it, in the same column. Scoring it
                # with the vertical rule (bar's left edge vs the axes' right edge) systematically
                # picked the column to the LEFT, which put all three of heldout_resid's bars
                # under column 0 and left the right column bare.
                d = (abs(0.5 * (cbb.x0 + cbb.x1) - 0.5 * (ab.x0 + ab.x1))
                     + abs(ab.y0 - cbb.y1))
            else:
                d = abs(cbb.x0 - ab.x1) + abs(cbb.y0 - ab.y0)
            if bestd is None or d < bestd:
                best, bestd = ax, d
        if best is None:
            continue
        try:
            tb = cb.get_tightbbox(fig.canvas.get_renderer())
            extent = ((tb.y1 - tb.y0) if horiz else (tb.x1 - tb.x0)) / dpi
        except Exception:
            extent = CBAR_W_IN
        out.setdefault(best, []).append((cb, max(CBAR_W_IN, extent), horiz))
    return out


def layout(fig, max_iter: int = 3) -> None:
    """Give every plot box exactly PLOT_W_IN x PLOT_H_IN and size the canvas around it.

    This replaces tight_layout/constrained_layout for sizing. Those engines distribute a FIXED
    canvas among the axes, which is precisely the behaviour that made the plot box a residual:
    add a colourbar and the plot shrinks. Here the plot boxes are placed at a fixed physical
    size and the canvas is whatever the measured decorations require.

    Skipped for 3-D and fixed-aspect axes, whose extent is not a data rectangle.
    """
    if any(_is_3d(ax) for ax in fig.axes):
        return
    axes = _data_axes(fig)
    if not axes:
        return
    for ax in axes:
        try:
            if ax.get_aspect() != "auto":
                return
        except Exception:
            return

    dpi = fig.dpi
    pad = [0.0, 0.0, 0.0, 0.0]      # extra outer margin L,R,B,T discovered from overhang
    for _ in range(max_iter):
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        cbars = _cbar_owner_map(fig, axes, dpi)

        # Grid from the SUBPLOTSPEC where there is one. Inferring it from positions is not
        # safe: `fig.colorbar(ax=...)` shrinks its parent before layout runs, which moved one
        # panel of a 1x2 down and made this read a 2-row grid. Position is kept only as the
        # fallback for hand-placed axes, which have no spec to ask.
        spec = {}
        for ax in axes:
            cell = getattr(ax, "_ps_cell", None)
            if cell is not None:
                spec[ax] = cell
                continue
            try:
                ss = ax.get_subplotspec()
                spec[ax] = (ss.rowspan.start, ss.colspan.start) if ss is not None else None
            except Exception:
                spec[ax] = None
        # All-identical specs mean the colourbar re-parenting above has flattened them; that is
        # not a 1x1 grid, it is a destroyed one. Fall through to positions.
        if len(axes) > 1 and len(set(spec.values())) == 1:
            spec = {ax: None for ax in axes}
        if all(v is not None for v in spec.values()):
            rs = sorted({v[0] for v in spec.values()})
            cs = sorted({v[1] for v in spec.values()})
            row_of = {ax: rs.index(spec[ax][0]) for ax in axes}
            col_of = {ax: cs.index(spec[ax][1]) for ax in axes}
            nrows, ncols = len(rs), len(cs)
        else:
            xs = sorted({round(ax.get_window_extent().x0 / dpi, 1) for ax in axes})
            ys = sorted({round(ax.get_window_extent().y0 / dpi, 1) for ax in axes}, reverse=True)
            col_of = {ax: xs.index(round(ax.get_window_extent().x0 / dpi, 1)) for ax in axes}
            row_of = {ax: ys.index(round(ax.get_window_extent().y0 / dpi, 1)) for ax in axes}
            ncols, nrows = len(xs), len(ys)

        L = [0.0] * ncols; R = [0.0] * ncols
        B = [0.0] * nrows; T = [0.0] * nrows
        dec = {}
        for ax in axes:
            l, r, b, t = _decor_in(ax, rend, dpi)
            dec[ax] = (l, r, b, t)
            mine = cbars.get(ax, [])
            ex_r = sum(w + CBAR_GAP_IN for _, w, hz in mine if not hz)
            ex_b = sum(w + CBAR_GAP_IN for _, w, hz in mine if hz)
            c, rw = col_of[ax], row_of[ax]
            L[c] = max(L[c], l); R[c] = max(R[c], r + ex_r)
            B[rw] = max(B[rw], b + ex_b); T[rw] = max(T[rw], t)

        # A figure-level legend is paid for by the canvas, never by the plots.
        # Which SIDE a figure legend is on is measured, not declared. Trusting
        # `_ps_legend_side` (set only by ps.shared_legend) reserved a top strip for
        # make_ir.py's raw `fig.legend(loc="outside lower center")`, leaving 0.36in of dead
        # white space at the top and printing the legend over the bottom colourbar.
        fh_now = fig.get_size_inches()[1]
        top_leg = bot_leg = 0.0
        for lg in fig.legends:
            try:
                lb = lg.get_window_extent(rend)
            except Exception:
                continue
            h = lb.height / dpi + 0.10
            if 0.5 * (lb.y0 + lb.y1) / dpi >= 0.5 * fh_now:
                top_leg = max(top_leg, h)
            else:
                bot_leg = max(bot_leg, h)
        leg_h = top_leg + bot_leg

        # Floor the OUTER margins: with nothing decorating that side the plot box lands
        # exactly on the canvas edge and the deliberate full-box frame (axes.spines.top) is
        # clipped away -- measured as a missing top spine on hpo_optima_summary.
        L[0] = max(L[0], SPINE_PAD_IN) + pad[0]
        R[ncols - 1] = max(R[ncols - 1], SPINE_PAD_IN) + pad[1]
        B[nrows - 1] = max(B[nrows - 1], SPINE_PAD_IN) + pad[2]
        T[0] = max(T[0], SPINE_PAD_IN) + pad[3]
        col_w = [L[c] + PLOT_W_IN + R[c] for c in range(ncols)]
        row_h = [B[r] + PLOT_H_IN + T[r] for r in range(nrows)]
        W = sum(col_w) + COL_GAP_IN * (ncols - 1)
        H = sum(row_h) + ROW_GAP_IN * (nrows - 1) + leg_h
        if not (0.5 < W < 40 and 0.5 < H < 40):
            return
        fig.set_size_inches(W, H, forward=True)

        # Place every plot box explicitly, in figure fractions of the new canvas.
        x_off = [sum(col_w[:c]) + COL_GAP_IN * c for c in range(ncols)]
        y_off = [sum(row_h[:r]) + ROW_GAP_IN * r for r in range(nrows)]
        for ax in axes:
            c, rw = col_of[ax], row_of[ax]
            x0 = x_off[c] + L[c]
            y0 = H - top_leg - (y_off[rw] + row_h[rw]) + B[rw]
            ax.set_position([x0 / W, y0 / H, PLOT_W_IN / W, PLOT_H_IN / H])
            # Colourbars ride alongside in their allocated strip.
            cx = x0 + PLOT_W_IN + CBAR_GAP_IN
            # Below this axes' OWN x tick labels and xlabel, not below its box -- otherwise
            # the bar lands on top of them and its label hangs off the canvas, which the
            # overhang pass then "fixes" by growing the figure on every iteration.
            cy = y0 - dec[ax][2] - CBAR_GAP_IN - CBAR_W_IN
            for cb, w, hz in cbars.get(ax, []):
                if hz:
                    # A horizontal bar belongs UNDER its plot, spanning it, and is paid for out
                    # of the row's bottom margin. Forcing it into the vertical strip turned it
                    # into a 0.13x0.01in sliver floating above the panels and blew the canvas
                    # out to 10.85in.
                    cb.set_position([x0 / W, cy / H, PLOT_W_IN / W, CBAR_W_IN / H])
                    cy -= w + CBAR_GAP_IN
                else:
                    cb.set_position([cx / W, y0 / H, CBAR_W_IN / W, PLOT_H_IN / H])
                    cx += w + CBAR_GAP_IN
        fig._ps_laid_out = True
        fig._ps_grid = (nrows, ncols, len(axes))
        # Anything still hanging off the canvas becomes outer margin on the next pass. The
        # plot boxes are never touched: an overhang costs canvas, like every other decoration.
        # (A colourbar's tick labels can reach a little above the bar, which the per-axes
        # tightbbox does not see because the colourbar is a separate axes.)
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        over = [max(0.0, -tb.x0), max(0.0, tb.x1 - W), max(0.0, -tb.y0), max(0.0, tb.y1 - H)]
        if max(over) < 0.01:
            return
        pad = [pad[i] + over[i] for i in range(4)]


def _fits_textwidth(verbose: bool = True) -> bool:
    """Assert the standard plot box still fits \textwidth in the WORST 2-column case.

    The worst case is two columns each carrying their own y-label and 4-character
    scientific-notation tick labels. If this fails, PLOT_W_IN is too big and SOME figure would
    have to break the invariant -- which is exactly how the previous two designs died, so it
    is checked rather than asserted in a comment.
    """
    import matplotlib.pyplot as _plt
    import numpy as _np
    fig, axes = _plt.subplots(1, 2, figsize=(6.5, 2.6))
    for ax in axes:
        ax.plot([1e-6, 1e-1], [1e-9, 1e-2])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$y_{\min}$")
        ax.set_ylabel(r"MSE$(\Delta\log|\mathcal{M}|^2)$")
    layout(fig)
    w = fig.get_size_inches()[0]
    _plt.close(fig)
    if verbose:
        print(f"worst-case 2-column canvas: {w:.2f}in (limit {TEXTWIDTH_IN})"
              f" -> {'OK' if w <= TEXTWIDTH_IN + 0.01 else 'TOO WIDE, reduce PLOT_W_IN'}")
    return w <= TEXTWIDTH_IN + 0.01


def _measure_panels(grids=((1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2))) -> None:
    """Print the achieved plot box per grid. Every row must read PLOT_W_IN x PLOT_H_IN."""
    import matplotlib.pyplot as _plt
    print(f"target {PLOT_W_IN:.2f}x{PLOT_H_IN:.2f}in")
    for nc, nr in grids:
        fig, _ = figure(ncols=nc, nrows=nr, squeeze=False)
        for ax in fig.axes:
            ax.plot([1, 2], [1, 2])
            ax.set_xlabel("x"); ax.set_ylabel("y")
        layout(fig)
        fig.canvas.draw()
        bb = fig.axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fw, fh = fig.get_size_inches()
        print(f"  {nc}x{nr}: canvas={fw:.2f}x{fh:.2f}  plot={bb.width:.2f}x{bb.height:.2f}")
        _plt.close(fig)


# --- colour -----------------------------------------------------------------

# Okabe-Ito, the standard colourblind-safe qualitative palette. Ordered so the first
# two (blue, vermillion) carry the usual baseline-vs-treatment contrast and stay
# distinguishable in greyscale.
C = SimpleNamespace(
    blue="#0072B2",
    vermillion="#D55E00",
    green="#009E73",
    orange="#E69F00",
    sky="#56B4E9",
    purple="#CC79A7",
    yellow="#F0E442",
    grey="#555555",
)

#: Cycle order for successive plot calls.
CYCLE = [C.blue, C.vermillion, C.green, C.orange, C.purple, C.sky, C.grey]

#: Sequential colormap for heatmaps / 2-D surfaces.
CMAP = "viridis"

#: Diverging colormap, for quantities with a meaningful zero (e.g. log ratios).
CMAP_DIV = "RdBu_r"


def sequence(n: int, cmap: str = "viridis", lo: float = 0.10, hi: float = 0.88) -> list:
    """`n` colours along a perceptually-ordered ramp, for series with a natural ORDER.

    Use for an ordered sweep (gamma = 1, 2, 3, 5, 10; dataset size; fraction f), where the
    qualitative palette `C` would hide the ordering. Endpoints are trimmed to keep both ends
    readable on white.
    """
    cm = plt.get_cmap(cmap)
    if n == 1:
        return [cm(0.5)]
    return [cm(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def use() -> None:
    """Apply the style. Called automatically on import."""
    mpl.rcParams.update({
        # --- typography: one family, one size, matching 11pt Computer Modern body text
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,          # cmr10 lacks U+2212
        "font.size": BASE_PT,
        "axes.titlesize": BASE_PT,
        "axes.labelsize": BASE_PT,
        "xtick.labelsize": BASE_PT,
        "ytick.labelsize": BASE_PT,
        "legend.fontsize": BASE_PT,
        "figure.titlesize": BASE_PT,

        # --- geometry
        "figure.figsize": figsize(1, 1),
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "figure.constrained_layout.use": False,   # we call tight_layout in save()

        # --- axes: light, unobtrusive frame; grid is a reading aid, not a feature
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.linewidth": 0.8,
        # Full box frame, as in the reference paper: all four spines drawn. The reference
        # draws the enclosing LINES only -- tick marks stay on the left and bottom, where the
        # numbers are. Mirroring ticks onto the top and right edges is a different choice and
        # was never asked for; it just adds clutter to every panel.
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.top": False,
        "ytick.right": False,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,

        # --- lines and markers
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "errorbar.capsize": 2.5,

        # --- ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # --- legend: no frame, so it sits on the plot without boxing it in
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.4,
        "legend.labelspacing": 0.3,

        # --- output
        # NOT "tight". bbox="tight" crops the canvas to its content, so a figure built at
        # 6.5in saves at ~6.19in; \includegraphics[width=\textwidth] then scales it back UP to
        # 6.5in and its 11pt text prints at 11.6pt. A figure whose legend overhangs saves at
        # 7.28in and is scaled DOWN to 9.8pt. Measured across the included set, that was a 19%
        # spread in effective font size -- which is exactly what "figures should look like a
        # set, text the same size as the body" rules out. Saving the full canvas makes the
        # scale factor exactly 1.000, and makes overhang show up as visible clipping instead
        # of a silent rescale on the page.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


use()


# --- construction and output -------------------------------------------------

def figure(ncols: int = 1, nrows: int = 1, width: float | str = "full", **kwargs):
    """`plt.subplots` at the repo's standard width and aspect for this grid.

    `width="half"` for a figure that LaTeX will include at 0.49\textwidth beside another.
    Returns whatever `plt.subplots` returns: `(fig, ax)` for a single panel,
    `(fig, axes)` otherwise.
    """
    # sharey changes the width budget, so the derived height must know about it.
    kwargs.setdefault("figsize", figsize(ncols, nrows, width,
                                         shared_y=bool(kwargs.get("sharey"))))
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    tag_grid(fig, nrows, ncols)
    return fig, axes


def panels(n: int, **kwargs):
    """`n` INDEPENDENT one-plot figures, to be included side by side in one LaTeX figure.

    Use this whenever the panel count does not fill a rectangle -- in practice, any 3. A 3-panel
    set forced into a 2x2 leaves a hole where the fourth would go, and that hole is the single
    most-remarked-on defect in this document's figures. Emitting three separate files and
    letting LaTeX pack them (two on the first line, the third centred under them) is what the
    same three plots would have looked like had they been written as three figures, which is
    what they are.

        figs = ps.panels(3)
        for (fig, ax), d in zip(figs, datasets):
            ax.plot(...)
        ps.save_panels(figs, "analysis/divergences/figs/my_set")

    Each panel carries its own axis labels, its own legend and its own colourbar; nothing is
    shared, because sharing is what forced them into one canvas in the first place.
    """
    return [figure(**kwargs) for _ in range(n)]


#: Suffixes for the files `save_panels` writes. `_a`, `_b`, ... so the LaTeX include order is
#: obvious and a caption can say "(a)", "(b)", "(c)" without counting.
PANEL_SUFFIXES = "abcdefghij"


def save_panels(figs, base: str, repo: str | None = None) -> list:
    """Save each of `figs` as `<base>_a`, `<base>_b`, ... and print the LaTeX to include them.

    `figs` is what `panels()` returned (a list of `(fig, ax)`), or a plain list of figures.
    """
    out = []
    for i, item in enumerate(figs):
        fig = item[0] if isinstance(item, tuple) else item
        out.append(save(fig, f"{base}_{PANEL_SUFFIXES[i]}", repo=repo))
    stem = os.path.basename(base)
    rows = [PANEL_SUFFIXES[i:i + 2] for i in range(0, len(figs), 2)]
    body = " \\\\[1ex]\n  ".join(
        "\\hfill".join(f"\\includegraphics{{{stem}_{s}.pdf}}" for s in row) for row in rows)
    print(f"  LaTeX:\n  \\centering\n  {body}")
    return out


def tag_grid(fig, nrows: int, ncols: int) -> None:
    """Record each axes' (row, col) on the axes itself, for `layout()`.

    Necessary because `fig.colorbar(ax=...)` RE-PARENTS its parent axes into a fresh 1x1
    gridspec: after adding a colourbar to each cell of a 2x2, all four report subplotspec
    (0, 0), and any grid inferred from the spec collapses to a single cell. Positions are no
    better -- a colourbar shrinks and shifts its parent before layout ever runs. The only
    reliable moment is creation, so the grid is recorded there.

    Producers that build their own GridSpec should call this after adding their subplots.
    """
    for i, ax in enumerate(fig.axes[:nrows * ncols]):
        ax._ps_cell = (i // ncols, i % ncols)


def process_label(ax, text: str, loc: str = "upper right", **kwargs):
    """Put the process label inside the axes: the one annotation a figure may carry.

    This replaces the title. Use it to say WHICH process is shown, e.g.
    `process_label(ax, r"$e^+e^-\\to u\\bar u gg$")`. Do not use it to explain the plot.
    """
    xy = {
        "upper left": (0.03, 0.97, "left", "top"),
        "upper right": (0.97, 0.97, "right", "top"),
        "lower left": (0.03, 0.03, "left", "bottom"),
        "lower right": (0.97, 0.03, "right", "bottom"),
    }[loc]
    kwargs.setdefault("fontsize", BASE_PT)
    lbl = ax.text(xy[0], xy[1], text, transform=ax.transAxes,
                  ha=xy[2], va=xy[3], **kwargs)
    ax._ps_label = lbl          # make_room() keeps data out from under it
    return lbl


#: The only legend positions allowed. A legend at `center left` / `center right` / `center`
#: floats in the middle of the plot with data on both sides of it, and it reads as a mistake
#: even when it happens not to overlap anything -- two of them ("weirdly placed") were spotted
#: on sight in the compiled document. A corner is always either clear or made clear by
#: `make_room`, which grows the axis; the middle of a plot cannot be cleared at all.
LEGEND_LOCS = ("upper left", "upper right", "lower left", "lower right")


def legend(ax, loc: str = "upper left", **kwargs):
    """The one way to put a legend on an axes: inside it, in a corner.

    `loc` must be a corner (see LEGEND_LOCS). `make_room`, called by `save`, then grows the
    y-range until the legend is clear of the data, so the corner does not have to be chosen to
    suit this particular dataset -- which is what made every script pick a different one.

    If the legend comes out wider than the plot box it is rebuilt at `ncol=1`: a legend that
    overhangs the axes is the "pops out horizontally" failure, and dropping to one column is
    the fix that does not shrink the font.
    """
    if loc not in LEGEND_LOCS:
        raise ValueError(f"legend loc {loc!r} is not a corner; use one of {LEGEND_LOCS}")
    kwargs.setdefault("loc", loc)
    leg = ax.legend(**kwargs)
    _shrink_wide_legend(ax, leg, kwargs)
    return leg


def _shrink_wide_legend(ax, leg, kwargs):
    """Rebuild a legend at fewer columns while it is wider than the plot box."""
    if leg is None or kwargs.get("ncol", 1) <= 1:
        return leg
    fig = ax.figure
    for ncol in range(int(kwargs["ncol"]) - 1, 0, -1):
        try:
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            if leg.get_window_extent(rend).width <= ax.get_window_extent().width:
                return leg
        except Exception:
            return leg
        kwargs["ncol"] = ncol
        leg = ax.legend(**kwargs)
    return leg


def shared_legend(fig, ax, ncol: int = 3, **kwargs):
    """One legend above the panels, for a multi-panel figure whose panels share series.

    Per-axes legends in a 2- or 3-panel figure almost always land on the data. When every
    panel plots the same series, take the handles from one axes and put a single legend
    across the top instead. Call BEFORE `save`.
    """
    handles, labels = ax.get_legend_handles_labels()
    # `loc="lower center"` puts the strip BELOW the panels. Producers that want that used to
    # hand-roll `fig.legend(...)` + `tight_layout(rect=[0, 0.30, 1, 1])`, which reserves a
    # fraction rather than inches and is invisible to the geometry solve. Supporting it here
    # means there is one path, and one place that knows how much room the legend needs.
    bottom = "lower" in str(kwargs.get("loc", "upper center"))
    kwargs.setdefault("loc", "upper center")
    kwargs.setdefault("bbox_to_anchor", (0.5, 0.0) if bottom else (0.5, 1.0))
    kwargs.setdefault("frameon", False)
    leg = fig.legend(handles, labels, ncol=ncol, **kwargs)
    fig._ps_legend_side = "bottom" if bottom else "top"
    # GROW the figure by the legend's measured height; do not carve the strip out of the
    # panels. Reserving a rect on the existing canvas pays for the legend with data area: on a
    # figsize(3,1) that turned a 1.77x0.91in panel into 1.77x0.52in, i.e. aspect 1.95 -> 3.41,
    # and left l2_sigma_vs_divergence rendering as three letterbox strips at aspect 3.48
    # against a target of 1.25. Adding the height instead keeps the panel at PANEL_ASPECT,
    # which is the whole point of deriving figsize.
    pad_in = 0.10
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        leg_in = leg.get_window_extent(rend).height / fig.dpi + pad_in
        w, h = fig.get_size_inches()
        fig.set_size_inches(w, h + leg_in, forward=True)
    except Exception:
        leg_in = 0.10 * fig.get_size_inches()[1]
    # Stored in INCHES, not as a fraction of the height: enforce_panels resizes the figure
    # afterwards, and a cached fraction would quietly shrink the strip on every growth step
    # until the legend sat back on the panels.
    fig._ps_legend_in = leg_in
    return leg


#: Fraction of the plot box left clear above the topmost datum and below the lowest.
#: Fixed so headroom does not depend on what the data happened to do -- "a lot of figures have
#: very little space between the line and the top edge, while others don't".
DATA_MARGIN = 0.06


def _y_data_extent(ax):
    """(min, max) of the REAL plotted y-data, or None.

    Read from the artists' data arrays, not by transforming display points back through
    `transData`. The round trip was the source of three separate corruptions:
      - a QuadMesh/PolyCollection/errorbar LineCollection reports `get_offsets() == [[0, 0]]`,
        so a heatmap's only "datum" was the origin and the panel was rescaled to +/-0.06,
      - `axvline`'s y-data is (0, 1) in AXES coordinates; transforming it with `transData`
        injected fake data at y=0 and y=1 and pushed the real curve off-screen,
      - subsampling the point list could step straight over the true maximum.
    """
    import numpy as np
    ys = []
    for ln in ax.get_lines():
        # axvline/axhline use a blended transform; their y-data is not in data space.
        if ln.get_transform() is not ax.transData:
            continue
        y = np.asarray(ln.get_ydata(), dtype=float)
        y = y[np.isfinite(y)]
        if y.size:
            ys.append(y)
    for coll in ax.collections:
        get_off = getattr(coll, "get_offsets", None)
        if get_off is None:
            continue
        try:
            off = np.asarray(get_off(), dtype=float)
        except Exception:
            continue
        if off.ndim != 2 or off.shape[0] == 0:
            continue
        if off.shape[0] == 1 and not np.any(off):
            continue                      # the [[0, 0]] "no offsets" sentinel
        y = off[:, 1][np.isfinite(off[:, 1])]
        if y.size:
            ys.append(y)
    if not ys:
        return None
    allv = np.concatenate(ys)
    return float(allv.min()), float(allv.max())


def _shared_y_groups(fig, axes):
    """Partition `axes` into shared-y groups (a lone axes is a group of one)."""
    groups, seen = [], set()
    for ax in axes:
        if ax in seen:
            continue
        try:
            sib = [a for a in ax.get_shared_y_axes().get_siblings(ax) if a in axes]
        except Exception:
            sib = [ax]
        seen.update(sib)
        groups.append(sib or [ax])
    return groups


def apply_headroom(fig, margin: float = DATA_MARGIN) -> None:
    """`headroom` for a whole figure, one shared-y group at a time.

    Per-axes is WRONG when `sharey=True`: `set_ylim` propagates across the group and turns
    autoscale off for all of it, so the first panel's data set the limits and every later panel
    took the `not autoscaley_on` early return and had its curve clipped. Measured on a 1x2
    sharey pair with y in [0, 0.2] and [0, 10], both ended at (-0.012, 0.212) -- the second
    panel's data ran off the top. It reached results.tex in l2_sigma_vs_divergence.
    """
    for grp in _shared_y_groups(fig, _data_axes(fig)):
        headroom(grp, margin)


def headroom(ax, margin: float = DATA_MARGIN) -> None:
    """Give every plot the same clear band above and below its data.

    Only where that is meaningful and wanted:
      - never on a 2-D map (image/QuadMesh): there is no "curve" and the extent IS the data,
      - never when the script set the limits itself (`autoscaley_on` False) -- e.g. the 3-seed
        panel whose tight ylim is the whole point of the figure, or `set_ylim(bottom=0)`,
      - never on an inverted axis, whose orientation `set_ylim` would silently flip.
    """
    import numpy as np
    from matplotlib.collections import QuadMesh
    group = ax if isinstance(ax, (list, tuple)) else [ax]
    ax = group[0]
    if any(a.images or any(isinstance(c, QuadMesh) for c in a.collections) for a in group):
        return
    if not ax.get_autoscaley_on():
        return
    lo0, hi0 = ax.get_ylim()
    if lo0 > hi0:
        return
    if ax.get_yscale() not in ("linear", "log"):
        return
    exts = [e for e in (_y_data_extent(a) for a in group) if e is not None]
    if not exts:
        return
    lo_d, hi_d = min(e[0] for e in exts), max(e[1] for e in exts)
    if ax.get_yscale() == "log":
        if lo_d <= 0 or hi_d <= 0:
            return
        span = np.log10(hi_d / lo_d) or 1.0
        ax.set_ylim(10 ** (np.log10(lo_d) - margin * span),
                    10 ** (np.log10(hi_d) + margin * span))
    else:
        span = (hi_d - lo_d) or (abs(hi_d) or 1.0)
        ax.set_ylim(lo_d - margin * span, hi_d + margin * span)
    ax.set_autoscaley_on(True)


def _data_points(ax):
    """Every plotted point in DISPLAY coordinates, for overlap tests."""
    import numpy as np
    pts = []
    for ln in ax.get_lines():
        xy = ln.get_xydata()
        if xy is not None and len(xy):
            pts.append(ax.transData.transform(xy))
    for coll in ax.collections:
        try:
            off = coll.get_offsets()
        except Exception:
            continue
        if off is not None and len(off):
            pts.append(ax.transData.transform(np.asarray(off)))
    if not pts:
        return None
    P = np.vstack(pts)
    return P[np.isfinite(P).all(axis=1)]


def _collides(ax, artist, renderer) -> bool:
    """Does `artist` (a legend or the process label) sit on top of any drawn data?"""
    import numpy as np
    try:
        bb = artist.get_window_extent(renderer)
    except Exception:
        return False
    P = _data_points(ax)
    if P is not None and len(P):
        hit = ((P[:, 0] >= bb.x0) & (P[:, 0] <= bb.x1) &
               (P[:, 1] >= bb.y0) & (P[:, 1] <= bb.y1))
        if bool(np.any(hit)):
            return True
    for p in ax.patches:                      # bars, spans
        try:
            if p.get_window_extent(renderer).overlaps(bb):
                return True
        except Exception:
            pass
    return False


def make_room(ax, max_iter: int = 12, step: float = 0.10, max_growth: float = 1.8) -> None:
    """Grow the y-range until the legend and process label no longer cover data.

    The alternative -- hand-picking `loc=` per panel -- does not survive a data change and
    was the source of most of the overlapping labels here. Expanding the axis instead
    compresses the curves slightly and always leaves the annotation legible.
    """
    fig = ax.figure
    movers = [a for a in (ax.get_legend(), getattr(ax, "_ps_label", None)) if a is not None]
    if not movers:
        return
    # Never touch a colourbar, nor a panel whose content is a 2-D map: growing the y-range
    # of a pcolormesh/imshow does not "make room", it stretches the image and leaves a band
    # of blank axes. Those panels need a smaller legend, not a bigger axis.
    if ax.get_label() == "<colorbar>" or getattr(ax, "_colorbar", None) is not None:
        return
    if ax.images:
        return
    from matplotlib.collections import QuadMesh
    if any(isinstance(c, QuadMesh) for c in ax.collections):
        return
    # Bound the growth. Without a cap a tall legend on a log axis spanning several decades
    # keeps demanding room and ends up squashing the data into a corner, which is worse
    # than a little overlap. Past the cap, leave it: the panel needs a smaller legend
    # (ncol=2) or to be split into two figures.
    lo0, hi0 = ax.get_ylim()
    log_y = ax.get_yscale() == "log" and lo0 > 0 and hi0 > 0
    # Measure the span the same way on both scales: DECADES on a log axis, absolute range
    # on a linear one. Comparing a raw ratio (e.g. 7 decades = 1e7) against max_growth=1.8
    # made the cap fire after a single step on any axis wider than ~2.6 decades, so
    # make_room silently became "one 10% expansion" instead of "grow until clear".
    import math
    span0 = math.log10(hi0 / lo0) if log_y else (hi0 - lo0)
    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bad = [m for m in movers if _collides(ax, m, renderer)]
        if not bad:
            return
        ax_bb = ax.get_window_extent(renderer)
        mid = 0.5 * (ax_bb.y0 + ax_bb.y1)
        grow_top = grow_bot = False
        for m in bad:
            bb = m.get_window_extent(renderer)
            if 0.5 * (bb.y0 + bb.y1) >= mid:
                grow_top = True
            else:
                grow_bot = True
        lo, hi = ax.get_ylim()
        span = math.log10(hi / lo) if log_y else (hi - lo)
        if span0 > 0 and span / span0 >= max_growth:
            return
        if ax.get_yscale() == "log":
            if lo <= 0 or hi <= 0:
                return
            r = hi / lo
            if grow_top:
                hi *= r ** step
            if grow_bot:
                lo /= r ** step
        else:
            span = hi - lo
            if grow_top:
                hi += span * step
            if grow_bot:
                lo -= span * step
        ax.set_ylim(lo, hi)


#: Colourbar geometry: horizontal, under its own panel. `pad` is in fractions of the axes
#: height and has to clear the panel's x tick labels and x-label.
CBAR_KW = dict(orientation="horizontal", location="bottom", fraction=0.07, pad=0.32)


def colorbar(ax, mappable, label: str = "", **kwargs):
    """The one way to put a colourbar on a plot: HORIZONTAL, directly under its own panel.

    One rule with no branches, because every branch this had before turned into an
    inconsistency someone noticed on the page: bars on the right of some panels and under
    others, one bar serving two panels while a third had its own, a bar above a panel because
    that was the only place it fitted.

    Horizontal-below is the placement that always fits. A vertical bar costs ~0.8in of COLUMN,
    which puts a panel at ~3.9in so two can never share a line; a horizontal one costs height
    and leaves the panel at ~3.1in. Every panel that needs a scale gets its own bar.
    """
    kw = dict(CBAR_KW)
    kw.update(kwargs)
    cb = ax.figure.colorbar(mappable, ax=ax, **kw)
    if label:
        cb.set_label(label)
    return cb


def tidy_colorbars(fig) -> None:
    """Keep colourbar tick labels from colliding, without shrinking the font.

    A horizontal bar under a 2.40in panel has room for about four labels. Left to matplotlib
    the error map's bar asked for six at full decimal precision
    ("0.000000 0.00005 0.00010 0.00015 0.00020") and they ran into each other. Fewer ticks and
    mathtext scientific notation is the fix the style rules already name for crowded ticks.
    """
    from matplotlib.ticker import MaxNLocator, ScalarFormatter, LogLocator
    for cb in fig.axes:
        if not _is_cbar(cb):
            continue
        bar = getattr(cb, "_colorbar", None)
        horiz = getattr(bar, "orientation", "vertical") == "horizontal"
        axis = cb.xaxis if horiz else cb.yaxis
        if isinstance(axis.get_major_locator(), LogLocator):
            continue                       # a log bar's decade ticks are already sparse
        axis.set_major_locator(MaxNLocator(nbins=4))
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 3))       # 0.00020 -> 2 x 10^-4, in the document's mathtext
        axis.set_major_formatter(fmt)


def save(fig, base: str, repo: str | None = None) -> str:
    """Save `fig` as BOTH `<base>.png` and `<base>.pdf` (repo convention, no exceptions).

    `base` is a path without extension; if relative and `repo` is given (or the module
    can find the repo root), it is resolved against the repo root. Applies
    `tight_layout()` first, so callers never need to.
    """
    if not os.path.isabs(base):
        root = repo or os.path.dirname(os.path.abspath(__file__))
        base = os.path.join(root, base)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    # A constrained/tight layout engine would fight `layout()` for control of the axes
    # positions, and it is the one that makes the plot box a residual. Drop it.
    try:
        fig.set_layout_engine("none")
    except Exception:
        pass
    # Headroom, then legends clear of the data, then geometry. `make_room` only changes DATA
    # limits, never sizes, so it cannot disturb the invariant.
    try:
        apply_headroom(fig)
    except Exception:
        pass
    try:
        tidy_colorbars(fig)
    except Exception:
        pass
    for _ax in _data_axes(fig):
        try:
            make_room(_ax)
        except Exception:
            pass
    # LAST and authoritative: every plot box to exactly PLOT_W_IN x PLOT_H_IN, canvas sized
    # around whatever the decorations turned out to need.
    try:
        layout(fig)
    except Exception as exc:
        print(f"  !! {os.path.basename(base)}: layout failed: "
              f"{type(exc).__name__}: {exc}")
    _warn_if_squeezed(fig, base)
    fig.savefig(base + ".png")
    fig.savefig(base + ".pdf")
    print(f"saved {base}.png / .pdf")
    return base


#: A data panel narrower than this (inches) is not a figure, it is a sliver.
MIN_PANEL_IN = 1.05


def check_panels(fig, name: str) -> None:
    """Public squeeze check, for figures that do NOT exit through `save()`.

    `save()` calls this for you. Multi-page producers write via `PdfPages.savefig` and so
    never touch `save()` -- which is exactly how `phase1_scaling.pdf` shipped as five 0.6in
    slivers while `rebuild_figures.sh` reported `ok` and `squeezed 0`. Call this immediately
    before every `pdf.savefig(fig)`.
    """
    try:
        fig.set_layout_engine("none")
    except Exception:
        pass
    try:
        layout(fig)
    except Exception as exc:
        print(f"  !! {name}: layout failed: {type(exc).__name__}: {exc}")
    _warn_if_squeezed(fig, name)


def _warn_if_squeezed(fig, base: str) -> None:
    """Check THE INVARIANT: every plot box is PLOT_W_IN x PLOT_H_IN, and nothing is clipped.

    Everything this used to check was a proxy -- a minimum size, a spread, an aspect. All three
    pass happily on a document where every figure is internally fine and no two agree with each
    other, which is the state that shipped. There is only one thing worth asserting now, and it
    is exact.
    """
    name = os.path.basename(base)
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        inv = fig.dpi_scale_trans.inverted()
        boxes = [(ax, ax.get_window_extent().transformed(inv))
                 for ax in _data_axes(fig) if not _is_3d(ax)]
        if boxes:
            bad = [b for _, b in boxes
                   if abs(b.width - PLOT_W_IN) > 0.02 or abs(b.height - PLOT_H_IN) > 0.02]
            if bad:
                ws = f"{min(b.width for b in bad):.2f}-{max(b.width for b in bad):.2f}"
                hs = f"{min(b.height for b in bad):.2f}-{max(b.height for b in bad):.2f}"
                print(f"  !! {name}: {len(bad)}/{len(boxes)} plot box(es) at {ws} x {hs}in, "
                      f"not the {PLOT_W_IN:.2f}x{PLOT_H_IN:.2f}in standard -- layout() did not "
                      f"run or was overridden (tight_layout/constrained_layout after save?)")
        fw, fh = fig.get_size_inches()
        if fw > TEXTWIDTH_IN + 0.01:
            print(f"  !! {name}: canvas {fw:.2f}in exceeds \\textwidth ({TEXTWIDTH_IN}in) -- "
                  f"too many columns for the standard plot box, or an outside legend that "
                  f"should be inside the axes")
        tb = fig.get_tightbbox(rend)
        if tb.x0 < -0.02 or tb.y0 < -0.02 or tb.x1 > fw + 0.02 or tb.y1 > fh + 0.02:
            print(f"  !! {name}: content overhangs the canvas ({tb.x0:.2f}..{tb.x1:.2f} x "
                  f"{tb.y0:.2f}..{tb.y1:.2f}in vs {fw:.2f}x{fh:.2f}in) -- it WILL be clipped")
        # A grid with an empty cell. Three panels in a 2x2 leave a visible hole where the
        # fourth would be; the fix is ps.panels(3) + ps.save_panels, so LaTeX packs them two
        # on the first line and the third centred underneath, as three figures would have.
        nr, nc, n = getattr(fig, "_ps_grid", (1, 1, 1))
        # A spare cell holding the figure's legend is not a hole -- it is the legend's home,
        # and it is cheaper than a strip above the panels. Count it as occupied.
        n += sum(1 for ax in fig.axes
                 if ax not in [b[0] for b in boxes] and ax.get_legend() is not None)
        if nr * nc > n:
            print(f"  !! {name}: {n} panels in a {nr}x{nc} grid leaves {nr * nc - n} empty "
                  f"cell(s) -- use ps.panels({n}) + ps.save_panels() instead")
        # A legend wider than its plot box hangs out over the neighbouring panel or off the
        # canvas. ps.legend drops ncol for you; a raw ax.legend(ncol=...) does not.
        for ax, box in boxes:
            lg = ax.get_legend()
            if lg is None:
                continue
            try:
                if lg.get_window_extent(rend).width > ax.get_window_extent().width + 1:
                    print(f"  !! {name}: legend is wider than its plot box -- use ps.legend(), "
                          f"which drops to ncol=1, or shorten the labels")
            except Exception:
                pass
    except Exception as exc:
        print(f"  !! {name}: layout check failed to run: {type(exc).__name__}: {exc}")
