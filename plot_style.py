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

Size. THE PANEL IS THE FIXED QUANTITY, NOT THE FIGURE. Every data panel is drawn at
PANEL_W_IN x PANEL_H_IN whatever grid it sits in, and the figure comes out as wide as that
grid needs -- so a 1-column figure is ~3.3in wide and a 2-column one ~6.5in, and a panel
looks the same in both. Pinning the figure at \\textwidth instead, as this module used to,
gives a lone panel 5.4in and a cell of a 3-wide grid 1.7in: the same object at 3x different
size on facing pages, which is what "the sizes are not consistent" means.

Because the saved width now varies, figures are included in LaTeX at their NATURAL size
(`\\includegraphics{f.pdf}`, no `width=`), not at `width=\\textwidth`. That keeps the scale
factor exactly 1.000, so 11pt in the figure prints as 11pt on the page, matching body text.
Forcing `width=\\textwidth` on a 3.3in figure would blow it up 2x and print its labels at
21pt. `enforce_panels` guarantees no figure is ever wider than \\textwidth.

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

#: THE standard data panel, in inches. Every panel of every figure is drawn at this size --
#: this is the quantity that is held fixed, and the FIGURE size is whatever follows from it.
#:
#: It used to be the other way round: the figure was pinned at \textwidth and the panels got
#: whatever was left over after the decorations. That is what made the set incoherent, and the
#: arithmetic says so plainly -- at 6.5in a lone panel comes out 5.4in wide, a cell of a 3-wide
#: grid 1.7in. Those are the SAME KIND OF OBJECT rendered at 3x different linear size, on
#: facing pages. Fixing the panel and letting a 1-column figure be 3.3in wide instead of 6.5in
#: is what makes a panel look the same wherever it appears.
#:
#: The value is the panel a 2-column figure gets at \textwidth, because ncols=2 is by far the
#: commonest layout here -- so the modal figure is unchanged and everything else moves to meet
#: it.
PANEL_W_IN = 2.55
PANEL_H_IN = PANEL_W_IN / PANEL_ASPECT

# Starting guess for the decorations, in inches. These are ONLY a starting guess now:
# enforce_panels() measures what the layout engine actually did and corrects the figure size,
# so an error here costs an iteration, not a misshapen figure. Before that existed these
# constants were load-bearing and every figure that deviated from the grid they were fitted on
# (a colourbar, a shared legend, a long label) silently shipped at the wrong panel size.
_W_FIRST, _W_PER_COL, _H_ROW = 0.73, 0.68, 0.63
#: Per-extra-column cost when the y-axis is SHARED: just the gap, no second tick-label block.
_W_GAP_ONLY = 0.30


def figsize(ncols: int = 1, nrows: int = 1, width: float | str = "full",
            shared_y: bool = False) -> tuple[float, float]:
    """Figure size in inches holding PANEL_W_IN x PANEL_H_IN per panel.

    Capped at `width` (\\textwidth by default): a grid too wide to hold the standard panel
    gets uniformly smaller panels AT THE SAME ASPECT, rather than a wider-than-page figure.
    So aspect is invariant everywhere, and size is invariant everywhere it can be.
    """
    frac = {"full": 1.0, "half": 0.49}.get(width, width)
    w_max = TEXTWIDTH_IN * float(frac)
    per_col = _W_GAP_ONLY if shared_y else _W_PER_COL
    deco_w = _W_FIRST + per_col * (ncols - 1)
    panel_w = min(PANEL_W_IN, (w_max - deco_w) / ncols)
    panel_h = panel_w / PANEL_ASPECT
    return (round(deco_w + ncols * panel_w, 2), round(nrows * (panel_h + _H_ROW), 2))


def _is_3d(ax) -> bool:
    """A 3-D axes. Its window extent is the projection's bounding square, not a data panel."""
    return hasattr(ax, "get_proj")


def _data_axes(fig):
    """The axes that hold data: everything except colourbars, spacers and empty cells."""
    out = []
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>" or not ax.get_visible():
            continue
        if getattr(ax, "_colorbar", None) is not None:
            continue
        if not (ax.lines or ax.collections or ax.images or ax.patches):
            continue
        out.append(ax)
    return out


def _relayout(fig) -> None:
    """Re-run whatever layout engine this figure uses, preserving any reserved strip.

    MUST honour `_ps_layout_done`. Producers set that flag to mean "I have laid this figure
    out myself, do not run tight_layout on it" -- a raw `tight_layout(rect=...)` reserving a
    legend strip, or a GridSpec-plus-colourbar arrangement that tight_layout fights. Calling a
    bare tight_layout here regardless is what re-broke the collinear figures: the axes expanded
    straight over the reserved strip and the 7-entry legend printed on top of both panels and
    their x-labels. Resizing alone is safe for those figures -- axes positions are stored as
    figure FRACTIONS, so they scale with the canvas and the reservation survives.
    """
    engine = fig.get_layout_engine()
    if engine is not None and engine.__class__.__name__ != "PlaceHolderLayoutEngine":
        return                                   # constrained layout re-runs itself on draw
    # Recompute the legend strip as a FRACTION of the current height. Caching the fraction
    # instead would silently shrink the strip every time the figure grew.
    leg_in = getattr(fig, "_ps_legend_in", 0.0)
    if leg_in:
        frac = min(0.50, leg_in / fig.get_size_inches()[1])
        rect = ([0, frac, 1, 1] if getattr(fig, "_ps_legend_side", "top") == "bottom"
                else [0, 0, 1, 1 - frac])
        try:
            fig.tight_layout(rect=rect)
        except Exception:
            pass
        return
    if getattr(fig, "_ps_layout_done", False):
        return                                   # the producer owns this layout; leave it
    try:
        fig.tight_layout()
    except Exception:
        pass


def enforce_panels(fig, max_iter: int = 6, tol: float = 0.02) -> None:
    """Resize the FIGURE until its panels actually measure PANEL_W_IN x PANEL_H_IN.

    This is the piece that was missing. `figsize()` can only ever PREDICT the panel size from
    a model of what the decorations cost; the layout engine then allocates whatever it likes,
    and every figure carrying something the model did not know about -- a colourbar, a shared
    legend, a two-line label -- came out at a different panel size and a different aspect. The
    prediction was right only for the grids the constants were fitted on.

    Measuring instead makes the model self-correcting: read back the panel boxes the engine
    produced, treat `fig_size - grid_of_panels` as the true decoration cost, and solve for the
    figure size that puts the panels on target. Two or three passes converge, because the
    decorations have a fixed physical size and do not scale with the canvas.
    """
    import statistics as _st
    # 3-D axes are not data panels: mplot3d reports the bounding square of the projection, not
    # the plotted region, so solving for "panel = 2.55x2.04" against that measurement drove the
    # ir3d and phase-space 3-D figures down to a 1.2in-tall canvas with their labels sliced off.
    # A figure holding any 3-D axes keeps the size its producer chose.
    if any(_is_3d(ax) for ax in fig.axes):
        return
    # A panel with a pinned data aspect (imshow, set_aspect("equal")) cannot be solved for both
    # width and height: the two solves fight the constraint and the iteration walks the figure
    # DOWN instead of converging -- measured, a 10x8 imshow figure ended at 6.50x1.91in with a
    # 1.10in panel, just clear of MIN_PANEL_IN and so unreported. Leave those to their producer.
    for ax in fig.axes:
        try:
            if ax.get_aspect() != "auto" and not _is_3d(ax):
                return
        except Exception:
            pass
    for _ in range(max_iter):
        fig.canvas.draw()
        inv = fig.dpi_scale_trans.inverted()
        axes = _data_axes(fig)
        if not axes:
            return
        boxes = [ax.get_window_extent().transformed(inv) for ax in axes]
        # Grid shape by POSITION, not by gridspec: figures built with a raw GridSpec, nested
        # subfigures or hand-placed axes have no single gridspec to ask, and those are exactly
        # the composite figures that drifted worst.
        ncols = len({round(b.x0, 1) for b in boxes}) or 1
        nrows = len({round(b.y0, 1) for b in boxes}) or 1
        w_r, h_r = _st.median([b.width for b in boxes]), _st.median([b.height for b in boxes])
        fw, fh = fig.get_size_inches()
        deco_w, deco_h = fw - ncols * w_r, fh - nrows * h_r
        panel_w = min(PANEL_W_IN, (TEXTWIDTH_IN - deco_w) / ncols)
        panel_h = panel_w / PANEL_ASPECT
        if abs(w_r - panel_w) < tol and abs(h_r - panel_h) < tol:
            return
        new_w = min(deco_w + ncols * panel_w, TEXTWIDTH_IN)
        new_h = deco_h + nrows * panel_h
        if not (0.5 < new_w < 20 and 0.5 < new_h < 30):
            return                                # degenerate measurement; leave it alone
        fig.set_size_inches(new_w, new_h, forward=True)
        _relayout(fig)


def _expand_to_content(fig, max_iter: int = 3) -> None:
    """Grow the canvas until nothing hangs off it. Safety net for `savefig.bbox=None`.

    Sizing the figure from the panels leaves the margins to the layout engine, and it can miss
    by a few hundredths of an inch -- compute_scan's y-label sat at x0=-0.05in on a 6.40in
    canvas, i.e. shipped shaved. Since bbox="tight" is deliberately off (it silently rescales
    the figure on the page), an overhang is a real loss of ink and has to be paid for in canvas.
    Width stays capped at \\textwidth: a wider figure would overrun the text block, so if the
    overhang cannot be covered within that, the panels give up the difference instead.
    """
    for _ in range(max_iter):
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        fw, fh = fig.get_size_inches()
        need_w = fw + max(0.0, -tb.x0) + max(0.0, tb.x1 - fw)
        need_h = fh + max(0.0, -tb.y0) + max(0.0, tb.y1 - fh)
        if need_w <= fw + 0.01 and need_h <= fh + 0.01:
            return
        # never SHRINK: this function only ever adds margin. Clamping to \textwidth
        # unconditionally would narrow a producer-chosen wider canvas whose content needs the
        # room, causing exactly the clipping this exists to prevent.
        fig.set_size_inches(max(fw, min(need_w, TEXTWIDTH_IN)), max(fh, need_h), forward=True)
        _relayout(fig)


def _measure_panels(grids=((1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2))) -> None:
    """Print the achieved panel size per grid. Run after touching the constants above: the
    whole point is that the last two columns come out the same for every grid."""
    import matplotlib.pyplot as _plt
    print(f"target {PANEL_W_IN:.2f}x{PANEL_H_IN:.2f}in, aspect {PANEL_ASPECT}")
    for nc, nr in grids:
        fig, _ = figure(ncols=nc, nrows=nr)
        _relayout(fig)
        enforce_panels(fig)
        fig.canvas.draw()
        bb = fig.axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fw, fh = fig.get_size_inches()
        print(f"  {nc}x{nr}: fig={fw:.2f}x{fh:.2f} panel={bb.width:.2f}x{bb.height:.2f}"
              f" aspect={bb.width / bb.height:.2f}")
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
    return plt.subplots(nrows, ncols, **kwargs)


def process_label(ax, text: str, loc: str = "upper left", **kwargs):
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
    _relayout(fig)
    fig._ps_layout_done = True
    return leg


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
    # Skip tight_layout when the figure already has a layout engine (constrained, used for
    # colourbar figures) or when shared_legend already reserved its strip — running it anyway
    # discards that work and drops the colourbar/legend back onto the panels.
    engine = fig.get_layout_engine()
    managed = engine is not None and engine.__class__.__name__ != "PlaceHolderLayoutEngine"
    if not managed and not getattr(fig, "_ps_layout_done", False):
        try:
            fig.tight_layout()
        except Exception:
            pass
    # Nothing may sit on top of the data: grow each panel's y-range until its legend and
    # process label are clear. Done here so every script gets it without asking.
    for _ax in fig.axes:
        try:
            make_room(_ax)
        except Exception:
            pass
    if not managed and not getattr(fig, "_ps_layout_done", False):
        try:
            fig.tight_layout()
        except Exception:
            pass
    # Grow the canvas if an axis label would be clipped. Runs for every figure, after layout
    # and make_room, so a long y-label on a narrow multi-column panel widens the figure a
    # little rather than being silently chopped.
    try:
        fit_labels(fig)
    except Exception:
        pass
    # LAST, after every other step that can change the canvas (shared_legend's strip,
    # make_room, fit_labels' growth). Each of those adds decoration height, and the panels are
    # what has to stay fixed, so the correction has to see the final decoration cost.
    try:
        enforce_panels(fig)
        _expand_to_content(fig)
    except Exception as exc:
        print(f"  !! {os.path.basename(base)}: enforce_panels failed: "
              f"{type(exc).__name__}: {exc}")
    _warn_if_squeezed(fig, base)
    fig.savefig(base + ".png")
    fig.savefig(base + ".pdf")
    print(f"saved {base}.png / .pdf")
    return base


#: A data panel narrower than this (inches) is not a figure, it is a sliver.
MIN_PANEL_IN = 1.05


def fit_labels(fig, max_iter: int = 2, grow: float = 1.16) -> bool:
    """Grow the figure until no axis label is cut off by the canvas edge. Returns True if it grew.

    A rotated 11pt y-label is ~1.8in of text. Once the derived `figsize` makes a multi-column
    panel shorter than that, the label overflows the figure and `savefig(bbox="tight")` does
    NOT rescue it -- it ships truncated ("MSE(dlog|M|^2) in dec"). No fixed aspect can prevent
    this in general, because label length is a property of the data, not the grid: at 3 columns
    a panel is 1.47in wide and holding the aspect would need a taller-than-wide panel to fit an
    unabbreviated label. So the aspect is the target, and this is the escape hatch when a
    particular label will not fit -- it costs a small aspect deviation on those figures only,
    which is much cheaper than an unreadable axis.
    """
    # An axis label taller than its own axes is clipped to the AXES box, not the figure, so it
    # ships with its ends sliced off ("...in decad") even though it sits well inside the canvas
    # and `bbox_inches="tight"` would have room for it. Turning clipping off on the labels is
    # what actually fixes the truncation; growing the figure below only handles the rarer case
    # where the label really does run past the canvas edge.
    for ax in fig.axes:
        for lbl in (ax.yaxis.label, ax.xaxis.label):
            lbl.set_clip_on(False)

    grew = False
    for _ in range(max_iter):
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        h_px = fig.get_size_inches()[1] * fig.dpi
        w_px = fig.get_size_inches()[0] * fig.dpi
        over = wide = False
        for ax in fig.axes:
            if ax.get_label() == "<colorbar>":
                continue
            axbb = ax.get_window_extent()
            for lbl in (ax.yaxis.label, ax.xaxis.label):
                if not lbl.get_text():
                    continue
                try:
                    bb = lbl.get_window_extent(rend)
                except Exception:
                    continue
                # Grow when the label runs off the canvas...
                if bb.y0 < -1.0 or bb.y1 > h_px + 1.0 or bb.x0 < -1.0 or bb.x1 > w_px + 1.0:
                    over = True
                    if bb.x0 < -1.0 or bb.x1 > w_px + 1.0:
                        wide = True
                # ...or when it collides with a figure-level legend. With clipping off, a
                # label taller than its panel does not get truncated any more -- it runs on
                # THROUGH the shared legend instead ("[%]" printed over the word "uniform").
                # Test the collision itself rather than a "label > x% of panel" proxy: the
                # proxy grew figures that were perfectly fine, and this catches the real case.
                else:
                    for _leg in fig.legends:
                        try:
                            lb = _leg.get_window_extent(rend)
                        except Exception:
                            continue
                        if (min(bb.x1, lb.x1) - max(bb.x0, lb.x0) > 0 and
                                min(bb.y1, lb.y1) - max(bb.y0, lb.y0) > 0):
                            over = True
        if not over:
            return grew
        w, h = fig.get_size_inches()
        # HEIGHT ONLY. Width is the panel grid's to set (see PANEL_W_IN) and is bounded by
        # TEXTWIDTH_IN, so widening here to chase a horizontal overhang would both break the
        # panel geometry and risk a canvas too wide for the text block. A horizontal overhang
        # is reported by _warn_if_squeezed, covered up to \textwidth by _expand_to_content, and
        # otherwise fixed in the producer (shorter label, fewer legend columns, more pad).
        if wide:
            return grew
        fig.set_size_inches(w, h * grow, forward=True)
        grew = True
        # Re-lay out preserving whatever strip shared_legend reserved. A bare tight_layout()
        # here drops that reservation and redraws the legend on top of the panels -- which it
        # did, on the flagship levers A/B among others.
        _relayout(fig)
    return grew


def check_panels(fig, name: str) -> None:
    """Public squeeze check, for figures that do NOT exit through `save()`.

    `save()` calls this for you. Multi-page producers write via `PdfPages.savefig` and so
    never touch `save()` -- which is exactly how `phase1_scaling.pdf` shipped as five 0.6in
    slivers while `rebuild_figures.sh` reported `ok` and `squeezed 0`. Call this immediately
    before every `pdf.savefig(fig)`.
    """
    try:
        enforce_panels(fig)
        _expand_to_content(fig)
    except Exception as exc:
        print(f"  !! {name}: enforce_panels failed: {type(exc).__name__}: {exc}")
    _warn_if_squeezed(fig, name)


def _warn_if_squeezed(fig, base: str) -> None:
    """Shout if the layout has squeezed any data panel down to nothing.

    Multi-column figures die silently: a colourbar plus a y-label per column, or one wide
    in-axes legend (tight_layout and constrained_layout both count legends), can drive the
    panels to a fraction of an inch. The figure still "builds" and still gets included in the
    document -- four figures shipped in results.tex with 0.2-0.3in panels, unreadable, and no
    script reported a problem. Cheap width check so that never passes unnoticed again.

    Fixes, in order of effectiveness: share the y-axis across a row (`sharey`, then
    `tick_params(labelleft=False)`); make colourbars horizontal (`orientation="horizontal",
    location="bottom"`) so they cost height, not width; move a wide legend out of the axes
    (`fig.legend(..., loc="outside lower center")`).
    """
    try:
        fig.canvas.draw()
        inv = fig.dpi_scale_trans.inverted()
        bad, short = [], []
        for ax in fig.axes:
            if ax.get_label() == "<colorbar>" or not ax.get_visible():
                continue
            if not (ax.lines or ax.collections or ax.images or ax.patches):
                continue                                  # legend-only / spacer axes
            bb_ax = ax.get_window_extent().transformed(inv)
            if bb_ax.width < MIN_PANEL_IN:
                bad.append(bb_ax.width)
            # Height too: a panel can be full-width and still be a 0.49in letterbox strip,
            # which passed as "squeezed 0" while rendering with two y-ticks and no room for
            # the data. Checking width alone is what let that ship.
            if bb_ax.height < MIN_PANEL_IN:
                short.append(bb_ax.height)
        if bad:
            print(f"  !! {os.path.basename(base)}: {len(bad)} panel(s) squeezed to "
                  f"{min(bad):.2f}in wide (want >= {MIN_PANEL_IN}in) -- see "
                  f"plot_style._warn_if_squeezed for the fixes")
        if short:
            print(f"  !! {os.path.basename(base)}: {len(short)} panel(s) only "
                  f"{min(short):.2f}in tall (want >= {MIN_PANEL_IN}in) -- a legend or "
                  f"colourbar is being paid for out of the panels")
        # OFF-TARGET PANELS. The check the old guard never made: it asked only whether a panel
        # had collapsed below a floor, so a figure whose panels were a perfectly healthy but
        # WRONG 5.4x4.6in passed silently, sitting opposite a figure with 1.7in panels. Report
        # the realised geometry against the standard panel, and the spread WITHIN the figure,
        # since panels of one figure disagreeing with each other is the more visible fault.
        # 3-D axes excluded for the same reason enforce_panels skips them: their extent is the
        # projection's bounding square, so they report aspect 1.00 by construction and would
        # warn on every 3-D figure forever.
        pan = [ax.get_window_extent().transformed(inv)
               for ax in _data_axes(fig) if not _is_3d(ax)]
        if pan:
            ws = [b.width for b in pan]
            hs2 = [b.height for b in pan]
            asp = [w / h for w, h in zip(ws, hs2) if h > 0]
            if max(ws) - min(ws) > 0.05 or max(hs2) - min(hs2) > 0.05:
                print(f"  !! {os.path.basename(base)}: panels disagree within the figure "
                      f"(width {min(ws):.2f}-{max(ws):.2f}in, height "
                      f"{min(hs2):.2f}-{max(hs2):.2f}in) -- unequal grid cells")
            if asp and (max(asp) > PANEL_ASPECT * 1.15 or min(asp) < PANEL_ASPECT / 1.15):
                print(f"  !! {os.path.basename(base)}: panel aspect {min(asp):.2f}-"
                      f"{max(asp):.2f} off the {PANEL_ASPECT} standard")
            # OFF THE STANDARD SIZE. The check that was missing: everything above compares a
            # figure against ITSELF (spread, aspect) or against a bare legibility floor, so a
            # figure whose panels are uniformly 1.36in -- half the standard, and the actual
            # complaint -- passed silently, clearing MIN_PANEL_IN by 0.04in. Report the gap to
            # PANEL_W_IN so the rebuild names the real fault instead of counting "squeezed".
            # Threshold at 2.0in, not "PANEL_W_IN minus a hair". A 2-column figure whose
            # y-labels are wide is already at \textwidth and lands at 2.26-2.39in: capped by
            # its own decorations, nothing to reflow, and warning on it buried the real cases
            # in 20 lines of noise. Below 2.0in means a third column is being carried.
            if max(ws) < 2.0:
                print(f"  !! {os.path.basename(base)}: panels {max(ws):.2f}x{max(hs2):.2f}in "
                      f"vs the {PANEL_W_IN:.2f}x{PANEL_H_IN:.2f}in standard -- too many "
                      f"columns to hold it at \\textwidth; reflow to <= 2 columns")
        rend = fig.canvas.get_renderer()
        fw, fh = fig.get_size_inches()
        # CANVAS OVERHANG. savefig writes the full canvas now, so anything outside it is CUT.
        # Compare POSITIONS, not sizes: l2_bbb_sweep's tight bbox ran x0=0.165 .. x1=6.653 on a
        # 6.500in canvas, so 0.153in was being destroyed on the right while its total WIDTH
        # (6.488) still measured under the canvas -- a size comparison saw nothing wrong and
        # the figure shipped with the gamma colourbar label sliced off.
        tb = fig.get_tightbbox(rend)
        eps = 0.02
        if tb.x0 < -eps or tb.y0 < -eps or tb.x1 > fw + eps or tb.y1 > fh + eps:
            print(f"  !! {os.path.basename(base)}: content overhangs the canvas "
                  f"(tight bbox {tb.x0:.2f}..{tb.x1:.2f} x {tb.y0:.2f}..{tb.y1:.2f}in vs "
                  f"canvas {fw:.2f}x{fh:.2f}in) -- it WILL BE CLIPPED on save; shrink the "
                  f"legend (fewer ncol) or grow the figure")
        # Width was not enough: a panel can be wide and still ship a y-label truncated off the
        # top of the canvas, which passed as "ok" on four figures. Check the labels too.
        h_px = fh * fig.dpi
        w_px = fw * fig.dpi
        clipped = []
        for ax in fig.axes:
            # Colourbars ARE checked. The old exemption was justified by a false positive under
            # bbox="tight", where nothing could be clipped; with bbox=None a colourbar label is
            # exactly what gets cut, and l2_bbb_sweep's gamma was the one true positive.
            for lbl in (ax.yaxis.label, ax.xaxis.label):
                if not lbl.get_text():
                    continue
                try:
                    bb = lbl.get_window_extent(rend)
                except Exception:
                    continue
                if bb.y0 < -1.0 or bb.y1 > h_px + 1.0 or bb.x0 < -1.0 or bb.x1 > w_px + 1.0:
                    clipped.append(lbl.get_text()[:40])
        if clipped:
            print(f"  !! {os.path.basename(base)}: axis label(s) cut off by the canvas: "
                  f"{clipped[:3]} -- plot_style.fit_labels should have grown the figure")
        # NOTE: there is deliberately no "label taller than its panel" warning here. Once
        # fit_labels turns off label clipping, such a label renders in full ('median rel.
        # error [%]' sits at 84% of its panel and is complete), so the check only produced
        # false alarms -- and acting on it grew figures and pushed shared legends onto data.
    except Exception as exc:
        # NEVER silent. A bare `pass` here hid a NameError (rend used before assignment) that
        # made this entire function dead code -- both the overhang and the label checks were
        # skipped on every figure while the rebuild still reported success.
        print(f"  !! {os.path.basename(base)}: layout check failed to run: "
              f"{type(exc).__name__}: {exc}")
