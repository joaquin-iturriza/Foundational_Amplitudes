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

Size. `docs/results.tex` is `article` at 11pt with 1in margins, so \\textwidth is exactly
6.5in. Figures are saved 6.5in wide and included at `width=\\textwidth`, which makes the
scale factor exactly 1.0: 11pt in the figure renders as 11pt on the page, matching body
text. Including a 6.5in figure at `width=0.62\\textwidth` instead shrinks its text to 6.8pt,
which is why every figure should be included at full \\textwidth and given the aspect ratio
it actually needs rather than being scaled down in LaTeX.

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

#: Default width:height of a single axes panel. Fixed so figures look like a set.
PANEL_ASPECT = 1.75

# Hand-tuned figure heights for the common grids, so that panels keep a consistent
# shape once axis labels and legends have taken their share of the canvas.
_SIZES = {
    (1, 1): (TEXTWIDTH_IN, 3.9),
    (2, 1): (TEXTWIDTH_IN, 2.9),
    (3, 1): (TEXTWIDTH_IN, 2.9),
    (4, 1): (TEXTWIDTH_IN, 2.0),
    (1, 2): (TEXTWIDTH_IN, 6.2),
    (2, 2): (TEXTWIDTH_IN, 5.2),
    (3, 2): (TEXTWIDTH_IN, 4.4),
    (2, 3): (TEXTWIDTH_IN, 8.8),
    (3, 3): (TEXTWIDTH_IN, 6.4),
}


def figsize(ncols: int = 1, nrows: int = 1, width: float | str = "full") -> tuple[float, float]:
    """Figure size in inches for an `ncols` x `nrows` panel grid.

    `width` is "full" (\\textwidth, the default) or "half", for the case where TWO figures
    sit side by side in one LaTeX figure environment at `0.49\\textwidth` each. A half-width
    figure must be SAVED at 3.25in, otherwise LaTeX scales it down and its 11pt text lands
    on the page at 5.5pt. A float is taken as a fraction of \\textwidth.
    """
    frac = {"full": 1.0, "half": 0.49}.get(width, width)
    w = TEXTWIDTH_IN * float(frac)
    if frac == 1.0 and (ncols, nrows) in _SIZES:
        return _SIZES[(ncols, nrows)]
    panel_h = (w / ncols) / PANEL_ASPECT
    return (w, round(panel_h * nrows + 0.9, 2))


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
        "figure.figsize": _SIZES[(1, 1)],
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
        # Full box frame, as in the reference paper: all four spines drawn.
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.top": True,
        "ytick.right": True,
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
        "savefig.bbox": "tight",
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
    kwargs.setdefault("figsize", figsize(ncols, nrows, width))
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
    kwargs.setdefault("loc", "upper center")
    kwargs.setdefault("bbox_to_anchor", (0.5, 1.0))
    kwargs.setdefault("frameon", False)
    leg = fig.legend(handles, labels, ncol=ncol, **kwargs)
    # Reserve the strip for it, and tell save() not to re-run a plain tight_layout(),
    # which would drop the rect and put the legend back on top of the panels.
    fig.tight_layout(rect=[0, 0, 1, 0.90])
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
    _warn_if_squeezed(fig, base)
    fig.savefig(base + ".png")
    fig.savefig(base + ".pdf")
    print(f"saved {base}.png / .pdf")
    return base


#: A data panel narrower than this (inches) is not a figure, it is a sliver.
MIN_PANEL_IN = 1.05


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
        bad = []
        for ax in fig.axes:
            if ax.get_label() == "<colorbar>" or not ax.get_visible():
                continue
            if not (ax.lines or ax.collections or ax.images or ax.patches):
                continue                                  # legend-only / spacer axes
            w = ax.get_window_extent().transformed(inv).width
            if w < MIN_PANEL_IN:
                bad.append(w)
        if bad:
            print(f"  !! {os.path.basename(base)}: {len(bad)} panel(s) squeezed to "
                  f"{min(bad):.2f}in wide (want >= {MIN_PANEL_IN}in) -- see "
                  f"plot_style._warn_if_squeezed for the fixes")
    except Exception:
        pass
