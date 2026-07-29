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
    (2, 3): (TEXTWIDTH_IN, 7.4),
    (3, 3): (TEXTWIDTH_IN, 6.4),
}


def figsize(ncols: int = 1, nrows: int = 1) -> tuple[float, float]:
    """Figure size in inches for an `ncols` x `nrows` panel grid, always \\textwidth wide."""
    if (ncols, nrows) in _SIZES:
        return _SIZES[(ncols, nrows)]
    panel_h = (TEXTWIDTH_IN / ncols) / PANEL_ASPECT
    return (TEXTWIDTH_IN, round(panel_h * nrows + 0.9, 2))


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
        "axes.spines.top": False,
        "axes.spines.right": False,

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

def figure(ncols: int = 1, nrows: int = 1, **kwargs):
    """`plt.subplots` at the repo's standard width and aspect for this grid.

    Returns whatever `plt.subplots` returns: `(fig, ax)` for a single panel,
    `(fig, axes)` otherwise.
    """
    kwargs.setdefault("figsize", figsize(ncols, nrows))
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
    return ax.text(xy[0], xy[1], text, transform=ax.transAxes,
                   ha=xy[2], va=xy[3], **kwargs)


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
    fig.savefig(base + ".png")
    fig.savefig(base + ".pdf")
    print(f"saved {base}.png / .pdf")
    return base
