---
name: figure-style
description: >-
  How to make any figure for Foundational_Amplitudes. Use whenever writing or editing a
  plotting script, producing a figure for docs/results.tex, or reviewing an existing figure.
  Covers the plot-box invariant, the shared plot_style module, the minimal-figure rules (no
  titles, no in-plot explanation, everything in the legend), and the png+pdf convention.
---

# Figure style

Every figure goes into `docs/results.tex` and must look like it belongs to the same document
as the prose and as every other figure.

## THE INVARIANT — read this before anything else

**The plot box is a constant: `PLOT_W_IN` x `PLOT_H_IN` (2.40 x 1.92 in). The canvas is
computed from it. Nothing may ever be paid for out of the plot box.**

A colourbar, a long y-label, rotated tick labels, an outside legend, scientific-notation tick
labels — every one of these grows the *canvas*. None of them changes the size of a plot.
`ps.layout()`, called by `ps.save()`, measures the decorations and positions every axes
explicitly in absolute inches.

This is the one rule the figures kept failing, in three different ways, so it is worth knowing
why. In the first two designs the plot box was a **residual**:

- **v1** pinned the figure at `\textwidth` and gave the panel whatever was left over. A lone
  panel got 5.4 in; a cell of a 3-wide grid got 1.7 in.
- **v2** measured the panel and solved for the figure size — but capped the solve at
  `\textwidth`. The moment decorations did not fit, the cap bound and the plot box absorbed the
  difference again. Fig 2 (no colourbars) rendered at 2.35 in and Fig 3 (a colourbar per panel)
  at 1.95 in, on the same page.

Both failed identically: anything competing for width won, and the plot paid.

**The corollary that matters when you are fixing a cramped figure.** The instinct is to move a
decoration to free up space — legend outside, colourbar to the bottom, colourbar to an inset.
Do not. That is how the document ended up with legends outside on some figures and inside on
others, and colourbars on the right, the bottom and the top. It trades a size inconsistency for
a placement inconsistency and makes both visible at once. If a figure does not fit, the answer
is **fewer columns**, never a smaller plot and never a relocated decoration.

### The width budget: the margins are fair game, the paper is not

**A figure may be wider than `\textwidth`.** The text block is 6.5 in but the paper is 8.5 in,
so there is 1 in of blank margin on each side; a row is centred on the text block and overhangs
both equally, via the `\widerow` macro in `results.tex`. `MAX_FIG_IN` is **8.2 in** — past that
the content runs off the *sheet* and is genuinely lost, which is the only hard limit.

This is the deliberate trade. Two panels side by side, each at the standard plot box, come to
~7.9 in once both carry a colourbar. Holding that to `\textwidth` would mean either shrinking
the plots (the invariant forbids it) or stacking them one per line, which turns a set of maps
into three pages. **Blank margin is cheaper than either.** So: never split a figure or stack a
row just because a colourbar pushed it past 6.5 in.

`ps.save()` prints `!! <name>: canvas 8.6in exceeds the printable width`. *That* means the grid
cannot hold standard plot boxes at that column count. Fix it structurally:

| symptom | fix |
|---|---|
| 3 panels, or any count that does not fill a rectangle | `ps.panels(n)` + `ps.save_panels`, two per row |
| a row between 6.5 and 8.2 in | nothing — wrap it in `\widerow` and let it overhang the margins |
| a row over 8.2 in | shorten the longest tick or legend label; only then fewer columns |
| 3 or more columns | 2 columns and more rows (never 3 across) |
| legend outside the axes | put it inside; use `ps.make_room` to open space |
| a panel spanning two cells | split the figure; every panel is one cell |

## Grids: one canvas holds 1, 2 or 4 panels. Anything else is separate files.

**`ps.figure` is only for 1x1, 1x2 and 2x2.** Never 3 across, never 4 across, never a panel
spanning two cells, and never a grid with an empty cell.

**For any other panel count — in practice any 3 — use `ps.panels(n)` and `ps.save_panels`.**
Each panel is written as its own file (`<base>_a.pdf`, `_b`, `_c`) and `results.tex` includes
them two per line, so LaTeX puts two on the first line and the third centred underneath:

```python
figs = ps.panels(3)
for (fig, ax), d in zip(figs, datasets):
    ax.plot(...)
ps.save_panels(figs, "analysis/divergences/figs/my_set")   # prints the LaTeX to paste
```

```latex
\includegraphics{my_set_a.pdf}\hfill\includegraphics{my_set_b.pdf} \\[1ex]
\includegraphics{my_set_c.pdf}
```

Three panels forced into a 2x2 leave a hole where the fourth would go, and that hole is the
first thing anyone notices about the figure. `ps.save()` warns when a grid has an empty cell.

Two panels share a line whenever their canvases add to under **8.2 in** — the text block
plus both margins. A panel carrying a colourbar is ~3.9 in, so a pair of maps is ~7.9 in and
still goes on one row, overhanging the margin. `ps.save` prints each canvas width and
`scripts/check_tex_figure_rows.py` checks every row's sum against 8.2 in.

Group panels into one figure only when they are the same quantity over one swept parameter.
Unrelated plots that happen to be discussed together go in separate figures.

**Do not share anything to buy width.** No `sharey` so one column can drop its tick labels, no
"axis labels on the outside edges only", no one legend or colourbar serving some panels and not
others. Sharing is how panels stop looking like each other: with `sharey` in a 2x2 only
column 0 carries tick labels, so panels (a) and (c) have numbers and (b) does not. Every panel
labels its own axes. `sharey` is fine in a 1x2 when both panels really are the same quantity —
there it is symmetric — but never as a width workaround.

## Always use the shared module

`plot_style.py` at the repo root. Never hand-set rcParams, figure sizes, colours, fonts, line
widths or marker sizes in a plotting script.

```python
import sys
REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps

fig, (axL, axR) = ps.figure(ncols=2)
axL.plot(x, y, color=ps.C.blue, label="uniform")
axL.set_xlabel(r"$\log_{10} y_{\min}$")
axL.set_ylabel(r"MSE$(\Delta\log|\mathcal{M}|^2)$")
axL.legend()
ps.process_label(axL, r"$e^+e^-\to u\bar u gg$")
ps.save(fig, "analysis/divergences/figs/my_figure")   # writes .png AND .pdf
```

- `ps.figure(ncols, nrows)` — do not pass your own `figsize`.
- **Never call `tight_layout()` or use `layout="constrained"`.** They distribute a *fixed*
  canvas among the axes, which is exactly what makes the plot box a residual. `ps.save` turns
  the layout engine off; a producer that re-enables it silently reintroduces the bug.
- `ps.C` — the palette. `ps.C.blue` = baseline, `ps.C.vermillion` = the thing being tested.
- **Never pass `lw=`, `ms=`, `linewidth=`, `markersize=` or `fontsize=`.** The shared style sets
  them. Two figures once shipped with `lw=2.8, ms=9` and visibly heavier lines than the rest of
  the document. The only legitimate exception is when line weight *encodes* something (a thick
  translucent "truth" band under a thin dashed "model"), and then it needs a comment saying so.
- `ps.save(fig, base)` — lays out, checks the invariant, writes **both** `.png` and `.pdf`. A
  `Stop` hook blocks the turn if one is missing.

## Standard placements — do not vary these per figure

These are fixed so that no figure has to negotiate for space, and so that nothing looks
different from its neighbour. Each was, at some point, decided ad hoc per figure, and that is
precisely what the reader notices.

| element | rule |
|---|---|
| legend | **`ps.legend(ax, "upper left")` — inside the axes, in a CORNER.** `ps.legend` rejects anything else. `ps.make_room` then grows the y-range until it is clear of the data. |
| legend that will not fit | let `ps.legend` drop the font (it goes to `LEGEND_MIN_PT`, 8.5pt). Do **not** rewrite the labels to save width if that costs the reader a distinction. |
| legend, which panel | **one legend per figure**, in whichever panel has room — often not the one the series were drawn on. Pass `handles=other_ax.get_legend_handles_labels()[0]`. |
| process label | `ps.process_label(ax, ...)`, default **upper right**. Do not pick a corner per figure. |
| colourbar | **`ps.colorbar(ax, mappable, label)` — vertical, immediately right of its own panel.** Always, and one per panel that needs a scale. Not below, not on top, not in an inset. |
| headroom | `ps.headroom` gives every plot the same clear band above and below the data. Do not hand-tune `ylim` for appearance. |
| tick labels | horizontal. **Never rotate them** — rotated labels are tall, and tall labels used to shrink the plot. If they collide, use fewer ticks or shorter text ($\log_{10}$ exponents, not `1e-6`). |

**Why colourbars have exactly one placement.** Every branch this rule ever had turned into
something a reader spotted: bars on the right of some panels and under others, one bar serving
two panels while a third had its own, a bar *above* a panel because that was the only place it
fitted. So: one rule, no exceptions, one bar per panel. Sharing a bar across panels is what
looked arbitrary, not the bar.

A vertical bar takes ~0.8 in of *column*, so a panel carrying one is ~3.9 in and a pair is
~7.9 in. That is fine: the row overhangs the margins (see the width budget above). Do **not**
turn a bar sideways, shrink a plot, or split a figure to bring a row back under `\textwidth`.

`ps.save` also thins colourbar ticks to four and switches them to mathtext scientific notation;
six labels at full decimal precision ran into each other under a 2.40 in panel.

**`center left` / `center right` / `center` are not legend positions.** A legend floating in the
middle of a plot with data on both sides of it reads as a mistake even when it happens not to
overlap anything; two were spotted on sight in the compiled document. A corner is always either
clear or made clear by growing the axis. The middle cannot be cleared at all.

**A legend wider than its plot box hangs over the neighbouring panel.** `ps.legend` fixes
that by dropping the legend font in half-point steps, keeping the column count you asked for,
and only reaches for fewer columns once it hits `LEGEND_MIN_PT`. Font first is deliberate:
dropping a column looks free but trades width for **height**, and six entries at `ncol=1` is
six rows tall, which lands straight on the curves in a 1.92 in box — the column drop passes
every width test while making the overlap worse.

Factorising a cross-product legend (colour carries the size, line style carries the arm) is
available but is **not** the default fix. It saved almost no width on the six-curve saturation
figure and made the reader assemble each curve's identity themselves. Prefer the smaller font;
reach for factorising only when the two factors are genuinely independent and the reader
already thinks in them.

An outside legend is a last resort for when the entries fit in no panel at all — eight process
labels, say. Then it is **one strip above the plot** via `ps.shared_legend`, never below, never
to the side, and it grows the canvas rather than the panel.

## Include at natural size in LaTeX

```latex
\includegraphics{my_figure.pdf}
```

**No `width=`.** Saved widths vary now (a 1-column figure is ~2.9 in), so `width=\textwidth`
would scale it up and print its 11 pt labels at 21 pt. Natural size keeps the scale factor
exactly 1.000, so figure text matches body text. `ps.save` guarantees nothing exceeds
`\textwidth`.

## The minimalism rules

The figures in the reference paper (arXiv:2601.13308) are very plain. Match them.

**No titles.** No `set_title`, no `suptitle`, no panel titles. Explanation lives in the LaTeX
caption and body text.

**No reading instructions.** Delete `(higher = better)`, `(lower = better)`, `(IR <-> bulk)`,
`(<1 = sigma wins)` from labels, ticks, legends and annotations.

**If a metric needs explaining, show its formula.** `MSE$(\Delta\log|\mathcal{M}|^2)$` beats a
word plus a hint in brackets.

**Everything drawn is identified by the legend.** No unlabeled guide lines, shaded bands, or
grey background scatter.

**No pointing at things.** No arrows, circles, callouts, or free text explaining the plot.

**The one allowed in-axes text is the process label.** A short label naming a line that cannot
go in the legend is a rare second exception. A third piece of in-plot text means you are
breaking the rule.

**Per-point value labels are clutter.** The axis carries the number; exact values go in a table.

**One font size everywhere.** 11 pt. Fix collisions with fewer ticks, never a smaller font.

## cmr10 glyph traps

- **En/em dashes** (`–`, `—`) render as a hollow box. Use `r"$3\!-\!15$"` or an ASCII hyphen.
- **Unicode minus** is handled (`axes.unicode_minus = False`); do not re-enable it.
- **`%` inside mathtext is a comment character.** `rf"$={x:.0%}$"` raises a parse error. Write
  `rf"$={100*x:.0f}\%$"`. In a plain (non-math) label you want a bare `%`.

A missing glyph is silent: no warning, just a box. Look at the PNG after any label change.

## Before you call it done

1. **Run the script and read its output.** `!!` lines are the invariant being violated. A
   `canvas exceeds \textwidth` line means restructure, not shrink; an `empty cell(s)` line means
   `ps.panels`; a `legend is wider than its plot box` line means factorise the labels.
2. **Look at the PNG.** Colliding tick labels, a legend on the data, a missing glyph and a
   colourbar in the wrong place are only visible in the render.
3. **Run `python scripts/check_tex_figure_rows.py`.** Panels included two per line have
   to *add up* to under `\textwidth`; if they do not, LaTeX silently drops the last one
   onto its own line and the caption's "(a) and (b)" stops describing the page. Nothing
   else warns about this: not `ps.save`, not the LaTeX log. `rebuild_figures.sh` runs it
   at the end.
4. **Look at the figure next to its neighbours in the compiled PDF.** Every one of the size
   complaints in this document's history was invisible in the single figure and obvious on the
   page. Rasterise a couple of pages (`gs -sDEVICE=png16m -r100 -dFirstPage=N -dLastPage=N`)
   and compare against the figure before and after it.
5. Ask: with the title gone, can a reader with the caption identify every line, band and marker
   from the axes and legend alone? If not, add a legend entry or a better axis label — not a
   note inside the plot.

## When editing an existing script

Convert it fully rather than patching around it: swap `plt.subplots` for `ps.figure` (or
`ps.panels` for a 3-panel set), delete local rcParams, colour constants, `figsize`, `lw`/`ms`
overrides and any `tight_layout` call, replace both `savefig` calls with `ps.save`, route
legends through `ps.legend` and colourbars through `ps.colorbar`, drop any `sharey` or
outer-edge-labels-only trick that exists to save width, and strip titles, annotations and
reading hints. Then regenerate and look at the PNG.

**A hand-rolled `GridSpec` is a sign the figure is the wrong shape.** Every one in this repo
existed to fit something that did not fit: a dedicated colourbar row, a panel spanning two
cells, a spacer column. Split the figure instead.

Moving explanation out of a figure usually means the caption in `docs/results.tex` must absorb
it, and a caption that says "left" and "right" must still be true after a reflow.
