---
name: figure-style
description: >-
  How to make any figure for Foundational_Amplitudes. Use whenever writing or editing a
  plotting script, producing a figure for docs/results.tex, or reviewing an existing figure.
  Covers the shared plot_style module, the minimal-figure rules (no titles, no in-plot
  explanation, everything in the legend), and the png+pdf convention.
---

# Figure style

Every figure in this repo goes into `docs/results.tex` and must look like it belongs to the
same document as the prose and as every other figure. Two things achieve that: the shared
style module, and a discipline of minimalism.

## Always use the shared module

`plot_style.py` at the repo root. Never hand-set rcParams, figure sizes, colours, or fonts
in a plotting script.

```python
import sys
REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps

fig, (axL, axR) = ps.figure(ncols=2)          # \textwidth wide, standard aspect
axL.plot(x, y, color=ps.C.blue, label="uniform")
axL.set_xlabel(r"$\log_{10} y_{\min}$")
axL.set_ylabel(r"MSE$(\Delta\log|\mathcal{M}|^2)$")
axL.legend()
ps.process_label(axL, r"$e^+e^-\to u\bar u gg$")
ps.save(fig, "analysis/divergences/figs/my_figure")   # writes .png AND .pdf
```

- `ps.figure(ncols, nrows)` — always 6.5in wide (`\textwidth`), with a consistent panel
  aspect. Do not pass your own `figsize`.
- `ps.C` — the palette (`blue`, `vermillion`, `green`, `orange`, `sky`, `purple`, `grey`).
  Use `ps.C.blue` for the baseline and `ps.C.vermillion` for the thing being tested. Let the
  cycler handle it when you have many series.
- `ps.save(fig, base)` — calls `tight_layout()` and writes **both** `.png` and `.pdf`. The
  repo convention is both formats, same basename, same directory, no exceptions. A `Stop`
  hook blocks the turn if one is missing.
- `ps.process_label(ax, text)` — the process label, and the only text you may put inside the
  axes by default.

## Include at full width in LaTeX

```latex
\includegraphics[width=\textwidth]{my_figure.pdf}
```

`\textwidth` is exactly 6.5in and `ps.figure` saves at 6.5in, so the scale factor is 1.0 and
11pt in the figure lands as 11pt on the page, matching body text. **Never** use
`width=0.6\textwidth` to shrink a figure: that silently shrinks its text to ~7pt and breaks
the match. If a figure should be shorter, give it fewer panels or a shorter height, do not
scale it down in LaTeX.

## The minimalism rules

The figures in the reference paper (arXiv:2601.13308) are very plain. Match them. The
default is clean; the user will ask for an exception if they want one.

**No titles.** No `set_title`, no `suptitle`, no panel titles. The explanation of a figure
belongs in its LaTeX caption and in the body text, never in the figure. In particular never
title a plot with the question it answers ("Does sigma-steering beat uniform?").

**No reading instructions anywhere in the figure.** Delete parentheticals like
`(higher = better)`, `(lower = better)`, `(IR <-> bulk)`, `(<1 = sigma wins)` from axis
labels, tick labels, legends and annotations. The reader learns how to read it from the
caption.

**If a metric needs explaining, show its formula instead of prose.** Prefer
`MSE$(\Delta\log|\mathcal{M}|^2)$` or an explicit
`$\langle(\hat y-y)^2\rangle$` on the axis over a word plus a hint in brackets. This is what
the paper does with MSE.

**Everything drawn must be identified by the legend.** If it has a colour or a linestyle, it
has a legend entry. That kills:
- unlabeled dashed/dotted guide lines (a reference line at $y=1$ still gets a label such as
  "parity", or it should not be drawn),
- unlabeled shaded bands and `fill_between` regions,
- grey background scatter that no entry accounts for.

**No pointing at things.** No arrows, no circles or ellipses around features, no
`annotate` with a callout, no free text in the middle of the axes explaining what the reader
is looking at. If a feature matters, say so in the caption.

**The one allowed in-axes text is the process label**, via `ps.process_label`, e.g.
$e^+e^-\to u\bar u gg$. A second exception exists but is genuinely rare: a short label naming
a line that cannot sensibly go in the legend, such as "theoretical lower bound" sitting on
that line. If you are about to add a third piece of in-plot text, you are breaking the rule.

**Per-point value labels are clutter.** Do not annotate each marker with its number; the axis
already carries it. If the exact numbers matter, they belong in a table.

**One font size everywhere.** `plot_style` sets everything to 11pt. Do not pass `fontsize=`
to individual calls. If tick labels collide, use fewer ticks or shorter tick labels (e.g.
$\log_{10}$ exponents instead of `1e-6`), never a smaller font.

## cmr10 glyph traps

The figure font is cmr10, which has a small glyph set. Three things bite:

- **En/em dashes** (`–`, `—`) render as a hollow box. Write a range as `r"$3\!-\!15$"`
  (math minus) or a plain ASCII hyphen.
- **Unicode minus** is already handled (`axes.unicode_minus = False`); do not re-enable it.
- **`%` inside mathtext is a comment character.** `rf"$={x:.0%}$"` raises a parse error.
  Write `rf"$={100*x:.0f}\%$"`, or keep the percent outside math entirely. Note that in a
  plain (non-math) label, `\%` renders literally as `\%` — there you want a bare `%`.

After any label change, look at the PNG. A missing glyph is silent: no warning, just a box.

## Self-containedness check

Before saving, ask: with the title gone, can a reader who has the caption tell what every
line, band and marker is, purely from the axes and legend? If not, the fix is a legend entry
or a better axis label, not a note inside the plot.

## When editing an existing script

Convert it fully rather than patching around it: swap `plt.subplots` for `ps.figure`, drop
its local rcParams and colour constants, replace both `savefig` calls with `ps.save`, and
strip titles, annotations and reading hints. Then regenerate the figure and **look at the
PNG** before declaring it done; several of these problems (colliding tick labels, a legend
covering the data) are only visible in the render.

Moving explanation out of a figure usually means the caption in `docs/results.tex` must
absorb it. Check the caption still says what the deleted title said.
