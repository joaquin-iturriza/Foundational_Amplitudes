#!/usr/bin/env python
"""Check that every row of figures in docs/results.tex actually fits on one line.

Figures are included at their NATURAL size (no `width=`), and a set of panels is included two
per line. If a row's canvases add up to more than \\textwidth, LaTeX silently drops the last one
onto its own line: the page still looks tidy, so nothing warns, but the arrangement is no longer
the two-per-line one the caption describes ("(a) and (b) on the first line...").

That failure is invisible in any single figure and invisible in the LaTeX log, which is exactly
the class of thing that kept reaching the compiled document. Run it after rebuild_figures.sh.

    python scripts/check_tex_figure_rows.py            # exit 1 if any row overflows

Widths come from the PNGs, which are written at savefig.dpi and are the same canvas as the PDF.
"""
import glob
import os
import re
import sys

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_ROW_IN = 8.2      # \textwidth 6.5 + the two 1in margins, less a little paper
SAVEFIG_DPI = 200

# Multi-page PDFs (included with [page=N]) have a .png sibling that is a DIFFERENT
# artifact -- a wide contact sheet -- so measuring it says nothing about the page as
# included, and it reported 18.75in for a figure that fits fine. Skip them.
MULTIPAGE = {"phase1_scaling.pdf"}

# Where \graphicspath points. Cheaper and more predictable than a recursive glob.
ROOTS = ["analysis/divergences/figs", "analysis/hpo_optima", "analysis/scaling_compute",
         "compare_models/_levers_ab", "sweep/plots", "docs/figs"]

_cache = {}


def find_png(stem):
    png = stem.replace(".pdf", ".png")
    if png in _cache:
        return _cache[png]
    for r in ROOTS:
        p = os.path.join(REPO, r, png)
        if os.path.exists(p):
            _cache[png] = p
            return p
    hits = glob.glob(os.path.join(REPO, "**", png), recursive=True)
    _cache[png] = hits[0] if hits else None
    return _cache[png]


def main():
    tex = open(os.path.join(REPO, "docs/results.tex")).read()
    rows = over = missing = 0
    for line in tex.splitlines():
        # Both spellings: a bare \includegraphics and the \pnl{} top-align wrapper that
        # rows use. Matching only the former made this silently report "0 rows checked"
        # the moment the rows were wrapped -- a guard that passes by seeing nothing.
        gs = re.findall(r"\\(?:includegraphics(?:\[[^\]]*\])?|pnl)\{([^}]+)\}", line)
        if len(gs) < 2:
            continue                       # a lone figure per line always fits
        if any(g in MULTIPAGE for g in gs):
            continue
        widths = []
        for g in gs:
            p = find_png(g)
            widths.append(Image.open(p).width / SAVEFIG_DPI if p else None)
        if any(w is None for w in widths):
            print(f"  ?? no PNG for {[g for g, w in zip(gs, widths) if w is None]}")
            missing += 1
            continue
        rows += 1
        total = sum(widths)
        if total > MAX_ROW_IN:
            over += 1
            print(f"OVER {total:5.2f}in  {[round(w, 2) for w in widths]}  {gs[0]}\n"
                  f"     -> wider than the paper ({MAX_ROW_IN}in incl. margins). Shorten the "
                  f"longest tick or legend label in the wider panel.")
    print(f"{rows} multi-panel rows checked, {over} overflowing, {missing} with no PNG")
    return 1 if (over or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
