#!/usr/bin/env python3
"""The test MSE of the standardised log amplitude per network and Z + n g multiplicity, read from the vector data of
arXiv:2505.20280's Fig. (figs/amplitudes/scaled_multiplicity_scan_prep_big.pdf in its arXiv source; the paper's own
test set, 500k events, not the UQ repo's Zenodo split). Each marker's centre is mapped through the log y-axis ticks.

    python analysis/zng_uq/paper_mse.py <pdf> [--out analysis/zng_uq/data/paper_mse.json]   (needs pymupdf)
"""
import argparse
import json

import numpy as np

# legend order of the figure, top to bottom; matched to the series by the legend markers' y order
LEGEND = ["MLP-I", "L-GATr", "GNN", "LLoCa-GNN", "Transformer", "LLoCa-Transformer"]
XLAB = ["Z+1g", "Z+2g", "Z+3g", "Z+4g"]


def read(pdf):
    import pymupdf
    page = pymupdf.open(pdf)[0]
    spans = [s for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]
    # y ticks: '10' '−' '<k>' spans; the exponent span's centre row maps to 10^-k
    ticks = [(-int(s["text"]), 0.5 * (s["bbox"][1] + s["bbox"][3])) for s in spans if s["text"].strip().isdigit()
             and len(s["text"].strip()) == 1 and s["bbox"][0] < 60]
    lines = [d for d in page.get_drawings() if d.get("width") == 1.5 and d.get("fill") is None]
    series = {}
    for d in lines:
        pts = [(p.x, p.y) for it in d["items"] for p in it[1:] if hasattr(p, "x")]
        series.setdefault(tuple(d["color"]), []).append({round(x, 2): y for x, y in pts})
    # tick rows: the major tick marks (width 0.8, 4 pt long, left of the axis)
    major = sorted(d["items"][0][1].y for d in page.get_drawings()
                   if abs((d.get("width") or 0) - 0.8) < 1e-3 and len(d["items"]) == 1
                   and 3.5 < abs(d["items"][0][1].x - d["items"][0][2].x) < 5 and d["items"][0][1].x < 60)
    k = sorted(t for t, _ in ticks)[::-1]          # -3 (top) ... -9 (bottom)
    a, b = np.polyfit(np.array(major), np.array(k), 1)
    out = {}
    for color, segs in series.items():
        data = max(segs, key=len)                  # the data line (4 points), not the legend's 2-point stub
        legend = min(segs, key=len)
        out[color] = (list(legend.values())[0], [10 ** (a * data[x] + b) for x in sorted(data)])
    names = {}
    for name, (_, vals) in zip(LEGEND, sorted(out.values(), key=lambda v: v[0])):
        names[name] = dict(zip(XLAB, vals))
    return names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", default="analysis/zng_uq/data/paper_mse.json")
    a = ap.parse_args()
    res = read(a.pdf)
    for n, v in res.items():
        print(f"{n:18s} " + " ".join(f"{k} {x:.3g}" for k, x in v.items()))
    json.dump({"source": "arXiv:2505.20280 source, figs/amplitudes/scaled_multiplicity_scan_prep_big.pdf",
               "quantity": "test MSE of standardised log|M|^2 (the paper's test set, 500k events)", "mse": res},
              open(a.out, "w"), indent=1)
