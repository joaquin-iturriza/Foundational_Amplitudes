#!/usr/bin/env python3
"""Does the power law of a KIT singular limit already show in the bulk training data? On the truth only (no network).

    python tools/kit_bulk_slope.py <file> <target> [--n-bulk 200000]      (on a site: the KIT h5 files)

For the first n_bulk events of bulk_train: the singular variables of the hard families, as the hard bins use them,
  coll_y   min over final quark-gluon pairs of s_qg/s           (u u~ g, u u~ g g)
  soft_x   min over final gluons of 2 p_g.Q/s, Q = p_0 + p_1    (u u~ g, u u~ g g)
  beta     sqrt(1 - 4 m_t^2/s) of the t t~ pair                 (t t~)
then the median log10|M|^2 in quarter-decade bins of each (bins of at least 200 events) and the least-squares slope
of those medians against log10 of the variable over the bins of the lowest decade the bulk reaches (the bulk's edge
nearest the limit), printed beside the bulk's 0.5% quantile of the variable. Rows are tools/kit_hardps.py's (momenta,
PDG ids, |M|^2 last). Compare with the hard bins' truth slope (tools/kit_vs_mlp.py plot, SLOPE lines).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import kit_hardps as kh  # noqa: E402

ETA = np.array([1.0, -1.0, -1.0, -1.0])


def variables(rows, n):
    p = rows[:, :4 * n].reshape(-1, n, 4)
    pdg = rows[:, 4 * n:5 * n].round().astype(int)
    dot = lambda a, b: (a * b * ETA).sum(-1)
    Q = p[:, 0] + p[:, 1]
    s = dot(Q, Q)
    out = {}
    glu = [i for i in range(2, n) if pdg[0, i] == 21]
    qrk = [i for i in range(2, n) if 0 < abs(pdg[0, i]) < 6]
    if glu:
        out["soft_x"] = np.min([2 * dot(p[:, g], Q) / s for g in glu], 0)
        out["coll_y"] = np.min([2 * dot(p[:, q], p[:, g]) / s for q in qrk for g in glu], 0)
    top = [i for i in range(2, n) if abs(pdg[0, i]) == 6]
    if len(top) == 2:
        m2 = dot(p[:, top[0]], p[:, top[0]])
        out["beta"] = np.sqrt(np.clip(1 - 4 * m2 / s, 0, None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file")
    ap.add_argument("target")
    ap.add_argument("--n-bulk", type=int, default=200000)
    a = ap.parse_args()
    import h5py
    meta = kh.metadata(a.file)
    n = len(meta["particles"])
    with h5py.File(kh.h5_path(a.file), "r") as f:
        g = f["bulk_train/bulk"]
        sl = slice(0, min(a.n_bulk, g["momenta"].shape[0]))
        rows = kh.group_rows({k: g[k][sl] for k in ("momenta", a.target)}, a.target, meta)
    lm = np.log10(np.abs(rows[:, -1]))   # the one-loop finite part can be negative: KIT scores log10 abs
    for name, v in variables(rows, n).items():
        lv = np.log10(np.clip(v, 1e-300, None))
        e = np.arange(np.floor(lv.min() * 4) / 4, lv.max() + 0.25, 0.25)
        i = np.digitize(lv, e) - 1
        bins = [(e[b] + 0.125, float(np.median(lm[i == b]))) for b in range(len(e) - 1) if (i == b).sum() >= 200]
        low = [q for q in bins if q[0] <= bins[0][0] + 1.0]
        slope = np.polyfit([q[0] for q in low], [q[1] for q in low], 1)[0] if len(low) >= 2 else float("nan")
        print(f"BULKSLOPE {a.file} {name}: n={len(v)} 0.5% quantile {np.quantile(v, 0.005):.3g}, lowest decade "
              f"10^{low[0][0] - 0.125:.2f}..10^{low[-1][0] + 0.125:.2f} ({len(low)} bins) slope {slope:+.3f}; "
              "medians " + " ".join(f"{x:.2f}:{y:.2f}" for x, y in bins))


if __name__ == "__main__":
    main()
