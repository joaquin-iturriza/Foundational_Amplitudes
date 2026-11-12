#!/usr/bin/env python3
"""How far towards the t t~ threshold the physical-mass top classes of a recipe reach: per pool with an external t and
t~, the top velocity in the pair's rest frame beta = sqrt(1 - 4 m_t^2 / m_tt^2) of every val event (every role samples
the same distribution), its counts in bins of log10 beta, and the top mass read from the momenta.

    python tools/kit_beta_coverage.py collect [--recipe recipes/catalog_v4_datasaving_h3_train.yaml] --out <json>
    python tools/kit_beta_coverage.py plot <json> --out <stem>

KIT's ee_tt for comparison: bulk_train beta >= 0.2 (tools/kit_input_range.py: log beta^2 / 5 >= -0.64), the
threshold strata beta 1e-4..0.2.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
EDGES = np.arange(-5.0, 0.01, 0.25)        # log10 beta bins

SITE = r'''
import json, sys, numpy as np, yaml
sys.path.insert(0, ".")
from analysis.common.readers import find_pool, split_pool
EDGES = np.arange(-5.0, 0.01, 0.25)
out = {}
for p in yaml.safe_load(open(sys.argv[1]))["processes"]:
    f = find_pool(p["name"], p.get("sqrts"), role="val", any_window=True)
    if f is None:
        out[p["name"]] = None
        continue
    r = np.load(f, mmap_mode="r")
    n = (r.shape[1] - 1) // 5
    mom, pdg, _ = split_pool(np.asarray(r), n)
    it, itx = np.argmax(pdg == 6, 1), np.argmax(pdg == -6, 1)
    ok = (pdg == 6).any(1) & (pdg == -6).any(1)
    if not ok.any():
        out[p["name"]] = {"n": int(len(r)), "pair": 0}
        continue
    a = np.arange(len(r))
    pt, ptx = mom[a, it], mom[a, itx]
    P = pt + ptx
    mtt2 = P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(1)
    m2 = pt[:, 0] ** 2 - (pt[:, 1:] ** 2).sum(1)
    b = np.sqrt(np.clip(1 - 4 * m2 / mtt2, 0, None))[ok]
    lb = np.log10(np.clip(b, 1e-12, None))
    out[p["name"]] = {"n": int(len(r)), "pair": int(ok.sum()), "mt": float(np.median(np.sqrt(m2[ok]))),
                      "q": [float(x) for x in np.quantile(b, [0.0, 0.01, 0.5])],
                      "hist": np.histogram(lb, EDGES)[0].tolist(), "below": int((lb < EDGES[0]).sum())}
print("KITBETA " + json.dumps(out))
'''


def plot(path, stem):
    import plot_style as ps
    d = json.load(open(path))
    rows = [v for v in d.values() if v and v.get("pair")]
    h = np.sum([v["hist"] for v in rows], 0)
    x = 10 ** (0.5 * (EDGES[1:] + EDGES[:-1]))
    fig, ax = ps.figure()
    ax.step(x, np.maximum(h, 0.5), where="mid", color=ps.C.blue, label="top-pair events")
    ax.axvspan(0.2, 1.0, color=ps.C.grey, alpha=0.2, label="KIT bulk training")
    ax.axvspan(1e-4, 0.2, color=ps.C.vermillion, alpha=0.15, label="KIT threshold strata")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"top velocity $\beta=\sqrt{1-4m_t^2/m_{t\bar t}^2}$")
    ax.set_ylabel("events per bin")
    ps.legend(ax, "upper left")
    ps.save(fig, stem)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--recipe", default="recipes/catalog_v4_datasaving_h3_train.yaml")
    c.add_argument("--out", required=True)
    p = sub.add_parser("plot"); p.add_argument("data"); p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "collect":
        from analysis.common.readers import site_python
        d = site_python(SITE, [a.recipe], "KITBETA")
        json.dump(d, open(a.out, "w"))
        rows = {k: v for k, v in d.items() if v and v.get("pair")}
        tot = np.sum([v["hist"] for v in rows.values()], 0)
        lo = EDGES[:-1]
        print(f"{len(d)} pools, {sum(v is None for v in d.values())} without a val pool, {len(rows)} with a t t~ pair, "
              f"{sum(v['pair'] for v in rows.values())} pair events; top mass median range "
              f"{min(v['mt'] for v in rows.values()):.1f}-{max(v['mt'] for v in rows.values()):.1f}")
        for cut in (-1, -2, -3):
            print(f"  events with beta < 1e{cut}: {int(tot[lo < cut].sum())}; pools reaching it (1% quantile): "
                  f"{sum(v['q'][1] < 10 ** cut for v in rows.values())}; min beta over pools "
                  f"{min(v['q'][0] for v in rows.values()):.2e}")
    else:
        plot(a.data, a.out)


if __name__ == "__main__":
    main()
