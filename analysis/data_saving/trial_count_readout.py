#!/usr/bin/env python3
"""How a DyHPO search's best improves with its trial count, from the long searches on record (Joaquin 2026-11-12:
"we have results where we searched for ~50 trials and results kept improving"): every search on the site with at least
MIN_N observed trials at its longest horizon, trials in the order they were observed (the timestamp in the result
file's name), the running best val_loss after n trials over the best after all N. The running best of one search
against its own end is not a convergence claim: it shows how much the later trials still found.
Figure figs/trial_count_{png,pdf}: one thin line per search, the median over searches thick.

    python analysis/data_saving/trial_count_readout.py [--cached] [--site ccin2p3]
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from analysis.common.readers import site_python  # noqa: E402

MIN_N = 24
REMOTE = r'''
import glob, json, math, os, re, sys
sys.path.insert(0, os.getcwd())  # the checkout: site run's working directory
import siteconf
out, seen = [], set()
for root in {siteconf.SWEEP_DIR, siteconf.RESULTS_DIR}:
    for rd in glob.glob(os.path.join(root, "**", "results"), recursive=True):
        rd = os.path.realpath(rd)
        if rd in seen:
            continue
        seen.add(rd)
        rows = []
        for f in os.listdir(rd):
            m = re.match(r"hp(\d+)_t(\d+)_(\d+)\.json$", f)
            if not m:
                continue
            try:
                v = json.load(open(os.path.join(rd, f))).get("val_loss")
            except Exception:
                continue
            rows.append((int(m.group(2)), int(m.group(3)), int(m.group(1)), v))
        if not rows:
            continue
        T = max(r[0] for r in rows)
        top = sorted((ts, hp, v) for t, ts, hp, v in rows if t == T)
        if len(top) >= int(sys.argv[1]):
            out.append({"sweep": os.path.relpath(os.path.dirname(rd), root), "T": T, "n_fidelities": len({r[0] for r in rows}),
                        "trials": [[hp, v if isinstance(v, (int, float)) and math.isfinite(v) else None] for ts, hp, v in top]})
print("TCNT " + json.dumps(out))
'''


def running(trials):
    """Running best over the observed trials (diverged ones, None, never lower it), over the best of all."""
    best, out = math.inf, []
    for _, v in trials:
        if v is not None and v > 0:
            best = min(best, v)
        out.append(best)
    out = np.array(out)
    return out / out[-1] if np.isfinite(out[-1]) else None


def main():
    site = sys.argv[sys.argv.index("--site") + 1] if "--site" in sys.argv else "ccin2p3"
    cache = os.path.join(HERE, "data", f"trial_count_{site}.json")
    rows = json.load(open(cache)) if "--cached" in sys.argv else site_python(REMOTE, [str(MIN_N)], "TCNT", site=site,
                                                                              timeout=1800)
    json.dump(rows, open(cache, "w"))
    curves = {r["sweep"]: c for r in rows for c in [running(r["trials"])] if c is not None}
    N = sorted(len(c) for c in curves.values())
    print(f"{len(curves)} searches with >= {MIN_N} observed trials at their longest horizon; N from {N[0]} to {N[-1]}, "
          f"median {N[len(N) // 2]}")
    res = {"site": site, "min_n": MIN_N, "n_searches": len(curves), "N": N, "at": {}}
    for n in (8, 12, 16, 20, 24, 32, 40, 48):
        r = [c[n - 1] for c in curves.values() if len(c) >= n + 8]  # 8 later trials at least
        if not r:
            continue
        q = np.quantile(r, [0.25, 0.5, 0.75])
        res["at"][n] = {"n": len(r), "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2]), "at_final": int(sum(x <= 1.0001 for x in r))}
        print(f"best after {n:2d} / best after all: median {q[1]:.3g} (25-75% {q[0]:.3g}-{q[2]:.3g}) over {len(r)} searches; "
              f"already final in {res['at'][n]['at_final']}")
    # where the final best was found, as a fraction of N
    pos = [int(np.argmax(c <= 1.0001)) + 1 for c in curves.values()]
    res["final_at_frac"] = [float(p / len(c)) for p, c in zip(pos, curves.values())]
    print("final best found at trial / N: median %.2f, share found in the last third %.2f"
          % (np.median(res["final_at_frac"]), np.mean([f > 2 / 3 for f in res["final_at_frac"]])))
    json.dump(res, open(os.path.join(HERE, "data", f"trial_count_readout_{site}.json"), "w"), indent=1)

    import plot_style as ps  # noqa: E402
    ps.use()
    fig, ax = ps.figure()
    L = max(len(c) for c in curves.values())
    for c in curves.values():
        ax.plot(np.arange(1, len(c) + 1), c, color="0.75", lw=0.6)
    med = [np.median([c[n] for c in curves.values() if len(c) > n]) for n in range(L)]
    cnt = [sum(len(c) > n for c in curves.values()) for n in range(L)]
    k = max(i for i in range(L) if cnt[i] >= 5) + 1
    ax.plot(np.arange(1, k + 1), med[:k], color="black", lw=2, label="median")
    ax.axvline(8, color=ps.C.vermillion, ls="--", label="8 trials")
    ax.set_yscale("log")
    ax.set_xlabel("observed trials $n$")
    ax.set_ylabel(r"best val loss after $n$ / after all $N$")
    ps.legend(ax, loc="upper right")
    ps.save(fig, os.path.join(HERE, "figs", "trial_count"))


if __name__ == "__main__":
    main()
