#!/usr/bin/env python3
"""Readout of the D403 MLP ladder (tools/kit_mlp_ladder.py, scripts/job_kit_mlp_ladder.sh) under its passed design:
per process and hard family, the deepest-bin log-MSE of log10|M|^2 (KIT evaluate.py) of every arm and seed, against
the control arm mlp (his setup in the port), his own numpy src/baseline.py run (the port gate) and our transformer
reference (softpair inputs).

    python tools/kit_mlp_readout.py collect --out analysis/kit_benchmark/data/kit_mlp_ladder.json   (reads CC)
    python tools/kit_mlp_readout.py read analysis/kit_benchmark/data/kit_mlp_ladder.json [--figs <dir>]

Rules (the design): an arm differs from mlp in a family when its 3-seed range is disjoint from mlp's and the seed
medians differ by more than 0.3 decades; the port passes when mlp's median is within 0.3 decades of the numpy run in
every family's deepest bin; ours reproduces our failure when its median is within 0.5 decades of the transformer
reference in every family where his MLP wins (mlp's median below the reference by more than 0.3 decades).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.kit_figures import family, edges, XLAB, PROC  # noqa: E402

ARMS = ["mlp", "target", "mom", "raw", "gelu", "het", "cur", "recipe", "ours"]
REFS = {"ee_uug_me2": "kit_A_ee_uug_me2_n100000_t8000_softpair",          # FA-ccin2p3-c15b3337
        "ee_uugg_me2": "kit_A_ee_uugg_me2_n100000_t8000_softpair",        # FA-ccin2p3-c364a7ae
        "ee_tt_virt_finite": "kit_A_ee_tt_virt_finite_n100000_t8000_softpair"}   # FA-ccin2p3-f21fa163
NAME = re.compile(r"kit_A_(?P<proc>.+)_n100000_t\d+_mlp(?P<arm>[a-z]+)_s(?P<seed>\d)$")
UP = ("sudakov",)

SITE = r'''
import glob, json, os, sys
out = {}
def strata(p):
    r = json.load(open(p)); r = r.get("strata", r)
    return {k: v["logmse"] for k, v in r.items() if isinstance(v, dict) and "logmse" in v}
for d in sorted(glob.glob("runs/kit_mlp/kit_A_*_mlp*_s*")) + sorted(glob.glob("runs/kit_mlp/baseline_*")):
    p = os.path.join(d, "kit_benchmark", "kit_result.json")
    p = p if os.path.exists(p) else os.path.join(d, "kit_result.json")
    if os.path.exists(p):
        out[os.path.basename(d)] = strata(p)
for r in sys.argv[1:]:
    out[r] = strata(os.path.join("runs/kit", r, "kit_benchmark", "kit_result.json"))
print("KITMLP " + json.dumps(out))
'''


def deepest(st):
    """{family: logmse} at the deepest hard_test stratum of each family, and 'bulk'"""
    best = {}
    for k, v in st.items():
        sp, s = k.split("/")[:2]
        if sp == "bulk_test":
            best["bulk"] = (0, v)
            continue
        if sp != "hard_test":
            continue
        fam = family(s)
        x = np.sqrt(np.prod(edges(s)))
        key = -x if any(u in fam for u in UP) else x
        if fam not in best or key < best[fam][0]:
            best[fam] = (key, v)
    return {f: v for f, (_, v) in best.items()}


def table(d):
    """{proc: {"arms": {arm: {fam: [per seed]}}, "ref": {fam: v}, "numpy": {fam: v} or None}}"""
    out = {}
    for name, st in d.items():
        m = NAME.match(name)
        if m:
            p = out.setdefault(m["proc"], {"arms": {}, "ref": None, "numpy": None})
            for f, v in deepest(st).items():
                p["arms"].setdefault(m["arm"], {}).setdefault(f, []).append(v)
    for proc, p in out.items():
        p["ref"] = deepest(d[REFS[proc]]) if REFS.get(proc) in d else None
        # his numpy runs: the seeded ones (baseline_<proc>_s<seed>, kit_baseline_seeded.py) when present, else the
        # single seed-0 run of the ladder job; per family the seed median and the list
        seeded = [deepest(v) for k, v in sorted(d.items()) if re.fullmatch(f"baseline_{proc}_s\\d", k)]
        runs = seeded or ([deepest(d["baseline_" + proc])] if "baseline_" + proc in d else [])
        p["numpy_all"] = {f: [r[f] for r in runs] for f in runs[0]} if runs else None
        p["numpy"] = {f: float(np.median(v)) for f, v in p["numpy_all"].items()} if runs else None
    return out


def lg(a, b):
    return float(np.log10(a / b))


def read(path, figs):
    t = table(json.load(open(path)))
    for proc, p in sorted(t.items()):
        mlp = p["arms"]["mlp"]
        fams = sorted(mlp, key=lambda f: (f == "bulk", f))
        print(f"\n== {proc}  (deepest-bin log-MSE of log10|M|^2; seed median [min, max]; d = log10(arm / mlp))")
        wins = [f for f in fams if f != "bulk" and p["ref"] and lg(np.median(mlp[f]), p["ref"][f]) < -0.3]
        if p["numpy"]:
            port = {f: lg(np.median(mlp[f]), p["numpy"][f]) for f in fams}
            print("his numpy runs per family:", {f: [float(f"{x:.3g}") for x in v] for f, v in p["numpy_all"].items()})
            print("port gate (mlp median vs his numpy median, |d| <= 0.3):",
                  {f: round(v, 3) for f, v in port.items()}, "PASS" if all(abs(v) <= 0.3 for v in port.values()) else "FAIL")
        print("families where his MLP wins (mlp < our transformer by > 0.3 decades):", wins)
        for f in fams:
            ref = p["ref"][f] if p["ref"] else None
            print(f"  {f}: transformer {ref:.3g}" if ref is not None else f"  {f}:",
                  f" numpy {p['numpy'][f]:.3g}" if p["numpy"] else "")
            lo0, hi0 = min(mlp[f]), max(mlp[f])
            for a in ARMS:
                v = p["arms"].get(a, {}).get(f)
                if not v:
                    continue
                d = lg(np.median(v), np.median(mlp[f]))
                disj = max(v) < lo0 or min(v) > hi0
                flag = ("WORSE" if d > 0 else "BETTER") if (a != "mlp" and disj and abs(d) > 0.3) else ""
                extra = ""
                if a == "ours" and ref is not None:
                    extra = f"  vs transformer {lg(np.median(v), ref):+.2f}"
                print(f"    {a:7s} {np.median(v):.3g} [{min(v):.3g}, {max(v):.3g}]  d {d:+.2f} {flag}{extra}")
        if figs:
            plot(proc, p, fams, figs)


def plot(proc, p, fams, outdir):
    import plot_style as ps
    figs = ps.panels(len(fams))
    y = np.arange(len(ARMS))
    for (fig, ax), f in zip(figs, fams):
        for i, a in enumerate(ARMS):
            v = p["arms"].get(a, {}).get(f, [])
            c = ps.C.blue if a == "mlp" else ps.C.vermillion
            ax.plot(v, [i] * len(v), "o", color=c, mfc="none", label="seed" if i == 1 else None)
            if v:
                ax.plot([np.median(v)], [i], "|", color=c, markersize=14, label="seed median" if i == 1 else None)  # the median tick encodes the statistic
        if p["ref"]:
            ax.axvline(p["ref"][f], ls="--", color=ps.C.grey, label="transformer, softpair")
        if p["numpy"]:
            ax.axvline(p["numpy"][f], ls=":", color=ps.C.green, label="his numpy MLP")
        ax.set_xscale("log")
        ax.set_yticks(y, ARMS)
        ax.set_ylim(len(ARMS) - 0.5, -0.5)
        ax.set_xlabel("deepest-bin log-MSE" if f != "bulk" else "bulk log-MSE")
        ax.set_ylabel(XLAB.get(f, f.replace("_", " ")) if f != "bulk" else "bulk")
        e, f2, tg = proc.split("_", 2)
        ps.process_label(ax, PROC.get((f"{e}_{f2}", tg), proc))
        ps.legend(ax, "lower right")
        ps.make_room(ax)
    ps.save_panels(figs, os.path.join(outdir, f"kit_mlp_ladder_{proc}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--out", required=True)
    r = sub.add_parser("read"); r.add_argument("data"); r.add_argument("--figs", default=None)
    a = ap.parse_args()
    if a.cmd == "collect":
        from analysis.common.readers import site_python
        d = site_python(SITE, list(REFS.values()), "KITMLP")
        json.dump(d, open(a.out, "w"))
        print(f"{len(d)} runs -> {a.out}")
    else:
        read(a.data, a.figs)


if __name__ == "__main__":
    main()
