#!/usr/bin/env python3
"""Figures of the KIT hard phase-space benchmark: per process, each arm's KIT log10-MSE per hard_test stratum and its
median predicted log10|M|^2 against the truth's, along the stratum family's singular variable.

    python tools/kit_figures.py collect <run dir> ... > out.txt     (on the site: reads the runs and the KIT files)
    python tools/kit_figures.py plot <collected.json> [--out analysis/kit_benchmark/figs]

collect, per run dir (runs/kit/kit_<mode>_<file>_<target>_n<N>_t<T>[_tag]): its kit_benchmark/kit_result.json
(KIT's src/evaluate.py numbers, through tools/kit_predict.py) and predictions.h5, and the KIT file's truth and
singular_var per hard_test event. Per stratum: log-MSE, bias, the median of singular_var (the x of the MSE figure).
Per family (stratum name without its two bin edges), events of all its strata in bins of 4 per decade of
singular_var: the median truth and median prediction of log10|M|^2. Per process: the 'learned nothing' floor, the
log10-MSE of the truth clipped to its bulk_train [0.5%, 99.5%] range (tools/kit_target_slope.py's plain_edge), and
the same binned truth median. Prints one line 'KITFIG <json>'.

plot: one panel set per process and figure kind (ps.panels, one panel per family):
  kit_mse_<file>_<target>_{a,b,...}     log10-MSE per stratum against the median singular_var, one curve per arm,
                                        Bakar's reference MLP (slide 9) at its deepest bin, the floor
  kit_median_<file>_<target>_{a,b,...}  median log10|M|^2 per bin, truth and each arm
"""
import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RUN_RE = re.compile(r"kit_(?P<mode>A_ft|A|B_bulk|B)_(?P<file>ee_[a-z]+)_(?P<target>[a-z_]+?)_n(?P<n>\d+)_t(?P<t>\d+)"
                    r"(?:_(?P<tag>.+))?$")

# Bakar's reference MLP (3 x 128, 1e5 bulk events), slide 9 of the KIT benchmark slides: log-MSE of log10|M|^2 in the
# deepest bin of a family, log-invariant and raw-invariant inputs; {(file, target, family): (bin lo, bin hi, log, raw)}
MLP = {("ee_uug", "me2", "soft_x"): (1e-7, 1e-6, 0.32, 89.0),
       ("ee_uug", "me2", "coll_y"): (1e-7, 1e-6, 0.11, 18.0),
       ("ee_uugg", "me2", "soft_x"): (1e-7, 1e-6, 0.40, 97.0),
       ("ee_uugg", "me2", "coll_qg_y"): (1e-7, 1e-6, 0.08, 17.0),
       ("ee_uugg", "me2", "coll_gg_y"): (1e-7, 1e-6, 0.05, 18.0),
       ("ee_tt", "virt_finite", "threshold_beta"): (1e-4, 1e-3, 7.8, 8.0),
       ("ee_tt", "virt_finite", "sudakov_sqrts"): (15e3, 50e3, 0.05, 4.2e4)}


def family(stratum):
    return stratum.rsplit("_", 2)[0]


def edges(stratum):
    lo, hi = stratum.rsplit("_", 2)[1:]
    return float(lo), float(hi)


def truth_key(g, target):
    return f"log10_{target}" if f"log10_{target}" in g else target


def collect(run_dirs):
    import h5py
    from tools import kit_hardps as kh
    out = {"runs": {}, "truth": {}}
    for rd in run_dirs:
        m = RUN_RE.match(os.path.basename(rd.rstrip("/")))
        if not m:
            raise SystemExit(f"not a KIT run dir: {rd}")
        stem, target = m["file"], m["target"]
        res = json.load(open(os.path.join(rd, "kit_benchmark", "kit_result.json")))
        h5 = kh.h5_path(stem)
        key = f"{stem}/{target}"
        fams = {}
        strata = {}
        with h5py.File(h5, "r") as f, h5py.File(os.path.join(rd, "kit_benchmark", "predictions.h5"), "r") as p:
            for name, r in res["strata"].items():
                split, st, ds = name.split("/")
                if split != "hard_test":
                    strata[name] = dict(r)
                    continue
                g = f[f"hard_test/{st}"]
                tk = truth_key(g, target)
                t = np.asarray(g[tk][:], np.float64)
                if tk == target:
                    t = np.log10(t)
                y = np.asarray(p[f"hard_test/{st}/{ds}"][:], np.float64)
                sv = (np.asarray(g["singular_var"][:], np.float64) if "singular_var" in g
                      else np.full(len(t), np.sqrt(np.prod(edges(st)))))
                strata[name] = dict(r, x=float(np.median(sv)), lo=edges(st)[0], hi=edges(st)[1])
                fams.setdefault(family(st), []).append((sv, t, y, st))
            if key not in out["truth"]:
                b = f["bulk_train/bulk"]
                tk = truth_key(b, target)
                tb = np.asarray(b[tk][:200000], np.float64)
                if tk == target:
                    tb = np.log10(tb)
                lo, hi = np.quantile(tb, [0.005, 0.995])
                floor, tmed = {}, {}
                for fam, parts in fams.items():
                    for sv, t, _, st in parts:
                        floor[st] = {"x": float(np.median(sv)), "mse": float(np.mean((np.clip(t, lo, hi) - t) ** 2))}
                out["truth"][key] = {"bulk_range": [float(lo), float(hi)], "floor": floor, "median": tmed}
        meds = {}
        for fam, parts in fams.items():
            sv = np.concatenate([q[0] for q in parts])
            t = np.concatenate([q[1] for q in parts])
            y = np.concatenate([q[2] for q in parts])
            lx = np.log10(sv)
            bins = np.arange(np.floor(lx.min() * 4) / 4, lx.max() + 0.25, 0.25)
            idx = np.digitize(lx, bins)
            rows = []
            for i in np.unique(idx):
                s = idx == i
                if s.sum() >= 50:
                    rows.append([float(np.median(sv[s])), float(np.median(t[s])), float(np.median(y[s])),
                                 int(s.sum())])
            meds[fam] = rows
            out["truth"][key]["median"].setdefault(fam, [[r[0], r[1]] for r in rows])
        out["runs"][os.path.basename(rd.rstrip("/"))] = {
            "mode": m["mode"], "file": stem, "target": target, "n": int(m["n"]), "steps": int(m["t"]),
            "tag": m["tag"] or "", "strata": strata, "slopes": res.get("slopes", {}), "medians": meds}
    print("KITFIG " + json.dumps(out))


PROC = {("ee_tt", "virt_finite"): r"$e^+e^-\to t\bar t$, virtual", ("ee_tt", "born"): r"$e^+e^-\to t\bar t$, Born",
        ("ee_uug", "me2"): r"$e^+e^-\to u\bar u g$", ("ee_uugg", "me2"): r"$e^+e^-\to u\bar u gg$"}
XLAB = {"threshold_beta": r"top velocity $\beta$", "sudakov_sqrts": r"$\sqrt{s}$ [GeV]",
        "soft_x": r"soft gluon $x=2E_g/\sqrt{s}$", "coll_y": r"collinear $y=s_{qg}/s$",
        "coll_qg_y": r"collinear $y=s_{qg}/s$", "coll_gg_y": r"collinear $y=s_{gg}/s$",
        "azimuth_y": r"$g\to gg$ collinear $y=s_{gg}/s$"}


def arm_label(r):
    n = r"10^5" if r["n"] == 100000 else r"9{\times}10^5"
    lab = rf"scratch ${n}$"
    if r["mode"] in ("B", "B_bulk"):
        lab = {"B": "hard added", "B_bulk": "bulk continued"}[r["mode"]] + rf" ${n}$"
    tag = r["tag"]
    if tag in ("offoff", "nooffshell"):
        lab += ", off-shell. off"
    elif tag == "plain":
        lab += ", plain"
    return lab


def plot(path, outdir):
    import plot_style as ps
    d = json.load(open(path))
    os.makedirs(outdir, exist_ok=True)
    colors = [ps.C.blue, ps.C.vermillion, ps.C.green, ps.C.orange, ps.C.purple, ps.C.sky]
    by_proc = {}
    for name, r in sorted(d["runs"].items()):
        by_proc.setdefault((r["file"], r["target"]), []).append((name, r))
    written = []
    for (stem, target), runs in sorted(by_proc.items()):
        tr = d["truth"][f"{stem}/{target}"]
        fams = sorted({family(k.split("/")[1]) for _, r in runs for k in r["strata"] if k.startswith("hard_test/")})
        arms = [(arm_label(r), colors[i % len(colors)], r) for i, (_, r) in enumerate(runs)]
        base = f"{stem}_{target}"
        # (1) log10-MSE per stratum
        figs = ps.panels(len(fams))
        for (fig, ax), fam in zip(figs, fams):
            for lab, c, r in arms:
                pts = sorted((v["x"], v["logmse"]) for k, v in r["strata"].items()
                             if k.startswith("hard_test/") and family(k.split("/")[1]) == fam)
                ax.plot(*zip(*pts), "o-", color=c, label=lab)
            fl = sorted((v["x"], v["mse"]) for st, v in tr["floor"].items() if family(st) == fam)
            ax.plot(*zip(*fl), "--", color=ps.C.grey, label="truth held in bulk range")
            ref = MLP.get((stem, target, fam))
            if ref:
                x = np.sqrt(ref[0] * ref[1])
                ax.plot([x], [ref[2]], "*", color="k", label="MLP, log invariants")
                ax.plot([x], [ref[3]], "x", color="k", label="MLP, raw invariants")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(XLAB.get(fam, fam.replace("_", " ")))
            ax.set_ylabel(r"log-MSE of $\log_{10}|\mathcal{M}|^2$")
            ps.process_label(ax, PROC.get((stem, target), stem))
        ps.legend(figs[0][1], "upper left")
        ps.make_room(figs[0][1])
        written += ps.save_panels(figs, os.path.join(outdir, f"kit_mse_{base}"))
        # (2) median predicted vs truth log10|M|^2
        figs = ps.panels(len(fams))
        for (fig, ax), fam in zip(figs, fams):
            tm = tr["median"].get(fam, [])
            if tm:
                ax.plot(*zip(*tm), "-", color="k", label="truth")
            for lab, c, r in arms:
                rows = r["medians"].get(fam, [])
                if rows:
                    ax.plot([q[0] for q in rows], [q[2] for q in rows], "o-", color=c, label=lab)
            ax.set_xscale("log")
            ax.set_xlabel(XLAB.get(fam, fam.replace("_", " ")))
            ax.set_ylabel(r"median $\log_{10}|\mathcal{M}|^2$")
            ps.process_label(ax, PROC.get((stem, target), stem))
        ps.legend(figs[0][1], "upper left")
        ps.make_room(figs[0][1])
        written += ps.save_panels(figs, os.path.join(outdir, f"kit_median_{base}"))
    for w in written:
        print(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("run_dirs", nargs="+")
    p = sub.add_parser("plot")
    p.add_argument("data")
    p.add_argument("--out", default=os.path.join(ROOT, "analysis", "kit_benchmark", "figs"))
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.run_dirs)
    else:
        plot(a.data, a.out)


if __name__ == "__main__":
    main()
