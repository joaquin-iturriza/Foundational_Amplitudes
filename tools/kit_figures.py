#!/usr/bin/env python3
"""Figures of the KIT hard phase-space benchmark: per process, each arm's KIT log10-MSE per hard_test stratum and its
median predicted log10|M|^2 against the truth's, along the stratum family's singular variable.

    python tools/kit_figures.py collect <run dir> ... > out.txt     (on the site: reads the runs and the KIT files)
    python tools/kit_figures.py plot <collected.json> [--out analysis/kit_benchmark/figs]

collect, per run dir (runs/kit/kit_<mode>_<file>_<target>_n<N>_t<T>[_tag]): its kit_benchmark/kit_result.json
(KIT's src/evaluate.py numbers, through tools/kit_predict.py) and predictions.h5, and the KIT file's truth and
singular_var per hard_test event. Per stratum: log-MSE, bias, the median of singular_var (the x of the MSE figure).
Per family (stratum name without its two bin edges), events of all its strata in bins of 4 per decade of
singular_var: the median truth and median prediction of log10|M|^2. Per process: the reference line, the log10-MSE of
the truth clipped to its bulk_train [0.5%, 99.5%] range (tools/kit_target_slope.py's plain_edge): a model exact inside
the training range and flat at its edge outside it (zero in a bin whose truth stays inside the range); and the same
binned truth median. Prints one line 'KITFIG <json>'.

plot: one panel set per process and figure kind (ps.panels, one panel per family):
  kit_mse_<file>_<target>_{a,b,...}     log10-MSE per stratum against the median singular_var, one curve per arm,
                                        Bakar's reference MLP (slide 9) at its deepest bin, the reference line
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

RUN_RE = re.compile(r"kit_(?P<mode>A_ft|A|B_bulk|B)_(?P<file>ee_[a-z]+)_(?P<target>[a-z0-9_]+?)_n(?P<n>\d+)_t(?P<t>\d+)"
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


def _split(stratum):
    """(family, [bin edges]) of a stratum name: its trailing numeric tokens are the edges (one for a fixed-y set)"""
    tok = stratum.split("_")
    k = len(tok)
    while k > 0:
        try:
            float(tok[k - 1])
        except ValueError:
            break
        k -= 1
    return "_".join(tok[:k]), [float(t) for t in tok[k:]]


def family(stratum):
    return _split(stratum)[0]


def edges(stratum):
    e = _split(stratum)[1]
    return e[0], e[-1]


def truth(g, ds, target):
    """log10 truth of a KIT group: the dataset the prediction is named after, else log10|raw target|"""
    if ds in g:
        return np.asarray(g[ds][:], np.float64)
    return np.log10(np.abs(np.asarray(g[target][:], np.float64)))


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
                t = truth(g, ds, target)
                dsname = ds
                y = np.asarray(p[f"hard_test/{st}/{ds}"][:], np.float64)
                sv = (np.asarray(g["singular_var"][:], np.float64) if "singular_var" in g
                      else np.full(len(t), np.sqrt(np.prod(edges(st)))))
                strata[name] = dict(r, x=float(np.median(sv)), lo=edges(st)[0], hi=edges(st)[1])
                assert len(t) == len(y), (name, len(t), len(y))
                fams.setdefault(family(st), []).append((sv, t, y, st))
            if key not in out["truth"]:
                b = f["bulk_train/bulk"]
                tb = truth(b, dsname, target)[:200000]
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


PROC = {("ee_tt", "virt_finite"): r"$t\bar t$ virtual", ("ee_tt", "born"): r"$t\bar t$ Born",
        ("ee_uug", "me2"): r"$u\bar u g$", ("ee_uugg", "me2"): r"$u\bar u gg$"}
XLAB = {"threshold_beta": r"top velocity $\beta$", "sudakov_sqrts": r"$\sqrt{s}$ [GeV]",
        "soft_x": r"soft gluon $x=2E_g/\sqrt{s}$", "coll_y": r"collinear $y=s_{qg}/s$",
        "coll_qg_y": r"quark-gluon collinear $y=s_{qg}/s$", "coll_gg_y": r"gluon-gluon collinear $y=s_{gg}/s$",
        "azimuth_gg_y": r"$g\to gg$ collinear $y=s_{gg}/s$"}


# input-representation arms (D374, model.input_*): tag -> legend text
INPUT_ARMS = {"pairlogs": r"+ $\log(s_{ij}/s)$ inputs", "softrescale": "+ soft log rescale",
              "locallogs": "+ local-frame logs",
              "softpair": r"+ soft rescale + $\log(s_{ij}/s)$",
              "softpairthr": r"+ soft rescale + $\log(s_{ij}/s)$" + "\n" + r"+ $\log\beta^2$ of massive pairs",
              "softpairthr_vbeta": r"+ soft rescale + $\log(s_{ij}/s)$" + "\n" + r"+ $\log\beta^2$, target $V\beta$",
              "softpair_rezero": r"+ soft rescale + $\log(s_{ij}/s)$" + "\n" + "ReZero",
              "softpair_scalars": "MLP inputs only",
              "softpairthr_scalars": "MLP inputs only",
              # + the learned log-invariant polynomial term on the output (degree 2, or 1 for _logskiplin)
              "softpair_plain": "plain target",
              "softpair_plain_logskip": r"plain target + $P_2(L)$", "softpair_plain_logskiplin": r"plain target + $P_1(L)$",
              "softpairthr_logskip": r"+ $\log\beta^2$ + $P_2(L)$", "softpairthr_logskiplin": r"+ $\log\beta^2$ + $P_1(L)$",
              # step 18 (D435): log|M|^2 = log A + g, A the asymptotic factors (dipoles, 1/beta, log^2), on the plain target
              "softpair_asym": r"+ soft rescale + $\log(s_{ij}/s)$" + "\n" + "+ asymptotic factor",
              "softpair_plain_asym": r"+ soft rescale + $\log(s_{ij}/s)$" + "\n" + "+ asymptotic factor",
              "softpairthr_asym_plainlog": r"+ $\log(s_{ij}/s)$ + $\log\beta^2$" + "\n" + "+ asymptotic factor",
              "asym_cont": "asymptotic factor,\n" + r"no log inputs, $\gamma\leq4$",
              "asym_cont_plainlog": "asymptotic factor,\n" + r"no log inputs, $\gamma\leq4$",
              # A_ft: the parent's pretraining (catalog_v4 with the KIT families held out, + LO / one-loop top pairs near
              # threshold), then the inputs every such arm has (input_soft_eps, input_pair_logs=6; beta where tagged)
              "ftr5own": "held-out catalog\n" + r"+ soft rescale + $\log(s_{ij}/s)$",
              "ftr5thrown": "held-out catalog + LO tops\n" + r"+ soft rescale + $\log(s_{ij}/s)$",
              "ftr5thrbetaown": "held-out catalog + LO tops\n" + r"+ soft rescale + $\log(s_{ij}/s)$" + "\n" + r"+ $\log\beta^2$",
              "ftr5thrloopown": "held-out catalog\n+ LO and one-loop tops\n" + r"+ soft rescale + $\log(s_{ij}/s)$",
              "ftr5thrloopbetaown": "held-out catalog\n+ LO and one-loop tops\n" + r"+ soft rescale + $\log(s_{ij}/s)$" + "\n" + r"+ $\log\beta^2$"}


def arm_label(r, with_n=True):
    """legend text saying what the curve is (D368, D369): how the network was trained, then the bulk size (left out
    when every curve of the figure has the same one; the caption gives it)"""
    n = r"$10^5$" if r["n"] == 100000 else r"$9\times10^5$"
    how = {"A": "scratch", "A_ft": "pretrained", "B": "scratch + hard events",
           "B_bulk": "scratch + 3000 steps\non bulk only"}[r["mode"]]
    lab = f"{how}, {n} bulk" if with_n else how
    tag = r["tag"]
    if tag.endswith("lr0p1"):   # the A_ft design's lr-axis check (KIT_LR_SCALE=0.1)
        lab += r", lr $\times0.1$"
        tag = tag[:-len("lr0p1")].rstrip("_")
    if tag in ("offoff", "nooffshell"):
        lab += ",\nno off-shell input"
    elif tag == "plain":
        lab += ",\nplain target"
    elif tag in INPUT_ARMS:
        lab += ",\n" + INPUT_ARMS[tag]
    return lab


def plot(path, outdir, prefix="", split_offshell=False):
    import plot_style as ps
    from matplotlib.ticker import NullFormatter
    d = json.load(open(path))
    os.makedirs(outdir, exist_ok=True)
    colors = [ps.C.blue, ps.C.vermillion, ps.C.green, ps.C.orange, ps.C.purple, ps.C.sky]
    by_proc = {}
    for name, r in sorted(d["runs"].items()):
        off = split_offshell and r["tag"] in ("offoff", "nooffshell")
        by_proc.setdefault((r["file"], r["target"], off), []).append((name, r))
    written = []
    for (stem, target, off), runs in sorted(by_proc.items()):
        tr = d["truth"][f"{stem}/{target}"]
        fams = sorted({family(k.split("/")[1]) for _, r in runs for k in r["strata"] if k.startswith("hard_test/")})
        if off:     # every curve of this figure has the off-shellness input off: the caption says so, not the legend
            runs = [(nm, dict(r, tag="")) for nm, r in runs]
        with_n = len({r["n"] for _, r in runs}) > 1
        arms = [(arm_label(r, with_n), colors[i % len(colors)], r) for i, (_, r) in enumerate(runs)]
        ls = {}
        if len(runs) > len(colors):   # colours would repeat: a fine-tune takes its scratch twin's colour, dashed (lr x0.1 dotted)
            twin = {(r["n"], r["tag"]): c for _, c, r in arms if r["mode"] == "A"}
            # yellow only here, as a 7th colour: cycling the six repeated a colour on two arms (11c set, 2026-11-12)
            free = iter([c for c in colors + [ps.C.yellow] if c not in twin.values()] * 2)
            for i, (lab, c, r) in enumerate(arms):
                if r["mode"] == "A_ft":
                    lr = r["tag"].endswith("lr0p1")
                    key = (r["n"], r["tag"][:-len("lr0p1")].rstrip("_") if lr else r["tag"])
                    arms[i] = (lab, twin.get(key) or next(free), r)
                    ls[id(r)] = ":" if lr else "--"
                elif r["mode"] != "A":
                    arms[i] = (lab, next(free), r)
        base = f"{stem}_{target}" + ("_nooffshell" if off else "")
        # (1) log10-MSE per stratum
        figs = ps.panels(len(fams))
        for (fig, ax), fam in zip(figs, fams):
            for lab, c, r in arms:
                pts = sorted((v["x"], v["logmse"]) for k, v in r["strata"].items()
                             if k.startswith("hard_test/") and family(k.split("/")[1]) == fam)
                ax.plot(*zip(*pts), "o", ls=ls.get(id(r), "-"), color=c, label=lab)
            fl = sorted((v["x"], v["mse"]) for st, v in tr["floor"].items() if family(st) == fam)
            if any(m > 0 for _, m in fl):   # zero in every bin (truth inside the training range): not drawn, no entry
                ax.plot(*zip(*fl), "--", color=ps.C.grey, label="truth clipped to\ntraining range")
            ref = MLP.get((stem, target, fam))
            if ref:
                x = np.sqrt(ref[0] * ref[1])
                ax.plot([x], [ref[2]], "*", color="k", label="KIT MLP (log inputs)")
                ax.plot([x], [ref[3]], "x", color="k", label="KIT MLP (raw inputs)")
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_yscale("log")
            ax.set_xlabel(XLAB.get(fam, fam.replace("_", " ")))
            ax.set_ylabel(r"log-MSE of $\log_{10}|\mathcal{M}|^2$")
            ps.process_label(ax, PROC.get((stem, target), stem))
        # each panel file carries its own legend (any file may be shown alone; Joaquin 2026-11-12: 'The figure has no
        # legend'), listing what that panel draws (the reference line is absent where it is zero); inside the axes up
        # to 4 arms, else a strip above the panel on the same canvas (more entries fit in no 2.4 x 1.92 in box)
        for fig, ax in figs:
            if len(arms) > 4:
                ps.shared_legend(fig, ax, ncol=2)
            else:
                ps.legend(ax, "upper left")
                ps.make_room(ax)
        written += ps.save_panels(figs, os.path.join(outdir, f"kit_mse_{prefix}{base}"))
        # (2) median predicted vs truth log10|M|^2
        figs = ps.panels(len(fams))
        for (fig, ax), fam in zip(figs, fams):
            tm = tr["median"].get(fam, [])
            if tm:
                ax.plot(*zip(*tm), "-", color="k", label="truth")
            for lab, c, r in arms:
                rows = r["medians"].get(fam, [])
                if rows:
                    ax.plot([q[0] for q in rows], [q[2] for q in rows], "o", ls=ls.get(id(r), "-"), color=c, label=lab)
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(XLAB.get(fam, fam.replace("_", " ")))
            ax.set_ylabel(r"median $\log_{10}|\mathcal{M}|^2$")
            ps.process_label(ax, PROC.get((stem, target), stem))
        h0 = figs[0][1].get_legend_handles_labels()[0]
        for fig, ax in figs:
            if len(arms) > 4:
                ps.shared_legend(fig, figs[0][1], ncol=2)
            else:
                ps.legend(ax, "upper left", handles=h0)
                ps.make_room(ax)
        written += ps.save_panels(figs, os.path.join(outdir, f"kit_median_{prefix}{base}"))
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
    p.add_argument("--prefix", default="", help="file name prefix after kit_mse_ / kit_median_ (e.g. B_)")
    p.add_argument("--split-offshell", action="store_true",
                   help="the off-shellness-off runs in their own figures (<base>_nooffshell), not beside the on ones")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.run_dirs)
    else:
        plot(a.data, a.out, a.prefix, a.split_offshell)


if __name__ == "__main__":
    main()
