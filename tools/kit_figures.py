#!/usr/bin/env python3
"""Figures of the KIT hard phase-space benchmark: per process, each arm's KIT log10-MSE per hard_test stratum and its
median predicted log10|M|^2 against the truth's, along the stratum family's singular variable.

    python tools/kit_figures.py collect <run dir>[=<KIT run name>] ... > out.txt     (on the site: reads the runs and the KIT files)
    python tools/kit_figures.py plot <collected.json> [--out analysis/kit_benchmark/figs]

collect, per run dir (runs/kit/kit_<mode>_<file>_<target>_n<N>_t<T>[_tag]): its kit_benchmark/kit_result.json
(KIT's src/evaluate.py numbers, through tools/kit_predict.py) and predictions.h5, and the KIT file's truth and
singular_var per hard_test event. Per stratum: log-MSE, bias, the median of singular_var (the x of the MSE figure).
Per family (stratum name without its two bin edges), events of all its strata in bins of 4 per decade of
singular_var: the median truth and median prediction of log10|M|^2. Per process: the reference line, the log10-MSE of
the truth clipped to its bulk_train [0.5%, 99.5%] range (tools/kit_target_slope.py's plain_edge): a model exact inside
the training range and flat at its edge outside it (zero in a bin whose truth stays inside the range); and the same
binned truth median. Prints one line 'KITFIG <json>'.

plot: one file per process and figure kind, one panel per family and the legend in a spare cell:
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

# Off-scale rule of the KIT figures (Joaquin 2026-11-17: a bad model far off must not squash the good ones): an
# MSE(log10|M|^2) axis stops at YCAP (an error of 10 decades); a value beyond an axis limit is drawn on the edge as a
# hollow triangle pointing the way the value lies, the most extreme of each curve printed beside it, and the legend gets an "off the scale" entry
YCAP = 1e2
OFF = "off the scale, value printed"
# one name, colour (a plot_style C name, or matplotlib's), marker and line style per model in every KIT figure
# (Joaquin 2026-11-26: the deck's figures coherent); tools/kit_vs_mlp.py and tools/kit_bakar_figs.py draw from it
MODELS = {"truth": ("truth", "k", None, "-"),
          "factor": ("transformer + asymptotic factor", "blue", "o", "-"),
          "scratch": ("transformer", "vermillion", "s", "-"),
          "MLP thr": ("MLP, log invariants\n" + r"and $\log(s_{t\bar t}-4m_t^2)$", "purple", "^", "--"),
          "MLP log": ("MLP, log invariants", "green", "^", "--"),
          "MLP raw": ("MLP, raw invariants", "grey", "x", ":")}


def model_style(key):
    """(name, colour, marker, line style) of a MODELS key"""
    import plot_style as ps
    name, c, mk, ls = MODELS[key]
    return name, getattr(ps.C, c, c), mk, ls


def clip_plot(ax, x, y, lo, hi, fmt, **kw):
    """ax.plot(x, y, **kw) with y clipped to [lo, hi]; clipped points hollow, the most extreme one labelled fmt(value)
    (fmt None: no label, the paper's figures carry no in-panel text and the caption gives the value). True when a point
    was clipped."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    yc = np.clip(y, lo, hi)
    line, = ax.plot(x, yc, **kw)
    out = np.flatnonzero(y != yc)
    if not out.size:
        return False
    for side, mk in ((y > hi, "^"), (y < lo, "v")):   # the arrow's way: where the value lies
        if side.any():
            ax.plot(x[side], yc[side], ls="", marker=mk, color=line.get_color(), mfc="white", zorder=line.get_zorder() + 1)
    if fmt is None:
        return True
    i = out[np.argmax(np.abs(np.log10(y[out] / yc[out]) if ax.get_yscale() == "log" else y[out] - yc[out]))]
    top = y[i] > hi
    ax.annotate(fmt(y[i]), (x[i], yc[i]), textcoords="offset points", xytext=(-4, -12 if top else 4), ha="right",
                color=line.get_color())
    return True


def off_entry(ax, label=OFF):
    """the legend entry of clip_plot's hollow markers (an empty line: no data)"""
    ax.plot([], [], ls="", marker="^", color="0.4", mfc="white", label=label)


def sci(v):
    """a value as a power of ten in mathtext, e.g. 3x10^4"""
    e = int(np.floor(np.log10(v)))
    return rf"${v / 10 ** e:.0f}\times10^{{{e}}}$"


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
        # <dir>=<name>: a run dir not named as a KIT run (a kit_hpo trial scored as a control) under a KIT run name
        rd, _, alias = rd.partition("=")
        alias = alias or os.path.basename(rd.rstrip("/"))
        m = RUN_RE.match(alias)
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
                # med: median signed error (prediction - truth, log10|M|^2), the kit_benchmark design's outcome measure
                strata[name] = dict(r, x=float(np.median(sv)), lo=edges(st)[0], hi=edges(st)[1],
                                    med=float(np.median(y - t)))
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
        out["runs"][alias] = {
            "mode": m["mode"], "file": stem, "target": target, "n": int(m["n"]), "steps": int(m["t"]),
            "tag": m["tag"] or "", "strata": strata, "slopes": res.get("slopes", {}), "medians": meds}
    print("KITFIG " + json.dumps(out))


PROC = {("ee_tt", "virt_finite"): r"$t\bar t$ virtual", ("ee_tt", "born"): r"$t\bar t$ Born",
        ("ee_uug", "me2"): r"$u\bar u g$", ("ee_uugg", "me2"): r"$u\bar u gg$"}
XLAB = {"threshold_beta": r"top velocity $\beta$", "sudakov_sqrts": r"$\sqrt{s}$ [GeV]",
        "soft_x": r"soft gluon $x=2E_g/\sqrt{s}$", "coll_y": r"collinear $y=s_{qg}/s$",
        "coll_qg_y": r"quark-gluon collinear $y=s_{qg}/s$", "coll_gg_y": r"gluon-gluon collinear $y=s_{gg}/s$",
        "azimuth_gg_y": r"$g\to gg$ collinear $y=s_{gg}/s$"}


# input-representation arms (D374, model.input_*): tag -> legend symbol, one line, defined in the caption
# (soft: soft log rescale; log s_ij: log(s_ij/s) pair inputs; log beta^2: of massive pairs; factor: the asymptotic
# factor A of log|M|^2 = log A + g; plain: plain target; P_k(L): the learned log-invariant polynomial on the output;
# FT: fine-tuned from the held-out catalog parent, +LO / +1L: with LO / one-loop tops near threshold)
INPUT_ARMS = {"pairlogs": r"$\log s_{ij}$", "softrescale": "soft", "locallogs": "local logs",
              "softpair": r"soft, $\log s_{ij}$",
              "softpairthr": r"soft, $\log s_{ij}$, $\log\beta^2$",
              "softpairthr_vbeta": r"$\log\beta^2$, target $V\beta$",
              "softpair_rezero": "ReZero",
              "softpair_scalars": "MLP inputs", "softpairthr_scalars": "MLP inputs",
              "softpair_plain": "plain",
              "softpair_plain_logskip": r"plain, $P_2(L)$", "softpair_plain_logskiplin": r"plain, $P_1(L)$",
              "softpairthr_logskip": r"$\log\beta^2$, $P_2(L)$", "softpairthr_logskiplin": r"$\log\beta^2$, $P_1(L)$",
              "softpair_asym": "factor", "softpair_plain_asym": "factor",
              "softpairthr_asym_plainlog": r"factor, $\log\beta^2$",
              "asym_cont": r"factor, $\gamma\leq4$", "asym_cont_plainlog": r"factor, $\gamma\leq4$",
              # steps 14/19/21 at the kit_hpo chosen HPs (tags hpo_<arm>; the hpo_ prefix is dropped for the legend)
              "asym_cont_sud1": r"factor+log, $\gamma\leq4$",
              "softpair_plain_divform": "div-form", "softpair_plain_rezero": "plain, ReZero",
              "softpairthr_divform": r"div-form, $\log\beta^2$", "softpairthr_rezero": r"ReZero, $\log\beta^2$",
              # step 22: each arm at its own 16-trial search's chosen HPs (best against best)
              "softpairthr_divform_searched": r"div-form searched, $\log\beta^2$",
              "softpairthr_rezero_searched": r"ReZero searched, $\log\beta^2$",
              "ftr5own": "FT", "ftr5thrown": "FT+LO", "ftr5thrbetaown": r"FT+LO+$\beta$",
              "ftr5thrloopown": "FT+1L", "ftr5thrloopbetaown": r"FT+1L+$\beta$"}


# short legend symbols (--short; the caption defines them): steps 14/17/19/21 at the kit_hpo chosen HPs
SHORT = {"softpair_plain": "scratch", "softpair": "scratch", "softpairthr": r"scratch+$\beta$",
         "asym_cont": "factor", "asym_cont_sud1": "factor+log",
         "softpair_plain_divform": "div-form", "softpairthr_divform": "div-form",
         "softpair_plain_rezero": "ReZero", "softpairthr_rezero": "ReZero",
         "softpairthr_divform_searched": "div-form searched", "softpairthr_rezero_searched": "ReZero searched",
         "ftr5own": "FT", "ftr5own_search": "FT searched", "ftr5thrown": "FT+LO", "ftr5thrbetaown": r"FT+LO+$\beta$",
         "ftr5thrloopown": "FT+1L", "ftr5thrloopbetaown": r"FT+1L+$\beta$"}


def arm_label(r, with_n=True, short=False):
    """the legend symbol of a run, one line, defined in the caption: how it was trained, the bulk size (left out when
    every curve of the figure has the same one), its inputs; short: the SHORT symbol"""
    if short:
        return SHORT[r["tag"].removeprefix("hpo_")]
    tag = r["tag"].removeprefix("hpo_")
    lr = tag.endswith("lr0p1")      # the A_ft design's lr-axis check (KIT_LR_SCALE=0.1)
    tag = tag[:-len("lr0p1")].rstrip("_") if lr else tag
    how = {"A": "", "A_ft": "FT", "B": "+hard", "B_bulk": "+bulk"}[r["mode"]]     # from scratch: no prefix
    inputs = {"offoff": "no off-shell", "nooffshell": "no off-shell", "plain": "plain"}.get(tag, INPUT_ARMS.get(tag, ""))
    n = (r"$10^5$" if r["n"] == 100000 else r"$9\times10^5$") if with_n else ""
    parts = [how if not inputs.startswith("FT") else "", n, inputs or ("scratch" if not how else ""), r"lr$\times0.1$" if lr else ""]
    return ", ".join(p for p in parts if p)


def _grid(ps, n):
    """One figure for n family panels plus a spare cell for the legend (one file, legend inside; fa-figures):
    (fig, panel axes, legend cell)."""
    cells = n + 1
    nc = 1 if cells == 3 else min(cells, 2)   # three columns run past the printable width, 2x2 would leave a hole
    nr = -(-cells // nc)
    fig, axs = ps.figure(ncols=nc, nrows=nr)
    axs = list(np.ravel(axs))
    for a in axs[cells:]:
        a.set_axis_off()
    return fig, axs[:n], axs[n]


def plot(path, outdir, prefix="", split_offshell=False, short=False):
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
        arms = [(arm_label(r, with_n, short), colors[i % len(colors)], r) for i, (_, r) in enumerate(runs)]
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
        fig, axs, cell = _grid(ps, len(fams))
        for ax, fam in zip(axs, fams):
            for lab, c, r in arms:
                pts = sorted((v["x"], v["logmse"]) for k, v in r["strata"].items()
                             if k.startswith("hard_test/") and family(k.split("/")[1]) == fam)
                ax.plot(*zip(*pts), "o", ls=ls.get(id(r), "-"), color=c, label=lab)
            fl = sorted((v["x"], v["mse"]) for st, v in tr["floor"].items() if family(st) == fam)
            if any(m > 0 for _, m in fl):   # zero in every bin (truth inside the training range): not drawn, no entry
                ax.plot(*zip(*fl), "--", color=ps.C.grey, label="truth clipped to its bulk range")
            ref = MLP.get((stem, target, fam))
            if ref:
                x = np.sqrt(ref[0] * ref[1])
                ax.plot([x], [ref[2]], "*", color="k", label="KIT MLP, log inputs")
                ax.plot([x], [ref[3]], "x", color="k", label="KIT MLP, raw inputs")
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_yscale("log")
            ax.set_xlabel(XLAB.get(fam, fam.replace("_", " ")))
            ax.set_ylabel(r"MSE$(\log_{10}|\mathcal{M}|^2)$")
            ps.process_label(ax, PROC.get((stem, target), stem))
        ps.legend_cell(cell, axs, fontsize=ps.LEGEND_MIN_PT)
        written.append(ps.save(fig, os.path.join(outdir, f"kit_mse_{prefix}{base}")))
        # (2) median predicted vs truth log10|M|^2
        fig, axs, cell = _grid(ps, len(fams))
        for ax, fam in zip(axs, fams):
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
        ps.legend_cell(cell, axs, fontsize=ps.LEGEND_MIN_PT)
        written.append(ps.save(fig, os.path.join(outdir, f"kit_median_{prefix}{base}")))
    for w in written:
        print(w)


# summary: deepest-bin MSE per family, one panel per process (families grouped by process), one file
FAMS = {("ee_uug", "me2"): [("coll_y", "collinear"), ("soft_x", "soft")],
        ("ee_uugg", "me2"): [("coll_gg_y", "gg coll."), ("coll_qg_y", "qg coll."), ("soft_x", "soft")],
        ("ee_tt", "virt_finite"): [("threshold_beta", "threshold"), ("sudakov_sqrts", "high energy")]}
SUMMARY_PROC = {("ee_uug", "me2"): r"$e^+e^-\to u\bar u g$", ("ee_uugg", "me2"): r"$e^+e^-\to u\bar u gg$",
                ("ee_tt", "virt_finite"): r"$e^+e^-\to t\bar t$, one loop"}
# the fine-tunes in two markers by parent (none fixes a limit): the held-out parent keeps every other top class
# (recipes/gen_datasaving_parent.py), so only e+e- > t t~ is out; the other four add threshold-sampled tops
FT_HELD = "fine-tuned, $e^+e^-\\to t\\bar t$\nnot in pretraining (2 runs)"
FT_TOPS = "fine-tuned, threshold tops\nadded to pretraining (4 runs)"
# each arm in plain words, the same name as in tools/kit_vs_mlp.py (Joaquin 2026-11-17: a figure reads on its own)
WORDS = {"scratch": "transformer", r"scratch+$\beta$": "transformer",
         "scratch-nobeta": r"transformer without $\log\beta^2$ input",
         "factor": "transformer + asymptotic factor", "factor+log": "asymptotic factor + single log",
         "div-form": "divergence-form inputs", "ReZero": "ReZero layers",
         "FT": "fine-tuned", "FT searched": "fine-tuned, searched",
         "FT+LO": "fine-tuned, LO tops in pretraining", r"FT+LO+$\beta$": r"fine-tuned, LO tops, $\log\beta^2$",
         "FT+1L": "fine-tuned, LO and one-loop tops", r"FT+1L+$\beta$": "fine-tuned, LO and one-loop tops,\n"
                                                                         r"$\log\beta^2$"}


def summary(paths, out):
    """the deepest-bin MSE of every arm and control of kit_readout --verdict jsons, per family: one canvas, a panel per
    process and the legend in the spare cell"""
    import plot_style as ps
    d = {k: v for path in paths for k, v in json.load(open(path)).items()}
    vals = {}   # (file, target) -> {tag: {family: logmse}}
    for pair, v in d.items():
        for name, key in zip(pair.split(" vs "), ("logmse", "logmse_ref")):
            m = RUN_RE.match(name)
            for fam, x in v["families"].items():
                vals.setdefault((m["file"], m["target"]), {}).setdefault(m["tag"].removeprefix("hpo_"), {})[fam] = x[key]
    procs = [p for p in FAMS if p in vals]
    # one style per arm across the panels, the transformers' and the MLP's as MODELS; green and purple left to the
    # MLPs; the fine-tunes share hollow black markers, one shape per parent
    C = ps.C
    sty = {"scratch": (C.vermillion, "s"), r"scratch+$\beta$": (C.vermillion, "s"), "scratch-nobeta": (C.orange, "s"),
           "factor": (C.blue, "o"), "factor+log": (C.sky, "o"), "div-form": (C.orange, "D"), "ReZero": (C.sky, "P"),
           "FT": ("k", "v"), "FT searched": ("k", "v"), "FT+LO": ("k", "d"), r"FT+LO+$\beta$": ("k", "d"), "FT+1L": ("k", "d"),
           r"FT+1L+$\beta$": ("k", "d")}
    sym = lambda p, t: "scratch-nobeta" if p[0] == "ee_tt" and t == "softpair" else SHORT[t]
    fig, axs = ps.figure(ncols=2, nrows=2)
    axs = list(np.ravel(axs))
    clipped, labs = False, set()
    for ax, p in zip(axs, procs):
        order = list(sty)
        fams, arms = FAMS[p], sorted(vals[p], key=lambda t: order.index(sym(p, t)))
        w = 0.7 / len(arms)
        for j, t in enumerate(arms):
            xs = [(i - 0.35 + w * (j + 0.5), vals[p][t][f]) for i, (f, _) in enumerate(fams) if f in vals[p][t]]
            c, mk = sty[sym(p, t)]
            ft = sym(p, t).startswith("FT")
            lab = (FT_HELD if sym(p, t) in ("FT", "FT searched") else FT_TOPS) if ft else WORDS[sym(p, t)]
            clipped |= clip_plot(ax, *zip(*xs), 0, YCAP, sci, ls="", marker=mk, color=c, mfc="none" if ft else c,
                                 label=lab if lab not in labs else None)
            labs.add(lab)
            print(f"DEEPEST {p[0]} {WORDS[sym(p, t)]!r}: " + ", ".join(f"{f} {vals[p][t][f]:.3g}" for f, _ in fams
                                                                       if f in vals[p][t]))
        ref = [(i, MLP[(*p, f)][2]) for i, (f, _) in enumerate(fams) if (*p, f) in MLP]
        mlp, mc, _, mls = model_style("MLP log")   # green and dashed, as on every KIT figure (MODELS)
        ax.hlines([y for _, y in ref], [i - 0.4 for i, _ in ref], [i + 0.4 for i, _ in ref], color=mc, ls=mls,
                  label=mlp)
        ax.set_xticks(range(len(fams)), [lab for _, lab in fams])
        ax.set_xlim(-0.6, len(fams) - 0.4)
        ax.set_yscale("log")
        ax.set_ylabel(r"MSE$(\log_{10}|\mathcal{M}|^2)$")
        ps.process_label(ax, SUMMARY_PROC[p])
        # a quarter of the span clear above the highest marker (make_room tests marker centres only)
        ys = [min(y, YCAP) for t in arms for y in vals[p][t].values()] + [y for _, y in ref]
        lo, hi = np.log10(min(ys)), np.log10(max(ys))
        ax.set_ylim(10 ** (lo - 0.1 * (hi - lo)), 10 ** (hi + 0.25 * (hi - lo)))
    if clipped:
        off_entry(axs[len(procs) - 1])   # last in the legend
    ps.legend_cell(axs[3], axs[:3], fontsize=ps.LEGEND_MIN_PT)
    print(ps.save(fig, out))
    # the numbers for % claim: lines (tools/verify_claims.py reads the last JSON line): MSE(log10|M|^2) in the deepest
    # bin, by process, run tag and family
    print(json.dumps({"deepest": {f"{p[0]}_{p[1]}": vals[p] for p in procs},
                      "mlp": {f"{p[0]}_{p[1]}": {f: MLP[(*p, f)][2] for f, _ in FAMS[p] if (*p, f) in MLP} for p in procs}}))


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
    p.add_argument("--short", action="store_true", help="legend symbols of SHORT (the caption defines them)")
    s = sub.add_parser("summary")
    s.add_argument("verdict", nargs="+", help="tools/kit_readout.py --verdict jsons")
    s.add_argument("--out", default=os.path.join(ROOT, "analysis", "kit_benchmark", "figs", "kit_deepest_A_hpo"))
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.run_dirs)
    elif a.cmd == "summary":
        summary(a.verdict, a.out)
    else:
        plot(a.data, a.out, a.prefix, a.split_offshell, a.short)


if __name__ == "__main__":
    main()
