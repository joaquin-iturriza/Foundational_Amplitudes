#!/usr/bin/env python3
"""The asymptotic-factor transformer against Bakar's reference MLP on every KIT hard family, with the searched
from-scratch transformer as the reference (kit_benchmark step 21 read beside the MLP). Eval-only, no GPU.

    python tools/kit_vs_mlp.py collect      (reads CC and lxplus: per-event histograms of one bin per family)
    python tools/kit_vs_mlp.py plot [--figs analysis/kit_benchmark/figs/vs_mlp]

Sources (all existing):
  factor, scratch    kit_fig_A_hpo.json (tools/kit_figures.py collect): KIT evaluate.py log-MSE per stratum, the
                     median prediction per bin of 4 per decade of the family's singular variable. factor =
                     kit_A_<p>_n100000_t1831_hpo_asym_cont; scratch = the searched chosen trial, plain or emission-scale
                     target (uug hp1918, uugg hp0798, tt virtual hp0131), aliased hpo_softpair_plain / hpo_softpairthr
  MLP (log-MSE)      Bakar's own evaluate.py files (data/full/baseline/*.json) as kept in kit_bakar_figs.json: log
                     invariants, raw invariants, and on tt virtual log invariants + log(s_tt - 4 m_t^2)
  MLP (medians, histograms)   his per-event predictions are not in the KIT files: our reruns of his MLP (D403 ladder,
                     runs/kit_mlp/*_t14400_mlp{mlp,raw}_s{0,1,2}; kit_fig_mlp_reruns.json by tools/kit_figures.py
                     collect @a1603add); medians are the median over the three seeds, histograms seed 0 (his numpy seed)
Figures (one file each, legends inside; off-scale values on the edge, kit_figures.clip_plot):
  kitvm_mse_<p>_*      log-MSE per stratum against the stratum's median singular variable, one panel per family
  kitvm_median_<p>_*   median predicted log10|M|^2 per bin and the truth's
  kitvm_bulk           bulk_test log-MSE per process
  kitvm_dist_<p>_*     one stratum per family (the deepest) and the bulk: truth and predicted log10|M|^2 histograms
                       over the union of the truth's and the predictions' ranges, predicted over true directly below
                       (ps.true_pred_hist, shared x); every model drawn, those near the truth (90% of their events
                       in its range) left, those far from it right
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.kit_figures import OFF, XLAB, YCAP, clip_plot, family, off_entry, sci  # noqa: E402

DATA = os.path.join(ROOT, "analysis/kit_benchmark/data")
PROCS = {"ee_uug_me2": ("ee_uug", "me2"), "ee_uugg_me2": ("ee_uugg", "me2"), "ee_tt_virt_finite": ("ee_tt", "virt_finite")}
FACTOR = "kit_A_{p}_n100000_t1831_hpo_asym_cont"
SCRATCH = {"ee_uug_me2": "kit_A_ee_uug_me2_n100000_t1831_hpo_softpair_plain",
           "ee_uugg_me2": "kit_A_ee_uugg_me2_n100000_t1831_hpo_softpair_plain",
           "ee_tt_virt_finite": "kit_A_ee_tt_virt_finite_n100000_t1831_hpo_softpairthr"}
# run dirs of the histograms: (site, dir)
HIST_RUNS = {p: {"factor": ("ccin2p3", "runs/kit/" + FACTOR.format(p=p)),
                 "MLP log": ("ccin2p3", f"runs/kit_mlp/kit_A_{p}_n100000_t14400_mlpmlp_s0"),
                 "MLP raw": ("ccin2p3", f"runs/kit_mlp/kit_A_{p}_n100000_t14400_mlpraw_s0")} for p in PROCS}
HIST_RUNS["ee_uug_me2"]["scratch"] = ("ccin2p3", "runs/kit_hpo_uug_plain/trial_1918")
HIST_RUNS["ee_uugg_me2"]["scratch"] = ("ccin2p3", "runs/kit_hpo_uugg_plain/trial_0798")
HIST_RUNS["ee_tt_virt_finite"]["scratch"] = ("lxplus", "runs/kit_hpo_ttvirt_thr/trial_0131")
# one stratum per family (the deepest bin) and each process's bulk, a true-against-predicted histogram each; the t t~
# threshold family is read against beta instead (EX_X): in a bin of beta log10|M|^2 is close to flat (|M|^2 ~ 1/beta,
# beta log-uniform), so its histogram is a box whatever the model does
EXAMPLES = [("ee_uug_me2", "hard_test", "coll_y_1e-07_1e-06"), ("ee_uug_me2", "hard_test", "soft_x_1e-07_1e-06"),
            ("ee_uug_me2", "bulk_test", "bulk"),
            ("ee_uugg_me2", "hard_test", "coll_qg_y_1e-07_1e-06"), ("ee_uugg_me2", "hard_test", "coll_gg_y_1e-07_1e-06"),
            ("ee_uugg_me2", "hard_test", "soft_x_1e-07_1e-06"), ("ee_uugg_me2", "hard_test", "azimuth_gg_y_1e-07"),
            ("ee_uugg_me2", "bulk_test", "bulk"),
            ("ee_tt_virt_finite", "hard_test", "sudakov_sqrts_15000_50000"), ("ee_tt_virt_finite", "bulk_test", "bulk")]
EX_X = [("ee_tt_virt_finite", "hard_test", "threshold_beta*")]
THR = r"MLP, log + $\log(s_{t\bar t}-4m_t^2)$"
# legend symbol -> (Bakar's label in kit_bakar_figs.json base, our rerun tag), per process
MLPS = {"ee_uug_me2": [("MLP log", "MLP, log invariants", "mlpmlp"), ("MLP raw", "MLP, raw invariants", "mlpraw")],
        "ee_uugg_me2": [("MLP log", "MLP, log invariants", "mlpmlp"), ("MLP raw", "MLP, raw invariants", "mlpraw")],
        "ee_tt_virt_finite": [(r"MLP log+$\beta$", THR, "mlpmlp"), ("MLP log", "MLP, log invariants", None),
                              ("MLP raw", "MLP, raw invariants", "mlpraw")]}

SITE = r'''
import json, sys
import numpy as np, h5py
sys.path.insert(0, ".")
from tools import kit_hardps as kh
EX, RUNS = json.loads(sys.argv[1]), json.loads(sys.argv[2])
out = []
for proc, split, st in EX:
    stem, tk = (proc[:-4], "log10_me2") if proc.endswith("_me2") else ("ee_tt", "log10_abs_" + proc[len("ee_tt_"):])
    if st.endswith("*"):    # a whole family against its singular variable: 10^(pred - truth) per bin of 4 a decade
        with h5py.File(kh.h5_path(stem), "r") as T:
            sts = [x for x in T[split] if x.startswith(st[:-1])]
            x = np.concatenate([T[split][x]["singular_var"][:] for x in sts])
            t = np.concatenate([T[split][x][tk][:] for x in sts])
        lx = np.log10(x)
        e = np.arange(np.floor(lx.min() * 4) / 4, lx.max() + 0.25, 0.25)
        i = np.digitize(lx, e) - 1
        keep = [b for b in range(len(e) - 1) if (i == b).sum() >= 50]
        r = {"proc": proc, "split": split, "stratum": st, "n": int(len(t)), "x": [float(10 ** (e[b] + 0.125)) for b in keep],
             "ratio": {}}
        for lab, d in RUNS.get(proc, {}).items():
            with h5py.File(d + "/kit_benchmark/predictions.h5", "r") as P:
                q = 10 ** (np.concatenate([P[split][x][tk][:] for x in sts]) - t)
            r["ratio"][lab] = [np.quantile(q[i == b], [0.16, 0.5, 0.84]).tolist() for b in keep]
        out.append(r)
        continue
    with h5py.File(kh.h5_path(stem), "r") as T:
        t = T[split][st][tk][:]
    # edges from the truth alone (both sites draw the same): 40 bins over its 0.1-99.9% range, the same width out to
    # 12 decades beyond it each side (a model far off still lands in a bin; the plot crops to the occupied range)
    q0, q1 = np.quantile(t, [0.001, 0.999])
    w = (q1 - q0) / 40
    k = int(np.ceil(12 / w))
    e = q0 + w * np.arange(-k, 40 + k + 1)
    h = lambda v: np.histogram(np.clip(v, e[0], e[-1]), e)[0].tolist()    # outer bins hold the overflow
    r = {"proc": proc, "split": split, "stratum": st, "n": int(len(t)), "edges": e.tolist(), "truth": h(t), "pred": {}}
    for lab, d in RUNS.get(proc, {}).items():
        with h5py.File(d + "/kit_benchmark/predictions.h5", "r") as P:
            v = P[split][st][tk][:]
        r["pred"][lab] = h(v)
        r.setdefault("outside", {})[lab] = float(np.mean((v < e[0]) | (v > e[-1])))
    out.append(r)
print("KITVM " + json.dumps(out))
'''


def collect():
    from analysis.common.readers import site_python
    import shlex
    ex = EXAMPLES + EX_X
    out = os.path.join(DATA, "kit_vs_mlp_hist.json")
    prev = {(e["proc"], e["stratum"]): e for e in json.load(open(out))} if os.path.exists(out) else {}
    merged = None
    for site in ("ccin2p3", "lxplus"):
        runs = {p: {lab: d for lab, (s, d) in m.items() if s == site} for p, m in HIST_RUNS.items()}
        try:
            part = site_python(SITE, [shlex.quote(json.dumps(ex)), shlex.quote(json.dumps(runs))], "KITVM", site=site,
                               timeout=3000)
        except SystemExit as err:   # a site down: its models keep the previous collection's histograms where they match
            print(f"{site} not read ({str(err)[-200:]}); its models from the previous collection where the bins match")
            for a in merged:
                o = prev.get((a["proc"], a["stratum"]))
                if o and "edges" in a and o.get("edges") == a["edges"] and o["truth"] == a["truth"]:
                    for lab in runs.get(a["proc"], {}):
                        if lab in o["pred"]:
                            a["pred"][lab] = o["pred"][lab]; a["outside"][lab] = o["outside"][lab]
            continue
        if merged is None:
            merged = part
            continue
        for a, b in zip(merged, part):
            if "ratio" in a:
                assert a["x"] == b["x"], f"{a['stratum']}: the sites' KIT files differ"
                a["ratio"].update(b["ratio"])
                continue
            assert a["edges"] == b["edges"] and a["truth"] == b["truth"], f"{a['stratum']}: the sites' KIT files differ"
            a["pred"].update(b["pred"]); a.setdefault("outside", {}).update(b.get("outside", {}))
    json.dump(merged, open(out, "w"))
    print(f"-> {out}")


def _st(k):
    return k.split("/")[1]


# one plain-words name per model, the same in every figure (Joaquin 2026-11-17: a figure reads on its own)
NAME = {"factor": "transformer + asymptotic factor", "scratch": "transformer",
        "MLP log": "MLP, log invariants", "MLP raw": "MLP, raw invariants",
        r"MLP log+$\beta$": "MLP, log invariants\n" + r"and $\log(s_{t\bar t}-4m_t^2)$"}
PROCESS = {"ee_uug_me2": r"$e^+e^-\to u\bar u g$", "ee_uugg_me2": r"$e^+e^-\to u\bar u gg$",
           "ee_tt_virt_finite": r"$e^+e^-\to t\bar t$, one loop"}
# panels in reading order; the azimuth strata are a scan in the gluons' azimuth at three fixed y (not a y scan of
# their own): their x is that fixed y
FAM_ORDER = ["coll_y", "coll_gg_y", "coll_qg_y", "soft_x", "azimuth_gg_y", "sudakov_sqrts", "threshold_beta"]
XLAB = dict(XLAB, azimuth_gg_y=r"azimuthal scan at fixed $y=s_{gg}/s$")
# the deepest bin of each family, as the panel says it
STRATUM = {"coll_y_1e-07_1e-06": r"collinear, $10^{-7}<y<10^{-6}$", "soft_x_1e-07_1e-06": r"soft, $10^{-7}<x<10^{-6}$",
           "coll_gg_y_1e-07_1e-06": r"$gg$ collinear, $10^{-7}<y<10^{-6}$",
           "coll_qg_y_1e-07_1e-06": r"$qg$ collinear, $10^{-7}<y<10^{-6}$",
           "azimuth_gg_y_1e-07": r"azimuthal scan, $y=10^{-7}$",
           "sudakov_sqrts_15000_50000": r"$15<\sqrt{s}<50$ TeV"}
MSE_LAB = r"MSE$(\log_{10}|\mathcal{M}|^2)$"
LEG_PT = 7.5   # ps.LEGEND_MIN_PT: the legends sit inside panels whose data leave little room (Joaquin 2026-11-17)
LOGM = r"$\log_{10}(|\mathcal{M}|^2/\mathrm{GeV}^{-2})$"


def _grid(n):
    """one canvas for n panels (1x1, 1x2, 2x2): (fig, axes in reading order, spare cell or None)"""
    import plot_style as ps
    fig, axs = ps.figure(ncols=1 if n == 1 else 2, nrows=2 if n > 2 else 1)
    axs = list(np.ravel(axs))
    return fig, axs[:n], (axs[n] if len(axs) > n else None)


def _legend(axs, spare, process=None, at=(0, "upper left")):
    """the one legend of the canvas: the spare cell of a 3-panel 2x2, else the corner `at` = (panel, corner), the one
    the data leave free; there, with `process`, the legend's title names the process in place of the panel's process
    label (they collided)"""
    import plot_style as ps
    if spare is not None:
        ps.legend_cell(spare, axs, fontsize=ps.LEGEND_MIN_PT)
        return
    hl = {}
    for a in axs:
        for h, l in zip(*a.get_legend_handles_labels()):
            hl.setdefault(l, h)
    if OFF in hl:   # last
        hl[OFF] = hl.pop(OFF)
    ax = axs[at[0]]
    if process and getattr(ax, "_ps_label", None) is not None:
        ax._ps_label.remove()
        del ax._ps_label
    ps.legend(ax, at[1], handles=list(hl.values()), labels=list(hl), fontsize=LEG_PT, title=process,
              title_fontsize=LEG_PT, alignment="left")
    ps.make_room(ax)


# the corner each canvas's legend goes to: one the data leave free once the off-scale values sit on the edge
LEG_AT = {"mse": {"ee_uug_me2": (1, "lower left"), "ee_uugg_me2": (2, "lower left"),
                  "ee_tt_virt_finite": (1, "lower left")},
          "median": {"ee_uug_me2": (0, "upper left"), "ee_uugg_me2": (0, "upper left"),
                     "ee_tt_virt_finite": (1, "upper right")}}


def _far(e, k):
    """a model far from the truth in a histogram: under 90% of its events inside the truth's filled range; drawn in
    its own panel beside the truth, so the models near it keep a readable x range (Joaquin 2026-11-17)"""
    ed, t, q = np.asarray(e["edges"]), np.asarray(e["truth"]), np.asarray(e["pred"][k], float)
    f = np.flatnonzero(t)
    c = 0.5 * (ed[:-1] + ed[1:])
    return q[(c >= ed[f[0]]) & (c <= ed[f[-1] + 1])].sum() < 0.9 * q.sum()


def plot(figs):
    import plot_style as ps
    from matplotlib.ticker import NullFormatter
    from tools.kit_bakar_figs import metrics
    os.makedirs(figs, exist_ok=True)
    A = json.load(open(os.path.join(DATA, "kit_fig_A_hpo.json")))
    M = json.load(open(os.path.join(DATA, "kit_fig_mlp_reruns.json")))
    B = json.load(open(os.path.join(DATA, "kit_bakar_figs.json")))
    C = ps.C
    sty = {"factor": (C.blue, "o", "-"), "scratch": (C.vermillion, "s", "-"), r"MLP log+$\beta$": ("k", "^", "--"),
           "MLP log": (C.green, "^", "--"), "MLP raw": (C.grey, "x", ":")}
    written = []
    bulk = {}
    for p, (stem, target) in PROCS.items():
        fac, scr = A["runs"][FACTOR.format(p=p)], A["runs"][SCRATCH[p]]
        tr = A["truth"][f"{stem}/{target}"]
        x_of = {_st(k): v["x"] for k, v in fac["strata"].items() if k.startswith("hard_test/")}
        series = {"factor": {_st(k): v["logmse"] for k, v in fac["strata"].items()},
                  "scratch": {_st(k): v["logmse"] for k, v in scr["strata"].items()}}
        for sym, his, _ in MLPS[p]:
            series[sym] = metrics(B[p]["base"][his])[0]
        bulk[p] = {s: v["bulk"] for s, v in series.items()}
        fams = sorted({family(s) for s in x_of}, key=FAM_ORDER.index)
        # (1) MSE per hard bin, every family of the process on one canvas (the azimuth scan at its three fixed y)
        fig, axs, spare = _grid(len(fams))
        for ax, fam in zip(axs, fams):
            for sym, v in series.items():
                pts = sorted((x_of[s], m) for s, m in v.items() if s in x_of and family(s) == fam)
                c, mk, ls = sty[sym]
                if clip_plot(ax, *zip(*pts), 0, YCAP, sci, marker=mk, ls=ls, color=c, label=NAME[sym]):
                    off_entry(ax)
                print(f"MSE {p} {fam} {sym}: " + " ".join(f"{x:.3g}:{m:.3g}" for x, m in pts))
            ax.set_xscale("log"); ax.set_yscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(XLAB.get(fam, fam)); ax.set_ylabel(MSE_LAB)
            ps.process_label(ax, PROCESS[p], loc="upper right")
        _legend(axs, spare, PROCESS[p], LEG_AT["mse"][p])
        written.append(ps.save(fig, os.path.join(figs, f"kitvm_mse_{p}")))
        # (2) median per bin and (3) its shift from the truth's median; the azimuth scan has no singular-variable bins
        mf = [f for f in fams if f in fac["medians"]]
        med = {"factor": fac["medians"], "scratch": scr["medians"]}
        for sym, _, tag in MLPS[p]:
            if tag is None:
                continue
            seeds = [r["medians"] for k, r in M["runs"].items() if r["file"] == stem and r["tag"].startswith(tag + "_")]
            med[sym] = {f: [[row[0], row[1], float(np.median([s[f][i][2] for s in seeds])), row[3]]
                            for i, row in enumerate(seeds[0][f])] for f in mf}
            assert all([q[0] for q in s[f]] == [q[0] for q in seeds[0][f]] for s in seeds for f in mf)
        # the y range is the truth's, widened by its own span (at least a decade) each side; beyond it, off the scale
        fig, axs, spare = _grid(len(mf))
        for ax, fam in zip(axs, mf):
            ty = [y for _, y in tr["median"][fam]]
            pad = max(1.0, max(ty) - min(ty))
            lo, hi = min(ty) - pad, max(ty) + pad
            ax.plot(*zip(*tr["median"][fam]), "-", color="k", label="truth")
            for sym, m in med.items():
                c, mk, ls = sty[sym]
                if clip_plot(ax, [q[0] for q in m[fam]], [q[2] for q in m[fam]], lo, hi, lambda v: f"{v:+.0f}",
                             marker=mk, ls=ls, color=c, label=NAME[sym]):
                    off_entry(ax)
            ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(XLAB.get(fam, fam))
            ax.set_ylabel(r"median $\log_{10}|\mathcal{M}|^2$")
            ps.process_label(ax, PROCESS[p], loc="upper right")
        _legend(axs, spare, PROCESS[p], LEG_AT["median"][p])
        written.append(ps.save(fig, os.path.join(figs, f"kitvm_median_{p}")))
    # (4) bulk MSE per process
    fig, ax = ps.figure()
    for j, sym in enumerate(sty):
        pts = [(i - 0.3 + 0.15 * j, bulk[p][sym]) for i, p in enumerate(PROCS) if sym in bulk[p]]
        c, mk, _ = sty[sym]
        ax.plot(*zip(*pts), mk, ls="", color=c, label=NAME[sym])
        print(f"BULK {sym}: " + " ".join(f"{v:.3g}" for _, v in pts))
    ax.set_xticks(range(3), [r"$u\bar ug$", r"$u\bar ugg$", r"$t\bar t$, one loop"]); ax.set_xlim(-0.6, 2.6)
    ax.set_yscale("log"); ax.set_ylabel(MSE_LAB)
    ps.legend(ax, "upper left", fontsize=LEG_PT); ps.make_room(ax)
    written.append(ps.save(fig, os.path.join(figs, "kitvm_bulk")))
    # (5) true against predicted log10|M|^2: the deepest bin of each hard family, one row each, the models near the
    # truth on the left and those far from it (_far) on the right, each panel on the union of its own models' and the
    # truth's ranges (ps.true_pred_hist); the three bulks on one canvas, every model near the truth there
    hp = os.path.join(DATA, "kit_vs_mlp_hist.json")
    if os.path.exists(hp):
        H = json.load(open(hp))
        sym_of = lambda p, k: r"MLP log+$\beta$" if k == "MLP log" and p.startswith("ee_tt") else k
        def draw(ax, p, e, keys, label):
            models = [dict(label=NAME[sym_of(p, k)], edges=e["edges"], truth=e["truth"], pred=e["pred"][k], n=e["n"],
                           style=dict(color=sty[sym_of(p, k)][0], ls=sty[sym_of(p, k)][2])) for k in keys]
            ps.true_pred_hist(ax, models, ratio=(0.1, 10), log_ratio=True, xlabel=LOGM,
                              ratio_label=r"$\frac{\rm pred}{\rm true}$")
            ps.process_label(ax, label, loc="upper right")
            print(f"DIST {p} {e['stratum']} {keys}: x {ax.get_xlim()[0]:.3g}..{ax.get_xlim()[1]:.3g}")
        order = ("factor", "MLP log", "scratch", "MLP raw")
        for p in PROCS:
            ex = sorted((e for e in H if e["proc"] == p and "edges" in e and e["split"] == "hard_test"),
                        key=lambda e: FAM_ORDER.index(family(e["stratum"])))
            fig, axs = ps.figure(ncols=2, nrows=len(ex))
            axs = np.reshape(axs, (len(ex), 2))
            for row, e in zip(axs, ex):
                ks = [k for k in order if k in e["pred"]]
                near, far = [k for k in ks if not _far(e, k)], [k for k in ks if _far(e, k)]
                assert near and far, f"{p} {e['stratum']}: a column would be empty ({near} | {far})"
                for ax, keys in zip(row, (near, far)):
                    draw(ax, p, e, keys, PROCESS[p] + "\n" + STRATUM[e["stratum"]])
            for c in (0, 1):   # each column its own models: a legend each, in its top panel, titled with its label
                _legend([axs[0][c]], None, PROCESS[p] + "\n" + STRATUM[ex[0]["stratum"]])
            written.append(ps.save(fig, os.path.join(figs, f"kitvm_dist_{p}")))
        ex = [e for e in H if "edges" in e and e["split"] == "bulk_test"]
        fig, axs, spare = _grid(len(ex))
        for ax, e in zip(axs, ex):
            draw(ax, e["proc"], e, [k for k in order if k in e["pred"]], PROCESS[e["proc"]])
        _legend(axs, spare)
        written.append(ps.save(fig, os.path.join(figs, "kitvm_dist_bulk")))
    for w in written:
        print("FIGURE", w)


def main():   # dup-ok: argparse collect/plot boilerplate, every kit tool has it
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    p = sub.add_parser("plot"); p.add_argument("--figs", default=os.path.join(ROOT, "analysis/kit_benchmark/figs/vs_mlp"))
    a = ap.parse_args()
    collect() if a.cmd == "collect" else plot(a.figs)


if __name__ == "__main__":
    main()
