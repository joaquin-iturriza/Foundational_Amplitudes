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
Figures (each panel its own file, its own legend):
  kitvm_mse_<p>_*      log-MSE per stratum against the stratum's median singular variable, one panel per family
  kitvm_median_<p>_*   median predicted log10|M|^2 per bin and the truth's
  kitvm_dev_<p>_*      median prediction minus the truth's median per bin (symlog)
  kitvm_bulk           bulk_test log-MSE per process
  kitvm_dist_*         one stratum per family (the deepest) and each process's bulk: truth and predicted log10|M|^2
                       histograms, predicted over true below, over the range any of them occupies
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.kit_figures import PROC, XLAB, family  # noqa: E402

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
# one stratum per family (the deepest bin) and the bulk; canvases of two, grouped by process
CANVASES = [[("ee_uug_me2", "hard_test", "coll_y_1e-07_1e-06"), ("ee_uug_me2", "hard_test", "soft_x_1e-07_1e-06")],
            [("ee_uugg_me2", "hard_test", "coll_qg_y_1e-07_1e-06"), ("ee_uugg_me2", "hard_test", "coll_gg_y_1e-07_1e-06")],
            [("ee_uugg_me2", "hard_test", "soft_x_1e-07_1e-06"), ("ee_uugg_me2", "hard_test", "azimuth_gg_y_1e-07")],
            [("ee_tt_virt_finite", "hard_test", "threshold_beta_0.0001_0.001"),
             ("ee_tt_virt_finite", "hard_test", "sudakov_sqrts_15000_50000")],
            [("ee_uug_me2", "bulk_test", "bulk"), ("ee_tt_virt_finite", "bulk_test", "bulk")]]
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
    ex = [x for c in CANVASES for x in c]
    merged = None
    for site in ("ccin2p3", "lxplus"):
        runs = {p: {lab: d for lab, (s, d) in m.items() if s == site} for p, m in HIST_RUNS.items()}
        part = site_python(SITE, [shlex.quote(json.dumps(ex)), shlex.quote(json.dumps(runs))], "KITVM", site=site,
                           timeout=3000)
        if merged is None:
            merged = part
            continue
        for a, b in zip(merged, part):
            assert a["edges"] == b["edges"] and a["truth"] == b["truth"], f"{a['stratum']}: the sites' KIT files differ"
            a["pred"].update(b["pred"]); a.setdefault("outside", {}).update(b.get("outside", {}))
    out = os.path.join(DATA, "kit_vs_mlp_hist.json")
    json.dump(merged, open(out, "w"))
    print(f"-> {out}")


def _st(k):
    return k.split("/")[1]


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
        fams = sorted({family(s) for s in x_of})
        # (1) log-MSE per stratum, every family (the azimuth scan at its three fixed y)
        panels = ps.panels(len(fams))
        for (fig, ax), fam in zip(panels, fams):
            for sym, v in series.items():
                pts = sorted((x_of[s], m) for s, m in v.items() if s in x_of and family(s) == fam)
                c, mk, ls = sty[sym]
                ax.plot(*zip(*pts), marker=mk, ls=ls, color=c, label=sym)
                print(f"MSE {p} {fam} {sym}: " + " ".join(f"{m:.3g}" for _, m in pts))
            ax.set_xscale("log"); ax.set_yscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(XLAB.get(fam, fam)); ax.set_ylabel(r"log-MSE of $\log_{10}|\mathcal{M}|^2$")
            ps.process_label(ax, PROC[(stem, target)])
            ps.shared_legend(fig, ax, ncol=3)
        written += ps.save_panels(panels, os.path.join(figs, f"kitvm_mse_{p}"))
        # (2) median per bin and (3) its deviation from the truth's median; the azimuth scan has no singular-variable bins
        mf = [f for f in fams if f in fac["medians"]]
        med = {"factor": fac["medians"], "scratch": scr["medians"]}
        for sym, _, tag in MLPS[p]:
            if tag is None:
                continue
            seeds = [r["medians"] for k, r in M["runs"].items() if r["file"] == stem and r["tag"].startswith(tag + "_")]
            med[sym] = {f: [[row[0], row[1], float(np.median([s[f][i][2] for s in seeds])), row[3]]
                            for i, row in enumerate(seeds[0][f])] for f in mf}
            assert all([q[0] for q in s[f]] == [q[0] for q in seeds[0][f]] for s in seeds for f in mf)
        for kind in ("median", "dev"):
            panels = ps.panels(len(mf))
            for (fig, ax), fam in zip(panels, mf):
                tm = dict((round(x, 12), y) for x, y in tr["median"][fam])
                if kind == "median":
                    ax.plot(*zip(*tr["median"][fam]), "-", color="k", label="truth")
                for sym, m in med.items():
                    rows = m[fam]
                    c, mk, ls = sty[sym]
                    y = [q[2] if kind == "median" else q[2] - tm[round(q[0], 12)] for q in rows]
                    ax.plot([q[0] for q in rows], y, marker=mk, ls=ls, color=c, label=sym)
                    if kind == "dev":
                        print(f"DEV {p} {fam} {sym}: deepest {y[0]:+.3g}, max |dev| {max(map(abs, y)):.3g}")
                ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
                if kind == "median":    # the truth's span and half of it again each side: a series beyond it leaves the axes
                    ty = [y for _, y in tr["median"][fam]]
                    pad = max(1.0, 0.5 * (max(ty) - min(ty)))
                    ax.set_ylim(min(ty) - pad, max(ty) + pad)
                if kind == "dev":
                    ax.set_yscale("symlog", linthresh=0.01); ax.axhline(0, color="k", lw=0.8)
                ax.set_xlabel(XLAB.get(fam, fam))
                ax.set_ylabel(r"median $\log_{10}|\mathcal{M}|^2$" if kind == "median"
                              else r"median$_{\rm pred}-$median$_{\rm true}$")
                ps.process_label(ax, PROC[(stem, target)])
                ps.shared_legend(fig, ax, ncol=3)
            written += ps.save_panels(panels, os.path.join(figs, f"kitvm_{kind}_{p}"))
    # (4) bulk log-MSE per process
    fig, ax = ps.figure()
    syms = list(sty)
    for j, sym in enumerate(syms):
        pts = [(i - 0.3 + 0.15 * j, bulk[p][sym]) for i, p in enumerate(PROCS) if sym in bulk[p]]
        c, mk, _ = sty[sym]
        ax.plot(*zip(*pts), mk, ls="", color=c, label=sym)
        print(f"BULK {sym}: " + " ".join(f"{v:.3g}" for _, v in pts))
    ax.set_xticks(range(3), [PROC[PROCS[p]] for p in PROCS]); ax.set_xlim(-0.6, 2.6)
    ax.set_yscale("log"); ax.set_ylabel(r"bulk log-MSE of $\log_{10}|\mathcal{M}|^2$")
    ps.shared_legend(fig, ax, ncol=3)
    written.append(ps.save(fig, os.path.join(figs, "kitvm_bulk")))
    # (5) distributions of one stratum per family, predicted over true below
    hp = os.path.join(DATA, "kit_vs_mlp_hist.json")
    if os.path.exists(hp):
        H = {(e["proc"], e["stratum"]): e for e in json.load(open(hp))}
        for i, canvas in enumerate(CANVASES):
            fig, axs = ps.figure(ncols=2, nrows=2)
            for j, (p, _, st) in enumerate(canvas):
                e = H[(p, st)]
                c = np.array(e["edges"]); t = np.array(e["truth"], float)
                hs = {k: np.array(v, float) for k, v in e["pred"].items()}
                occ = np.flatnonzero(t + sum(hs.values()))
                a, b = occ[0], occ[-1] + 1                       # crop to the occupied bins, merge to <= 300
                m = int(np.ceil((b - a) / 300))
                b = a + m * int(np.ceil((b - a) / m))
                b = min(b, len(t))
                a = b - m * ((b - a) // m)
                c = c[a:b + 1:m]
                t = t[a:b].reshape(-1, m).sum(1)
                e = dict(e, pred={k: v[a:b].reshape(-1, m).sum(1) for k, v in hs.items()})
                top, bot = axs[0][j], axs[1][j]
                top.stairs(t, c, color="k", label="truth")
                for sym in ("factor", "scratch", "MLP log", "MLP raw"):
                    if sym not in e["pred"]:
                        continue
                    h = np.array(e["pred"][sym], float)
                    col, _, ls = sty[sym]
                    lab = r"MLP log+$\beta$" if sym == "MLP log" and p.startswith("ee_tt") else sym
                    col, _, ls = sty[lab]
                    top.stairs(h, c, color=col, ls=ls, label=lab)
                    bot.stairs(np.where(t > 0, h / np.where(t > 0, t, 1), np.nan), c, color=col, ls=ls)
                    print(f"DIST {p} {st} {lab}: outside the edges {100 * e['outside'][sym]:.1f}%")
                top.set_yscale("log"); top.set_ylabel("events"); top.set_ylim(0.5, None)
                occ = np.flatnonzero(t)                   # the ratio exists where the truth has events
                bot.set_xlim(c[occ[0]], c[occ[-1] + 1])
                bot.set_yscale("log"); bot.set_ylabel("pred / true")
                for ax in (top, bot):
                    ax.set_xlabel(r"$\log_{10}(|\mathcal{M}|^2)$")
                fam = family(st)
                lo_hi = st[len(fam) + 1:].replace("_", ",")
                ps.process_label(top, PROC[PROCS[p]] + ("" if st == "bulk" else rf", {fam.split('_')[0]} [{lo_hi}]"))
            ps.shared_legend(fig, axs[0][0], ncol=3)
            written.append(ps.save(fig, os.path.join(figs, f"kitvm_dist_{'abcde'[i]}")))
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
