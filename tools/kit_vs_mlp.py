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
  kitvm_dist_<p>_*     one stratum per family (the deepest) and the bulk: truth and predicted log10|M|^2 histograms
                       over the union of the truth's and the predictions' ranges, predicted over true directly below
                       (ps.true_pred_hist, shared x); every model drawn
  kitvm_ratio_beta_*   t t~ virtual threshold family: |M|^2_pred/|M|^2_true per bin of beta, median and 16-84% band
                       (a histogram of a beta bin is a box: log10|M|^2 is near flat there)
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
                ys = {sym: [q[2] if kind == "median" else q[2] - tm[round(q[0], 12)] for q in m[fam]] for sym, m in med.items()}
                if kind == "median":    # the truth's span and half of it again each side
                    ty = [y for _, y in tr["median"][fam]]
                    pad = max(1.0, 0.5 * (max(ty) - min(ty)))
                    lo, hi = min(ty) - pad, max(ty) + pad
                else:                   # the models on the truth (factor, MLP with log inputs) and half again
                    lim = max(0.1, 1.5 * max(abs(v) for s_, y in ys.items() if s_ in ("factor", "MLP log", r"MLP log+$\beta$")
                                             for v in y))
                    lo, hi = -lim, lim
                for sym, m in med.items():
                    y = ys[sym]
                    if kind == "dev":
                        print(f"DEV {p} {fam} {sym}: deepest {y[0]:+.3g}, max |dev| {max(map(abs, y)):.3g}")
                    if not any(lo <= v <= hi for v in y):     # wholly off the axes: not drawn (caption)
                        print(f"OFF {kind} {p} {fam} {sym}: not drawn")
                        continue
                    c, mk, ls = sty[sym]
                    ax.plot([q[0] for q in m[fam]], y, marker=mk, ls=ls, color=c, label=sym)
                ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
                if kind == "dev":
                    ax.set_yscale("symlog", linthresh=0.01); ax.axhline(0, color="k", lw=0.8)
                ax.set_ylim(lo, hi)
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
    # (5) true against predicted log10|M|^2 of one stratum per family, predicted over true in the panel below
    # (ps.true_pred_hist: the union of the truth's and the predictions' ranges, every model drawn)
    hp = os.path.join(DATA, "kit_vs_mlp_hist.json")
    if os.path.exists(hp):
        H = json.load(open(hp))
        mlp_sym = lambda p, k: r"MLP log+$\beta$" if k == "MLP log" and p.startswith("ee_tt") else k
        for p in PROCS:
            ex = [e for e in H if e["proc"] == p and "edges" in e]
            panels = ps.panels(len(ex))
            for (fig, ax), e in zip(panels, ex):
                models = [dict(label=mlp_sym(p, k), edges=e["edges"], truth=e["truth"], pred=e["pred"][k], n=e["n"],
                               style=dict(color=sty[mlp_sym(p, k)][0], ls=sty[mlp_sym(p, k)][2]))
                          for k in ("factor", "MLP log", "scratch", "MLP raw") if k in e["pred"]]
                ps.true_pred_hist(ax, models, ratio=(0.1, 10), log_ratio=True,
                                               xlabel=r"$\log_{10}(|\mathcal{M}|^2/\mathrm{GeV}^{-2})$",
                                               ratio_label=r"$\frac{\rm pred}{\rm true}$")
                ps.process_label(ax, PROC[PROCS[p]])
                ps.shared_legend(fig, ax, ncol=3)
            written += ps.save_panels(panels, os.path.join(figs, f"kitvm_dist_{p}"))
        # (6) the t t~ threshold family: 10^(pred - truth) per bin of beta, median and 16-84% band
        for e in [e for e in H if "ratio" in e]:
            p = e["proc"]
            fig, ax = ps.figure()
            x = np.array(e["x"])
            for k in ("factor", "MLP log", "scratch", "MLP raw"):
                if k not in e["ratio"]:
                    continue
                q = np.array(e["ratio"][k])
                sym = mlp_sym(p, k)
                col, mk, ls = sty[sym]
                ax.plot(x, q[:, 1], marker=mk, ls=ls, color=col, label=sym)
                ax.fill_between(x, q[:, 0], q[:, 2], color=col, alpha=0.2, lw=0)
                print(f"BETA {sym}: median pred/true at beta {x[0]:.2g} {q[0, 1]:.3g}, at {x[-1]:.2g} {q[-1, 1]:.3g}")
            ax.axhline(1, color="k", lw=0.8)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(XLAB["threshold_beta"]); ax.set_ylabel(r"$|\mathcal{M}|^2_{\rm pred}/|\mathcal{M}|^2_{\rm true}$")
            ps.process_label(ax, PROC[PROCS[p]])
            ps.shared_legend(fig, ax, ncol=3)
            written.append(ps.save(fig, os.path.join(figs, f"kitvm_ratio_beta_{p}")))
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
