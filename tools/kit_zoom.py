#!/usr/bin/env python3
"""KIT comparison at fixed other divergence variable, in the divergence-free bulk, with a relative error measure
(kit_benchmark step 37, D598). Eval-only: existing predictions.h5 of each run and the KIT files' truth, read on CC.

    python tools/kit_zoom.py collect     (a CPU job on CC; call again once it has ended to read it)
    python tools/kit_zoom.py plot [--figs analysis/kit_benchmark/figs/zoom]

Per event D = log10 pred - log10 truth (|M|^2; t t~ one loop: |V|). Two measures of a set of events: MSE(D) and the
mean relative error <|10^D - 1|> = <|pred - true| / true>, unweighted (the largest amplitudes do not dominate it).
(a) Bulk core: bulk_test events with every singular variable above its cut; a variable's cut is the lower edge of the
    bins above which the truth's bulk median log10|M|^2 against log10 of it has local slope above half the limit
    exponent (soft -2, collinear -1, threshold -1); t t~ also sqrt s < 1739 GeV, the first Sudakov stratum's edge.
    Both measures also in bins of each variable over the whole bulk.
(b) Hard families binned in the family's singular variable (5 bins a decade) with the other divergence variable in a
    fixed band: u u~ g collinear at x in [0.45, 0.55], soft at angle y/x in [0.28, 0.32]; u u~ g g g->gg collinear at
    energy share z in [0.33, 0.37], q->qg collinear at the collinear gluon's x in [0.25, 0.35], soft at the softest
    gluon's min_i s_ig/(s x) in [0.14, 0.17]. t t~ has no second divergence variable (at fixed beta sqrt s is fixed,
    at high sqrt s beta = 1): its families keep every event.
(c) Every error is drawn as the median and the mean of |Delta|, Delta = pred/true - 1, and as MSE(ln), the draft's
    natural log (MSE(ln) = ln(10)^2 MSE(D)); medians of log|M|^2 in ln too.
Drawn arms (D604): the draft's KIT figures' (kit_vs_mlp paper figures, names and styles from kit_figures.MODELS):
transformer + asymptotic factor, transformer (uug hp1918, uugg hp0798, t t~ hp0131), the MLP on log invariants (t t~:
plus log(s_tt - 4 m_t^2)) and on raw invariants (our reruns, seeds 0-2: median over seeds of each per-seed number).
Collected, not drawn: the emission-scale transformers (searched, and at the factor's HPs: steps 37/38).
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "analysis/kit_benchmark/data")
OUT = os.path.join(DATA, "kit_zoom.json")
PROCS = ["ee_uug_me2", "ee_uugg_me2", "ee_tt_virt_finite"]
MLP = "runs/kit_mlp/kit_A_{p}_n100000_t14400_mlpmlp_s{s}"
MLPR = "runs/kit_mlp/kit_A_{p}_n100000_t14400_mlpraw_s{s}"
RUNS = {"ee_uug_me2": {"factor": ["runs/kit/kit_A_ee_uug_me2_n100000_t1831_hpo_asym_cont"],
                       "emission": ["runs/kit_hpo_uug_T/trial_1493"],
                       "emis_m": ["runs/kit/kit_A_ee_uug_me2_n100000_t1831_hpo_emis_matched"],
                       "scratch": ["runs/kit_hpo_uug_plain/trial_1918"],
                       "MLP log": [MLP.format(p="ee_uug_me2", s=s) for s in range(3)],
                       "MLP raw": [MLPR.format(p="ee_uug_me2", s=s) for s in range(3)]},
        "ee_uugg_me2": {"factor": ["runs/kit/kit_A_ee_uugg_me2_n100000_t1831_hpo_asym_cont"],
                        "emission": ["runs/kit_hpo_uugg_T/trial_1196"],
                        "emis_m": ["runs/kit/kit_A_ee_uugg_me2_n100000_t1831_hpo_emis_matched"],
                        "scratch": ["runs/kit_hpo_uugg_plain/trial_0798"],
                        "MLP log": [MLP.format(p="ee_uugg_me2", s=s) for s in range(3)],
                        "MLP raw": [MLPR.format(p="ee_uugg_me2", s=s) for s in range(3)]},
        "ee_tt_virt_finite": {"factor": ["runs/kit/kit_A_ee_tt_virt_finite_n100000_t1831_hpo_asym_cont"],
                              "scratch": ["runs/kit_hpo_ttvirt_thr/trial_0131"],
                              "MLP thr": [MLP.format(p="ee_tt_virt_finite", s=s) for s in range(3)],
                              "MLP raw": [MLPR.format(p="ee_tt_virt_finite", s=s) for s in range(3)]}}
# family -> (singular variable, other variable, its band)
FAMS = {"ee_uug": {"coll_y": ("y", "x", (0.45, 0.55)), "soft_x": ("x", "a", (0.28, 0.32))},
        "ee_uugg": {"coll_gg_y": ("ygg", "z", (0.33, 0.37)), "coll_qg_y": ("yqg", "xg", (0.25, 0.35)),
                    "soft_x": ("x", "a", (0.14, 0.17))},
        "ee_tt": {"threshold_beta": ("beta", None, None), "sudakov_sqrts": ("sqrts", None, None)},
        # t t~ g (step 39): soft at gluon-top 1 - cos in [0.8, 1.2] (uniform top direction in the pair frame), the
        # quasi-collinear family at x in [0.25, 0.35] (generated 0.2-0.4), the pair threshold has no second variable
        "fa_ee_ttg": {"soft_x": ("x", "ctg", (0.8, 1.2)), "coll_tg": ("ctg", "x", (0.25, 0.35)),
                      "threshold_beta": ("beta", None, None)}}
# bulk variable -> the limit's exponent of |M|^2 (None: an upper cut instead)
BULKV = {"ee_uug": {"x": -2, "y": -1}, "ee_uugg": {"x": -2, "ygg": -1, "yqg": -1},
         "ee_tt": {"beta": -1, "sqrts": None}, "fa_ee_ttg": {"x": -2}}
SQRTS_MAX = 1739.0

SITE = r'''
import json, os, sys
import numpy as np, h5py
sys.path.insert(0, ".")
from tools import kit_hardps as kh
RUNS, FAMS, BULKV, SQRTS_MAX = json.loads(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3]), float(sys.argv[4])
PD, NMIN = 5, 20
def dot(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)
def variables(stem, p):
    P = p[:, 0] + p[:, 1]; s = dot(P, P); rs = np.sqrt(s)
    if stem == "ee_uug":
        q, qb, g = p[:, 2], p[:, 3], p[:, 4]
        x, y = 2 * g[:, 0] / rs, np.minimum(2 * dot(q, g), 2 * dot(qb, g)) / s
        return {"x": x, "y": y, "a": y / x}
    if stem == "ee_uugg":
        q, qb, g1, g2 = p[:, 2], p[:, 3], p[:, 4], p[:, 5]
        e = np.stack([g1[:, 0], g2[:, 0]], 1)
        ygg = 2 * dot(g1, g2) / s
        yqg = np.stack([2 * dot(a, g) / s for g in (g1, g2) for a in (q, qb)], 1)   # (q g1, qb g1, q g2, qb g2)
        gj = yqg.argmin(1) // 2                       # the gluon of the most collinear quark-gluon pair
        k = e.argmin(1)                               # the softer gluon
        xs = 2 * e.min(1) / rs
        ymin = np.where(k == 0, np.minimum(yqg[:, :2].min(1), ygg), np.minimum(yqg[:, 2:].min(1), ygg))
        return {"x": xs, "ygg": ygg, "yqg": yqg.min(1), "z": e.min(1) / e.sum(1),
                "xg": 2 * e[np.arange(len(e)), gj] / rs, "a": ymin / xs, "yqg4": yqg, "xg2": 2 * e / rs[:, None]}
    if stem == "fa_ee_ttg":   # tools/kit_ttg.py: x of the gluon, 1 - cos of its angle to the top, beta of the pair
        t, tb, g = p[:, 2], p[:, 3], p[:, 4]
        ctg = 1 - (t[:, 1:] * g[:, 1:]).sum(-1) / (np.linalg.norm(t[:, 1:], axis=1) * np.linalg.norm(g[:, 1:], axis=1))
        tt = t + tb
        return {"x": 2 * g[:, 0] / rs, "ctg": ctg, "beta": np.sqrt(np.clip(1 - 4 * dot(t, t) / dot(tt, tt), 1e-30, None))}
    t = p[:, 2]
    return {"beta": np.sqrt(np.clip(1 - 4 * dot(t, t) / s, 1e-30, None)), "sqrts": rs}
def measures(d):
    r = np.abs(10.0 ** d - 1)
    return {"n": int(d.size), "mse": float(np.mean(d ** 2)), "rel": float(np.mean(r)), "relmed": float(np.median(r)),
            "med": float(np.median(d))}
def binned(v, t, preds, lo=None):
    lv = np.log10(v)
    e = np.arange(np.floor(lv.min() * PD) / PD, lv.max() + 1.0 / PD, 1.0 / PD)
    i = np.digitize(lv, e) - 1
    r = {"lo": [], "hi": [], "x": [], "truth": [], "arms": {a: [{"pred": [], "mse": [], "rel": [], "relmed": [], "n": []} for _ in P]
                                                             for a, P in preds.items()}}
    for b in range(len(e) - 1):
        m = i == b
        if m.sum() < NMIN:
            continue
        r["lo"].append(float(10 ** e[b])); r["hi"].append(float(10 ** e[b + 1]))
        r["x"].append(float(np.median(v[m]))); r["truth"].append(float(np.median(t[m])))
        for a, P in preds.items():
            for j, q in enumerate(P):
                M = measures(q[m] - t[m])
                o = r["arms"][a][j]
                o["pred"].append(float(np.median(q[m]))); o["mse"].append(M["mse"]); o["rel"].append(M["rel"])
                o["relmed"].append(M["relmed"]); o["n"].append(M["n"])
    return r
out = {}
for proc, arms in RUNS.items():
    stem, tk = (proc[:-4], "log10_me2") if proc.endswith("_me2") else ("ee_tt", "log10_abs_" + proc[len("ee_tt_"):])
    o = out[proc] = {"missing": [], "check": {}, "fams": {}, "bulk_bins": {}, "cuts": {}}
    H = {}
    for a, dirs in arms.items():
        H[a] = []
        for d in dirs:
            f = d + "/kit_benchmark/predictions.h5"
            if os.path.exists(f):
                H[a].append(h5py.File(f, "r"))
            else:
                o["missing"].append(d)
        if not H[a]:
            del H[a]
    with h5py.File(kh.h5_path(stem), "r") as T:
        g = T["bulk_test"]["bulk"]
        V, t = variables(stem, g["momenta"][:]), g[tk][:]
        preds = {a: [P["bulk_test"]["bulk"][tk][:] for P in Ps] for a, Ps in H.items()}
        keep = np.ones(len(t), bool)
        for v, ex in BULKV[stem].items():
            b = o["bulk_bins"][v] = binned(V[v], t, preds)
            if ex is None:
                o["cuts"][v] = ["max", SQRTS_MAX]
                keep &= V[v] < SQRTS_MAX
                continue
            lx, ly = np.log10(b["x"]), np.array(b["truth"])
            sl = np.diff(ly) / np.diff(lx)
            low = np.flatnonzero(sl < ex / 2.0)
            cut = b["lo"][low[-1] + 1] if low.size else b["lo"][0]
            o["cuts"][v] = ["min", cut, sl.round(3).tolist()]
            keep &= V[v] >= cut
        o["core"] = {"n": int(keep.sum()), "frac": float(keep.mean()),
                     "arms": {a: [measures(q[keep] - t[keep]) for q in P] for a, P in preds.items()}}
        o["bulk"] = {a: [measures(q - t) for q in P] for a, P in preds.items()}
        for fam, (sv, ov, band) in FAMS[stem].items():
            sts = sorted(s for s in T["hard_test"] if s.startswith(fam + "_"))
            Vh = {}
            for s in sts:
                for k, x in variables(stem, T["hard_test"][s]["momenta"][:]).items():
                    Vh.setdefault(k, []).append(x)
            Vh = {k: np.concatenate(x) for k, x in Vh.items()}
            x = np.concatenate([T["hard_test"][s]["singular_var"][:] for s in sts])
            th = np.concatenate([T["hard_test"][s][tk][:] for s in sts])
            ph = {a: [np.concatenate([P["hard_test"][s][tk][:] for s in sts]) for P in Ps] for a, Ps in H.items()}
            if ov == "xg":   # the file's collinear pair: the quark-gluon pair whose s_qg/s is its singular_var
                j = np.argmin(np.abs(np.log(Vh["yqg4"] / x[:, None])), 1)
                Vh["yqg"], Vh["xg"] = Vh["yqg4"][np.arange(len(x)), j], Vh["xg2"][np.arange(len(x)), j // 2]
                o["check_min_pair"] = float(np.mean(j != Vh["yqg4"].argmin(1)))
            o["check"][fam] = float(np.max(np.abs(Vh[sv] / x - 1)))   # our variable against the file's singular_var
            o["fams"][fam] = {"all": binned(x, th, ph)}
            if ov is not None:
                m = (Vh[ov] >= band[0]) & (Vh[ov] <= band[1])
                o["fams"][fam]["band"] = binned(x[m], th[m], {a: [q[m] for q in P] for a, P in ph.items()})
                o["fams"][fam]["band_frac"] = float(m.mean())
                # the deepest decade of the family against the other variable: 8 equal bins over its 1-99% range
                d = x < 10 ** (np.floor(np.log10(x.min()) + 1e-9) + 1)
                w = Vh[ov][d]
                e = np.linspace(*np.quantile(w, [0.01, 0.99]), 9)
                i = np.clip(np.digitize(w, e) - 1, 0, 7)
                sc = o["fams"][fam]["scan"] = {"var": ov, "deep": [float(x[d].min()), float(x[d].max())], "x": [], "n": [],
                                               "arms": {a: [{"mse": [], "rel": [], "relmed": []} for _ in P] for a, P in ph.items()}}
                for b in range(8):
                    mb = i == b
                    if mb.sum() < NMIN:
                        continue
                    sc["x"].append(float(np.median(w[mb]))); sc["n"].append(int(mb.sum()))
                    for a, P in ph.items():
                        for j, q in enumerate(P):
                            M = measures(q[d][mb] - th[d][mb])
                            sc["arms"][a][j]["mse"].append(M["mse"]); sc["arms"][a][j]["rel"].append(M["rel"]); sc["arms"][a][j]["relmed"].append(M["relmed"])
print("KITZOOM " + json.dumps(out))
'''


def collect():
    from analysis.common.readers import site_python
    import shlex
    args = [shlex.quote(json.dumps(x)) for x in (RUNS, FAMS, BULKV)] + [str(SQRTS_MAX)]
    d = site_python(SITE, args, "KITZOOM", timeout=3000)
    json.dump(d, open(OUT, "w"))
    print(f"-> {OUT}")


def _seedmed(rows, k):
    """per-bin value k of one arm: its one run, or the median over its seeds"""
    return np.median(np.array([r[k] for r in rows], float), axis=0)


VLAB = {"x": r"soft gluon $x=2E_g/\sqrt{s}$", "y": r"collinear $y=s_{qg}/s$",
        "ygg": r"gluon-gluon collinear $y=s_{gg}/s$", "yqg": r"quark-gluon collinear $y=s_{qg}/s$",
        "beta": r"top velocity $\beta$", "sqrts": r"$\sqrt{s}$ [GeV]",
        "a": r"gluon angle $\min_i s_{ig}/(s\,x)$", "z": r"gluon energy share $z=E_{\rm soft}/(E_{\rm soft}+E_{\rm hard})$",
        "xg": r"collinear gluon $x_g=2E_g/\sqrt{s}$"}
VLAB_UUG = dict(VLAB, a=r"gluon angle $y/x$, $y=\min s_{qg}/s$", x=r"gluon energy $x=2E_g/\sqrt{s}$")
# the draft's natural log (Sec. 7, kit_vs_mlp paper figures): MSE(ln) = ln(10)^2 MSE(log10), medians ln(10) log10
LN = float(np.log(10))
SCALE = {"rel": 1.0, "relmed": 1.0, "mse": LN ** 2}
CAP = {"rel": 1e6, "relmed": 1e6, "mse": 1e200}
MEAS = {"relmed": r"median $|\Delta|$, $\Delta=|\mathcal{M}|^2_{\rm pred}/|\mathcal{M}|^2_{\rm true}-1$",
        "rel": r"mean $|\Delta|$, $\Delta=|\mathcal{M}|^2_{\rm pred}/|\mathcal{M}|^2_{\rm true}-1$",
        "mse": r"MSE$(\log|\mathcal{M}|^2)$"}
MEAS_TT = {"relmed": r"median $|\Delta|$, $\Delta=V_{\rm pred}/V_{\rm true}-1$",
           "rel": r"mean $|\Delta|$, $\Delta=V_{\rm pred}/V_{\rm true}-1$", "mse": r"MSE$(\log|V|)$"}
CUT = "divergence-free cut"
# the draft's arms in its legend order (kit_vs_mlp.PAPER_ORDER), D604: the emission-scale arms are collected, not drawn
ORDER = ["factor", "scratch", "MLP log", "MLP thr", "MLP raw"]
ORDER_EMIS = ["factor", "emis_m", "emission", "scratch"]   # the factor-vs-emission-scale figures (steps 37/38)


def _arms(p):
    from tools.kit_figures import model_style
    return {a: model_style(a) for a in ORDER if a in RUNS[p]}


def _panels(ps, axs, rows, xlab, ylab, proc, logx=True, cap=1e200):
    """one panel per (x, {arm: y}) row, every arm in ORDER's style"""
    from matplotlib.ticker import NullFormatter
    from tools.kit_figures import clip_plot, off_entry
    from tools.kit_vs_mlp import PAPER_OFF
    for ax, (x, ys, extra) in zip(axs, rows):
        fin = [v for a in ys["y"] for v in np.ravel(ys["y"][a]) if np.isfinite(v)]
        # above `cap` a value sits on the top edge, hollow (the draft's convention for off-scale medians): |Delta| beyond
        # 1e6 (an overshoot by six decades, up to 10^231 for the raw-invariant MLP at 45 TeV) and any overflow
        hi = min(10 * max(fin), cap) if fin else 1.0
        off = False
        for a, (name, c, mk, ls) in ys["sty"].items():
            if a in ys["y"]:
                off |= clip_plot(ax, x, ys["y"][a], 0, hi, None, color=c, marker=mk, ls=ls, label=name)
        if off:
            off_entry(ax, PAPER_OFF)
        if extra:
            extra(ax)
        if logx:
            ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yscale("log")
        ax.set_xlabel(xlab(ax)); ax.set_ylabel(ylab)
        ps.process_label(ax, proc, loc="upper right")


def plot(figs):
    import plot_style as ps
    from matplotlib.ticker import NullFormatter
    from tools.kit_figures import XLAB, model_style
    from tools.kit_vs_mlp import PROCESS, _grid, _legend
    os.makedirs(figs, exist_ok=True)
    d = json.load(open(OUT))
    written = []
    for p in PROCS:
        o, sty = d[p], _arms(p)
        meas = MEAS_TT if p.startswith("ee_tt") else MEAS
        vl = VLAB_UUG if p == "ee_uug_me2" else VLAB
        print(f"{p}: missing {o['missing']}; variable check {o['check']}; cuts "
              + "; ".join(f"{v} {c[0]} {c[1]:.3g}" for v, c in o["cuts"].items())
              + f"; core {o['core']['n']} events ({100 * o['core']['frac']:.1f}% of the bulk)")
        for a, rows in o["core"]["arms"].items():
            b = o["bulk"][a]
            print(f"  CORE {p} {a}: rel {_seedmed(rows, 'rel'):.3g} (median {_seedmed(rows, 'relmed'):.3g}) "
                  f"mse {_seedmed(rows, 'mse'):.3g} | whole bulk rel {_seedmed(b, 'rel'):.3g} mse {_seedmed(b, 'mse'):.3g}")
        fams = list(o["fams"])
        key = lambda f: "band" if "band" in o["fams"][f] else "all"
        for m, ylab in meas.items():
            # (a) the bulk binned in each singular variable, the cut marked
            vs = list(o["bulk_bins"])
            fig, axs, spare = _grid(len(vs))
            rows = []
            for v in vs:
                b = o["bulk_bins"][v]
                cut = o["cuts"][v][1]
                rows.append((b["x"], {"sty": sty, "y": {a: SCALE[m] * _seedmed(r, m) for a, r in b["arms"].items()}},
                             lambda ax, cut=cut: ax.axvline(cut, color="0.5", ls=":", label=CUT)))
            _panels(ps, axs, rows, lambda ax, it=iter(vs): vl[next(it)], ylab, PROCESS[p], cap=CAP[m])
            _legend(axs, spare, PROCESS[p], (0, "upper left"))
            written.append(ps.save(fig, os.path.join(figs, f"kitzoom_bulk_{m}_{p}")))
            # (b, c) every family per bin of its singular variable, the other variable in its band
            fig, axs, spare = _grid(len(fams))
            rows = []
            for f in fams:
                b = o["fams"][f][key(f)]
                ys = {a: SCALE[m] * _seedmed(r, m) for a, r in b["arms"].items()}
                rows.append((b["x"], {"sty": sty, "y": ys}, None))
                for a, y in ys.items():
                    print(f"  {m.upper()} {p} {f} {key(f)} {a}: " + " ".join(f"{x:.2g}:{v:.3g}" for x, v in zip(b["x"], y)))
            _panels(ps, axs, rows, lambda ax, it=iter(fams): XLAB.get(next(it), ""), ylab, PROCESS[p], cap=CAP[m])
            _legend(axs, spare, PROCESS[p], (0, "upper right"))
            written.append(ps.save(fig, os.path.join(figs, f"kitzoom_fam_{m}_{p}")))
            # the deepest decade of each family against its other variable (the soft-collinear corners)
            sf = [f for f in fams if "scan" in o["fams"][f]]
            if sf:
                fig, axs, spare = _grid(len(sf))
                rows = []
                for f in sf:
                    sc = o["fams"][f]["scan"]
                    ys = {a: SCALE[m] * _seedmed(r, m) for a, r in sc["arms"].items()}
                    rows.append((sc["x"], {"sty": sty, "y": ys}, None))
                    for a, y in ys.items():
                        print(f"  SCAN {m} {p} {f} deepest {sc['deep'][0]:.1e}-{sc['deep'][1]:.1e} vs {sc['var']} {a}: "
                              + " ".join(f"{x:.2g}:{v:.3g}" for x, v in zip(sc["x"], y)))
                _panels(ps, axs, rows, lambda ax, it=iter(sf): vl[o["fams"][next(it)]["scan"]["var"]], ylab, PROCESS[p],
                        logx=False, cap=CAP[m])
                _legend(axs, spare, PROCESS[p], (0, "upper left"))
                written.append(ps.save(fig, os.path.join(figs, f"kitzoom_scan_{m}_{p}")))
        # the medians at fixed other variable
        if any(key(f) == "band" for f in fams):
            fig, axs, spare = _grid(len(fams))
            for ax, f in zip(axs, fams):
                b = o["fams"][f][key(f)]
                name, c, _, ls = model_style("truth")
                ax.plot(b["x"], LN * np.asarray(b["truth"]), color=c, ls=ls, label=name)
                for a, (name, c, mk, ls) in sty.items():
                    if a in b["arms"]:
                        ax.plot(b["x"], LN * _seedmed(b["arms"][a], "pred"), color=c, marker=mk, ls=ls, label=name)
                ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_xlabel(XLAB.get(f, f)); ax.set_ylabel(r"median $\log|\mathcal{M}|^2$")
                ps.process_label(ax, PROCESS[p], loc="upper right")
            _legend(axs, spare, PROCESS[p], (0, "lower left"))
            written.append(ps.save(fig, os.path.join(figs, f"kitzoom_median_{p}")))
    # the draft's median figure (kit_vs_mlp kitvm_paper_median: u u~ g collinear and soft, t t~ high energy in TeV and
    # threshold, whole strata) with median |Delta| in place of the median log|M|^2 (D604)
    fig, axs = ps.figure(ncols=2, nrows=2)
    axs = list(np.ravel(axs))
    for i, (p, fs) in enumerate((("ee_uug_me2", ("coll_y", "soft_x")), ("ee_tt_virt_finite", ("sudakov_sqrts", "threshold_beta")))):
        rows = []
        for f in fs:
            b = d[p]["fams"][f]["all"]
            sx = 1e-3 if f == "sudakov_sqrts" else 1.0
            rows.append((sx * np.asarray(b["x"]), {"sty": _arms(p), "y": {a: _seedmed(r, "relmed") for a, r in b["arms"].items()}}, None))
        labs = iter(fs)
        _panels(ps, axs[2 * i:2 * i + 2], rows, lambda ax: (lambda f: r"$\sqrt{s}$ [TeV]" if f == "sudakov_sqrts" else XLAB[f])(next(labs)),
                (MEAS_TT if p.startswith("ee_tt") else MEAS)["relmed"], PROCESS[p], cap=CAP["relmed"])
    axs[2].set_xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    from tools.kit_vs_mlp import LEG_PT, PAPER_OFF
    for i in (1, 3):   # one legend per process row, as the draft's
        hl = {}
        for a in axs[i - 1:i + 1]:
            for h, l in zip(*a.get_legend_handles_labels()):
                hl.setdefault(l, h)
        if PAPER_OFF in hl:
            hl[PAPER_OFF] = hl.pop(PAPER_OFF)
        ps.legend(axs[i], "lower left", handles=list(hl.values()), labels=list(hl), fontsize=LEG_PT)
    written.append(ps.save(fig, os.path.join(figs, "kitzoom_paper_relmed")))
    # D598/D600: factor vs emission scales, at the factor's HPs (step 38) and at their own search's (step 37), the plain
    # transformer beside; every family at fixed other variable, median |Delta| and MSE(ln)
    from tools.kit_figures import model_style as _ms
    for p in ("ee_uug_me2", "ee_uugg_me2"):
        o = d[p]
        sty = {a: _ms(a) for a in ORDER_EMIS if a in RUNS[p]}
        fams = list(o["fams"])
        for m in ("relmed", "mse"):
            fig, axs, spare = _grid(len(fams))
            rows = [(o["fams"][f]["band"]["x"], {"sty": sty, "y": {a: SCALE[m] * _seedmed(r, m) for a, r in
                     o["fams"][f]["band"]["arms"].items() if a in sty}}, None) for f in fams]
            _panels(ps, axs, rows, lambda ax, it=iter(fams): XLAB.get(next(it), ""), MEAS[m], PROCESS[p], cap=CAP[m])
            _legend(axs, spare, PROCESS[p], (0, "upper right"))
            written.append(ps.save(fig, os.path.join(figs, f"kitzoom_emis_{m}_{p}")))
        for a in sty:
            if a in o["core"]["arms"]:
                print(f"  EMIS {p} {a}: core relmed {_seedmed(o['core']['arms'][a], 'relmed'):.3g} mse(ln) "
                      f"{LN ** 2 * _seedmed(o['core']['arms'][a], 'mse'):.3g} | bulk mse(ln) {LN ** 2 * _seedmed(o['bulk'][a], 'mse'):.3g}")
    # the core, every process: both measures side by side
    fig, (a1, a2) = ps.figure(ncols=2)
    for k in ORDER:
        name, c, mk, _ = model_style(k)
        for ax, m in ((a1, "relmed"), (a2, "mse")):
            pts = [(i, SCALE[m] * _seedmed(d[p]["core"]["arms"][k], m)) for i, p in enumerate(PROCS) if k in d[p]["core"]["arms"]]
            if pts:
                ax.plot(*zip(*pts), ls="", color=c, marker=mk, label=name)
    for ax, lab in ((a1, MEAS["relmed"]), (a2, MEAS["mse"])):
        ax.set_yscale("log"); ax.set_xticks(range(len(PROCS)))
        ax.set_xticklabels([r"$u\bar ug$", r"$u\bar ugg$", r"$t\bar t$ 1L"]); ax.set_xlim(-0.5, len(PROCS) - 0.5)
        ax.set_ylabel(lab)
    ps.legend(a1, "upper left", fontsize=ps.LEGEND_MIN_PT)
    ps.make_room(a1)
    written.append(ps.save(fig, os.path.join(figs, "kitzoom_core")))
    for w in written:
        print("FIGURE", w)


def stability():
    """the largest validation jump of every KIT run with a mu-MSE curve (kit_curves*.json, 50 evaluations): J =
    max_t val_t / min_{s<=t} val_s, where it sits, and the final over the best value. NLL curves (negative values)
    are skipped: their ratios mean nothing."""
    import glob
    rows = {}
    for f in sorted(glob.glob(os.path.join(DATA, "kit_curves*.json"))):
        d = json.load(open(f))
        for k, r in (d.items() if isinstance(d, dict) else []):
            if not isinstance(r, dict) or "val" not in r:
                continue
            v = np.array([x for x in r["val"] if x is not None], float)
            if len(v) < 10 or (v <= 0).any():
                continue
            q = v / np.minimum.accumulate(v)
            rows[k] = (os.path.basename(f), len(v), float(q.max()), int(q.argmax()), float(v.min()), float(v[-1] / v.min()))
    for k, (f, n, j, at, best, last) in sorted(rows.items()):
        print(f"JUMP {k:75s} {j:8.2f}x at eval {at:2d}/{n}  best {best:.2e}  last/best {last:.2f}  ({f})")
    json.dump(rows, open(os.path.join(DATA, "kit_stability.json"), "w"), indent=0)


def main():   # dup-ok: argparse collect/plot boilerplate, every kit tool has it
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    sub.add_parser("stability")
    p = sub.add_parser("plot"); p.add_argument("--figs", default=os.path.join(ROOT, "analysis/kit_benchmark/figs/zoom"))
    a = ap.parse_args()
    {"collect": collect, "stability": stability}.get(a.cmd, lambda: plot(a.figs))()


if __name__ == "__main__":
    main()
