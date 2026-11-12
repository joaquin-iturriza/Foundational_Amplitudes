#!/usr/bin/env python3
"""Bakar's KIT benchmark slides (hardps_benchmark_slides.pdf, slides 5-9) redrawn with our transformers beside his
reference MLP (D415). Eval-only: existing predictions.h5 / kit_result.json of each run, the KIT files' truth.

    python tools/kit_bakar_figs.py collect --out analysis/kit_benchmark/data/kit_bakar_figs.json   (reads CC)
    python tools/kit_bakar_figs.py plot analysis/kit_benchmark/data/kit_bakar_figs.json --figs analysis/kit_benchmark/figs/bakar

Slides: 5 tt virtual V/B median per beta bin (with C_F pi alpha_s / (2 beta)); 6 uug median log10|M|^2 per soft x
and collinear y bin; 7 uugg g->gg azimuth (one event at y = 1e-5, rms over events of h = 2<log10|M|^2 cos 2phi>_phi
per y); 8 log-MSE per stratum, bulk dotted (uugg soft, tt virtual threshold); 9 the summary table (log-MSE bulk and
deepest bin, fitted exponents) printed and kept in the JSON. Medians and h of his MLP come from our reruns of it
(D403 ladder arms mlp = log invariants, raw = raw invariants, seeds 0-2, median over seeds); his own evaluate.py
numbers (data/full/baseline/*.json: raw, log, log + threshold) give slides 8 and 9 for him.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

T = "runs/kit/kit_A_{p}_n100000_t8000_{tag}"
M = "runs/kit_mlp/kit_A_{p}_n100000_t14400_mlp{arm}_s{s}"
# per process: (label, run dirs); MLP arms list their three seeds (medians over seeds)
MODELS = {
    "ee_uug_me2": [("transformer", [T.format(p="ee_uug_me2", tag="softpair")]),
                   ("no frames", [T.format(p="ee_uug_me2", tag="softpair_noframes")]),
                   ("no frames, no off-shellness", [T.format(p="ee_uug_me2", tag="softpair_noframesoff")])],
    "ee_uugg_me2": [("transformer", [T.format(p="ee_uugg_me2", tag="softpair")])],
    "ee_tt_virt_finite": [("transformer", [T.format(p="ee_tt_virt_finite", tag="softpairthr")]),
                          ("no frames", [T.format(p="ee_tt_virt_finite", tag="softpairthr_noframes")]),
                          ("no frames, no off-shellness", [T.format(p="ee_tt_virt_finite", tag="softpairthr_noframesoff")])]}
THR = r"MLP, log + $\log(s_{t\bar t}-4m_t^2)$"
for _p, _m in MODELS.items():
    # the ladder's 'mlp' arm is his setup: log invariants, plus the threshold column on tt (kit_mlp_ladder.py)
    _m += [(THR if _p.startswith("ee_tt") else "MLP, log invariants", [M.format(p=_p, arm="mlp", s=s) for s in range(3)]),
           ("MLP, raw invariants", [M.format(p=_p, arm="raw", s=s) for s in range(3)])]
# his evaluate.py outputs: (label, file) per process
BASE = {"ee_uug_me2": [("MLP, raw invariants", "ee_uug_me2_raw"), ("MLP, log invariants", "ee_uug_me2_log")],
        "ee_uugg_me2": [("MLP, raw invariants", "ee_uugg_me2_raw"), ("MLP, log invariants", "ee_uugg_me2_log")],
        "ee_tt_virt_finite": [("MLP, raw invariants", "ee_tt_abs_virt_finite_raw"),
                              ("MLP, log invariants", "ee_tt_abs_virt_finite_log"),
                              (THR, "ee_tt_abs_virt_finite_logthr")]}
PER_DECADE = 5      # his slide 6: 25 points over the five decades 1e-7..1e-2; a bin's median needs >= 20 events
CF, ALPHA_S = 4.0 / 3.0, 0.118

SITE = r'''
import json, os, sys
import numpy as np, h5py
sys.path.insert(0, ".")
from tools import kit_hardps as kh
MODELS, BASE, PD = json.loads(sys.argv[1]), json.loads(sys.argv[2]), int(sys.argv[3])
out = {}
def key(proc):
    return (proc[:-4], "log10_me2") if proc.endswith("_me2") else ("ee_tt", "log10_abs_" + proc[len("ee_tt_"):])
def fam(st):
    t = st.split("_")
    while t and t[-1].replace(".", "").replace("e-", "").replace("-", "").isdigit():
        t = t[:-1]
    return "_".join(t)
def binned(x, ys, lo, hi):
    e = 10 ** np.arange(np.floor(np.log10(lo) * PD) / PD, np.log10(hi) + 1e-9, 1 / PD)
    i = np.digitize(x, e) - 1
    r = {"x": [], **{k: [] for k in ys}}
    for b in range(len(e) - 1):
        m = i == b
        if m.sum() >= 20:
            r["x"].append(float(np.sqrt(e[b] * e[b + 1])))
            for k, y in ys.items():
                r[k].append(float(np.median(y[m])))
    return r
for proc, models in MODELS.items():
    stem, tk = key(proc)
    o = out[proc] = {"bins": {}, "result": {}, "base": {}, "azimuth": {}, "vb": None}
    with h5py.File(kh.h5_path(stem), "r") as T:
        preds = {}
        for lab, dirs in models:
            ps = [f"{d}/kit_benchmark/predictions.h5" for d in dirs]
            ps = [p for p in ps if os.path.exists(p)]
            if not ps:
                continue
            preds[lab] = [h5py.File(p, "r") for p in ps]
            o["result"][lab] = [json.load(open(os.path.dirname(p) + "/kit_result.json")) for p in ps]
        fams = sorted({fam(st) for st in T["hard_test"] if not st.startswith("azimuth")})
        for f in fams:
            sts = [st for st in T["hard_test"] if fam(st) == f]
            x = np.concatenate([T["hard_test"][st]["singular_var"][:] for st in sts])
            ys = {"truth": np.concatenate([T["hard_test"][st][tk][:] for st in sts])}
            for lab, Ps in preds.items():
                for j, P in enumerate(Ps):
                    ys[f"{lab}#{j}"] = np.concatenate([P["hard_test"][st][tk][:] for st in sts])
            o["bins"][f] = binned(x, ys, x.min(), x.max())
        if stem == "ee_tt":       # slide 5: V/B against beta, hard threshold strata and bulk_test, beta from the momenta
            sts = [("hard_test", st) for st in T["hard_test"] if st.startswith("threshold")] + \
                  [("bulk_test", st) for st in T["bulk_test"]]
            def beta(mom):
                pt = mom[:, 2] + mom[:, 3]
                s = pt[:, 0] ** 2 - (pt[:, 1:] ** 2).sum(1)
                m2 = mom[:, 2, 0] ** 2 - (mom[:, 2, 1:] ** 2).sum(1)
                return np.sqrt(np.clip(1 - 4 * m2 / s, 1e-30, None))
            b = np.concatenate([beta(T[sp][st]["momenta"][:]) for sp, st in sts])
            born = np.concatenate([T[sp][st]["log10_abs_born"][:] for sp, st in sts])
            ys = {"truth": 10 ** (np.concatenate([T[sp][st][tk][:] for sp, st in sts]) - born)}
            for lab, Ps in preds.items():
                for j, P in enumerate(Ps):
                    ys[f"{lab}#{j}"] = 10 ** (np.concatenate([P[sp][st][tk][:] for sp, st in sts]) - born)
            o["vb"] = binned(b, ys, 1e-4, 1.0)
        for st in [s for s in T["hard_test"] if s.startswith("azimuth")]:   # slide 7
            g = T["hard_test"][st]
            phi, grp = g["phi"][:], g["group"][:]
            order = np.lexsort((phi, grp))
            n = len(np.unique(grp))
            P_ = phi[order].reshape(n, -1)
            def h(v):
                v = v[order].reshape(n, -1)
                return 2 * np.mean(v * np.cos(2 * P_), 1), v - v.mean(1, keepdims=True)
            ht, dt = h(g[tk][:])
            a = {"y": float(st.rsplit("_", 1)[-1]), "rms": {"truth": float(np.sqrt(np.mean(ht ** 2)))}}
            k = int(np.argmin(np.abs(np.abs(ht) - np.median(np.abs(ht)))))
            a["event"] = {"phi": P_[k].tolist(), "truth": dt[k].tolist()}
            for lab, Ps in preds.items():
                hs = [h(P["hard_test"][st][tk][:]) for P in Ps]
                a["rms"][lab] = [float(np.sqrt(np.mean(x[0] ** 2))) for x in hs]
                a["event"][lab] = [x[1][k].tolist() for x in hs]
            o["azimuth"][st] = a
    bd = os.path.join(os.path.dirname(kh.kit_dir().rstrip("/")), "full", "baseline")
    bd = bd if os.path.isdir(bd) else os.path.join(kh.kit_dir(), "baseline")
    for lab, f in BASE.get(proc, []):
        o["base"][lab] = json.load(open(os.path.join(bd, f + ".json")))
print("KITBAKAR " + json.dumps(out))
'''


def collect(out):
    from analysis.common.readers import site_python
    import shlex
    d = site_python(SITE, [shlex.quote(json.dumps(MODELS)), shlex.quote(json.dumps(BASE)), str(PER_DECADE)],
                    "KITBAKAR", timeout=3600)
    json.dump(d, open(out, "w"))
    print(f"-> {out}")


def seeds(d, lab):
    """the per-seed series of a model label in a binned dict, median over seeds"""
    ks = [k for k in d if k.split("#")[0] == lab]
    return np.median([d[k] for k in ks], 0) if ks else None


def plot(path, figs):
    import plot_style as ps
    ps.use()
    os.makedirs(figs, exist_ok=True)
    d = json.load(open(path))
    labs = ["transformer", "no frames", "no frames, no off-shellness", "MLP, log invariants", "MLP, raw invariants"]
    style = dict(zip(labs, [("o-", ps.C.vermillion), ("v-", ps.C.purple), ("D-", ps.C.orange),
                            ("o-", ps.C.blue), ("s-", ps.C.green)]))
    style[THR] = ("^-", ps.C.sky)
    labs = labs + [THR]
    present = lambda dd: [lab for lab in labs if any(k.split("#")[0] == lab for k in dd)]

    # slide 5: V/B median per beta bin
    vb = d["ee_tt_virt_finite"]["vb"]
    fig, ax = ps.figure()
    x = np.array(vb["x"])
    ax.plot(x, CF * np.pi * ALPHA_S / (2 * x), "k--", lw=1, label=r"$C_F\pi\alpha_s/(2\beta)$")
    ax.plot(x, vb["truth"], "o", color="k", ms=3, label="MadLoop truth")
    for lab in present(vb):
        mk, c = style[lab]
        ax.plot(x, seeds(vb, lab), mk, color=c, ms=3, label=lab)
    ax.axvspan(0.2, 1.0, color=ps.C.grey, alpha=0.2, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"top velocity $\beta$"); ax.set_ylabel(r"$V/B$ (median per bin)")
    ps.legend_strip(ax, os.path.join(figs, "slide5_tt_coulomb_legend"), ncol=2)   # a legend box covers the curves
    print("FIGURE", ps.save(fig, os.path.join(figs, "slide5_tt_coulomb")))

    # slide 6: uug median log10|M|^2 per soft x and collinear y bin
    b = d["ee_uug_me2"]["bins"]
    panels = ps.panels(2)
    for (fig, ax), f, xl in zip(panels, ("soft_x", "coll_y"), (r"soft gluon: $x=2E_g/\sqrt{s}$",
                                                              r"collinear: $y=s_{qg}/s$ ($q$ or $\bar q$)")):
        x = np.array(b[f]["x"])
        ax.plot(x, b[f]["truth"], "-", color="k", lw=2, label="MadGraph truth")
        for lab in present(b[f]):
            mk, c = style[lab]
            ax.plot(x, seeds(b[f], lab), mk, color=c, ms=3, label=lab)
        ax.set_xscale("log"); ax.set_xlabel(xl); ax.set_ylabel(r"median $\log_{10}(|\mathcal{M}|^2/\mathrm{GeV}^{-2})$")
        if f == "soft_x":
            ps.legend_strip(ax, os.path.join(figs, "slide6_uug_limits_legend"), ncol=3)
    ps.save_panels(panels, os.path.join(figs, "slide6_uug_limits"))

    # slide 7: uugg g->gg azimuth
    az = d["ee_uugg_me2"]["azimuth"]
    panels = ps.panels(2)
    (f1, a1), (f2, a2) = panels
    ev = next(v for v in az.values() if abs(v["y"] - 1e-5) < 1e-9)["event"]
    a1.plot(ev["phi"], ev["truth"], "-", color="k", lw=2, label="MadGraph truth")
    for lab in [l for l in labs if l in ev]:
        mk, c = style[lab]
        a1.plot(ev["phi"], np.median(ev[lab], 0), mk, color=c, ms=2, label=lab)
    a1.set_xticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    a1.set_xlabel(r"azimuth $\varphi$ of the $gg$ pair")
    a1.set_ylabel(r"$\log_{10}|\mathcal{M}|^2-$ mean over $\varphi$")
    rows = sorted(az.values(), key=lambda v: v["y"])
    y = [v["y"] for v in rows]
    a2.plot(y, [v["rms"]["truth"] for v in rows], "o-", color="k", label="MadGraph truth")
    for lab in [l for l in labs if l in rows[0]["rms"]]:
        mk, c = style[lab]
        a2.plot(y, [np.median(v["rms"][lab]) for v in rows], mk, color=c, label=lab)   # true values, no floor
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel(r"collinearity $y=s_{gg}/s$"); a2.set_ylabel(r"rms over events of $h$")
    ps.legend_strip(a2, os.path.join(figs, "slide7_uugg_azimuth_legend"), ncol=4)
    ps.save_panels(panels, os.path.join(figs, "slide7_uugg_azimuth"))

    # slide 8: log-MSE per stratum, bulk dotted (ours from kit_result.json, his from his evaluate.py files)
    panels = ps.panels(2)
    for (fig, ax), proc, f, xl in zip(panels, ("ee_uugg_me2", "ee_tt_virt_finite"), ("soft_x", "threshold_beta"),
                                      (r"soft gluon in $u\bar ugg$: $x=2E_g/\sqrt{s}$",
                                       r"$t\bar t$ virtual at threshold: $\beta$")):
        for lab, rows, his in table_rows(d[proc]):
            mk, c = style.get(lab.replace(" (rerun)", ""), ("o-", ps.C.grey))
            dash = "--" if his else "-"
            st = sorted((k, v) for k, v in rows.items() if k.startswith(f))
            xs = [np.sqrt(np.prod(edges(k))) for k, _ in st]
            ax.plot(xs, [v for _, v in st], marker=mk[0], ls=dash, color=c, ms=3, label=lab)
            if "bulk" in rows:
                ax.axhline(rows["bulk"], color=c, ls=":", lw=1)
        lo = min([r["bulk"] for _, r, _ in table_rows(d[proc]) if "bulk" in r] + [ax.get_ylim()[0]])
        ax.set_yscale("log"); ax.set_ylim(lo / 3, None)     # the bulk lines are inside the axes, as on his slide
        ax.set_xscale("log"); ax.set_xlabel(xl); ax.set_ylabel("log-MSE per bin")
        # eight curves: the legend gets its own strip per panel (slide8_logmse_<panel>_legend)
        ps.legend_strip(ax, os.path.join(figs, f"slide8_logmse_{'ab'[f != 'soft_x']}_legend"), ncol=2)
    ps.save_panels(panels, os.path.join(figs, "slide8_logmse"))

    # slide 9: the summary table
    print(summary(d))


def edges(stratum):
    t = stratum.split("_")
    return [float(t[-2]), float(t[-1])]


def metrics(res):
    """{stratum or 'bulk': log-MSE} and {family: fitted exponent} of one evaluate.py result"""
    m = res.get("strata", res)
    lm = {}
    for k, v in m.items():
        if isinstance(v, dict) and "logmse" in v and "/" in k:
            sp, st = k.split("/")[:2]
            lm["bulk" if sp == "bulk_test" else st] = v["logmse"]
    return lm, {k.split("/")[0]: v["pred"] for k, v in res.get("slopes", {}).items()}


def table_rows(e):
    """(label, {stratum: log-MSE}, his) per model: ours as the median over seeds of kit_result.json (solid; our reruns of
    his MLP marked '(rerun)'), his evaluate.py files under his labels (drawn dashed)"""
    out = []
    for lab, rs in e["result"].items():
        ms = [metrics(r)[0] for r in rs]
        out.append((lab + (" (rerun)" if lab in e["base"] else ""),
                    {k: float(np.median([m[k] for m in ms if k in m])) for k in ms[0]}, False))
    for lab, r in e["base"].items():
        out.append((lab, metrics(r)[0], True))
    return out


def summary(d):
    deep = {"ee_uug_me2": ["soft_x_1e-07_1e-06", "coll_y_1e-07_1e-06"],
            "ee_uugg_me2": ["soft_x_1e-07_1e-06", "coll_qg_y_1e-07_1e-06", "coll_gg_y_1e-07_1e-06"],
            "ee_tt_virt_finite": ["threshold_beta_0.0001_0.001", "sudakov_sqrts_15000_50000"]}
    lines, rows = ["| process | model | bulk | " + " | ".join(["deepest bins (log-MSE; exponent)"]) + " |", "|---|---|---|---|"], []
    for proc, sts in deep.items():
        e = d[proc]
        models = [(lab, rs) for lab, rs in e["result"].items()] + [(lab + " [his]", [r]) for lab, r in e["base"].items()]
        for lab, rs in models:
            ms = [metrics(r) for r in rs]
            lm = lambda k: float(np.median([m[0][k] for m in ms if k in m[0]])) if any(k in m[0] for m in ms) else None
            ex = lambda f: float(np.median([m[1][f] for m in ms if f in m[1]])) if any(f in m[1] for m in ms) else None
            cells = [f"{st.rsplit('_', 2)[0]} {lm(st):.3g}; {ex(st.rsplit('_', 2)[0]) if ex(st.rsplit('_', 2)[0]) is None else round(ex(st.rsplit('_', 2)[0]), 2)}"
                     for st in sts if lm(st) is not None]
            rows.append({"process": proc, "model": lab, "bulk": lm("bulk"),
                         **{st: lm(st) for st in sts}, **{f"exp_{st.rsplit('_', 2)[0]}": ex(st.rsplit('_', 2)[0]) for st in sts}})
            lines.append(f"| {proc} | {lab} | {lm('bulk'):.2g} | " + "; ".join(cells) + " |")
    d["summary"] = rows
    return "\n".join(lines)


def main():   # dup-ok: argparse collect/plot boilerplate, every kit tool has it
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--out", required=True)
    p = sub.add_parser("plot"); p.add_argument("data"); p.add_argument("--figs", required=True)
    a = ap.parse_args()
    collect(a.out) if a.cmd == "collect" else plot(a.data, a.figs)


if __name__ == "__main__":
    main()
