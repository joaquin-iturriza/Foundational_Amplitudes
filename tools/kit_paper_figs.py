#!/usr/bin/env python3
"""The paper's KIT figures (draft Sec. 5), eval-only on existing results, drawn by Joaquin's figure rule (d2f6f84a):
absolute values on a log scale, rows ordered best to worst, and a few representative strata shown as the delta
histogram and the true against predicted distribution with a ratio panel below.

    python tools/kit_paper_figs.py collect --out analysis/kit_benchmark/data/kit_paper_examples.json   (reads CC)
    python tools/kit_paper_figs.py plot --figs <dir>          (prints every plotted number)

Sources: log-MSE, bias, 99th percentile and |M|^2-weighted error per stratum are KIT's evaluate.py outputs
(kit_result.json) as collected by tools/kit_figures.py (kit_fig_*.json) and tools/kit_bakar_figs.py (Bakar's own
evaluate.py files for his MLP); the examples are per-event histograms of predictions.h5 against the KIT truth.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "analysis/kit_benchmark/data")

# representative strata: (process, split, stratum); models: transformer (softpair inputs; beta input on tt) and
# Bakar's MLP as rerun by us (ladder arm 'mlp', seed 0 = his numpy seed)
EXAMPLES = [("ee_uug_me2", "hard_test", "coll_y_1e-07_1e-06"), ("ee_uug_me2", "bulk_test", "bulk"),
            ("ee_uugg_me2", "hard_test", "coll_qg_y_1e-07_1e-06"),
            ("ee_tt_virt_finite", "hard_test", "sudakov_sqrts_15000_50000"),
            ("ee_tt_virt_finite", "hard_test", "threshold_beta_0.0001_0.001"),
            ("ee_tt_virt_finite", "bulk_test", "bulk")]
EX_RUNS = {"ee_uug_me2": "runs/kit/kit_A_ee_uug_me2_n100000_t8000_softpair",
           "ee_uugg_me2": "runs/kit/kit_A_ee_uugg_me2_n100000_t8000_softpair",
           "ee_tt_virt_finite": "runs/kit/kit_A_ee_tt_virt_finite_n100000_t8000_softpairthr"}
MLP_RUN = "runs/kit_mlp/kit_A_{p}_n100000_t14400_mlpmlp_s0"

SITE = r'''
import json, sys
import numpy as np, h5py
sys.path.insert(0, ".")
from tools import kit_hardps as kh
EX, RUNS, MLP = json.loads(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
out = []
for proc, split, st in EX:
    stem, tk = (proc[:-4], "log10_me2") if proc.endswith("_me2") else ("ee_tt", "log10_abs_" + proc[len("ee_tt_"):])
    with h5py.File(kh.h5_path(stem), "r") as T:
        t = T[split][st][tk][:]
    p = {}
    for lab, d in (("transformer", RUNS[proc]), ("mlp", MLP.format(p=proc))):
        with h5py.File(d + "/kit_benchmark/predictions.h5", "r") as P:
            p[lab] = P[split][st][tk][:]
    dl = {k: v - t for k, v in p.items()}
    m = max(np.quantile(np.abs(v), 0.995) for v in dl.values())
    de = np.linspace(-m, m, 61)
    allv = np.concatenate([t] + list(p.values()))
    ve = np.linspace(np.quantile(allv, 0.001), np.quantile(allv, 0.999), 41)
    out.append({"proc": proc, "split": split, "stratum": st, "n": int(len(t)), "delta_edges": de.tolist(),
                "value_edges": ve.tolist(), "truth": np.histogram(t, ve)[0].tolist(),
                "pred": {k: np.histogram(v, ve)[0].tolist() for k, v in p.items()},
                "delta": {k: np.histogram(v, de)[0].tolist() for k, v in dl.items()},
                "stats": {k: {"median": float(np.median(v)), "mean": float(np.mean(v)), "mse": float(np.mean(v ** 2)),
                              "p99_abs": float(np.quantile(np.abs(v), 0.99))} for k, v in dl.items()}})
print("KITPAPER " + json.dumps(out))
'''

# rows of the extrapolation-attempts figure: (label, kit_fig file, run key), from scratch at 1e5 bulk events,
# 8000 steps, seed 42 unless the label says pretrained
U, TT = "kit_A_ee_uug_me2_n100000_t8000", "kit_A_ee_tt_virt_finite_n100000_t8000"
ATTEMPTS = {
    "ee_uug_me2": [
        ("parent inputs", "A_inputs", U),
        ("+ soft rescale", "A_inputs", U + "_softrescale"),
        (r"+ $\log s_{ij}$", "A_inputs", U + "_pairlogs"),
        (r"+ both (ref.)", "A_inputs", U + "_softpair"),
        ("ref., no frames", "A_route_uug", U + "_softpair_noframes"),
        ("ref., MLP inputs", "A_scalars_uug", U + "_softpair_scalars"),
        (r"ref., $\gamma\leq 4$", "A_gmax4_uug", U + "_softpair_gmax4"),
        (r"plain target", "A_logpoly_uug", U + "_softpair_plain"),
        (r"plain $+P_1(L)$", "A_logpolylin_uug", U + "_softpair_plain_logskiplin"),
        ("pretrained", "A_ft_lr", "kit_A_ft_ee_uug_me2_n100000_t8000_lr0p1")],
    "ee_tt_virt_finite": [
        ("parent inputs", "A_inputs_carry", TT + "_offon"),
        (r"+ rescale, $\log s_{ij}$", "A_inputs_carry", TT + "_softpair"),
        (r"+ $\log\beta^2$ (ref.)", "A_betainput", TT + "_softpairthr"),
        ("ref., no frames", "A_route_tt", TT + "_softpairthr_noframes"),
        ("ref., MLP inputs", "A_scalars_tt", TT + "_softpairthr_scalars"),
        (r"ref. $+P_1(L)$", "A_logpolylin_tt", TT + "_softpairthr_logskiplin"),
        (r"ref. $+P_2(L)$", "A_logpoly_tt", TT + "_softpairthr_logskip"),
        (r"ref., target $V\beta$", "A_vbeta", TT + "_softpairthr_vbeta"),
        ("pretr. + LO tops", "Aft_thr", "kit_A_ft_ee_tt_virt_finite_n100000_t8000_ftr5thrown"),
        ("pretr. + 1-loop tops", "Aft_thrloop", "kit_A_ft_ee_tt_virt_finite_n100000_t8000_ftr5thrloopown")]}
# (panel, process, stratum, Bakar MLP label in kit_bakar_figs.json, x label)
THR = r"MLP, log + $\log(s_{t\bar t}-4m_t^2)$"
LIMITS = [("coll", "ee_uug_me2", "coll_y_1e-07_1e-06", "MLP, log invariants",
           r"log-MSE, $u\bar ug$ collinear $y\in[10^{-7},10^{-6}]$"),
          ("soft", "ee_uug_me2", "soft_x_1e-07_1e-06", "MLP, log invariants",
           r"log-MSE, $u\bar ug$ soft $x\in[10^{-7},10^{-6}]$"),
          ("sudakov", "ee_tt_virt_finite", "sudakov_sqrts_15000_50000", THR,
           r"log-MSE, $t\bar t$ virtual $\sqrt{s}\in[15,50]$ TeV"),
          ("coulomb", "ee_tt_virt_finite", "threshold_beta_0.0001_0.001", THR,
           r"log-MSE, $t\bar t$ virtual $\beta\in[10^{-4},10^{-3}]$"),
          ("bulk_uug", "ee_uug_me2", "bulk", "MLP, log invariants", r"log-MSE, $u\bar ug$ bulk"),
          ("bulk_tt", "ee_tt_virt_finite", "bulk", THR, r"log-MSE, $t\bar t$ virtual bulk")]

# benchmark comparison against Bakar's MLP: (row label, process, stratum)
BENCH = [(r"$u\bar ug$ bulk", "ee_uug_me2", "bulk"),
         (r"$u\bar ug$ soft", "ee_uug_me2", "soft_x_1e-07_1e-06"),
         (r"$u\bar ug$ collinear", "ee_uug_me2", "coll_y_1e-07_1e-06"),
         (r"$u\bar ugg$ bulk", "ee_uugg_me2", "bulk"),
         (r"$u\bar ugg$ soft", "ee_uugg_me2", "soft_x_1e-07_1e-06"),
         (r"$u\bar ugg$ quark-gluon coll.", "ee_uugg_me2", "coll_qg_y_1e-07_1e-06"),
         (r"$u\bar ugg$ gluon-gluon coll.", "ee_uugg_me2", "coll_gg_y_1e-07_1e-06"),
         (r"$t\bar t$ virt. bulk", "ee_tt_virt_finite", "bulk"),
         (r"$t\bar t$ virt. Sudakov", "ee_tt_virt_finite", "sudakov_sqrts_15000_50000"),
         (r"$t\bar t$ virt. Coulomb", "ee_tt_virt_finite", "threshold_beta_0.0001_0.001")]
METRICS = [("logmse", "log-MSE"), ("bias", r"$|$bias$|$"), ("p99_abs", r"99th pct. of $|\Delta|$"),
           ("me2_weighted_rel_err", r"$|\mathcal{M}|^2$-weighted rel. error")]

# protocols A, A pretrained, B_bulk, B at 9e5 bulk events (parent inputs, off-shellness on), deepest bins
PROTO_ROWS = [(r"$u\bar ug$ soft", "ee_uug_me2", "soft_x_1e-07_1e-06", ""),
              (r"$u\bar ug$ collinear", "ee_uug_me2", "coll_y_1e-07_1e-06", ""),
              (r"$u\bar ugg$ soft", "ee_uugg_me2", "soft_x_1e-07_1e-06", ""),
              (r"$u\bar ugg$ quark-gluon coll.", "ee_uugg_me2", "coll_qg_y_1e-07_1e-06", ""),
              (r"$u\bar ugg$ gluon-gluon coll.", "ee_uugg_me2", "coll_gg_y_1e-07_1e-06", ""),
              (r"$t\bar t$ virt. Sudakov", "ee_tt_virt_finite", "sudakov_sqrts_15000_50000", "_offon"),
              (r"$t\bar t$ virt. Coulomb", "ee_tt_virt_finite", "threshold_beta_0.0001_0.001", "_offon"),
              (r"$t\bar t$ Born Sudakov", "ee_tt_born", "sudakov_sqrts_15000_50000", "_offon"),
              (r"$t\bar t$ Born threshold", "ee_tt_born", "threshold_beta_0.0001_0.001", "_offon")]
PROTOS = [("bulk, scratch", "kit_A_{p}_n900000_t8000{s}"), ("bulk, pretrained", "kit_A_ft_{p}_n900000_t8000{s}"),
          ("bulk continued", "kit_B_bulk_{p}_n900000_t3000{s}"), ("hard added", "kit_B_{p}_n900000_t3000{s}")]


def collect(out):
    from analysis.common.readers import site_python
    import shlex
    d = site_python(SITE, [shlex.quote(json.dumps(EXAMPLES)), shlex.quote(json.dumps(EX_RUNS)), MLP_RUN],
                    "KITPAPER", timeout=1800)
    json.dump(d, open(out, "w"))
    print(f"-> {out}")


def stratum(run, st, key="logmse"):
    """one evaluate.py number of a run's stratum ('bulk' = bulk_test)"""
    sp = "bulk_test" if st == "bulk" else "hard_test"
    k = [k for k in run["strata"] if k.startswith(f"{sp}/{st}/")]
    return run["strata"][k[0]][key] if k else None


def fig_runs(name):
    return json.load(open(os.path.join(DATA, f"kit_fig_{name}.json")))["runs"]


def dotrows(ax, rows, ps):
    """rows (label, {marker label: value}) top to bottom; markers by label"""
    style = {"transformer": ("o", ps.C.vermillion), "MLP (Bakar)": ("s", ps.C.blue), "MLP, raw (Bakar)": ("^", ps.C.green),
             "bulk, scratch": ("o", ps.C.vermillion), "bulk, pretrained": ("D", ps.C.purple),
             "bulk continued": ("v", ps.C.grey), "hard added": ("s", ps.C.blue)}
    n = len(rows)
    for i, (lab, vals) in enumerate(rows):
        for k, v in vals.items():
            if v is not None:
                mk, c = style[k]
                ax.plot([abs(v)], [n - 1 - i], mk, color=c, label=k if i == 0 or k not in [l.get_label() for l in ax.lines] else None)
    ax.set_yticks(range(n), [r[0] for r in rows][::-1])
    ax.set_xscale("log")


def save_rows(figs, base, ps):
    """ps.save, then the pdf again with a tight box: the pdf backend sets the long row labels wider than the
    layout measured and the fixed canvas cut them on the left (seen in the draft); the plot box is unchanged"""
    figs = figs if isinstance(figs, list) else [(figs, None)]
    for i, (fig, _) in enumerate(figs):
        b = base if len(figs) == 1 else f"{base}_{'abcdefgh'[i]}"
        ps.save(fig, b)
        fig.savefig(b + ".pdf", bbox_inches="tight", pad_inches=0.02)


def plot(figs):
    import plot_style as ps
    ps.use()
    os.makedirs(figs, exist_ok=True)
    bak = json.load(open(os.path.join(DATA, "kit_bakar_figs.json")))

    # 1. benchmark against Bakar's MLP, four metrics, rows by log10(transformer / MLP) of the log-MSE, best first
    def bvals(proc, st, key):
        e = bak[proc]
        mlp = THR if proc.startswith("ee_tt") else "MLP, log invariants"
        return {"transformer": float(np.median([stratum({"strata": r["strata"]}, st, key) for r in e["result"]["transformer"]])),
                "MLP (Bakar)": stratum(e["base"][mlp], st, key),
                "MLP, raw (Bakar)": stratum(e["base"]["MLP, raw invariants"], st, key)}
    order = sorted(BENCH, key=lambda b: np.log10(bvals(b[1], b[2], "logmse")["transformer"] / bvals(b[1], b[2], "logmse")["MLP (Bakar)"]))
    print("BENCH  row | metric: transformer / MLP log (Bakar) / MLP raw (Bakar) | log10(transformer/MLP)")
    panels = ps.panels(len(METRICS))
    for (fig, ax), (key, xl) in zip(panels, METRICS):
        rows = [(lab, bvals(p, st, key)) for lab, p, st in order]
        for lab, v in rows:
            print(f"BENCH {lab} | {key}: {v['transformer']:.3g} / {v['MLP (Bakar)']:.3g} / {v['MLP, raw (Bakar)']:.3g} | "
                  f"{np.log10(abs(v['transformer']) / abs(v['MLP (Bakar)'])):+.2f}")
        dotrows(ax, rows, ps)
        ax.set_xlabel(xl)
        ps.legend_strip(ax, os.path.join(figs, "kit_bench_legend"), ncol=3) if key == "logmse" else None   # a corner box covers the rows
    save_rows(panels, os.path.join(figs, "kit_bench"), ps)

    # 2. protocols at 9e5: deepest-bin log-MSE
    fb = fig_runs("B")
    rows = []
    print("PROTO  row: " + " / ".join(p for p, _ in PROTOS))
    for lab, p, st, s in PROTO_ROWS:
        v = {pl: stratum(fb[f.format(p=p, s=s)], st) if f.format(p=p, s=s) in fb else None for pl, f in PROTOS}
        rows.append((lab, v))
        print(f"PROTO {lab}: " + " / ".join("-" if x is None else f"{x:.3g}" for x in v.values()))
    fig, ax = ps.figure()
    dotrows(ax, rows, ps)
    ax.set_xlabel(r"log-MSE, deepest bin")
    ps.legend_strip(ax, os.path.join(figs, "kit_protocols_legend"), ncol=2)
    save_rows(fig, os.path.join(figs, "kit_protocols"), ps)

    # 3. extrapolation attempts: per limit, rows best to worst, Bakar's MLP as the last row
    panels = ps.panels(len(LIMITS))
    for (fig, ax), (name, proc, st, mlp, xl) in zip(panels, LIMITS):
        rows = []
        for lab, f, run in ATTEMPTS[proc]:
            rows.append((lab, stratum(fig_runs(f)[run], st)))
        rows.sort(key=lambda r: r[1])
        mv = stratum(bak[proc]["base"][mlp], st)
        print(f"ATTEMPT {name}: " + "; ".join(f"{l} {v:.3g}" for l, v in rows) + f"; MLP (Bakar) {mv:.3g}")
        n = len(rows)
        ax.plot([v for _, v in rows], range(n - 1, -1, -1), "o", color=ps.C.vermillion, label="transformer")
        ax.plot([mv], [-1], "s", color=ps.C.blue, label="MLP (Bakar)")
        ax.set_yticks(range(-1, n), ["MLP (Bakar)"] + [r[0] for r in rows][::-1])
        ax.set_xscale("log"); ax.set_xlabel(xl)
    save_rows(panels, os.path.join(figs, "kit_attempts"), ps)

    # 4. representative examples: delta histograms, and true vs predicted distributions with a ratio panel
    ex = json.load(open(os.path.join(DATA, "kit_paper_examples.json")))
    name = {("ee_uug_me2", "coll_y_1e-07_1e-06"): r"$u\bar ug$, $y\in[10^{-7},10^{-6}]$",
            ("ee_uug_me2", "bulk"): r"$u\bar ug$, bulk",
            ("ee_uugg_me2", "coll_qg_y_1e-07_1e-06"): r"$u\bar ugg$, $y_{qg}\in[10^{-7},10^{-6}]$",
            ("ee_tt_virt_finite", "sudakov_sqrts_15000_50000"): r"$t\bar t$ virt., $\sqrt{s}\in[15,50]$ TeV",
            ("ee_tt_virt_finite", "threshold_beta_0.0001_0.001"): r"$t\bar t$ virt., $\beta\in[10^{-4},10^{-3}]$",
            ("ee_tt_virt_finite", "bulk"): r"$t\bar t$ virt., bulk"}
    lab = {"transformer": "transformer", "mlp": "MLP (Bakar, rerun)"}
    col = {"transformer": ps.C.vermillion, "mlp": ps.C.blue}
    for e in ex:
        print(f"EXAMPLE {e['proc']} {e['stratum']} n={e['n']}: " + "; ".join(
            f"{k} median {s['median']:+.3f} mse {s['mse']:.3g} p99 {s['p99_abs']:.3g}" for k, s in e["stats"].items()))
    # delta histograms, two canvases of four
    for part, sel in (("a", ex[:3] + ex[5:6]), ("b", ex[3:5])):
        fig, axs = ps.figure(ncols=2, nrows=2) if len(sel) == 4 else ps.figure(ncols=2)
        for ax, e in zip(np.ravel(axs), sel):
            c = np.array(e["delta_edges"])
            for k, h in e["delta"].items():
                ax.stairs(h, c, color=col[k], label=lab[k])
            ax.set_yscale("log"); ax.set_xlabel(r"$\Delta=\log_{10}|\mathcal{M}|^2_{\rm pred}-\log_{10}|\mathcal{M}|^2_{\rm true}$")
            ax.set_ylabel("events")
            ps.process_label(ax, name[(e["proc"], e["stratum"])])
        ps.legend_strip(np.ravel(axs)[0], os.path.join(figs, f"kit_delta_{part}_legend"), ncol=2)   # corners hold the labels
        print("FIGURE", ps.save(fig, os.path.join(figs, f"kit_delta_{part}")))
    # distributions with the ratio below: per canvas two strata, top true and predicted, bottom predicted / true
    for part, sel in (("a", [ex[0], ex[3]]), ("b", [ex[2], ex[5]])):
        fig, axs = ps.figure(ncols=2, nrows=2)
        for j, e in enumerate(sel):
            c = np.array(e["value_edges"]); t = np.array(e["truth"], float)
            top, bot = axs[0][j], axs[1][j]
            top.stairs(t, c, color="k", label="truth")
            for k, h in e["pred"].items():
                top.stairs(h, c, color=col[k], label=lab[k])
                r = np.where(t > 0, np.array(h) / np.where(t > 0, t, 1), np.nan)
                bot.stairs(r, c, color=col[k])
            top.set_yscale("log"); top.set_ylabel("events")
            rr = np.concatenate([np.array(h) / t[t > 0] for h in (np.array(v)[t > 0] for v in e["pred"].values())])
            bot.set_yscale("log" if np.nanmax(rr) > 10 * max(np.nanmin(rr[rr > 0]), 1e-12) else "linear")
            bot.set_ylabel("pred / truth")
            bot.set_xlabel(r"$\log_{10}(|\mathcal{M}|^2)$")
            top.set_xlabel(r"$\log_{10}(|\mathcal{M}|^2)$")
            ps.process_label(top, name[(e["proc"], e["stratum"])])
        ps.legend_strip(axs[0][0], os.path.join(figs, f"kit_dist_{part}_legend"), ncol=3)
        print("FIGURE", ps.save(fig, os.path.join(figs, f"kit_dist_{part}")))


def main():   # dup-ok: argparse collect/plot boilerplate, every kit tool has it
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--out", required=True)
    p = sub.add_parser("plot"); p.add_argument("--figs", required=True)
    a = ap.parse_args()
    collect(a.out) if a.cmd == "collect" else plot(a.figs)


if __name__ == "__main__":
    main()
