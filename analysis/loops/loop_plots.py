"""Figures of the synthetic one-loop investigation (D44; docs/results.tex sec:syn-loops).
  loop_validation  (a) fraction of negative events against the spread of log|x|, (b) Spearman correlation of the
                   signed log with s and with t/s, real probes against the synthetic loop pools; (c) the standardised
                   signed log, the four real pools and the synthetic ones pooled; (d) V/|M0|^2 against sqrt(s) for four
                   synthetic processes. Input: the CPU job's results_loop_val.json (analysis/loops/loop_validate.py).
    python analysis/loops/loop_plots.py validation <results_loop_val.json>
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

FIGS = os.path.join(ROOT, "analysis", "loops", "figs")
REAL_LAB = {"ee_dd_nlo": r"$e^+e^-\to d\bar d$ (QCD)", "ee_bb_nlo": r"$e^+e^-\to b\bar b$ (QCD)",
            "ee_dd_ew_nlo": r"$e^+e^-\to d\bar d$ (EW)", "ee_ttbar_nlo_thr": r"$e^+e^-\to t\bar t$ (thr.)"}
REAL_C = {"ee_dd_nlo": ps.C.blue, "ee_bb_nlo": ps.C.sky, "ee_dd_ew_nlo": ps.C.vermillion,
          "ee_ttbar_nlo_thr": ps.C.orange}
SHOW = ["synl_00000", "synl_00011", "synl_00015", "synl2_00004"]


def validation(path):
    R = json.load(open(path))
    real, syn = R["real"], R["synthetic"]
    fig, axs = ps.figure(ncols=2, nrows=2)
    (a, b), (c, d) = axs
    for r in real:
        a.plot(r["log_abs_std"], r["neg_frac"], "D", color=REAL_C[r["name"]], label=REAL_LAB[r["name"]])
        b.plot(r["rho_sl_s"], r["rho_sl_t"], "D", color=REAL_C[r["name"]])
    for typ, mk, col in (("QCD", "o", ps.C.green), ("EW", "s", ps.C.purple)):
        sel = [x for x in syn if x["type"] == typ]
        a.plot([x["log_abs_std"] for x in sel], [x["neg_frac"] for x in sel], mk, color=col, mfc="none",
               label=f"synthetic, {typ} loop")
        b.plot([x["rho_sl_s"] for x in sel], [x["rho_sl_t"] for x in sel], mk, color=col, mfc="none")
    a.set_xlabel(r"std of $\log|x|$")
    a.set_ylabel("fraction $x<0$")
    b.set_xlabel(r"$\rho_S(\mathrm{slog}\,x,\ s)$")
    b.set_ylabel(r"$\rho_S(\mathrm{slog}\,x,\ t/s)$")
    e = np.asarray(R["edges_sl"]); mid = 0.5 * (e[1:] + e[:-1]); w = e[1] - e[0]
    for r in real:
        h = np.asarray(r["hist_sl"], float); c.step(mid, h / h.sum() / w, where="mid", color=REAL_C[r["name"]])
    hs = np.sum([np.asarray(x["hist_sl"], float) / np.sum(x["hist_sl"]) for x in syn], axis=0) / len(syn)
    c.step(mid, hs / w, where="mid", color=ps.C.grey, label="synthetic, all")
    c.set_xlabel(r"$\mathrm{slog}\,x=\mathrm{sgn}(x)\log(1+|x|/x_{1\%})$")
    c.set_ylabel("density")
    for nm, col in zip(SHOW, (ps.C.green, ps.C.purple, ps.C.orange, ps.C.blue)):
        x = next((y for y in syn if y["name"] == nm), None)
        if x is None:
            continue
        d.plot(x["sub"]["sqrts"], x["v_over_b"], ".", color=col, ms=1.5,  # point size: a dense scatter
               label=nm.replace("_", r"\_") + f" ({x['type']})", rasterized=True)
    d.set_xlabel(r"$\sqrt{s}$ [GeV]")
    d.set_ylabel(r"$V/|\mathcal{M}_0|^2$")
    ps.legend(a, "upper left")
    ps.legend(c, "upper left")
    ps.legend(d, "lower right", markerscale=6)
    os.makedirs(FIGS, exist_ok=True)
    ps.save(fig, os.path.join(FIGS, "loop_validation"))
    print(f"{'pool':20s} {'type':4s} {'neg':>5s} {'std':>5s} {'rho_s':>6s} {'rho_t':>6s} {'rhoneg_s':>8s}")
    for x in real + syn:
        rn = x.get("rho_neg_s")
        print(f"{x['name']:20s} {x.get('type', 'real'):4s} {x['neg_frac']:5.2f} {x['log_abs_std']:5.2f} "
              f"{x['rho_sl_s']:+6.2f} {x['rho_sl_t']:+6.2f} {('%+8.2f' % rn) if rn is not None else '       -'}")


if __name__ == "__main__":
    {"validation": validation}[sys.argv[1]](*sys.argv[2:])
