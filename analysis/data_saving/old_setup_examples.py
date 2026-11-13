"""Data saving in the old (transfer-ladder) setup, four examples for the draft (D433): from scratch against fine-tuned,
MSE of log|M|^2 against the fine-tune events D, on the ladder's 8k-step grid (the draft's ladder figures' cells).
Per panel: scratch and the fine-tune from the pretraining with every structure (rung 9 plus every star arm's
processes, "finale" in analysis/transfer), with the target precision the lowest of 1e-4, 1e-3, 1e-2 that both fits
reach within the measured D (Joaquin 2026-11-13: 1e-3 where possible rather than jumping to 1e-2); the saving S = D*_scratch/D*_fine-tune is read from fitted power laws (D438): each curve fitted
alone, L(D) = A D^-alpha + L_inf with its own exponent (dstar_fit.fit_each; Joaquin 2026-11-13 "Separate exponents"), D* where each fit
meets the line, band the 16-84% bootstrap over cells; the b b~ one-loop D=10 point is left out of both its fits (D438,
drawn open). Cells are read exactly as
analysis/transfer/finale_synth.py reads them (cell(): the best of a search's first 5 trials, a fixed run at the
reference point, or the two chosen points of the W-pole arm); scratch is the 8-trial search (cells.scratch).
    python analysis/data_saving/old_setup_examples.py   -> figs/old_setup_examples; per-cell ratios on stdout
"""
import io, contextlib, os, sys
os.environ["TP_USE_32K"] = "0"      # the 8k grid, as the draft's ladder figures
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "analysis", "transfer"))
sys.path.insert(0, ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    from finale_synth import cell, ARMLAB  # noqa: E402
    from rung_grid import LAB  # noqa: E402
    from cells import scratch  # noqa: E402
import plot_style as ps  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "analysis", "data_saving"))
from dstar_fit import saving, dstar, fit_each  # noqa: E402

PROBES = ["uubar_Zggg", "ee_ddbarg", "udbar_enu", "ee_bb_nlo"]
GAP = "#9c86a6"                    # the gap arrow, muted (Joaquin 2026-11-13: the orange popped too much)
SKIP = {"ee_bb_nlo": 10}          # D left out of the fits (Joaquin, D438: the first b b~ one-loop point)
K = range(2, 9)
LEVELS = [1e-4, 1e-3, 1e-2]
# what an MSE of log|M|^2 means as a relative error of |M|^2, if the residual r = log(pred/true) is Gaussian with
# variance L (an assumption): Delta = exp(r) - 1, its RMS and its median |Delta|, from 10^6 draws
_r = np.random.default_rng(0).standard_normal(10 ** 6)
for L in LEVELS:
    d = np.expm1(np.sqrt(L) * _r)
    print(f"MSE {L:g}: rms log ratio {np.sqrt(L):.3f}, rms Delta {np.sqrt(np.mean(d ** 2)):.4f}, "
          f"median |Delta| {np.median(np.abs(d)):.4f}")
D = [10 ** (k / 2) for k in K]

fig, axs = ps.figure(ncols=2, nrows=2)
ALL, S = [], []
for ax, p in zip(axs.flat, PROBES):
    sc = [scratch(p, k, steered=False)[0] for k in K]
    ft = [cell("tp3_finfte", p, k) for k in K]
    arms, curves = {}, {}
    for key, vals, c in (("scr", sc, "k"), ("ft", ft, ps.C.blue)):
        pts = [(x, y) for x, y in zip(D, vals) if y]
        curves[key] = pts
        fit_pts = [q for q in pts if not np.isclose(q[0], SKIP.get(p, -1))]
        arms[key] = (np.array([q[0] for q in fit_pts]), np.array([q[1] for q in fit_pts]))
        ax.plot(*arms[key], "o", color=c)
        out = [q for q in pts if q not in fit_pts]
        if out:
            ax.plot([q[0] for q in out], [q[1] for q in out], "o", mfc="none", color=c)
    # the line: the lowest of 1e-4, 1e-3, 1e-2 where both fits cross it within the measured D (Joaquin, 2026-11-13)
    f = fit_each(arms)
    level = next((l for l in LEVELS if all(dstar(*f[k], f["alpha_" + k], l) <= max(D) for k in ("scr", "ft"))), None)
    if level is None:
        sys.exit(f"{p}: neither fit reaches any of {LEVELS} within D <= {max(D):g}")
    r = saving(arms, level, shared=False)
    f = r["fit"]
    xs = np.geomspace(min(D), max(D), 200)
    for key, c in (("scr", "k"), ("ft", ps.C.blue)):
        A, Li = f[key]
        ax.plot(xs, A * xs ** -f["alpha_" + key] + Li, "-", color=c)
    ax.axhline(level, color="0.5", ls="--")
    ds, df = (dstar(*f[k], f["alpha_" + k], level) for k in ("scr", "ft"))
    s, (lo, _, hi) = 10 ** r["log10_S"], (10 ** b for b in r["band"])
    S.append(s)
    ax.annotate("", xy=(df, level), xytext=(ds, level), arrowprops=dict(arrowstyle="<|-|>", color=GAP,
                                                                          shrinkA=0, shrinkB=0))
    g = [a / b for a, b in zip(sc, ft) if a and b]
    ALL += g
    print(f"{p}: level {level:g}  alpha scr {f['alpha_scr']:.2f} ft {f['alpha_ft']:.2f}  rms {f['rms']:.2f} dec  L_inf scr {f['scr'][1]:.1e} "
          f"ft {f['ft'][1]:.1e}  D*_scratch {ds:.0f}  D*_ft {df:.0f}  S = D*_scr/D*_ft {s:.1f} "
          f"[16-84%: {lo:.1f}, {hi:.1f}] boot {r['boot']}  "
          f"median L_scratch/L_ft over D {np.median(g):.1f} (range {min(g):.1f}-{max(g):.1f}, {len(g)} cells)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    # the saving's value under the process label, where make_room keeps the data clear of it
    ps.process_label(ax, {**LAB, **ARMLAB}[p].replace(" (1-loop)", " one loop") + "\n" + rf"$\times{s:.1f}$")
print(f"median fitted saving over the 4 panels {np.median(S):.1f} (each read at its own panel's level, not one precision); median L_scratch/L_ft over all {len(ALL)} cells "
      f"{np.median(ALL):.1f}")
H = [Line2D([], [], color="k", marker="o", label="scratch"), Line2D([], [], color=ps.C.blue, marker="o", label="fine-tune"),
     Line2D([], [], color="0.5", marker="o", mfc="none", ls="none", label="not fitted"),
     Line2D([], [], color="0.5", ls="--", label="target precision"),
     Line2D([], [], color=GAP, label=r"$D^*_{\rm scr}/D^*_{\rm ft}$")]
for ax in axs.flat:
    ps.make_room(ax)
for h in H:                          # the strip on top takes its handles from the first panel (Joaquin 2026-11-13)
    axs.flat[0].plot([], [], color=h.get_color(), marker=h.get_marker(), mfc=h.get_markerfacecolor(),
                     ls=h.get_linestyle(), label=h.get_label())
ps.shared_legend(fig, axs.flat[0], ncol=5)
ps.save(fig, os.path.join(ROOT, "analysis", "data_saving", "figs", "old_setup_examples"))
