"""Validation MSE against training step, one curve per training-set size D, with each curve's minimum marked: the
steps-per-D calibration of the draft Fig. 22 targets (D385-D387; rule D391: T(D) = 1.1 x the minimum's step,
non-decreasing in D, capped at 8500). One panel per target, each in its trained target's unit (ln|M|^2 for LO,
ln|V| for e+e- > u u~ V trained on V (D375), ln|V/B| for e+e- > t t~ V). Reads horizons.py --ladder outputs.

    python analysis/data_saving/plot_horizons.py
"""
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

DATA = os.path.join(REPO, "analysis/data_saving/data")
PANELS = [("ee_uu", "horizons_cal_f22_cal_f22_lrhalf.json", r"$e^+e^-\to u\bar u$", r"val MSE$(\ln|\mathcal{M}|^2)$"),
          ("epem_ttx", "horizons_cal2_ladder.json", r"$e^+e^-\to t\bar t$", r"val MSE$(\ln|\mathcal{M}|^2)$"),
          ("ee_uu_nlo", "horizons_cal_f22_cal_f22_lrhalf.json", r"$e^+e^-\to u\bar u$, $V$", r"val MSE$(\ln|V|)$"),
          ("ee_ttbar_nlo", "horizons_cal_f22_cal_f22_lrhalf.json", r"$e^+e^-\to t\bar t$, $V/B$",
           r"val MSE$(\ln|V/B|)$")]
KS = list(range(2, 11))
COL = dict(zip(KS, ps.sequence(len(KS))))

fig, axes = ps.figure(ncols=2, nrows=2)
for n, (ax, (tgt, f, lab, ylab)) in enumerate(zip(axes.flat, PANELS)):
    rows = sorted((r for r in json.load(open(os.path.join(DATA, f))) if r["target"] == tgt and r.get("curve")),
                  key=lambda r: r["k"])
    for r in rows:
        x = [(i + 1) * r["every"] for i in range(len(r["curve"]))]
        ax.plot(x, r["curve"], color=COL[r["k"]])
        ax.plot([r["s_b"]], [r["best"]], marker="o", linestyle="none", color=COL[r["k"]])
    if n == 0:   # one legend for the canvas: every D of any panel, and the minimum's marker
        for k in KS:
            ax.plot([], [], color=COL[k], label=rf"$10^{{{k / 2:g}}}$")
        ax.plot([], [], marker="o", linestyle="none", color="gray", label="min")
    for r in json.load(open(os.path.join(DATA, f))):
        if r["target"] == tgt and not r.get("curve"):
            print("not plotted:", tgt, r.get("k"), "blowup" if "blowup_step" in r else "missing")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(ylab)
    ps.process_label(ax, lab)
ps.legend(axes.flat[0], "lower left", ncol=3, title=r"$D$")
ps.save(fig, os.path.join(REPO, "analysis/data_saving/figs/horizons_fig22_targets"))
