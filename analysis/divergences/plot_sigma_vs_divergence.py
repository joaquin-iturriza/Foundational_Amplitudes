#!/usr/bin/env python
"""Does the learned sigma actually rise IN the divergence region? -- and can its profile PREDICT
whether sigma-steering will pay, before spending the A/B compute?

Both questions are answered from data already on disk: score_and_save() stores sigma row-aligned with
the divergence coordinate (y_min for the IR, sqrt_s for the s-channel resonance) in
heldout_eval_<label>.npz. No GPU, no new runs.

Panels (a)-(c): mean sigma and RMSE against the coordinate that DEFINES each divergence, both in
log|M|^2 units (sigma is stored in preprocessed units, so it is multiplied by that run's amplitude
standardization scale) -- so the two curves are directly comparable and the vertical gap IS the
calibration. All three are the BASE (uniform-keep) arm: this is what the model knows before any
sigma-driven generation happens.

Panel (d): the pre-flight diagnostic. sigma-CONTRAST (how much higher sigma is in the singular region
than in the bulk, measured on the base arm alone) against the sigma-arm/base MSE ratio actually
measured in that region. The three processes order monotonically, which turns the concentration
principle from a post-hoc explanation into a test you can run BEFORE the A/B.

Emits png+pdf.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MZ = 91.1876

# Amplitude standardization scale std(log|M|^2) of each process's round-0 pool -- converts the stored
# (preprocessed-unit) sigma back to log|M|^2 units, so sigma and RMSE are on one axis.
AMP_STD = {"uug": 3.3545, "uugg": 6.1200, "uuggg": 8.9193}

SIG_C = ps.C.vermillion
ERR_C = ps.C.blue


def load(tag):
    d = np.load(os.path.join(HERE, tag + ".npz"))
    return dict(
        y=d["y_min"], sigma=d["sigma"],
        sqrt_s=(d["sqrt_s"] if "sqrt_s" in d.files and d["sqrt_s"].size else None),
        err2=(d["pred_logamp"] - d["true_logamp"]) ** 2,
    )


def profile(x, sigma, err2, edges, min_n=40):
    """Per-bin mean sigma and RMSE over bins `edges` in the divergence coordinate x."""
    xc, sm, rm = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < min_n:
            continue
        xc.append(np.sqrt(lo * hi))                 # geometric bin centre (log axis)
        sm.append(sigma[m].mean())
        rm.append(np.sqrt(err2[m].mean()))
    return np.array(xc), np.array(sm), np.array(rm)


def panel(ax, x, sigma_log, rmse, xlabel, process):
    # (marker size left at the shared default; the ms=3 here was a workaround for 1.5in
    # default markers merge into a band and hide the curve they are meant to mark.
    ax.plot(x, sigma_log, "o-", color=SIG_C, label=r"learned $\sigma$")
    ax.plot(x, rmse, "s--", color=ERR_C, label=r"RMSE$(\Delta\log|\mathcal{M}|^2)$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\log|\mathcal{M}|^2$ units")
    ps.process_label(ax, process, loc="upper right")
    return ax


# FOUR independent panels: the three sigma-vs-divergence profiles and the pre-flight diagnostic
# they feed. They belong in one figure -- the diagnostic is the conclusion of the other three,
# not a separate result -- and results.tex includes the four files two per line, so on the page
# it reads as a 2x2.
#
# Not ps.figure(2, 2): four panels each carrying a y-label and log tick labels measure 6.79in
# as one canvas, which is wider than the paper. As four files it is two lots of 3.2in, which
# fits with room for the \hfill. And nothing is shared: no sharey (with a shared y-axis only
# column 0 carries tick labels, so in a 2x2 that is (a) and (c) but not (b), and the panels
# stop looking like each other), no shared legend strip.
figs = ps.panels(4)
axes = [f[1] for f in figs]

# ---------------------------------------------------------------- (a) uug: the Z resonance
d = load("heldout_eval_uug_base_s0")
A = AMP_STD["uug"]
dist = np.abs(d["sqrt_s"] - MZ)                      # distance from the pole = the divergence variable
edges = np.logspace(np.log10(0.02), np.log10(900.0), 22)
x, s, r = profile(dist, d["sigma"] * A, d["err2"], edges)
panel(axes[0], x, s, r, r"$|\sqrt{s}-M_Z|$  [GeV]", r"$e^+e^-\to u\bar u g$")

# ---------------------------------------------------------------- (b) uugg: concentrated IR
d = load("heldout_eval_base_s0")
A = AMP_STD["uugg"]
edges = np.logspace(-7.2, 0.0, 18)
x, s, r = profile(d["y"], d["sigma"] * A, d["err2"], edges)
panel(axes[1], x, s, r, r"$y_{\min}=\min_{ij} s_{ij}/s$", r"$e^+e^-\to u\bar u gg$")

# ---------------------------------------------------------------- (c) uuggg: diffuse IR
d = load("heldout_eval_uuggg_base_s0")
A = AMP_STD["uuggg"]
x, s, r = profile(d["y"], d["sigma"] * A, d["err2"], edges)
panel(axes[2], x, s, r, r"$y_{\min}=\min_{ij} s_{ij}/s$", r"$e^+e^-\to u\bar u ggg$")

# ---------------------------------------------------------------- (d) the pre-flight diagnostic
# sigma-contrast measured on the BASE arm (singular region vs bulk) against the sigma-arm/base MSE
# ratio measured in that same region -- i.e. "how peaked is sigma" vs "how much did steering pay".
def contrast_and_gain(base_tag, sig_tag, reg_fn, bulk_fn, coord):
    b, a = load(base_tag), load(sig_tag)
    xb = b[coord]
    reg, bulk = reg_fn(xb), bulk_fn(xb)
    c = b["sigma"][reg].mean() / b["sigma"][bulk].mean()
    g = a["err2"][reg_fn(a[coord])].mean() / b["err2"][reg].mean()
    return c, g

POINTS = [
    (r"$u\bar u g$" + "\n(Z peak)", "heldout_eval_uug_base_s0", "heldout_eval_uug_g3_s0",
     lambda x: np.abs(x - MZ) < 3.0, lambda x: np.abs(x - MZ) >= 15.0, "sqrt_s", r"$\gamma{=}3$"),
    (r"$u\bar u gg$" + "\n(deep IR)", "heldout_eval_base_s0", "heldout_eval_g10_s0",
     lambda x: x < 1e-6, lambda x: x > 1e-1, "y", r"$\gamma{=}10$"),
    (r"$u\bar u ggg$" + "\n(deep IR)", "heldout_eval_uuggg_base_s0", "heldout_eval_uuggg_g10_s0",
     lambda x: x < 1e-6, lambda x: x > 1e-2, "y", r"$\gamma{=}10$"),
]
# The two series are the same in all three profile panels, so ONE legend, inside panel (a),
# in its free corner -- not a strip above the figure. Both curves start high on the left and
# fall, so the lower left is clear.
ps.legend(axes[0], "lower left")

# ---------------------------------------------------------------- (d) the pre-flight panel
ax = axes[3]
cs, gs = [], []
MARKS = ["o", "s", "D"]
COLS = [ps.C.blue, ps.C.vermillion, ps.C.green]
for (name, bt, st, rf, bf, coord, glab), mk, col in zip(POINTS, MARKS, COLS):
    c, g = contrast_and_gain(bt, st, rf, bf, coord)
    cs.append(c); gs.append(g)
    # Just the process and the exponent: the region each point refers to is already the
    # x-axis of the panel it came from, and the long "(Z peak)" / "(deep IR)" labels made the
    # legend nearly as wide as the plot box.
    lab = name.split("\n")[0] + f", {glab}"
    ax.plot(c, g, mk, color=col, zorder=3, label=lab)
    print(f"  {name.replace(chr(10),' '):22s} sigma-contrast={c:.2f}  sigma/base MSE in region={g:.3f}")
ax.plot(cs, gs, "-", color=ps.C.grey, lw=1.0, zorder=2)
ax.axhline(1.0, color=ps.C.grey, ls="--", label="parity")
ax.set_xlabel(r"$\langle\sigma\rangle_{\rm singular}\,/\,\langle\sigma\rangle_{\rm bulk}$")
ax.set_ylabel(r"$\mathrm{MSE}_{\sigma}\,/\,\mathrm{MSE}_{\rm base}$")
ax.set_xlim(1.1, 5.0); ax.set_ylim(0.30, 1.15)
# Lower left is the free corner: the points fall left-to-right and the parity line sits at
# the top.
ps.legend(ax, "lower left")

ps.save_panels(figs, os.path.join(HERE, "figs", "l2_sigma_vs_divergence"))
