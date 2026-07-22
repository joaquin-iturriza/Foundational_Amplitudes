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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MZ = 91.1876

# Amplitude standardization scale std(log|M|^2) of each process's round-0 pool -- converts the stored
# (preprocessed-unit) sigma back to log|M|^2 units, so sigma and RMSE are on one axis.
AMP_STD = {"uug": 3.3545, "uugg": 6.1200, "uuggg": 8.9193}

SIG_C = "crimson"
ERR_C = "steelblue"


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


def panel(ax, x, sigma_log, rmse, xlabel, title, note=None):
    ax.plot(x, sigma_log, "o-", color=SIG_C, lw=2.0, ms=5.5,
            label=r"learned $\sigma$  (model's own uncertainty)")
    ax.plot(x, rmse, "s--", color=ERR_C, lw=2.0, ms=5.5,
            label=r"actual RMSE$(\Delta\log|\mathcal{M}|^2)$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\log|\mathcal{M}|^2$ units")
    ax.set_title(title, fontsize=10.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    if note:
        ax.text(0.03, 0.04, note, transform=ax.transAxes, fontsize=8, va="bottom")


fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))

# ---------------------------------------------------------------- (a) uug: the Z resonance
d = load("heldout_eval_uug_base_s0")
A = AMP_STD["uug"]
dist = np.abs(d["sqrt_s"] - MZ)                      # distance from the pole = the divergence variable
edges = np.logspace(np.log10(0.02), np.log10(900.0), 22)
x, s, r = profile(dist, d["sigma"] * A, d["err2"], edges)
panel(axes[0, 0], x, s, r,
      r"$|\sqrt{s}-M_Z|$  [GeV]",
      r"(a) $e^+e^-\to u\bar u g$: the $s$-channel $Z$ resonance",
      note="$\\sigma$ rises $\\sim\\!4.5\\times$ into the pole,\ntracking a $10\\times$ rise in the true error")

# ---------------------------------------------------------------- (b) uugg: concentrated IR
d = load("heldout_eval_base_s0")
A = AMP_STD["uugg"]
edges = np.logspace(-7.2, 0.0, 18)
x, s, r = profile(d["y"], d["sigma"] * A, d["err2"], edges)
panel(axes[0, 1], x, s, r,
      r"$y_{\min}=\min_{ij} s_{ij}/s$",
      r"(b) $e^+e^-\to u\bar u gg$: a CONCENTRATED IR divergence",
      note="$\\sigma$ rises $\\sim\\!1.8\\times$ into the deep IR")

# ---------------------------------------------------------------- (c) uuggg: diffuse IR
d = load("heldout_eval_uuggg_base_s0")
A = AMP_STD["uuggg"]
x, s, r = profile(d["y"], d["sigma"] * A, d["err2"], edges)
panel(axes[1, 0], x, s, r,
      r"$y_{\min}=\min_{ij} s_{ij}/s$",
      r"(c) $e^+e^-\to u\bar u ggg$: a DIFFUSE divergence",
      note="$\\sigma$ is nearly FLAT ($\\sim\\!1.3\\times$):\nno thin region to point at")

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
LABEL_OFF = [(8, 8), (8, 8), (10, -34)]     # nudge the uuggg label clear of the baseline
ax = axes[1, 1]
cs, gs = [], []
for (name, bt, st, rf, bf, coord, glab), off in zip(POINTS, LABEL_OFF):
    c, g = contrast_and_gain(bt, st, rf, bf, coord)
    cs.append(c); gs.append(g)
    ax.plot(c, g, "o", ms=13, color="darkorange", mec="k", mew=1.2, zorder=3)
    ax.annotate(f"{name}\n{glab}", xy=(c, g), xytext=off, textcoords="offset points", fontsize=8.5)
    print(f"  {name.replace(chr(10),' '):22s} sigma-contrast={c:.2f}  sigma/base MSE in region={g:.3f}")
ax.plot(cs, gs, "-", color="darkorange", lw=1.4, alpha=0.6, zorder=2)
ax.axhline(1.0, color="k", ls="--", lw=1.2)
ax.text(4.85, 1.012, "no gain over uniform generation", fontsize=8, ha="right")
ax.set_xlabel(r"$\sigma$-CONTRAST: $\langle\sigma\rangle_{\rm singular}\,/\,\langle\sigma\rangle_{\rm bulk}$"
              "\n(measured on the base arm alone — one forward pass, no A/B)")
ax.set_ylabel(r"measured $\sigma$-arm / base  MSE ratio in that region")
ax.set_title(r"(d) the $\sigma$ profile PREDICTS whether steering pays", fontsize=10.5)
ax.grid(True, alpha=0.25)
ax.set_xlim(1.1, 5.0); ax.set_ylim(0.30, 1.10)

fig.suptitle(r"The learned $\sigma$ rises inside the divergence — and how sharply it does so "
             r"predicts whether $\sigma$-driven generation helps", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.97])
base = os.path.join(HERE, "figs", "l2_sigma_vs_divergence")
os.makedirs(os.path.dirname(base), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{base}.{ext}", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {base}.png/.pdf")
