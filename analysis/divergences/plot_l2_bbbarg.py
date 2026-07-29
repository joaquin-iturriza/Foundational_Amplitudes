#!/usr/bin/env python
"""ee -> b bbar g: the MASSIVE-process L2 test, and the confirmation of the sigma-contrast prediction.

bbbarg is the massive analogue of uug -- same Z resonance and soft-gluon singularity, but the collinear
limit is regulated by m_b (the dead cone), which FLATTENS one of the two singular directions. Before the
sigma arm ran, the sigma-contrast measured on the base arm alone (Sec. sigma-vs-divergence) predicted:
a moderate Z-peak gain (contrast 2.51, between uug's 4.48 and uugg's 1.83) and a WEAK deep-IR gain
(contrast 1.61, dead-cone-softened). Both held.

Left  : held-out MSE per y_min decade, base (uniform) vs sigma^3.
Right : the same error split by sqrt(s) region, where the gain actually lives -- the Z-peak, not the
        IR continuum. Annotated with the pre-recorded predictions.
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
DEC = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]

base = np.load(os.path.join(HERE, "heldout_eval_bbbarg_base_s0.npz"))
sig = np.load(os.path.join(HERE, "heldout_eval_bbbarg_g3_s0.npz"))
y = base["y_min"]; ss = base["sqrt_s"]
eb = (base["pred_logamp"] - base["true_logamp"]) ** 2
es = (sig["pred_logamp"] - sig["true_logamp"]) ** 2

fig, axes = ps.figure(ncols=2)

# ---------------------------------------------------------------- per y_min decade
ax = axes[0]
xc, mb, ms = [], [], []
for lo, hi in DEC:
    m = (y >= lo) & (y < hi)
    if m.sum() < 20:
        continue
    xc.append(np.sqrt(max(lo, 1e-7) * hi)); mb.append(eb[m].mean()); ms.append(es[m].mean())
ax.plot(xc, mb, "o-", color=ps.C.grey, label="uniform")
ax.plot(xc, ms, "s-", color=ps.C.vermillion, label=r"$\sigma$-driven ($\gamma{=}3$)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$y_{\min}=\min_{ij}2p_i\!\cdot\!p_j/s$")
ax.set_ylabel(r"MSE$(\Delta\log|\mathcal{M}|^2)$")
ax.legend(loc="lower left")
ps.process_label(ax, r"$e^+e^-\to b\bar b g$", loc="upper right")

# ---------------------------------------------------------------- per sqrt(s) region
ax = axes[1]
REG = [(r"$|\sqrt{s}-M_Z|<3$", np.abs(ss - MZ) < 3.0),
       (r"$3$–$15$", (np.abs(ss - MZ) >= 3) & (np.abs(ss - MZ) < 15)),
       (r"$>15$", np.abs(ss - MZ) >= 15)]
xs = np.arange(len(REG)); w = 0.38
bvals = [eb[m].mean() for _, m in REG]
svals = [es[m].mean() for _, m in REG]
ax.bar(xs - w / 2, bvals, w, color=ps.C.grey, label="uniform")
ax.bar(xs + w / 2, svals, w, color=ps.C.vermillion, label=r"$\sigma$-driven ($\gamma{=}3$)")
ax.set_yscale("log")
ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in REG])
ax.set_xlabel(r"$|\sqrt{s}-M_Z|$  [GeV]")
ax.set_ylabel(r"MSE$(\Delta\log|\mathcal{M}|^2)$")
ax.grid(True, axis="y", which="both")
ax.legend(loc="upper left")
ax.set_ylim(top=ax.get_ylim()[1] * 2.2)

base_path = os.path.join(HERE, "figs", "l2_bbbarg")
ps.save(fig, base_path)
for (nm, m) in REG:
    print(f"  {nm}: ratio {es[m].mean()/eb[m].mean():.3f}")
