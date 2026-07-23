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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MZ = 91.1876
DEC = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]

base = np.load(os.path.join(HERE, "heldout_eval_bbbarg_base_s0.npz"))
sig = np.load(os.path.join(HERE, "heldout_eval_bbbarg_g3_s0.npz"))
y = base["y_min"]; ss = base["sqrt_s"]
eb = (base["pred_logamp"] - base["true_logamp"]) ** 2
es = (sig["pred_logamp"] - sig["true_logamp"]) ** 2

fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))

# ---------------------------------------------------------------- per y_min decade
ax = axes[0]
xc, mb, ms = [], [], []
for lo, hi in DEC:
    m = (y >= lo) & (y < hi)
    if m.sum() < 20:
        continue
    xc.append(np.sqrt(max(lo, 1e-7) * hi)); mb.append(eb[m].mean()); ms.append(es[m].mean())
ax.plot(xc, mb, "o-", color="0.4", lw=2, ms=6, label="base (uniform)")
ax.plot(xc, ms, "s-", color="crimson", lw=2, ms=6, label=r"$\sigma$-driven ($\gamma{=}3$)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$y_{\min}=\min_{ij}2p_i\!\cdot\!p_j/s$")
ax.set_ylabel(r"held-out MSE$(\Delta\log|\mathcal{M}|^2)$")
ax.set_title(r"(a) per $y_{\min}$ decade", fontsize=10.5)
ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=9)

# ---------------------------------------------------------------- per sqrt(s) region
ax = axes[1]
REG = [(r"Z-peak" "\n" r"$|\sqrt{s}-M_Z|<3$", np.abs(ss - MZ) < 3.0, 0.50),
       ("shoulder\n3–15", (np.abs(ss - MZ) >= 3) & (np.abs(ss - MZ) < 15), None),
       ("continuum\n>15", np.abs(ss - MZ) >= 15, None)]
xs = np.arange(len(REG)); w = 0.38
bvals = [eb[m].mean() for _, m, _ in REG]
svals = [es[m].mean() for _, m, _ in REG]
ax.bar(xs - w / 2, bvals, w, color="0.4", label="base (uniform)")
ax.bar(xs + w / 2, svals, w, color="crimson", label=r"$\sigma$-driven ($\gamma{=}3$)")
for i, (_, m, pred) in enumerate(REG):
    r = es[m].mean() / eb[m].mean()
    ax.text(i + w/2, svals[i] * 1.08, f"×{r:.2f}", ha="center", fontsize=9, color="crimson")
    if pred is not None:
        ax.text(i - w/2, bvals[i] * 1.08, f"pred ×{pred:.2f}", ha="center",
                fontsize=8, color="navy")
ax.set_yscale("log")
ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in REG], fontsize=9)
ax.set_ylabel(r"held-out MSE$(\Delta\log|\mathcal{M}|^2)$")
ax.set_title(r"(b) per $\sqrt{s}$ region — the gain is Z-driven, as predicted", fontsize=10.5)
ax.grid(True, axis="y", which="both", alpha=0.25); ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(top=ax.get_ylim()[1]*1.6)

fig.suptitle(r"$e^+e^-\to b\bar b g$ (massive): $\sigma$-steering pays where the $\sigma$-contrast "
             r"predicted, and the $m_b$ dead cone softens the IR", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
base_path = os.path.join(HERE, "figs", "l2_bbbarg")
os.makedirs(os.path.dirname(base_path), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{base_path}.{ext}", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {base_path}.png/.pdf")
