#!/usr/bin/env python
"""Extended gamma-response: sigma/base error ratio vs the keep exponent gamma, for uugg (2 gluons,
concentrated divergence) and uuggg (3 gluons, diffuse). Shows the two behaviours the concentration
principle predicts -- uugg's deepest decade improves then SATURATES while its bulk penalty keeps
growing; uuggg barely responds and TURNS OVER above gamma~3, going worse than uniform. Emits png+pdf."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
UUGG = [(1, "heldout_eval_sigma_s0"), (2, "heldout_eval_g2_s0"), (3, "heldout_eval_g3_s0"),
        (5, "heldout_eval_g5_s0"), (10, "heldout_eval_g10_s0"),
        (20, "heldout_eval_uugg_g20_s0"), (30, "heldout_eval_uugg_g30_s0")]
UUGGG = [(1, "heldout_eval_uuggg_sigma_s0"), (3, "heldout_eval_uuggg_g3_s0"),
         (10, "heldout_eval_uuggg_g10_s0"), (20, "heldout_eval_uuggg_g20_s0")]


def bands(tag):
    d = np.load(os.path.join(HERE, tag + ".npz"))
    e2 = (d["pred_logamp"] - d["true_logamp"]) ** 2
    ym = d["y_min"]
    deep = e2[ym < 1e-3].mean()
    deepest = e2[ym < 1e-6].mean() if (ym < 1e-6).any() else np.nan
    bulk = e2[ym > 1e-1].mean() if (ym > 1e-1).any() else np.nan
    return deep, deepest, bulk


def series(base_tag, rows):
    b = bands(base_tag)
    g, out = [], []
    for gam, tag in rows:
        if os.path.exists(os.path.join(HERE, tag + ".npz")):
            s = bands(tag)
            g.append(gam)
            out.append([s[i] / b[i] if np.isfinite(b[i]) and b[i] > 0 else np.nan for i in range(3)])
    return np.array(g), np.array(out)


fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharey=True)
for ax, (title, base_tag, rows) in zip(axes, [
        (r"$e^+e^-\to u\bar u gg$  (2 gluons: CONCENTRATED)", "heldout_eval_base_s0", UUGG),
        (r"$e^+e^-\to u\bar u ggg$  (3 gluons: DIFFUSE)", "heldout_eval_uuggg_base_s0", UUGGG)]):
    g, r = series(base_tag, rows)
    for j, (lab, c, m) in enumerate([(r"deep IR $y_{\min}<10^{-3}$", "steelblue", "o"),
                                     (r"deepest $y_{\min}<10^{-6}$", "crimson", "s"),
                                     (r"bulk $y_{\min}>10^{-1}$", "darkorange", "^")]):
        if np.isfinite(r[:, j]).any():
            ax.plot(g, r[:, j], m + "-", color=c, lw=2.0, ms=7, label=lab)
    ax.axhline(1.0, color="k", ls="--", lw=1.2)
    ax.text(g[0] * 1.05, 1.012, "uniform baseline (no gain)", fontsize=8, color="k")
    ax.set_xscale("log"); ax.set_xlabel(r"keep exponent $\gamma$   ($p\propto\sigma^\gamma$)")
    ax.set_xticks(g); ax.set_xticklabels([str(int(x)) for x in g])
    ax.set_title(title, fontsize=11); ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
axes[0].set_ylabel(r"$\sigma$-arm / base   MSE ratio  (lower = better)")
axes[0].annotate("deepest decade SATURATES\n(0.53→0.51 over γ=10→30)\nwhile the bulk penalty keeps growing",
                 xy=(20, 0.52), xytext=(0.06, 0.30), textcoords="axes fraction", fontsize=8,
                 color="crimson", arrowprops=dict(arrowstyle="->", color="crimson", lw=1.1))
axes[1].annotate("TURNS OVER above γ≈3:\nharder concentration is WORSE\nthan uniform",
                 xy=(20, 1.079), xytext=(0.10, 0.72), textcoords="axes fraction", fontsize=8,
                 color="steelblue", arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.1))
fig.suptitle(r"Pushing the concentration exponent $\gamma$: saturation vs reversal", fontsize=12)
fig.tight_layout()
base = os.path.join(HERE, "figs", "l2_gamma_response_extended")
os.makedirs(os.path.dirname(base), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{base}.{ext}", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {base}.png/.pdf")
for nm, bt, rows in [("uugg", "heldout_eval_base_s0", UUGG), ("uuggg", "heldout_eval_uuggg_base_s0", UUGGG)]:
    g, r = series(bt, rows)
    print(f"{nm}: gamma={list(g)}")
    print(f"   deep   ={np.round(r[:,0],3).tolist()}")
    print(f"   deepest={np.round(r[:,1],3).tolist()}")
    print(f"   bulk   ={np.round(r[:,2],3).tolist()}")
