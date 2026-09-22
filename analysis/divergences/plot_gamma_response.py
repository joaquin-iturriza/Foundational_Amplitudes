#!/usr/bin/env python
"""Extended gamma-response: sigma/base error ratio vs the keep exponent gamma, for uugg (2 gluons,
concentrated divergence) and uuggg (3 gluons, diffuse). Shows the two behaviours the concentration
principle predicts: uugg's deepest decade improves then saturates while its bulk penalty keeps
growing; uuggg barely responds and turns over above gamma~3, going worse than uniform. png+pdf."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

REPO = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

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


TRACKS = [(r"$\log_{10} y_{\min}<-3$", ps.C.blue, "o"),
          (r"$\log_{10} y_{\min}<-6$", ps.C.vermillion, "s"),
          (r"$\log_{10} y_{\min}>-1$", ps.C.green, "^")]

fig, axes = ps.figure(ncols=2, sharey=True)
for ax, (proc, base_tag, rows) in zip(axes, [
        (r"$e^+e^-\to u\bar u gg$", "heldout_eval_base_s0", UUGG),
        (r"$e^+e^-\to u\bar u ggg$", "heldout_eval_uuggg_base_s0", UUGGG)]):
    g, r = series(base_tag, rows)
    ax.axhline(1.0, color=ps.C.grey, lw=0.8, label="parity")
    for j, (lab, c, m) in enumerate(TRACKS):
        if np.isfinite(r[:, j]).any():
            ax.plot(g, r[:, j], m + "-", color=c, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$")
    ax.set_xticks(g); ax.set_xticklabels([str(int(x)) for x in g])
    ax.minorticks_off()
    ps.process_label(ax, proc, loc="upper left")
axes[0].set_ylabel(r"MSE ratio")
# The legend goes in the RIGHT panel, which is empty: the uuggg curves all sit on parity, and
# that flatness is the point of the panel. In the left panel the three tracks fan across the
# whole box, so any corner there costs a y-range expansion big enough to push the axis below
# zero -- meaningless for a ratio. Handles come from the left panel, which has all three
# tracks; the right one only ever draws two.
ps.legend(axes[1], "lower left", handles=axes[0].get_legend_handles_labels()[0])

ps.save(fig, "analysis/divergences/figs/l2_gamma_response_extended")
for nm, bt, rows in [("uugg", "heldout_eval_base_s0", UUGG), ("uuggg", "heldout_eval_uuggg_base_s0", UUGGG)]:
    g, r = series(bt, rows)
    print(f"{nm}: gamma={list(g)}")
    print(f"   deep   ={np.round(r[:,0],3).tolist()}")
    print(f"   deepest={np.round(r[:,1],3).tolist()}")
    print(f"   bulk   ={np.round(r[:,2],3).tolist()}")
