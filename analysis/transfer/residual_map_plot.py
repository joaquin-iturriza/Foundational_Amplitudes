"""Where the error sits, ee->WW (t-channel factor on, the cell's best HPs) vs ee->Za (factor off),
D = 10^4, 8000 steps: the validation MSE of log|M|^2 per (sqrt(s), cos theta) bin (theta between the
e- beam and the first final-state particle), from analysis/transfer/residual_map.py on the runs'
saved predictions. Empty bins are blank. Also printed: the share of the squared error carried by
the worst 1% of events.
Data: analysis/transfer/residual_map_ee_WW.json, residual_map_ee_Za.json.
    python analysis/transfer/residual_map_plot.py
"""
import json, os, sys
import numpy as np
from matplotlib.ticker import NullFormatter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

figs = ps.panels(2)
for (fig, ax), (p, lab) in zip(figs, (("ee_WW", r"$e^+e^-\to W^+W^-$"), ("ee_Za", r"$e^+e^-\to Z\gamma$"))):
    d = json.load(open(os.path.join(ROOT, "analysis", "transfer", f"residual_map_{p}.json")))
    M = np.array(d["mse_map"], float); M[M <= 0] = np.nan
    im = ax.pcolormesh(d["cos_edges"], d["sqrts_edges"], np.log10(M), cmap="viridis")
    ax.set_yscale("log")
    # plain GeV ticks: the mathtext 6x10^2 labels made the pair wider than the page
    ax.set_yticks([200, 400, 800], ["200", "400", "800"]); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$\cos\theta$"); ax.set_ylabel(r"$\sqrt{s}$ [GeV]")
    ps.process_label(ax, lab)
    ps.colorbar(ax, im, r"$\log_{10}$ MSE$(\log|\mathcal{M}|^2)$")
    print(f"{p}: MSE {d['mse']:.3g}, worst 1% of events carry {100 * d['share_top1pct']:.0f}% of it")
ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", "residual_map"))
