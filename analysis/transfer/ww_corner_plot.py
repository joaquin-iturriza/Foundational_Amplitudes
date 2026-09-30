"""ee->WW's forward corner (analysis/transfer/ww_corner.py): training events whose |t| (e- to W-) lies
below the t-channel factor's floor (the mixture pool's 1e-3 quantile of |t|) and within 3x and 10x of
it, in the first D events of the pool, against D; the 11 validation events below the floor carry 98%
of the D = 10^4 run's squared error.
Data: analysis/transfer/ww_corner_ee_WW.json.
    python analysis/transfer/ww_corner_plot.py
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

d = json.load(open(os.path.join(ROOT, "analysis", "transfer", "ww_corner_ee_WW.json")))
D = np.array(sorted(int(k) for k in d["per_D"]))
fig, ax = ps.figure()
for key, lab, col in (("below", r"$|t|<t_{\rm floor}$", ps.C.vermillion),
                      ("within3", r"$|t|<3\,t_{\rm floor}$", ps.C.orange),
                      ("within10", r"$|t|<10\,t_{\rm floor}$", ps.C.blue)):
    y = np.array([d["per_D"][str(k)][key] for k in D], float)
    ok = y > 0
    ax.plot(D[ok], y[ok], "o-", color=col, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"training events $D$"); ax.set_ylabel("events in the corner")
ps.process_label(ax, r"$e^+e^-\to W^+W^-$")
ps.legend(ax, "upper left")
ps.make_room(ax)
ps.save(fig, os.path.join(ROOT, "analysis", "transfer", "figs", "ww_corner"))
