"""The four window-trained short arms on 13 TeV pp events, per group of classes: the median per-class MSE of log|M|^2
on the class's window pool, on its pp pool inside the windows' coverage, and outside it (analysis/catalog_v3/
pp_compare.py's summary; docs/results.tex, catalog_v3's short arms read on pp events).
    python analysis/catalog_v3/pp_arms_fig.py
"""
import json, os, subprocess, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

out = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "catalog_v3", "pp_compare.py")],
                     capture_output=True, text=True, cwd=ROOT).stdout.strip().splitlines()[-1]
d = json.loads(out)
ARMS = [("baseline", "baseline"), ("dimensionless", "dimensionless"),
        ("dimensionless_context", "with context"), ("dimensionless_twin", "twin")]
GROUPS = [("train", "training classes"), ("H1", "new combinations"), ("H2", r"$2\to4$ families"), ("H3", "the top")]
figs = ps.panels(4)
for (fig, ax), (g, gname) in zip(figs, GROUPS):
    x = np.arange(len(ARMS))
    for j, (key, lab, col) in enumerate([("mse_window", "window pool", ps.C.blue),
                                          ("mse_pp_inside", "pp, inside the windows", ps.C.green),
                                          ("mse_pp_outside", "pp, outside the windows", ps.C.vermillion)]):
        ax.bar(x + (j - 1) * 0.27, [d[a][g][key] for a, _ in ARMS], 0.27, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["baseline", "dimen-\nsionless", "with\ncontext", "twin"])
    ax.set_yscale("log")
    ax.set_ylabel(r"median MSE$(\log|\mathcal{M}|^2)$")
    ax.set_xlabel(gname)
ps.legend(figs[0][1], "upper left")
ps.make_room(figs[0][1])
ps.save_panels(figs, os.path.join(ROOT, "analysis", "catalog_v3", "figs", "pp_arms"))
