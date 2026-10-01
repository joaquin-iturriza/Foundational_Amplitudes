"""Steering A/B on ee->WW (docs/results.tex, sec:ladder): the sigma-steered pool against the mixture pool,
both scored on the mixture test split (MSE of log|M|^2 at the best checkpoint). Baseline: each cell's search
best (8 trials); steered: one run at the baseline best's HPs (the A/B shortcut), so the comparison is biased
against the steered arm. The steered runs are scored with their own off-shellness stats pinned
(tools/rebuild_run.py). Data: analysis/transfer/steer_ab.json.
    python analysis/transfer/steer_ab.py
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

J = json.load(open(os.path.join(ROOT, "analysis", "transfer", "steer_ab.json")))
fig, ax = ps.figure()
for arm, col, lab in [("baseline", ps.C.blue, "mixture pool (search best)"), ("steered", ps.C.vermillion, r"$\sigma$-steered pool (same HPs)")]:
    k = sorted(J[arm], key=int)
    ax.plot([10 ** (int(x) / 2) for x in k], [J[arm][x] for x in k], "o-", color=col, label=lab)
for x in sorted(J["baseline"], key=int):
    print(f"D=10^{int(x)/2:.1f}  steered/baseline {J['steered'][x] / J['baseline'][x]:.2f}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
ps.legend(ax, "lower left")
ps.process_label(ax, r"$e^+e^-\to W^+W^-$")
ps.make_room(ax)
ps.save(fig, os.path.join(ROOT, "analysis", "transfer", "figs", "steer_ab"))
