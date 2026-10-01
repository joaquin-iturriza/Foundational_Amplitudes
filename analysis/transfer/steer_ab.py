"""Steering A/B on ee->WW (docs/results.tex, sec:ladder): the sigma-steered pool against the mixture pool,
both scored on the mixture test split (MSE of log|M|^2 at the best checkpoint). Baseline: each cell's search
best (8 trials); steered: the best of its own 8-trial search where the shortcut lost (D <= 10^3, line),
else one run at the baseline best's HPs (the A/B shortcut, open markers), which is biased against it. The steered runs are scored with their own off-shellness stats pinned
(tools/rebuild_run.py). Data: analysis/transfer/steer_ab.json.
    python analysis/transfer/steer_ab.py
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

J = json.load(open(os.path.join(ROOT, "analysis", "transfer", "steer_ab.json")))
fig, ax = ps.figure()
k = sorted(J["baseline"], key=int)
ax.plot([10 ** (int(x) / 2) for x in k], [J["baseline"][x] for x in k], "o-", color=ps.C.blue, label="mixture pool")
st = {**{x: v for x, v in J["steered"].items() if x not in J["steered_search"]}, **J["steered_search"]}
ks = sorted(st, key=int)
ax.plot([10 ** (int(x) / 2) for x in ks], [st[x] for x in ks], "-", color=ps.C.vermillion)
ax.plot([10 ** (int(x) / 2) for x in J["steered_search"]], list(J["steered_search"].values()), "o", color=ps.C.vermillion,
        ls="none", label="steered, own search")
sc = [x for x in ks if x not in J["steered_search"]]
ax.plot([10 ** (int(x) / 2) for x in sc], [st[x] for x in sc], "o", mfc="none", color=ps.C.vermillion, ls="none",
        label="steered, baseline HPs")
for x in k:
    print(f"D=10^{int(x)/2:.1f}  steered/baseline {st[x] / J['baseline'][x]:.2f}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
ps.legend(ax, "lower left")
ps.process_label(ax, r"$e^+e^-\to W^+W^-$")
ps.make_room(ax)
ps.save(fig, os.path.join(ROOT, "analysis", "transfer", "figs", "steer_ab"))
