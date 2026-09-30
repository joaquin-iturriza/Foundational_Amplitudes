"""Target factors on vs off, three seeds per arm (seed 1, 2, 3), each cell at its best trial's full
HPs from the factor-on scratch sweep (tp_scr), 8000 steps: the Breit-Wigner factor
(data.target_propagators) on ee->dd~ and ee->nu_e nu_e~ at D = 10^3 and 10^4, the t-channel factor
(data.target_propagator_tchannel) on uu~->gg and ee->Za at D = 10^4. Loss is MSE of log|M|^2 at the
best checkpoint (val_loss_no_reg times prepd_std^2; the factors are exact and divide out of
log|M|^2, so the arms compare directly). Points are the seeds, bars their geometric mean.
Data: analysis/transfer/seeded_ab.json (collect_calib.py on lxplus and jeanzay, runs named *_seed<n>).
    python analysis/transfer/seeded_ab.py
"""
import collections, json, math, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

d = json.load(open(os.path.join(ROOT, "analysis", "transfer", "seeded_ab.json")))
G = collections.defaultdict(list)
for n, r in d.items():
    m = re.match(r"tp2?_calib_(\w+?)_d(\d+)_t8000_lr[^_]+_(target_propagators|target_propagator_tchannel)-(true|false)_seed\d", n)
    G[(m.group(1), int(m.group(2)), m.group(4))].append(r["val"] * r["std"] ** 2)
PANELS = [[("ee_ddbar", 6, "$d\\bar d$\n$10^3$"), ("ee_ddbar", 8, "$d\\bar d$\n$10^4$"),
           ("ee_nnbar", 6, "$\\nu_e\\bar\\nu_e$\n$10^3$"), ("ee_nnbar", 8, "$\\nu_e\\bar\\nu_e$\n$10^4$")],
          [("uubar_gg", 8, "$u\\bar u\\to gg$\n$10^4$"), ("ee_Za", 8, "$e^+e^-\\to Z\\gamma$\n$10^4$")]]
figs = ps.panels(2)
for (fig, ax), cells, lab in zip(figs, PANELS, (r"Breit-Wigner factor, $e^+e^-\to$", "t-channel factor")):
    for arm, col, dx in (("true", ps.C.blue, -0.13), ("false", ps.C.vermillion, 0.13)):
        for i, (p, k, _) in enumerate(cells):
            y = np.array(G[(p, k, arm)])
            ax.plot(np.full(len(y), i + dx), y, "o", color=col, mfc="none", ls="none")
            ax.plot([i + dx - 0.1, i + dx + 0.1], [math.exp(np.log(y).mean())] * 2, color=col)
        ax.plot([], [], "o", mfc="none", color=col, ls="none", label=f"factor {'on' if arm == 'true' else 'off'}")
    ax.set_xticks(range(len(cells))); ax.set_xticklabels([c[2] for c in cells])
    ax.set_xlim(-0.5, len(cells) - 0.5)
    ax.set_yscale("log"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(ax, lab)
    ps.legend(ax, "lower left")
ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", "seeded_ab"))
