"""Target factors on vs off, three seeds per arm (seed 1, 2, 3), each cell at its best trial's full
HPs from the factor-on scratch sweep (tp_scr), 8000 steps: the Breit-Wigner factor
(data.target_propagators) on ee->dd~ and ee->nu_e nu_e~ at D = 10^3 and 10^4, the t-channel factor
(data.target_propagator_tchannel) on uu~->gg and ee->Za at D = 10^4. Loss is MSE of log|M|^2 at the
best checkpoint (val_loss_no_reg times prepd_std^2; the factors are exact and divide out of
log|M|^2, so the arms compare directly). Points are the seeds, bars their geometric mean. The arm is
read from each run's recorded flags (collect_calib.py), not from its name; failed runs are listed.
A seed whose validation loss jumps five-fold or more between two validations after its best
checkpoint (the blow-up the notes describe for the catalog's t-channel seeds: a jump by five to
ten) is kept at its best checkpoint, drawn filled, and printed with the step of the jump.
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
on = lambda v: str(v).lower() == "true"
for n, r in sorted(d.items()):
    if r.get("failed"):
        print(f"not plotted, run failed: {n} ({r['failed']})")
        continue
    m = re.match(r"tp2?_calib_(\w+?)_d(\d+)_t8000_lr[^_]+_(target_propagators|target_propagator_tchannel)-", n)
    # the factor under test is the one the name overrides; its value comes from what ran
    arm = "true" if on(r["tchannel"] if m.group(3) == "target_propagator_tchannel" else r["target_propagators"]) else "false"
    v = np.array(r["curve"]); ib = int(np.argmin(v))
    jump = next((i for i in range(ib + 1, len(v)) if v[i] >= 5 * v[i - 1]), None)
    if jump is not None:
        print(f"blow-up: {n}: best at step {r['best_step']}, jump at step {(jump + 1) * r['every']}, "
              f"ends {v[-1] / v[ib]:.1f}x above its best")
    G[(m.group(1), int(m.group(2)), arm)].append((r["val"] * r["std"] ** 2, jump is not None))
PANELS = [[("ee_ddbar", 6, "$d\\bar d$\n$10^3$"), ("ee_ddbar", 8, "$d\\bar d$\n$10^4$"),
           ("ee_nnbar", 6, "$\\nu_e\\bar\\nu_e$\n$10^3$"), ("ee_nnbar", 8, "$\\nu_e\\bar\\nu_e$\n$10^4$")],
          [("uubar_gg", 8, "$u\\bar u\\to gg$\n$10^4$"), ("ee_Za", 8, "$e^+e^-\\to Z\\gamma$\n$10^4$")]]
figs = ps.panels(2)
for (fig, ax), cells, lab in zip(figs, PANELS, (r"Breit-Wigner factor, $e^+e^-\to$", "t-channel factor")):
    for arm, col, dx in (("true", ps.C.blue, -0.13), ("false", ps.C.vermillion, 0.13)):
        for i, (p, k, _) in enumerate(cells):
            y = np.array([v for v, _ in G[(p, k, arm)]]); bl = np.array([b for _, b in G[(p, k, arm)]], bool)
            ax.plot(np.full((~bl).sum(), i + dx), y[~bl], "o", color=col, mfc="none", ls="none")
            ax.plot(np.full(bl.sum(), i + dx), y[bl], "o", color=col, ls="none")
            ax.plot([i + dx - 0.1, i + dx + 0.1], [math.exp(np.log(y).mean())] * 2, color=col)
        ax.plot([], [], "o", mfc="none", color=col, ls="none", label=f"factor {'on' if arm == 'true' else 'off'}")
    ax.plot([], [], "o", color="k", ls="none", label="blew up after best")
    ax.set_xticks(range(len(cells))); ax.set_xticklabels([c[2] for c in cells])
    ax.set_xlim(-0.5, len(cells) - 0.5)
    ax.set_yscale("log"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(ax, lab)
    ps.legend(ax, "lower left")
ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", "seeded_ab"))
