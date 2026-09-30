"""t-channel target factor, on vs off: the four scratch probes whose curves plateaued near 1e-5
(massless t/u-channel exchange), at D = 10^3 and 10^4, 8000 steps, the same HPs in both arms (the
cell's best lr, warm-up 0.1, lambda 1e-8, eta_min 1e-8; scripts/job_transfer_calib.sh with
data.target_propagator_tchannel=true/false). Loss is MSE of log|M|^2 at the best checkpoint
(val_loss_no_reg times prepd_std^2; the factor is exact and divides out of log|M|^2, so the two arms
compare directly).
  <base>_a  loss per probe and D, factor on vs off
  <base>_b  validation curves of uu~->gg at D = 10^4, on vs off
Data: analysis/transfer/tchannel_ab.json.
    python analysis/transfer/tchannel_ab.py
"""
import json, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

d = json.load(open(os.path.join(ROOT, "analysis", "transfer", "tchannel_ab.json")))
R = {}
for n, r in d.items():
    m = re.match(r"tp_calib_(\w+?)_d(\d+)_t8000_lr[^_]+_target_propagator_tchannel-(true|false)", n)
    R[(m.group(1), int(m.group(2)), m.group(3))] = r
PROBES = [("uubar_gg", r"$u\bar u\,gg$"), ("uubar_Zg", r"$u\bar u\,Zg$"),
          ("ee_Za", r"$ee\,Z\gamma$"), ("ee_WW", r"$ee\,WW$")]
phys = lambda r: r["val"] * r["std"] ** 2

figs = ps.panels(2)
fig, ax = figs[0]
x = np.arange(len(PROBES))
for k, dx, mk in ((6, -0.12, "o"), (8, 0.12, "s")):
    for arm, col in (("true", ps.C.blue), ("false", ps.C.vermillion)):
        y = [phys(R[(p, k, arm)]) for p, _ in PROBES]
        ax.plot(x + dx, y, mk, color=col, ls="none",
                label=f"factor {'on' if arm == 'true' else 'off'}, $D=10^{{{k / 2:g}}}$")
ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in PROBES])
ax.set_yscale("log"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
ps.shared_legend(fig, ax, ncol=2)

fig, ax = figs[1]
for arm, col in (("true", ps.C.blue), ("false", ps.C.vermillion)):
    r = R[("uubar_gg", 8, arm)]
    y = np.array(r["curve"]) * r["std"] ** 2
    ax.plot(r["every"] * np.arange(1, len(y) + 1), y, color=col, label=f"factor {'on' if arm == 'true' else 'off'}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("step"); ax.set_ylabel(r"val MSE$(\log|\mathcal{M}|^2)$")
ps.process_label(ax, r"$u\bar u\to gg$, $D=10^4$")
ps.legend(ax, "lower left")
ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", "tchannel_ab"))
