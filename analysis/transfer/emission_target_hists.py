"""How the emission-scale target deforms log|M|^2 (D599): for a few example processes, the distribution of ln|M|^2 and
of the target the runs with data.target_physics_factor, target_physics_dim_ref=qperp, target_physics_dim_only and
target_physics_orders train on, and both against the variables where the deformation acts: the smallest pair
invariant y_min = min_{i<j} 2|p_i.p_j| / s_hat over every pair but the two beams (soft and collinear limits), and
sqrt(s_hat).

Target, per event (experiment._physics_logF with s0 = 1 GeV^2 and no coupling product, the kit configs'
sweep/kit_base/*/config.yaml): T = ln|M|^2 + L_q - q ln(4 pi alpha_s(sqrt(s_hat))) + const, L_q = sum over the N - 4
emission steps of ln q_k^2 (physics_scales.qperp_logs), q the alpha_s power, alpha_s run at one loop from 0.118 at
M_Z (mg5_pipeline_final.compute_alphas, the pools' labelling). The alpha power is a constant here and is dropped.

Processes (catalog_v3 validation pools, the shaped _smix ones training reads): e+e- -> u u~ g and u u~ g g (one and
two emissions, the KIT comparison's processes), u u~ -> Z g g and Z g g g (two and three emissions off quark beams,
the ladder's multi-leg probes).
    python analysis/transfer/emission_target_hists.py collect   # read-only site run on lxplus -> data/emission_target.npz
    python analysis/transfer/emission_target_hists.py plot      # -> figs/emission_target_<process>
"""
import base64, io, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
NPZ = os.path.join(HERE, "data", "emission_target.npz")
# name: (pool, alpha_s power of the tree |M|^2 at minimal QED order, label)
PROC = {"ee_uug": ("ee_uug_25-1000GeV_val_smix_amplitudes.npy", 1, r"$e^+e^-\to u\bar ug$"),
        "ee_uugg": ("ee_uugg_25-1000GeV_val_smix_amplitudes.npy", 2, r"$e^+e^-\to u\bar ugg$"),
        "uubar_Zgg": ("uubar_Zgg_96-1000GeV_val_smix_amplitudes.npy", 2, r"$u\bar u\to Zgg$"),
        "uubar_Zggg": ("uubar_Zggg_96-1000GeV_val_smix_amplitudes.npy", 3, r"$u\bar u\to Zggg$")}

REMOTE = r'''
import base64, io, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from physics_scales import qperp_logs
out = {}
for f in sys.argv[1:]:
    a = np.load(os.path.join(os.environ["DATA_DIR"], f))
    n = (a.shape[1] - 1) // 5
    P = a[:, :4 * n].reshape(-1, n, 4)
    g = P * np.array([1.0, -1, -1, -1])
    G = np.einsum("npc,nqc->npq", g, P)
    sh = np.abs(G[:, 0, 0] + G[:, 1, 1] + 2 * G[:, 0, 1])
    iu = np.triu_indices(n, 1)
    y = 2 * np.abs(G[:, iu[0], iu[1]]) / sh[:, None]
    y = np.where((iu[0] == 0) & (iu[1] == 1), np.inf, y).min(1)
    b = io.BytesIO()
    np.savez_compressed(b, lnM2=np.log(np.abs(a[:, -1])), Lq=qperp_logs(P), lnshat=np.log(sh), ymin=y,
                        neg=np.array((a[:, -1] <= 0).sum()))
    out[f] = base64.b64encode(b.getvalue()).decode()
print("EMTGT " + json.dumps(out))
'''


def collect():
    from common.readers import site_python
    got = site_python(REMOTE, [p for p, _, _ in PROC.values()], "EMTGT", site="lxplus")
    have = {}
    for k, (p, _, _) in PROC.items():
        z = np.load(io.BytesIO(base64.b64decode(got[p])))
        have.update({f"{k}/{x}": z[x] for x in z.files})
    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    np.savez_compressed(NPZ, **have)


def target(z, k):
    from mg5_pipeline_final import compute_alphas
    q = PROC[k][1]
    return z[f"{k}/lnM2"] + z[f"{k}/Lq"] - q * np.log(4 * np.pi * compute_alphas(np.exp(0.5 * z[f"{k}/lnshat"])))


def _band(ax, x, v, nb, **st):
    """Median and 16-84% band of v in nb equal-count bins of x."""
    e = np.quantile(x, np.linspace(0, 1, nb + 1))
    k = np.clip(np.digitize(x, e[1:-1]), 0, nb - 1)
    c = np.array([np.median(x[k == i]) for i in range(nb)])
    q = np.array([np.quantile(v[k == i], [0.16, 0.5, 0.84]) for i in range(nb)])
    ax.plot(c, q[:, 1], marker="o", **st)
    ax.fill_between(c, q[:, 0], q[:, 2], alpha=0.2, color=st["color"], linewidth=0)
    return q


def plot():
    import matplotlib.pyplot as plt
    import plot_style as ps
    z = np.load(NPZ)
    S = {"m": dict(color=ps.C.blue, label=r"$\ln|\mathcal{M}|^2$"),
         "t": dict(color=ps.C.vermillion, label=r"target with emission scales")}
    for k, (_, q, lab) in PROC.items():
        lm, T = z[f"{k}/lnM2"], target(z, k)
        ly, sq = np.log10(z[f"{k}/ymin"]), np.exp(0.5 * z[f"{k}/lnshat"])
        v = {"m": lm - np.median(lm), "t": T - np.median(T)}
        fig, axs = ps.figure(2, 2)
        axs = axs.ravel()
        e = np.linspace(min(v["m"].min(), v["t"].min()), max(v["m"].max(), v["t"].max()), 61)
        for s in ("m", "t"):
            axs[0].stairs(np.histogram(v[s], e)[0], e, **S[s])
        axs[0].set_yscale("log"); axs[0].set_xlabel("value minus its median"); axs[0].set_ylabel("events")
        slopes, qs = {}, {1: [], 2: []}
        for s in ("m", "t"):
            qs[1].append(_band(axs[1], ly, v[s], 12, **S[s]))
            slopes[s] = np.polyfit(ly, v[s], 1)[0]
            qs[2].append(_band(axs[2], sq, v[s], 12, **S[s]))
        for i in (1, 2):   # the bands set the range: never drawn off the frame
            lo, hi = min(x.min() for x in qs[i]), max(x.max() for x in qs[i])
            axs[i].set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        from matplotlib.patches import Patch
        axs[1].set_xlabel(r"$\log_{10}y_{\min}$, $y_{\min}=\min_{i<j}2|p_i\cdot p_j|/\hat s$")
        axs[2].set_xlabel(r"$\sqrt{\hat s}$ [GeV]")
        for ax in axs[1:3]:
            ax.set_ylabel("value minus its median")
        ps.process_label(axs[0], lab)
        h, l = axs[0].get_legend_handles_labels()
        axs[3].legend(h + [Patch(color="0.5", alpha=0.3)], l + ["16-84% of events per bin"], loc="center", frameon=False)
        axs[3].set_axis_off()
        ps.save(fig, os.path.join(HERE, "figs", f"emission_target_{k}"))
        plt.close(fig)
        iq = lambda x: np.subtract(*np.quantile(x, [0.84, 0.16])) / 2
        r = np.corrcoef(ly, v["m"])[0, 1], np.corrcoef(ly, v["t"])[0, 1]
        print(f"PROC {k}: n={lm.size} (|M|^2 <= 0: {int(z[f'{k}/neg'])}); half 16-84% width ln|M|^2 {iq(v['m']):.2f}, "
              f"target {iq(v['t']):.2f}; std {v['m'].std():.2f}, {v['t'].std():.2f}; slope against log10 y_min "
              f"{slopes['m']:.2f}, {slopes['t']:.2f} (corr {r[0]:.2f}, {r[1]:.2f}); log10 y_min 1-99% "
              f"{np.quantile(ly, 0.01):.1f} to {np.quantile(ly, 0.99):.1f}; slope against ln s_hat "
              f"{np.polyfit(z[f'{k}/lnshat'], v['m'], 1)[0]:.2f}, {np.polyfit(z[f'{k}/lnshat'], v['t'], 1)[0]:.2f}")


if __name__ == "__main__":
    collect() if sys.argv[1:] == ["collect"] else plot()
