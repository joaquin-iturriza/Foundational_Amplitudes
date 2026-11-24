"""Plan finetune_mechanism, the momentum log-rescale question (D539): what model.input_soft_eps does to the inputs of
q q~ -> Z+4g and e+e- -> u u~ g. The rescale (wrappers._soft_rescale) maps p_i -> f_i p_i, f_i = log1p(x_i/eps) /
(x_i log1p(1/eps)), x_i = 2 p_i.Q/s, Q = p_0 + p_1, so the network sees the energy fraction x'_i = x_i f_i =
log1p(x_i/eps)/log1p(1/eps). Per pool (first N events of the val pool on CC) and eps: quantiles of x and x' over the
final-state particles, the bulk spread kept S = (x'_95 - x'_5)/(x_95 - x_5), the Z's factor f (its mass becomes f m_Z),
and the spread of log(2 p'_i.p'_j/s) against log(2 p_i.p_j/s) over final-state pairs.

    python3 analysis/finetune_mechanism/soft_rescale_inputs.py [--cached]
        -> data/soft_rescale_inputs.json, figs/soft_rescale_inputs
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from analysis.common.readers import site_python  # noqa: E402

POOLS = {"zgggg": "kit_uq_zgggg__me2_1000-1000GeV_val_n100000_kit_amplitudes.npy",
         "uug": "kit_ee_uug__me2_1000-1000GeV_val_n100000_kit_amplitudes.npy"}
EPS = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]
N = 20000

REMOTE = r'''
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd()); import siteconf
N = int(sys.argv[1]); eps_l = [float(e) for e in sys.argv[2].split(",")]
eta = np.array([1., -1., -1., -1.])
out = {}
for a in sys.argv[3:]:
    lab, f = a.split("#")
    rows = np.load(os.path.join(siteconf.DATA_DIR, f), mmap_mode="r")[:N].astype(np.float64)
    n = (rows.shape[1] - 1) // 5
    p = rows[:, :4 * n].reshape(-1, n, 4); pid = rows[:, 4 * n:5 * n]
    Q = p[:, 0] + p[:, 1]; s = (Q * Q * eta).sum(-1)
    x = np.abs(2 * (p * Q[:, None] * eta).sum(-1) / s[:, None])[:, 2:]          # final state
    m2 = (p * p * eta).sum(-1)[:, 2:]
    dots = np.einsum("bic,bjc,c->bij", p[:, 2:], p[:, 2:], eta)
    iu = np.triu_indices(n - 2, 1)
    L = np.log(np.abs(2 * dots[:, iu[0], iu[1]] / s[:, None]).clip(1e-30))
    r = {"n": n, "pids": [int(v) for v in pid[0]], "x_hist": np.histogram(x, 60, (0, 1.05))[0].tolist(),
         "x_q": np.quantile(x, [0.05, 0.5, 0.95]).tolist(), "m2_over_s_q": np.quantile(m2 / s[:, None], [0.05, 0.5, 0.95]).tolist(),
         "massive": int((m2 / s[:, None] > 1e-6).any(0).sum()), "sdL": float(L.std()), "eps": {}}
    mass = (m2 / s[:, None] > 1e-6)
    for e in eps_l:
        f = np.log1p(x / e) / (x * np.log1p(1 / e)); xp = x * f
        Lp = L + np.log(f[:, iu[0]] * f[:, iu[1]])
        r["eps"][repr(e)] = {"xp_q": np.quantile(xp, [0.05, 0.5, 0.95]).tolist(),
                             "xp_hist": np.histogram(xp, 60, (0, 1.05))[0].tolist(),
                             "f_massive_q": np.quantile(f[mass], [0.05, 0.5, 0.95]).tolist() if mass.any() else None,
                             "sdLp": float(Lp.std())}
    out[lab] = r
print("SRIN " + json.dumps(out))
'''


def main():
    cache = os.path.join(HERE, "data", "soft_rescale_inputs_raw.json")
    raw = json.load(open(cache)) if "--cached" in sys.argv else site_python(
        REMOTE, [str(N), ",".join(map(repr, EPS))] + [f"{k}#{v}" for k, v in POOLS.items()], "SRIN")
    json.dump(raw, open(cache, "w"))
    res = {}
    for k, r in raw.items():
        x5, _, x95 = r["x_q"]
        res[k] = {"n_particles": r["n"], "x_q": r["x_q"], "sd_log_pair": r["sdL"], "eps": {}}
        for e, v in r["eps"].items():
            a, _, b = v["xp_q"]
            res[k]["eps"][e] = {"xp_q": v["xp_q"], "spread_kept": (b - a) / (x95 - x5), "f_massive_q": v["f_massive_q"],
                                "sd_log_pair": v["sdLp"]}
    json.dump(res, open(os.path.join(HERE, "data", "soft_rescale_inputs.json"), "w"), indent=1)
    print(json.dumps(res))   # one line: verify_claims reads the last {...} line
    plot(raw)


def plot(raw):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2, nrows=2)
    names = {"zgggg": r"$q\bar q\to Z+4g$", "uug": r"$e^+e^-\to u\bar ug$"}
    cols = ps.sequence(len(EPS))
    m, h1, h2, leg = axs[0][0], axs[0][1], axs[1][0], axs[1][1]
    xs = np.logspace(-8, 0, 400)
    for e, c in zip(EPS, cols):
        m.plot(xs, np.log1p(xs / e) / np.log1p(1 / e), color=c, label=rf"$\epsilon=10^{{{int(round(np.log10(e)))}}}$")
    m.plot(xs, xs, color="k", ls="--", label="no rescale")
    for k, ls in (("zgggg", "-"), ("uug", ":")):
        lo, _, hi = raw[k]["x_q"]
        m.axvspan(lo, hi, color=ps.C.grey, alpha=0.15 if k == "zgggg" else 0.3, lw=0,
                  label="5-95% of $x$, " + (r"$q\bar q\to Z+4g$" if k == "zgggg" else r"$e^+e^-\to u\bar ug$"))
    m.set_xscale("log"); m.set_xlabel(r"energy fraction $x=2p\cdot Q/s$"); m.set_ylabel(r"network's fraction $x'$")
    edges = np.linspace(0, 1.05, 61); ctr = (edges[1:] + edges[:-1]) / 2
    for ax, k in ((h1, "zgggg"), (h2, "uug")):
        for e, c in zip(EPS, cols):
            ax.plot(ctr, raw[k]["eps"][repr(e)]["xp_hist"], color=c, drawstyle="steps-mid")
        ax.plot(ctr, raw[k]["x_hist"], color="k", ls="--", drawstyle="steps-mid")
        ax.set_xlabel(r"network's fraction $x'$"); ax.set_ylabel("final-state particles"); ax.set_yscale("log")
        ps.process_label(ax, names[k])
    ps.legend_cell(leg, [m])
    ps.save(fig, os.path.join(HERE, "figs", "soft_rescale_inputs"))


if __name__ == "__main__":
    main()
