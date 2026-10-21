"""The sigma-only stage on the 32k het model (D69): sigma's readout retrained on the frozen trunk and mu. Calibration on
the test split of the joint 32k model, the 4k-step refit (trial lr 2.3e-3) and the best 250-step refit of
sweep_config_tp_het_sigonly_ee_ttbar_d8.yaml (hp1953, lr 0.097). Inputs: tools/eval_delta.py outputs in
het_ab_data/long_eval/<run>/sigma_test.npz. Figure analysis/transfer/figs/het_sigonly: the RMS actual error against the
median sigma per sigma decile (left), the test pulls against N(0,1) (right).
    python analysis/transfer/het_sigonly.py
"""
import os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402

D = os.path.join(HERE, "het_ab_data", "long_eval")
RUNS = (("joint, 32k", "zz_het_ab__tp_het_ee_ttbar_d8__trial_1550_s1_t32kfix", "k"),
        (r"$\sigma$ refit, 4000 steps", "zz_het_ab__tp_het_ee_ttbar_d8__trial_1550_s1_t32k_sigonly", ps.C.blue),
        (r"$\sigma$ refit, 250 steps", "tp_het_sigonly_ee_ttbar_d8__trial_1953", ps.C.vermillion))


def main():
    fig, (a1, a2) = ps.figure(ncols=2)
    lo, hi = np.inf, 0
    for lab, run, c in RUNS:
        z = np.load(os.path.join(D, run, "sigma_test.npz"))
        mu, tr, sg = (z[k].reshape(-1).astype(float) for k in ("pred", "truth", "sigma"))
        r = np.abs(mu - tr)
        q = np.quantile(sg, np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(sg, q[1:-1]), 0, 9)
        cs = np.array([np.median(sg[idx == k]) for k in range(10)])
        rms = np.array([np.sqrt(np.mean(r[idx == k] ** 2)) for k in range(10)])
        a1.plot(cs, rms, "o-", color=c)
        a2.plot([], [], "-", color=c, label=lab)
        lo, hi = min(lo, cs.min(), rms.min()), max(hi, cs.max(), rms.max())
        pull = (mu - tr) / sg
        hh, ee = np.histogram(pull, bins=np.linspace(-5, 5, 41), density=True)
        a2.stairs(hh, ee, color=c)
        print(f"{lab}: test pull std {pull.std():.2f}, |pull|<1 {np.mean(np.abs(pull) < 1):.2f}, RMS/sigma per decile "
              f"{np.round(rms / cs, 2)}, NLL {np.mean(np.log(sg) + r ** 2 / (2 * sg ** 2)):.3f}")
    # the same limits on both axes, so the calibrated line y = x is the plot box's diagonal; the legend sits on the
    # right panel, so it does not stretch this one
    a1.set_xscale("log"); a1.set_yscale("log")
    lim = (lo / 1.5, hi * 1.5)
    a1.set_xlim(*lim); a1.set_ylim(*lim)
    a1.plot(lim, lim, "--", color="k")
    a2.plot([], [], "--", color="k", label=r"calibrated, $\mathcal{N}(0,1)$")
    a1.set_xlabel(r"predicted $\sigma$"); a1.set_ylabel("actual error")
    x = np.linspace(-5, 5, 201)
    a2.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), "--", color="k")
    a2.set_yscale("log"); a2.set_ylim(1e-4, 5)
    a2.set_xlabel(r"pull $(\mu-\log|\mathcal{M}|^2)/\sigma$"); a2.set_ylabel("density")
    ps.shared_legend(fig, a2, ncol=4); ps.process_label(a1, r"$e^+e^-\to t\bar t$", loc="lower right")
    ps.save(fig, os.path.join(HERE, "figs", "het_sigonly"))


if __name__ == "__main__":
    main()
