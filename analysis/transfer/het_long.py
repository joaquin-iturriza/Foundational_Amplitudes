"""The user's heteroscedastic loss at the GP sweep's best HPs (tp_het_ee_ttbar_d8 hp1550: lr 2.3e-3, EMA off) trained
longer, no sweep (decision D61): 8k, 16k and 32k steps, all seed 1 (the sweep's own 8k trial was seed 42), e+e- -> t tbar
at D=10^4. Inputs: het_ab_data/long/t{8,16,32}k.json, the runs' result.json (runs/zz_het_ab/
tp_het_ee_ttbar_d8__trial_1550_s1_t*kfix on CC: training.increment_steps cleared, so the cosine spans the whole run),
brought home by `site run`.
Test predictions of each run's best checkpoint (tools/eval_delta.py, FA-ccin2p3-0207ccee): het_ab_data/long_eval/
zz_het_ab__tp_het_ee_ttbar_d8__trial_1550_s1_t*kfix/{preds,sigma}_test.npz.
Figures (analysis/transfer/figs/):
  het_long        validation NLL (left) and validation MSE of mu (right) against step
  het_long_calib  the actual error against the predicted sigma, per sigma decile, at each horizon
The MSE baselines it prints are scratch_sweeps.json's test losses re-based to log|M|^2 (times prepd_std^2): 8k the best of
tp_scr_ee_ttbar_d8's sweep (trial 144), 32k the paper's rule, the better on validation of the two chosen points of
tp3_scr32k_ee_ttbar_d8 (horizon32k_chosen.json).
    python analysis/transfer/het_long.py
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402

D = os.path.join(HERE, "het_ab_data")


def main():
    runs = [("8k", os.path.join(D, "long", "t8k.json"), ps.C.sky),
            ("16k", os.path.join(D, "long", "t16k.json"), ps.C.blue),
            ("32k", os.path.join(D, "long", "t32k.json"), ps.C.vermillion)]
    fig, (a1, a2) = ps.figure(ncols=2)
    for lab, f, c in runs:
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        st = d["validate_every"] * np.arange(1, len(d["val_curve"]) + 1)
        a1.plot(st, d["val_curve"], color=c, label=lab)
        a2.plot(st, d["val_mse_curve"], color=c)
        i = int(np.argmin(d["val_curve"]))
        print(f"{lab}: best val NLL {d['val_loss']:.3f} at step {d['best_step']}, val MSE(mu) there {d['val_mse_curve'][i]:.3g},"
              f" train-split MSE(mu) there {d['train_eval_mse_curve'][i]:.3g}")
    a1.set_ylim(-9.5, 0)
    a1.set_xlabel("step"); a1.set_ylabel("validation NLL")
    a2.set_yscale("log"); a2.set_xlabel("step"); a2.set_ylabel(r"validation MSE of $\mu$")
    ps.legend(a1, "upper right"); ps.process_label(a2, r"$e^+e^-\to t\bar t$")
    ps.save(fig, os.path.join(HERE, "figs", "het_long"))

    fig, ax = ps.figure()
    lo, hi = np.inf, 0
    for lab, _, c in runs:
        e = os.path.join(D, "long_eval", f"zz_het_ab__tp_het_ee_ttbar_d8__trial_1550_s1_t{lab}fix")
        z, zp = np.load(os.path.join(e, "sigma_test.npz")), np.load(os.path.join(e, "preds_test.npz"))
        mu, tr, sg = (z[k].reshape(-1).astype(float) for k in ("pred", "truth", "sigma"))
        r = np.abs(mu - tr)
        q = np.quantile(sg, np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(sg, q[1:-1]), 0, 9)
        cs = np.array([np.median(sg[idx == k]) for k in range(10)])
        rms = np.array([np.sqrt(np.mean(r[idx == k] ** 2)) for k in range(10)])
        ax.plot(cs, rms, "o-", color=c, label=lab)
        lo, hi = min(lo, cs.min(), rms.min()), max(hi, cs.max(), rms.max())
        t, pr = zp["raw_truth"].reshape(-1).astype(float), zp["raw_pred"].reshape(-1).astype(float)
        pull = (mu - tr) / sg
        print(f"{lab}: test MSE(log|M|^2) {np.mean((np.log(pr) - np.log(t)) ** 2):.3g}, median |Delta| "
              f"{np.median(np.abs(pr - t) / t):.3g}, pull std {pull.std():.2f}, median sigma / RMS error "
              f"{np.median(sg) / np.sqrt(np.mean(r ** 2)):.2f}")
    ax.plot([lo / 2, hi * 2], [lo / 2, hi * 2], "--", color="k", label="calibrated")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"predicted $\sigma$"); ax.set_ylabel("actual error")
    ps.process_label(ax, r"$e^+e^-\to t\bar t$", loc="lower right"); ps.legend(ax, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "het_long_calib"))

    sw = json.load(open(os.path.join(HERE, "scratch_sweeps.json")))
    ch = json.load(open(os.path.join(HERE, "horizon32k_chosen.json")))["tp3_scr32k_ee_ttbar_d8"]
    b8 = [r for r in sw["tp_scr_ee_ttbar_d8"] if r["hp"] == 144][0]
    b32 = min((r for r in sw["tp3_scr32k_ee_ttbar_d8"] if r["hp"] in ch), key=lambda r: r["val_loss"])
    for lab, b in (("8k", b8), ("32k", b32)):
        print(f"MSE baseline {lab} (hp {b['hp']}): test MSE(log|M|^2) {b['test_loss'] * b['prepd_std'] ** 2:.3g}")


if __name__ == "__main__":
    main()
