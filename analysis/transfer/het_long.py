"""The user's heteroscedastic loss at the GP sweep's best HPs (tp_het_ee_ttbar_d8 hp1550: lr 2.3e-3, EMA off) trained
longer, no sweep (decision D61): 8k, 16k and 32k steps, all seed 1 (the sweep's own 8k trial was seed 42), e+e- -> t tbar
at D=10^4. Inputs: het_ab_data/long/t{8,16,32}k.json, the runs' result.json (runs/zz_het_ab/
tp_het_ee_ttbar_d8__trial_1550_s1_t*kfix on CC: training.increment_steps cleared, so the cosine spans the whole run),
brought home by `site run`.
Figure analysis/transfer/figs/het_long: validation NLL (left) and validation MSE of mu (right) against step.
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


if __name__ == "__main__":
    main()
