"""The pretraining sweep on the arXiv:2606.23791 catalog (D59; sweep/sweep_config_dancat_het_pre.yaml: 1352 training
processes, the user's het loss ranked on the NLL, Daniel's diagram embedding, 32k steps, GP DyHPO, 8 trials in waves
3+3+2). Left: the validation NLL against the step, one line per trial, coloured by lr. Right: the validation MSE of mu
on the standardized log|M|^2 (the het fit's mean). Input: data/dancat_het_pre_curves.json (each trial's result record
on CC: val_curve, val_mse_curve, validate_every).
    python analysis/catalog_daniel/dancat_het_pre.py   -> figs/dancat_het_pre
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402

LR = {"hp0058": 8.94e-3, "hp1733": 8.91e-3, "hp1333": 6.92e-3, "hp1269": 7.22e-3, "hp1537": 4.42e-3, "hp0353": 4.04e-3,
      "hp1860": 1.73e-3, "hp1466": 8.83e-3}   # the trials' training.lr (sweeps_local/dancat_het_pre/summary.txt)


def main():
    d = json.load(open(os.path.join(HERE, "data", "dancat_het_pre_curves.json")))
    fig, (a1, a2) = ps.figure(ncols=2)
    order = sorted(d, key=lambda k: d[k]["val_loss"])
    cols = dict(zip(order, ps.sequence(len(order))))
    for k in order:
        r = d[k]
        s = np.arange(1, len(r["val_curve"]) + 1) * r["validate_every"]
        nll, mse = np.array(r["val_curve"], float), np.array(r["val_mse_curve"], float)
        lab = f"hp{k[2:].lstrip('0')}, lr {LR[k] * 1e3:.1f}e-3"
        if r["best_step"] < s[-1]:
            # a run that blew up: drawn to its best step (x), the blow-up after it named in the legend, never dropped (CLAUDE.md, reported values)
            keep = s <= r["best_step"]
            s, nll, mse = s[keep], nll[keep], mse[keep]
            lab += f", blew up after {s[-1] / 1e3:.1f}k"
            a1.plot(s[-1:], nll[-1:], "x", color=cols[k]); a2.plot(s[-1:], mse[-1:], "x", color=cols[k])
        a1.plot(s, nll, color=cols[k], label=lab)
        a2.plot(s, mse, color=cols[k])
        print(f"{k} best {r['val_loss']:.3f} at step {r['best_step']}, mu-MSE at the best step {r["val_mse_curve"][r["best_step"] // r["validate_every"] - 1]:.3g}, test {r['test_loss']:.3g}, "
              f"{r['traintime_hours']:.1f} h")
    a1.set_xlabel("step"); a1.set_ylabel("validation NLL")
    a2.set_yscale("log"); a2.set_xlabel("step"); a2.set_ylabel(r"validation MSE of $\mu$")
    ps.shared_legend(fig, a1, ncol=3)
    ps.save(fig, os.path.join(HERE, "figs", "dancat_het_pre"))


if __name__ == "__main__":
    main()
