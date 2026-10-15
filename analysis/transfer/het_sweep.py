"""The user's heteroscedastic loss, GP sweep on e+e- -> t tbar at D=10^4, 8k steps (decision D57;
sweep/sweep_config_tp_het_ee_ttbar_d8.yaml: 8 trials in waves 3+3+2, 2000 candidates, ranked on the validation NLL).
Inputs: the trials' result JSONs, brought home into analysis/transfer/het_ab_data/sweep/ by `site run` (tar of
sweeps/tp_het_ee_ttbar_d8/results). training.track_train_eval scores a training subsample with the weights validation
uses (EMA when on), so the train and validation curves are the same model.
Figures (analysis/transfer/figs/):
  het_sweep_curves    best trial: NLL and the MSE of mu (standardized log|M|^2), validation and training split
  het_sweep_trials    validation NLL of every trial, coloured by its lr
    python analysis/transfer/het_sweep.py
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

D = os.path.join(HERE, "het_ab_data", "sweep", "sweeps", "tp_het_ee_ttbar_d8", "results")
OUT = os.path.join(HERE, "figs")
LAB = r"$e^+e^-\to t\bar t$"


def lrs():
    """lr per hp from the driver's summary (sweeps_local/tp_het_ee_ttbar_d8/summary.txt)."""
    out = {}
    for line in open(os.path.join(ROOT, "sweeps_local", "tp_het_ee_ttbar_d8", "summary.txt")):
        if line.strip().startswith("hp_") and "lr=" in line:
            out["hp" + line.split()[0][3:]] = float(line.split("lr=")[1].split()[0])
    return out


def load():
    rs = []
    for f in sorted(glob.glob(os.path.join(D, "*.json"))):
        d = json.load(open(f))
        d["hp"] = os.path.basename(f).split("_")[0]
        d["steps"] = d["validate_every"] * np.arange(1, len(d["val_curve"]) + 1)
        rs.append(d)
    return sorted(rs, key=lambda d: d["val_loss"])


def main():
    os.makedirs(OUT, exist_ok=True)
    rs = load()
    b = rs[0]
    fig, (a1, a2) = ps.figure(ncols=2)
    a1.plot(b["steps"], b["val_curve"], color=ps.C.vermillion, label="validation")
    a1.plot(b["steps"], b["train_eval_loss_no_reg_curve"], color=ps.C.blue, label="training split")
    a1.set_xlabel("step"); a1.set_ylabel("NLL")
    a2.plot(b["steps"], b["val_mse_curve"], color=ps.C.vermillion, label="validation")
    a2.plot(b["steps"], b["train_eval_mse_curve"], color=ps.C.blue, label="training split")
    a2.set_yscale("log"); a2.set_xlabel("step"); a2.set_ylabel(r"MSE of $\mu$")
    ps.process_label(a2, LAB); ps.legend(a1, "upper right")
    ps.save(fig, os.path.join(OUT, "het_sweep_curves"))

    fig, ax = ps.figure()
    import matplotlib as mpl
    lr = lrs()
    norm = mpl.colors.Normalize(np.log10(min(lr.values())), np.log10(max(lr.values())))
    cmap = mpl.colormaps["viridis"]
    for d in rs:
        ax.plot(d["steps"], d["val_curve"], color=cmap(norm(np.log10(lr[d["hp"]]))))
    # loss spikes early in training reach +20; the axis stops at 0 so the endpoints that rank the trials can be read
    ax.set_ylim(min(min(d["val_curve"]) for d in rs) - 0.5, 0)
    ax.set_xlabel("step"); ax.set_ylabel("validation NLL")
    ps.process_label(ax, LAB)
    ps.colorbar(ax, mpl.cm.ScalarMappable(norm=norm, cmap=cmap), r"$\log_{10}$ lr")
    ps.save(fig, os.path.join(OUT, "het_sweep_trials"))
    for d in rs:
        i = int(np.argmin(d["val_curve"]))   # the best checkpoint's validation (CLAUDE.md "Reported values")
        print(f"{d['hp']}: val NLL {d['val_loss']:.3f} at step {d['best_step']}, val MSE(mu) {d['val_mse_curve'][i]:.3g}, "
              f"train-split MSE(mu) {d['train_eval_mse_curve'][i]:.3g}")


if __name__ == "__main__":
    main()
