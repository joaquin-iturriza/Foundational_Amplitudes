"""Part (c) of llm_compare: validation NLL (val_loss_no_reg) of the main stage against training step, the control
(trial_1262's HPs) beside all 8 search trials, from their logs' 'Val loss (combined)' lines (data/c_val_trial_*.txt,
saved from runs/llmcmp_c_qperp/trial_<hp>/out_0.log); the sigma stage (after 'sigma_stage' in the log) is left out.
Prints each run's best value, where it sits, its largest value and the steps above 0."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

EVERY = 3042
RUNS = (("9000", "control, lr 3.65e-3"), ("1860", "lr 1.60e-3"), ("1736", "lr 2.57e-3"), ("0563", "lr 1.56e-3"),
        ("0756", "lr 2.02e-3"), ("0308", "lr 2.07e-3"), ("0902", "lr 7.01e-3"), ("1333", "lr 6.02e-3"),
        ("0353", "lr 8.56e-3"))   # control, then the 8 trials best first
VAL = re.compile(r"val_loss_no_reg: ([-0-9.e+]+)")


def main_stage(path):
    out = []
    for line in open(path):
        if "sigma_stage" in line:
            break
        m = VAL.search(line)
        if m:
            out.append(float(m.group(1)))
    return out


def main():
    ps.use()
    fig, axs = ps.figure(ncols=1, nrows=3)
    ax, az = axs[0], axs[1]     # full range (blow-ups) and a linear zoom on the trained range
    cols = (ps.C.vermillion, ps.C.blue, ps.C.green, ps.C.orange, ps.C.purple, ps.C.sky, ps.C.grey, ps.C.yellow, "k")
    stats = {}
    for (hp, lab), c in zip(RUNS, cols):
        v = main_stage(os.path.join(HERE, "data", f"c_val_trial_{hp}.txt"))
        x = [EVERY * (i + 1) for i in range(len(v))]
        ax.plot(x, v, "o-", color=c, ms=3, label=lab)
        az.plot(x, v, "o-", color=c, ms=3)
        i = min(range(len(v)), key=v.__getitem__)
        stats[hp] = {"best": v[i], "best_step": x[i], "max": max(v), "n_val": len(v), "last": v[-1],
                     "steps_above_0": [x[j] for j in range(len(v)) if v[j] > 0]}
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation NLL")
    az.set_ylim(-6.2, -1.5)       # points above -1.5 (the blow-ups) are on the full-range panel
    az.set_xlabel("training step")
    az.set_ylabel("validation NLL")
    ps.legend_cell(axs[2], [ax])
    ps.save(fig, os.path.join(HERE, "figs", "c_val_curves"))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
