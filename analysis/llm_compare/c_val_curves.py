"""Part (c) of llm_compare: validation NLL (val_loss_no_reg) of the main stage against training step, the control
(trial_1262's HPs) beside the first-wave trials, from their logs' 'Val loss (combined)' lines (data/c_val_trial_*.txt,
saved from runs/llmcmp_c_qperp/trial_<hp>/out_0.log); the sigma stage (after 'sigma_stage' in the log) is left out.
Prints each run's best value, where it sits, its largest value and the steps above 0."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

EVERY = 3042
RUNS = (("9000", "control, lr 3.65e-3"), ("1860", "lr 1.60e-3"), ("1333", "lr 6.02e-3"), ("0353", "lr 8.56e-3"))
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
    fig, axs = ps.figure(ncols=2)
    ax = axs[0]
    cols = (ps.C.vermillion, ps.C.blue, ps.C.green, ps.C.orange)
    stats = {}
    for (hp, lab), c in zip(RUNS, cols):
        v = main_stage(os.path.join(HERE, "data", f"c_val_trial_{hp}.txt"))
        x = [EVERY * (i + 1) for i in range(len(v))]
        ax.plot(x, v, "o-", color=c, ms=3, label=lab)
        i = min(range(len(v)), key=v.__getitem__)
        stats[hp] = {"best": v[i], "best_step": x[i], "max": max(v), "n_val": len(v), "last": v[-1],
                     "steps_above_0": [x[j] for j in range(len(v)) if v[j] > 0]}
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation NLL")
    ps.legend_cell(axs[1], [ax])
    ps.save(fig, os.path.join(HERE, "figs", "c_val_curves"))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
