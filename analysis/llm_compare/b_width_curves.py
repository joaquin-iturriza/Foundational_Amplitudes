"""Part (b) of llm_compare: validation MSE of z (val_loss_no_reg, his target, plain MSE) against training step for the
muP width scale-up at part (a)'s best HPs, trial 5 (8 heads, FA-ccin2p3-c891c0c2) beside each width whose log lines are
saved. Data: the logs' 'Val loss (combined)' lines, data/a_val_trial_1163.txt (trial 5) and data/b_val_h<H>.txt
(runs/llmcmp_b_width/trial_<H>/out_0.log). Prints each run's best value, where it sits and last over best.
    python analysis/llm_compare/b_width_curves.py
Output: figs/b_width_curves.{png,pdf}.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402

EVERY = 3042
RUNS = (("a_val_trial_1163", "8 heads"), ("b_val_h16", "16 heads"), ("b_val_h32", "32 heads"),
        ("b_val_h64", "64 heads"), ("b_val_h120", "120 heads"))
VAL = re.compile(r"val_loss_no_reg: ([-0-9.e+naif]+)")


def curve(path):
    return [float(m.group(1)) for m in map(VAL.search, open(path)) if m]


def main():
    ps.use()
    fig, axs = ps.figure(ncols=2)
    ax = axs[0]
    stats = {}
    for (f, lab), c in zip(RUNS, ps.CYCLE):
        p = os.path.join(HERE, "data", f + ".txt")
        if not os.path.exists(p):
            continue
        v = curve(p)
        x = [EVERY * (i + 1) for i in range(len(v))]
        ax.plot(x, v, "o-", color=c, ms=3, label=lab)
        fin = [i for i in range(len(v)) if v[i] == v[i]]     # NaN validations stay in the curve, not in the best
        i = min(fin, key=v.__getitem__)
        stats[lab] = {"best": v[i], "best_step": x[i], "n_val": len(v), "last_over_best": v[-1] / v[i]}
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"validation MSE of $z$")
    ps.legend_cell(axs[1], [ax])
    ps.save(fig, os.path.join(HERE, "figs", "b_width_curves"))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
