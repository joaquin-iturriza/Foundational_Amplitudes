"""Stage 1 of the het fine-tune benchmark (D91): the 8 het DyHPO sweeps of the finale fine-tune (sweep/gen_finale_het_sweeps.py:
e+e- -> d dbar, e+e- -> W+W-, e+e- -> d dbar one-loop, u ubar -> Z ggg at D = 10^2 and 10^4, 5 trials each, checkpoint on the
validation mu MSE). Per cell: the best trial's HPs, the spread of the 5 trials (worst over best) and whether the best is the lowest
learning rate tried. Figure: each trial's validation mu MSE over its cell's best, against its learning rate, one panel per D.
Inputs: sweeps_local/tp3_finfte_*_het/{driver_state.json,results/*.json}. Outputs: the table printed,
analysis/transfer/het_finale_s1.json, analysis/transfer/figs/het_finale_s1.{png,pdf}.
    .venv-driver/bin/python analysis/transfer/het_finale_s1.py
"""
import glob, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

PROBES = {"ee_ddbar": r"$e^+e^-\to d\bar d$", "ee_WW": r"$e^+e^-\to W^+W^-$",
          "ee_dd_nlo": r"$e^+e^-\to d\bar d$, one loop", "uubar_Zggg": r"$u\bar u\to Zggg$"}
COL = dict(zip(PROBES, (ps.C.blue, ps.C.vermillion, ps.C.green, ps.C.orange)))


def cell(p, k):
    d = os.path.join(ROOT, "sweeps_local", f"tp3_finfte_{p}_d{k}_het")
    rows = []
    for t in json.load(open(os.path.join(d, "driver_state.json")))["trials"].values():
        r = json.load(open(os.path.join(d, "results", f"hp{t['hp_idx']:04d}_t{t['t_steps']}.json")))
        rows.append(dict(val=float(r["val_loss"]), **t["hp"]))
    return sorted(rows, key=lambda x: x["val"])


def main():
    res = {}
    fig, axes = ps.figure(ncols=2)
    for ax, k in zip(axes, (4, 8)):
        for p, lab in PROBES.items():
            rows = cell(p, k)
            b = rows[0]
            res[f"{p}_d{k}"] = dict(best=b, spread=rows[-1]["val"] / b["val"],
                                    best_at_lowest_lr=b["training.lr"] == min(x["training.lr"] for x in rows), trials=rows)
            print(f"{p:11s} D=10^{k // 2}: best mu MSE {b['val']:.3g}, worst/best {rows[-1]['val'] / b['val']:6.2f}, "
                  f"lr* {b['training.lr']:.2e} (tried {min(x['training.lr'] for x in rows):.1e} to "
                  f"{max(x['training.lr'] for x in rows):.1e}), lambda {b['training.regularization_lambda']:.1e}, "
                  f"warmup {b['training.cosanneal_warmup_frac']:.2f}, ema {b['ema']}")
            ax.plot([math.log10(x["training.lr"]) for x in rows], [x["val"] / b["val"] for x in rows], "o", color=COL[p],
                    label=lab)
        ax.set_yscale("log")
        ax.set_xlim(-3.1, -2.0); ax.set_xticks([-3.0, -2.5, -2.0])
        ax.set_xlabel(r"$\log_{10}$ learning rate")
        ax.set_ylabel(r"$\mu$ MSE / cell best")
        ps.process_label(ax, rf"$D=10^{k // 2}$")
    ps.legend(axes[0], "upper left")
    n = sum(v["best_at_lowest_lr"] for v in res.values())
    print(f"best trial at the lowest learning rate tried: {n} of {len(res)} cells "
          f"({', '.join(c for c, v in res.items() if v['best_at_lowest_lr'])})")
    ps.save(fig, os.path.join(HERE, "figs", "het_finale_s1"))
    json.dump(res, open(os.path.join(HERE, "het_finale_s1.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
