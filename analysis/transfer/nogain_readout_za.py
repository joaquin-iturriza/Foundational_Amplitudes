"""par:nogain, ee -> Z gamma (2026-10-25): does any pretraining's frozen trunk already carry ee_Za's target at
D = 10^1.5 and 10^2, where every fine-tune loses to scratch? The read-out of tools/linear_readout.py (the input of the
output head, mean-pooled per event, a ridge fit on the cell's D training events, the penalty by leave-one-out on them,
scored on the validation split) on every rung's frozen pretraining (rung 0, the ee -> u ubar pretraining, rungs 1 to 9,
the finale), in the MSE of log|M|^2 as cells.py reports a cell, against the same rung's fine-tune (cells.best) and
scratch. If no rung's frozen features read out near scratch, the dip is in the pretrained representation; if some do,
the fine-tune fails to reach what its start already holds.
Inputs: data/nogain_readout_za_runs.json (the READOUT lines of run FA-ccin2p3-b0302fe9).
    .venv-driver/bin/python analysis/transfer/nogain_readout_za.py -> data/nogain_readout_za.json, figs/nogain_readout_za
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from cells import best, scratch  # noqa: E402

ARMS = [("uufte", "rung 0"), *[(f"r{r}fte", f"rung {r}") for r in range(1, 10)], ("finfte", "finale")]
KS = (3, 4)


def main():
    R = {r["sweep"]: r for r in json.load(open(os.path.join(HERE, "data", "nogain_readout_za_runs.json")))
         if "sweep" in r}
    out = {}
    for k in KS:
        sc, _ = scratch("ee_Za", k, steered=False)
        row = {"D": 10 ** (k / 2), "scratch": sc, "arms": {}}
        for a, lab in ARMS:
            sw = f"tp3_{a}_ee_Za_d{k}"
            if sw not in R:
                row["arms"][a] = {"label": lab, "missing": True}
                continue
            x = R[sw]
            s2 = x["prepd_std"] ** 2
            ft, t = best(sw)                     # None: the cell's fine-tune results are not collected here
            if t is not None:
                assert abs(t["prepd_std"] - x["prepd_std"]) < 1e-6 * x["prepd_std"], sw
            row["arms"][a] = {"label": lab, "readout": x["readout_val_mse"] * s2,
                              "head_affine": x["head_affine_val_mse"] * s2, "finetune": ft, "var": x["val_var"] * s2}
        out[k] = row
        print(f"d{k} D={row['D']:.0f} scratch {sc:.2e}")
        for a, lab in ARMS:
            v = row["arms"][a]
            if v.get("missing"):
                print(f"  {lab:8s} MISSING")
                continue
            f = v["finetune"]
            print(f"  {lab:8s} read-out {v['readout']:.2e} ({v['readout'] / sc:5.1f}x scratch)  fine-tune "
                  + (f"{f:.2e} ({f / sc:5.1f}x)" if f else "not collected") + f"  head+affine {v['head_affine']:.2e}")
    json.dump(out, open(os.path.join(HERE, "data", "nogain_readout_za.json"), "w"), indent=1)
    plot(out)


def plot(out):
    import numpy as np
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    for ax, k in zip(axs, KS):
        row = out[k]
        x = np.arange(len(ARMS))
        ro = [row["arms"][a].get("readout", np.nan) for a, _ in ARMS]
        ft = [row["arms"][a].get("finetune") or np.nan for a, _ in ARMS]
        ax.plot(x, ro, "o", color=ps.C.vermillion, label="frozen read-out")
        ax.plot(x, ft, "s", mfc="none", color=ps.C.blue, label="fine-tune")
        ax.plot([x[0], x[-1]], [row["scratch"]] * 2, ls="--", color=ps.C.grey, label="scratch")
        ax.set_yscale("log")
        ax.set_xticks(x[::2]); ax.set_xticklabels([str(i) if i < 10 else "fin" for i in x[::2]])
        ax.set_xlabel("rung"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, rf"$D=10^{{{k / 2:g}}}$")
    ps.legend(axs[0], "upper left")
    for ax in axs:
        ps.make_room(ax)
    ps.save(fig, os.path.join(HERE, "figs", "nogain_readout_za"))


if __name__ == "__main__":
    main()
