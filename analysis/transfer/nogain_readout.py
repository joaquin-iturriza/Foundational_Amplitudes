"""D108, par:nogain: does rung 7's frozen trunk already carry u ubar -> Z g where rung 6's does not? Per D of the 8k grid, a
ridge read-out of each pretraining's frozen per-event features (tools/linear_readout.py: the input of the output head,
mean-pooled per event, fitted on the cell's D training events, the penalty by leave-one-out on them) scored on the
validation split, in the MSE of log|M|^2 (as cells.py), against the same rung's fine-tune (cells.best) and scratch.
Inputs: data/nogain_readout_runs.json (the READOUT lines of run FA-ccin2p3-139138ee).
    .venv-driver/bin/python analysis/transfer/nogain_readout.py   -> data/nogain_readout.json, the table printed
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cells import best, scratch  # noqa: E402

R = {r["sweep"]: r for r in json.load(open(os.path.join(HERE, "data", "nogain_readout_runs.json")))}


def main():
    out = {}
    for k in range(2, 9):
        sc, _ = scratch("uubar_Zg", k, steered=False)
        row = {"D": 10 ** (k / 2), "scratch": sc}
        for r in (6, 7):
            x = R[f"tp3_r{r}fte_uubar_Zg_d{k}"]
            s2 = x["prepd_std"] ** 2
            ft, t = best(f"tp3_r{r}fte_uubar_Zg_d{k}")
            assert abs(t["prepd_std"] - x["prepd_std"]) < 1e-6 * x["prepd_std"]
            row[f"r{r}"] = {"readout": x["readout_val_mse"] * s2, "head_affine": x["head_affine_val_mse"] * s2,
                            "finetune": ft, "var": x["val_var"] * s2}
        out[k] = row
        print(f"d{k} scratch {sc:.2e} | " + " | ".join(
            f"r{r}: read-out {row[f'r{r}']['readout']:.2e}, head+affine {row[f'r{r}']['head_affine']:.2e}, "
            f"fine-tune {row[f'r{r}']['finetune']:.2e}" for r in (6, 7)))
    json.dump(out, open(os.path.join(HERE, "data", "nogain_readout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
