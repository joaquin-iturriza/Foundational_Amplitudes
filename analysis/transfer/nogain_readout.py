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
            ft, t = best(f"tp3_r{r}fte_uubar_Zg_d{k}")
            assert abs(t["prepd_std"] - x["prepd_std"]) < 1e-6 * x["prepd_std"]
            # tools/linear_readout.py already reports these times prepd_std^2 (the MSE of log|M|^2, as cells.py)
            row[f"r{r}"] = {"readout": x["readout_val_mse"], "head_affine": x["head_affine_val_mse"],
                            "finetune": ft, "var": x["val_var"]}
        out[k] = row
        print(f"d{k} scratch {sc:.2e} | " + " | ".join(
            f"r{r}: read-out {row[f'r{r}']['readout']:.2e}, head+affine {row[f'r{r}']['head_affine']:.2e}, "
            f"fine-tune {row[f'r{r}']['finetune']:.2e}" for r in (6, 7)))
    json.dump(out, open(os.path.join(HERE, "data", "nogain_readout.json"), "w"), indent=1)
    # the numbers par:nogain quotes, as the last JSON line (tools/verify_claims.py)
    g = lambda x: float(f"{x:.3g}")
    ratio = [out[k]["r6"]["readout"] / out[k]["r7"]["readout"] for k in range(3, 9)]
    hi = range(6, 9)
    ha = [out[k][f"r{r}"]["head_affine"] for k in out for r in (6, 7)]
    print(json.dumps({
        "d4": {"scratch": g(out[4]["scratch"]), "r7_readout": g(out[4]["r7"]["readout"]),
               "r6_readout": g(out[4]["r6"]["readout"]), "r7_over_scratch": round(out[4]["r7"]["readout"] / out[4]["scratch"], 2),
               "r6_over_r7": round(ratio[1], 1)},
        "r6_over_r7_min": round(min(ratio), 1), "r6_over_r7_max": round(max(ratio), 1),
        "hiD": {"r7_min": g(min(out[k]["r7"]["readout"] for k in hi)), "r7_max": g(max(out[k]["r7"]["readout"] for k in hi)),
                "r6_min": g(min(out[k]["r6"]["readout"] for k in hi)), "r6_max": g(max(out[k]["r6"]["readout"] for k in hi))},
        "head_affine_min": round(min(ha), 2), "head_affine_max": round(max(ha), 2)}))


if __name__ == "__main__":
    main()
