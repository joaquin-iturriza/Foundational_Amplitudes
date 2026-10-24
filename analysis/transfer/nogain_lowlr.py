"""D108, par:nogain: the ee -> Z gamma fine-tunes searched again with the learning-rate window lowered to scratch's own
(tp3_<fam>ftelo_ee_Za_d3/d4, sweep/drive.py, 5 DyHPO trials per rung as the grid's cells). Per rung and D: the new
search's best (val_loss times prepd_std^2, the MSE of log|M|^2 at the best checkpoint, as cells.py) and its lr, the
grid cell's best (cells.best) and scratch's (cells.scratch); the gain log10(scratch / fine-tune) with the grid window, the
lowered one, and the better of the two. Inputs: sweeps_local/tp3_*ftelo_ee_Za_d*/results/*.json; data/nogain_lowlr_runs.json
(each trial's prepd_std and lr from its run dir on CC, data_stats.json and config.yaml).
    .venv-driver/bin/python analysis/transfer/nogain_lowlr.py   -> data/nogain_lowlr.json, the table printed
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cells import ROOT, S, best, scratch  # noqa: E402
from rung_grid import fam  # noqa: E402

RUNS = json.load(open(os.path.join(HERE, "data", "nogain_lowlr_runs.json")))


def lowlr(r, k):
    name = f"{fam(r)}_ee_Za_d{k}".replace("fte_", "ftelo_", 1)
    rows = []
    for f in glob.glob(os.path.join(ROOT, "sweeps_local", name, "results", "*.json")):
        x = json.load(open(f))
        rd = x["run_dir"].split("Foundational_Amplitudes/", 1)[1]
        meta = RUNS[rd]
        rows.append((x["val_loss"] * meta["prepd_std"][0] ** 2, meta["lr"], meta["prepd_std"][0]))
    return min(rows) if rows else None, len(rows)


def main():
    out = {}
    for k in (3, 4):
        sc, _ = scratch("ee_Za", k, steered=False)
        g_old, g_new, g_both = [], [], []
        for r in range(10):
            ob, ot = best(f"{fam(r)}_ee_Za_d{k}")
            (nb, nlr, nstd), n = lowlr(r, k)
            assert abs(nstd - ot["prepd_std"]) < 1e-6 * nstd, (r, k, nstd, ot["prepd_std"])   # one target per cell
            bb = min(ob, nb)
            row = {"grid": ob, "grid_lr": ot["lr"], "lowlr": nb, "lowlr_lr": nlr, "trials": n,
                   "gain_grid": float(np.log10(sc / ob)), "gain_lowlr": float(np.log10(sc / nb)),
                   "gain_both": float(np.log10(sc / bb))}
            out.setdefault(f"d{k}", {"scratch": sc, "rungs": {}})["rungs"][r] = row
            g_old.append(row["gain_grid"]); g_new.append(row["gain_lowlr"]); g_both.append(row["gain_both"])
            print(f"d{k} r{r}: grid {ob:.2e} (lr {ot['lr']:.1e})  lowered {nb:.2e} (lr {nlr:.1e})  "
                  f"gain {row['gain_grid']:+.2f} -> {row['gain_lowlr']:+.2f}")
        out[f"d{k}"]["median"] = {"gain_grid": float(np.median(g_old)), "gain_lowlr": float(np.median(g_new)),
                                  "gain_both": float(np.median(g_both)), "lowlr_wins": int(sum(n > o for o, n in zip(g_old, g_new)))}
        print(f"d{k} scratch {sc:.2e}; median gain grid {np.median(g_old):+.2f}, lowered {np.median(g_new):+.2f}, "
              f"better of both {np.median(g_both):+.2f}; lowered window better on {out[f'd{k}']['median']['lowlr_wins']} of 10")
    json.dump(out, open(os.path.join(HERE, "data", "nogain_lowlr.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
