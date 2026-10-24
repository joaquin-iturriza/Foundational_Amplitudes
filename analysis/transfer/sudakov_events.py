"""par:nogain, the electroweak-Sudakov arm on the electroweak one-loop ee -> d dbar, per event: does each arm's error sit at
the pool's zero crossings? At D = 10^4 (the 8k grid, d8), the per-event test predictions eval_delta.py wrote for the
Delta plots ($SCRATCH/zz_delta/<sweep>__<trial>/preds_test.npz: the target the run trained on and its raw value V):
scratch (tp3_scr), rung 1 (tp3_r1fte, the pruned trial's rerun) and the arm (tp3_ssudakovewfte). Per event x = log10 of
|V| over the pool's median |V| (the distance from the crossing); in bins of x: the events' share, the mean squared error
of the trained target, its share of the total, and the fraction with the wrong sign.
CC login node, CPU: prints JSON after a __B64__ line.
    python analysis/transfer/sudakov_events.py
"""
import base64, gzip, json, os

import numpy as np

RUNS = {"scratch": "tp3_scr_ee_dd_ew_nlo_d8__trial_0154", "rung 1": "tp3_r1fte_ee_dd_ew_nlo_d8__trial_0098_rerun",
        "Sudakov arm": "tp3_ssudakovewfte_ee_dd_ew_nlo_d8__trial_0131"}
EDGES = [-np.inf, -3, -2, -1.5, -1, -0.5, 0, 0.5, np.inf]


def main():
    out = {"edges": [str(e) for e in EDGES], "arms": {}}
    for arm, d in RUNS.items():
        z = np.load(os.path.join(os.environ["SCRATCH"], "zz_delta", d, "preds_test.npz"))
        t, p = z["truth"][:, 0].astype(float), z["pred"][:, 0].astype(float)
        V, Vp = z["raw_truth"][:, 0].astype(float), z["raw_pred"][:, 0].astype(float)
        x = np.log10(np.abs(V) / np.median(np.abs(V)))
        e2 = (p - t) ** 2
        rows = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (x >= lo) & (x < hi)
            rows.append({"share": float(m.mean()), "mse": float(e2[m].mean()) if m.any() else None,
                         "err_share": float(e2[m].sum() / e2.sum()),
                         "wrong_sign": float(np.mean(np.sign(Vp[m]) != np.sign(V[m]))) if m.any() else None})
        out["arms"][arm] = {"mse": float(e2.mean()), "neg_frac": float(np.mean(V < 0)),
                            "wrong_sign": float(np.mean(np.sign(Vp) != np.sign(V))), "bins": rows}
    print("__B64__"); print(base64.b64encode(gzip.compress(json.dumps(out).encode())).decode())


if __name__ == "__main__":
    main()
