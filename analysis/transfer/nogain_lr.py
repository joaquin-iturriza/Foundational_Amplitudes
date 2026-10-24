"""par:nogain, the learning-rate window on every probe: is ee -> Z gamma's case (scratch's best learning rate below every
fine-tune trial's, where the gain dips) general? Per probe and D cell of the 8k grid (cells.py): scratch's best trial's
lr, the range of lr over the ten rungs' fine-tune trials, the log10 distance of scratch's lr below that range (0 when
inside or above), and the median gain over the ten rungs (data/nogain.json, nogain.py).
    .venv-driver/bin/python analysis/transfer/nogain_lr.py   -> data/nogain_lr.json, the table printed
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cells import scratch, S  # noqa: E402
from rung_grid import P, fam  # noqa: E402

K = range(2, 9)


def main():
    ng = json.load(open(os.path.join(HERE, "data", "nogain.json")))
    out, rows = {}, []
    for p in P:
        for k in K:
            s, st = scratch(p, k, steered=False)
            if not s or str(k) not in ng[p]["cells"]:
                continue
            ft = [t["lr"] for r in range(10) for t in S.get(f"{fam(r)}_{p}_d{k}", []) + S.get(f"{fam(r)}_{p}_d{k}_002", [])
                  if t.get("val_loss") is not None and t.get("lr")]
            if not ft:
                continue
            g = [x for x in ng[p]["cells"][str(k)]["gain"].values() if x is not None]
            below = max(0.0, float(np.log10(min(ft) / st["lr"])))
            r = {"scratch_lr": st["lr"], "ft_lo": min(ft), "ft_hi": max(ft), "below": below,
                 "gain": float(np.median(g)), "drop": ng[p]["scratch_drop"].get(str(k))}
            out.setdefault(p, {})[k] = r
            rows.append(r)
            print(f"{p:12s} d{k}  scratch lr {st['lr']:.1e}  fine-tunes [{min(ft):.1e}, {max(ft):.1e}]  below {below:.2f}"
                  f"  gain {r['gain']:+.2f}")
    b = np.array([r["below"] for r in rows]); g = np.array([r["gain"] for r in rows])
    summ = {"cells": len(rows), "below_cells": int((b > 0).sum()),
            "gain_median_below": float(np.median(g[b > 0])) if (b > 0).any() else None,
            "gain_median_inside": float(np.median(g[b == 0])), "r_below_gain": float(np.corrcoef(b, g)[0, 1])}
    dr = [(r["below"], r["drop"], r["gain"]) for r in rows if r["drop"] is not None]
    if dr:
        X = np.array([[1, a, c] for a, c, _ in dr]); y = np.array([z for *_, z in dr])
        summ["fit_gain_on_below_and_drop"] = [float(c) for c in np.linalg.lstsq(X, y, rcond=None)[0]]
    print("SUMMARY", json.dumps(summ))
    json.dump({"summary": summ, "cells": out}, open(os.path.join(HERE, "data", "nogain_lr.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
