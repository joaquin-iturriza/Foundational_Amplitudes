"""par:nogain, why a fine-tune's learning curve in D is smoother than scratch's: per probe and D cell of the 8k grid, the
first validation and the best of each search's trials (median over trials; the fine-tunes pooled over the ten rungs),
in log10 of the cell's MSE, and the improvement log10(first / best) within the run. If the fine-tunes start low and
improve little where scratch drops steeply (its steepest half-decade, nogain.json), the dip is the pretrained start
holding the run near it, not the end point scratch reaches.
    .venv-driver/bin/python analysis/transfer/nogain_start.py   -> data/nogain_start.json, the summary printed
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cells import ARM, S  # noqa: E402
from rung_grid import P, fam  # noqa: E402

K = range(2, 9)


def curves(names):
    return [t["val_curve"] for n in names for t in S.get(n, []) + S.get(n + "_002", []) if t.get("val_curve")]


def stats(cs):
    if not cs:
        return None
    f = np.log10([c[0] for c in cs]); b = np.log10([min(c) for c in cs])
    return {"first": float(np.median(f)), "best": float(np.median(b)), "improve": float(np.median(f - b)), "n": len(cs)}


def main():
    ng = json.load(open(os.path.join(HERE, "data", "nogain.json")))
    out, rows = {}, []
    for p in P:
        for k in K:
            sc = stats(curves([f"{ARM[p]}_{p}_d{k}", f"tp3_scrh_{p}_d{k}", f"tp3_scrh10_{p}_d{k}"]))
            ft = stats(curves([f"{fam(r)}_{p}_d{k}" for r in range(10)]))
            if not sc or not ft or str(k) not in ng[p]["cells"]:
                continue
            g = [x for x in ng[p]["cells"][str(k)]["gain"].values() if x is not None]
            drop = ng[p]["scratch_drop"].get(str(k))
            r = {"scratch": sc, "finetune": ft, "gain": float(np.median(g)), "drop": drop}
            out.setdefault(p, {})[k] = r
            rows.append(r)
    st = [r for r in rows if r["drop"] is not None]
    dr = np.array([r["drop"] for r in st]); top = dr >= np.quantile(dr, 0.75)
    summ = {}
    for lab, m in (("steep (top quarter of scratch's drop)", top), ("the rest", ~top)):
        sel = [r for r, x in zip(st, m) if x]
        summ[lab] = {"cells": len(sel),
                     "start_gap": float(np.median([r["scratch"]["first"] - r["finetune"]["first"] for r in sel])),
                     "improve_scratch": float(np.median([r["scratch"]["improve"] for r in sel])),
                     "improve_finetune": float(np.median([r["finetune"]["improve"] for r in sel])),
                     "gain": float(np.median([r["gain"] for r in sel]))}
        print(lab, json.dumps(summ[lab]))
    za = out.get("ee_Za", {})
    for k in (3, 4):
        if k in za:
            print("ee_Za d%d" % k, json.dumps(za[k]))
    json.dump({"summary": summ, "cells": out}, open(os.path.join(HERE, "data", "nogain_start.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
