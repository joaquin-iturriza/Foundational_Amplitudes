#!/usr/bin/env python3
"""The held-out multiplicity gap against capacity and training length (plan v3_prep_explore, step (a), catalog_v4).

No new training: catalog_coverage's catalog_v4 runs (het loss, share-0.25 default mixture, whole training set), each
read at its best checkpoint (lowest aggregate val_loss_no_reg):
  held-out side  the 218 H2 pools' zero-shot MSE of log|M|^2 (catalog-sampled, 2000 validation events per pool),
                 data/reads/<run>_H2.json (mixture_ab.py extract)
  training side  each training pool's validation MSE of log|M|^2 from the run's sigma-stage predictions (mu frozen at
                 the best checkpoint, 200 events per pool), data/size_mse_preds.json (size_figs.py pull_preds)
Per run, cells c = (particle count, initial state) of analysis/catalog_v4/data/tree_levels.json:
  F_c  = the median training-pool MSE over the run's training pools of cell c (the 7 cells of H2: 716 pools)
  g_p  = log10(MSE_p / F_c(p)) per H2 pool,  G = median_p g_p  (the gap; par:v4_shape_gap's d on MSE, offset kept)
Width (seeds 42, 43 paired): dG(w) = G(w) - G(w8) per seed, beside the numerator's and the floor's own moves:
  dH(w) = median_p log10(MSE_p(w) / MSE_p(w8)),  dF(w) = median_c log10(F_c(w) / F_c(w8)),  so dG ~ dH - dF.
  The seed margin m = max(0.05, 2 x the largest |G(seed 43) - G(seed 42)| over the three widths).
  Verdict (the design of 2026-11-04): NARROWS when dG(w32) <= -m on both seeds; WIDENS when dG(w32) >= +m on both;
  LEVEL when |dG(w32)| < m on both; otherwise BETWEEN. w16 reported beside. Bootstrap over H2 pools (1000 draws,
  seed 0) gives each G's 16-84% band.
Length (information: one seed, tuned HPs, length and tuning together): the 4x best trial (17056 steps, its eval read)
and, once its read is in, the 16x best (68224 steps), G against w8 seed 42.
    python3 analysis/catalog_v4/gap_capacity.py [--null-only]  -> data/gap_capacity.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
from mixture_ab import WIDTH, WIDTH_RUN_DIR  # noqa: E402

FLOOR_KEY = {r: f"runs/catalog_v4_long/{d}" for r, d in WIDTH_RUN_DIR.items()}
# length points: (H2 read run, training-floor key in size_mse_preds.json); the 16x best is added when its read ends
LENGTH = {"x4_best": ("FA-ccin2p3-6f8a1eed", "runs/cov_v4_long_hpo8_cc_17056/trial_1262")}
MARGIN_MIN = 0.05


def cells():
    tl = json.load(open(os.path.join(DATA, "tree_levels.json")))
    info = dict(tl["train"])
    for ax in tl["heldout"].values():
        info.update(ax)
    return {p: f"{v['n']}{v['initial']}" for p, v in info.items()}, set(tl["train"]), set(tl["heldout"]["H2"])


def gap(run, fkey, preds, cell, train, h2, boot=None):
    held = json.load(open(os.path.join(DATA, "reads", f"{run}_H2.json")))["mse"]
    hp = sorted(p for p in held if p in h2)
    cs = {cell[p] for p in hp}
    pools = preds[fkey]["pools"]
    byc = {}
    for p, v in pools.items():
        if p in train and cell.get(p) in cs and v[0] is not None and v[0] > 0:
            byc.setdefault(cell[p], []).append(v[0])
    F = {c: float(np.median(v)) for c, v in byc.items()}
    g = np.array([np.log10(held[p] / F[cell[p]]) for p in hp])
    out = {"run": run, "floor": fkey, "n_h2": len(hp), "n_train_pools": sum(len(v) for v in byc.values()),
           "G": round(float(np.median(g)), 4), "F": {c: round(F[c], 5) for c in sorted(F)},
           "G_per_cell": {c: round(float(np.median([g[i] for i, p in enumerate(hp) if cell[p] == c])), 4)
                          for c in sorted(cs)},
           "_held": {p: held[p] for p in hp}}
    if boot is not None:
        assert all(len(i) == len(g) for i in boot[:1]), "bootstrap drawn over a different pool count"
        out["G_band"] = [round(float(np.quantile([np.median(g[i]) for i in boot], q)), 4) for q in (0.16, 0.84)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null-only", action="store_true", help="only the seed pair of width 8 (the design's check)")
    a = ap.parse_args()
    preds = json.load(open(os.path.join(DATA, "size_mse_preds.json")))["runs"]
    cell, train, h2 = cells()
    rng = np.random.default_rng(0)
    boot = [rng.integers(len(h2), size=len(h2)) for _ in range(1000)]
    widths = ("w8",) if a.null_only else ("w8", "w16", "w32")
    R = {w: [gap(r, FLOOR_KEY[r], preds, cell, train, h2, boot) for r in WIDTH[w]] for w in widths}
    seed_diff = {w: round(R[w][1]["G"] - R[w][0]["G"], 4) for w in widths}
    if a.null_only:
        print(json.dumps({"G_w8": [x["G"] for x in R["w8"]], "seed_diff_w8": seed_diff["w8"]}))
        return
    m = max(MARGIN_MIN, 2 * max(abs(v) for v in seed_diff.values()))
    res = {"design": "plan v3_prep_explore step (a), design of 2026-11-04", "margin": round(m, 4),
           "seed_diff": seed_diff, "widths": {}, "length": {}}
    for w in widths:
        res["widths"][w] = {}
        for i, s in enumerate(("seed42", "seed43")):
            x, b = R[w][i], R["w8"][i]
            common = sorted(set(x["_held"]) & set(b["_held"]))
            dH = float(np.median([np.log10(x["_held"][p] / b["_held"][p]) for p in common]))
            dF = float(np.median([np.log10(x["F"][c] / b["F"][c]) for c in x["F"] if c in b["F"]]))
            res["widths"][w][s] = {k: v for k, v in x.items() if k != "_held"}
            res["widths"][w][s].update({"dG": round(x["G"] - b["G"], 4), "dH": round(dH, 4), "dF": round(dF, 4)})
    d32 = [res["widths"]["w32"][s]["dG"] for s in ("seed42", "seed43")]
    res["verdict"] = ("NARROWS" if all(d <= -m for d in d32) else "WIDENS" if all(d >= m for d in d32)
                      else "LEVEL" if all(abs(d) < m for d in d32) else "BETWEEN")
    b42 = R["w8"][0]
    for name, (run, fkey) in LENGTH.items():
        if fkey not in preds or not os.path.exists(os.path.join(DATA, "reads", f"{run}_H2.json")):
            res["length"][name] = "read missing"
            continue
        x = gap(run, fkey, preds, cell, train, h2, boot)
        x["dG_vs_w8_seed42"] = round(x["G"] - b42["G"], 4)
        res["length"][name] = {k: v for k, v in x.items() if k != "_held"}
    json.dump(res, open(os.path.join(DATA, "gap_capacity.json"), "w"), indent=1)
    print(json.dumps({"margin": res["margin"], "seed_diff": seed_diff, "verdict": res["verdict"],
                      "G": {w: [res["widths"][w][s]["G"] for s in ("seed42", "seed43")] for w in widths},
                      "dG": {w: [res["widths"][w][s]["dG"] for s in ("seed42", "seed43")] for w in widths},
                      "dH": {w: [res["widths"][w][s]["dH"] for s in ("seed42", "seed43")] for w in widths},
                      "dF": {w: [res["widths"][w][s]["dF"] for s in ("seed42", "seed43")] for w in widths},
                      "length": {k: (v if isinstance(v, str) else (v["G"], v["dG_vs_w8_seed42"]))
                                 for k, v in res["length"].items()}}))


if __name__ == "__main__":
    main()
