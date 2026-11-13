#!/usr/bin/env python3
"""pretrain_data_saving lower-lr-centre test readout (design passed 2026-11-12 12:47): e+e- > t t~ LO, D=3162, T=8500,
scratch and fine-tune searches at the lr centre c/3 and c/10, each against its control search (scratch: dsave_s2_scr,
fine-tune: dsave_s4_ft, same cell), by Joaquin's two reads (2026-11-12, curve_quality.py): the best validation loss of
the search (LOWER / HIGHER beyond 2x the control's, else SAME) and whether that best trial's validation falls with
training (curve_quality.py: good / bad); per search the trials it calls good.
Validation minimum in ln units (min of the val mu-MSE curve x prepd_std^2). Failed trials (diverged) listed apart.
Figure figs/lr_centre_curves_{a,b} (a scratch, b fine-tune): each search's best trial, val solid and train dashed.

    python analysis/data_saving/lr_centre_readout.py [--test lr|muonly|adamw] [--cached]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from analysis.common.readers import site_python  # noqa: E402
from curve_quality import quality, together_min  # noqa: E402
from s4_readout import failed  # noqa: E402
from suspects_readout import YLAB, ps  # noqa: E402
from trial_overfit import REMOTE, driven_rows  # noqa: E402

# test -> (cells: {cell: {arm: control search prefix}}, variants: {label: sweep tag}); the design of each test is on the
# plan entry: lr = the lower-lr-centre test, muonly = the one-change mu-only MSE test (training.heterosc_mu_only)
TTX = {"scr": "dsave_s2_scr", "ft": "dsave_s4_ft"}
UUV = {"scr": "dsave_s2_scr", "ft": "dsave_s2_ft"}
TESTS = {"lr": ({"epem_ttx_d7_t8500": TTX}, {"c/3": "lrthird", "c/10": "lrtenth"}),
         "muonly": ({"epem_ttx_d7_t8500": TTX, "ee_uu_nlo_d7_t8500": UUV, "ee_uu_nlo_d8_t8500": UUV},
                    {"mu-only": "muonlymse"}),
         # adamw = the one-change Muon-off test (training.optimizer AdamW, muon_weight_decay out of the space)
         "adamw": ({"epem_ttx_d7_t8500": TTX}, {"AdamW": "adamw"}),
         # lrx3 = the lr window x3 (gen_data_saving --lr-factor 3), validation-with-training test after D420
         "lrx3": ({c: UUV for c in ("epem_ttx_d5_t8500", "epem_ttx_d7_t8500", "ee_uu_d5_t7310", "ee_uu_d7_t8500",
                                    "bbx_zz_nlo_d5_t3840", "bbx_zz_nlo_d7_t6400")}, {"lr x3": "lrx3"}),
         # tmpl2 = the revised template (hp_ranges_readout.py) without the batch search (D426): WSD, and for the
         # fine-tunes lr_scale [0.03, 30] and layer_decay [0.5, 1] (dsave_wideft_ft; scratch has no such search)
         "tmpl2": ({"epem_ttx_d5_t8500": UUV, "epem_ttx_d7_t8500": UUV}, {"WSD": "wsd", "wide ft ranges": "wideft"})}
TEST = sys.argv[sys.argv.index("--test") + 1] if "--test" in sys.argv else "lr"
CELLS, VARIANTS = TESTS[TEST]
SUFFIX = "" if TEST == "lr" else f"_{TEST}"
DATA = os.path.join(HERE, "data")


def sweep(cell, arm, centre):
    return f"{CELLS[cell][arm]}_{cell}" if centre == "control" else f"dsave_{VARIANTS[centre]}_{arm}_{cell}"


def rows(cached):
    cache = os.path.join(DATA, f"lr_centre_rows{SUFFIX}.json")
    if cached:
        return json.load(open(cache))
    names = sorted(sweep(c, a, v) for c in CELLS for a in CELLS[c] for v in ["control", *VARIANTS])
    # driven searches (sweeps_local) are read on every site that ran a trial (lxplus too), the rest on CC
    drv = [n for n in names if os.path.isdir(os.path.join(REPO, "sweeps_local", n, "results"))]
    r = driven_rows(drv) + (site_python(REMOTE, sorted(set(names) - set(drv)), "TOVF", timeout=1800)
                            if set(names) - set(drv) else [])
    json.dump(r, open(cache, "w"))
    return r


def search(rr, arm, name):
    out = []
    for r in rr:
        if r["sweep"] != name or not r.get("val"):
            continue
        p = r["prepd_std"] ** 2
        r["val_ln"] = [x * p for x in r["val"]]
        r["train_ln"] = [x * p for x in (r.get("train") or [])]
        r["val_min"] = min(r["val_ln"])
        hp = r["hp"]
        r["peak_lr"] = hp["lr"] * (hp["lr_scale"] if arm == "ft" else 1.0)
        r.update(quality(r["val"], r.get("train")))
        r["together_min"] = together_min(r["val"], r["train"]) * p
        out.append(r)
    return out


def issue_order(name, r):
    """A trial's place in its search's issue order: the driver's trial idx (sweeps_local/<search>/driver_state.json);
    the trial_NNNN of the run dir is the candidate index, not the order."""
    st = json.load(open(os.path.join(REPO, "sweeps_local", name, "driver_state.json")))["trials"].values()
    by_hp = {t["hp_idx"]: t["idx"] for t in st}
    return by_hp[int(r["run_dir"].rstrip("/").rsplit("_", 1)[1])]


def ratio(a, b, bar):
    """a's best val over b's, read against the per-cell 5x spread: LOWER / HIGHER beyond `bar`, else WITHIN."""
    r = a["best_min"] / b["best_min"]
    return {"ratio": r, "read": "LOWER" if r < 1 / bar else "HIGHER" if r > bar else "WITHIN",
            "a_best_good": a["best"]["well"], "b_best_good": b["best"]["well"]}


def summary(g):
    b = min(g, key=lambda r: r["val_min"])
    return {"n": len(g), "n_good": sum(r["well"] for r in g), "best_min": b["val_min"],
            "together_min": min(r["together_min"] for r in g),
            "best": {**{k: b[k] for k in ("run_dir", "peak_lr", "separation", "end_up", "well")}, "batchsize": b["hp"].get("batchsize")}}


def verdict(s, c):
    """The arm against its control by Joaquin's two reads (curve_quality.py): the best validation loss (LOWER / HIGHER
    at 2x, else SAME) and whether that best trial's validation falls with training to the end."""
    r = s["best_min"] / c["best_min"]
    return ("LOWER" if r < 0.5 else "HIGHER" if r > 2 else "SAME") + f" ({r:.2g}x)" + \
        (", best good" if s["best"]["well"] else ", best bad")


def main():
    rr = rows("--cached" in sys.argv)
    res, best = {}, {}
    for cell in CELLS:
        for arm in CELLS[cell]:
            S = {}
            for centre in ["control", *VARIANTS]:
                name = sweep(cell, arm, centre)
                g = search(rr, arm, name)
                if not g:
                    print(f"{cell} {arm} {centre}: no trials with curves ({name})")
                    continue
                S[centre] = summary(g)
                if len(g) > 8:   # D427 test: the same search read at its first 8 issued trials
                    S[centre]["first8"] = summary(sorted(g, key=lambda r: issue_order(name, r))[:8])
                S[centre]["failed"] = failed(name)
                best[(cell, arm, centre)] = min(g, key=lambda r: r["val_min"])
            for centre, s in S.items():
                s["verdict"] = None if centre == "control" or "control" not in S else verdict(s, S["control"])
                b = s["best"]
                print(f"{cell:20s} {arm:3s} {centre:7s} n={s['n']} good {s['n_good']}  best min {s['best_min']:.2e} "
                      f"(peak lr {b['peak_lr']:.2e}: separation {b['separation']:.3g} end_up {b['end_up']:.3g})"
                      + (f"  failed {s['failed']}" if s["failed"] else "") + (f"  -> {s['verdict']}" if s["verdict"] else ""))
            res[f"{cell}/{arm}"] = S
    if TEST == "tmpl2":
        # design after D427: (A) each WSD search at 20 trials over its first 8; (B) wide fine-tune ranges over WSD at
        # 20, per fine-tune cell; (C) WSD's first 8 over the stage-2 control; per cell 5x, pooled 5^(1/sqrt(n))
        cmp = {"A": [], "B": [], "C": []}
        for k, S in res.items():
            w = S.get("WSD")
            if w and "first8" in w:
                cmp["A"].append((k, ratio(w, w["first8"], 5)))
            if w and "wide ft ranges" in S:
                cmp["B"].append((k, ratio(S["wide ft ranges"], w, 5)))
            if w and "control" in S:
                cmp["C"].append((k, ratio(w.get("first8", w), S["control"], 5)))
        res["comparisons"] = {}
        for part, L in cmp.items():
            if not L:
                continue
            G = float(np.exp(np.mean([np.log(r["ratio"]) for _, r in L])))
            bar = 5 ** (1 / np.sqrt(len(L)))
            res["comparisons"][part] = {"cells": dict(L), "pooled": G, "bar": bar,
                                        "read": "LOWER" if G < 1 / bar else "HIGHER" if G > bar else "WITHIN"}
            for k, r in L:
                print(f"({part}) {k:24s} ratio {r['ratio']:.3g} {r['read']}  best good {r['a_best_good']} vs {r['b_best_good']}")
            print(f"({part}) pooled {G:.3g} (bar {bar:.3g}) {res['comparisons'][part]['read']}")
    q = [S["control"]["best_min"] and S[v]["best_min"] / S["control"]["best_min"] for S in res.values() for v in VARIANTS
         if v in S and "control" in S]
    if TEST == "lrx3" and len(q) > 1:
        # pooled read (design 2026-11-12, after review): geometric mean of variant/control over cells, against the 5x
        # search-to-search spread (results.tex ladder) taken as 2 sd per cell: sd/sqrt(n), bar at 2 pooled sd
        G, bar = float(np.exp(np.mean(np.log(q)))), 10 ** (2 * np.log10(5) / 2 / np.sqrt(len(q)))
        good = sum(S[v]["best"]["well"] for S in res.values() for v in VARIANTS if v in S)
        res["pooled"] = {"G": G, "bar": bar, "n": len(q), "best_good": good,
                         "branch": ("LR WINDOW" if G <= bar else "GOOD BUT WORSE") if good >= 8 else "NO"}
        print(f"pooled variant/control G {G:.2f} over {len(q)} cells (bar {bar:.2f}); search bests good {good}"
              f" -> {res['pooled']['branch']}")
    json.dump(res, open(os.path.join(DATA, f"lr_centre_readout{SUFFIX}.json"), "w"), indent=1)
    COL = dict(zip(["control", *VARIANTS], ["black", ps.C.vermillion, ps.C.blue]))
    for cell in CELLS:
        target, D = cell.rsplit("_d", 1)[0], round(10 ** (int(cell.rsplit("_d", 1)[1].split("_")[0]) / 2))
        figs = ps.panels(len(CELLS[cell]))
        for (fig, ax), arm in zip(figs, CELLS[cell]):
            for centre, col in COL.items():
                r = best.get((cell, arm, centre))
                if r is None:
                    continue
                st = np.arange(len(r["val_ln"])) * (r.get("every") or 1)
                ax.plot(st, r["val_ln"], color=col, label=f"{centre}, val")
                if r["train_ln"]:
                    ax.plot(st[:len(r["train_ln"])], r["train_ln"], color=col, linestyle="--", label=f"{centre}, train")
            ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel(YLAB.get(target, r"val MSE$(\ln|V|)$"))  # u u~ V is trained on V (D375)
            ps.process_label(ax, {"scr": "scratch", "ft": "fine-tune"}[arm] + rf", $D={D}$", loc="upper right")
            ps.legend(ax, "lower left")
        name = "lr_centre_curves" if TEST == "lr" else f"{TEST}_curves_{cell.rsplit('_t', 1)[0]}"  # '_ttx' also holds '_t'
        ps.save_panels(figs, os.path.join(HERE, "figs", name))


if __name__ == "__main__":
    main()
