#!/usr/bin/env python3
"""pretrain_data_saving lower-lr-centre test readout (design passed 2026-11-12 12:47): e+e- > t t~ LO, D=3162, T=8500,
scratch and fine-tune searches at the lr centre c/3 and c/10, each read by curve_quality.py's bars on every trial
against its control search (scratch: dsave_s2_scr, fine-tune: dsave_s4_ft, same cell), in the design's order:
  WELL             best trial passes all four bars, >= 3 of its trials pass, best val min <= 2x the control's best
  CLEAN BUT WORSE  best passes the bars (or >= 3 trials do), best val min > 2x the control's best
  BETTER, NOT WELL >= 3 trials at or below the spike bar (control fewer), or median jitter < control's / 3
  NO CHANGE        none of the above
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
from curve_quality import BAR, quality  # noqa: E402
from s4_readout import failed  # noqa: E402
from suspects_readout import YLAB, ps  # noqa: E402
from trial_overfit import REMOTE  # noqa: E402

# test -> (cells: {cell: {arm: control search prefix}}, variants: {label: sweep tag}); the design of each test is on the
# plan entry: lr = the lower-lr-centre test, muonly = the one-change mu-only MSE test (training.heterosc_mu_only)
TTX = {"scr": "dsave_s2_scr", "ft": "dsave_s4_ft"}
UUV = {"scr": "dsave_s2_scr", "ft": "dsave_s2_ft"}
TESTS = {"lr": ({"epem_ttx_d7_t8500": TTX}, {"c/3": "lrthird", "c/10": "lrtenth"}),
         "muonly": ({"epem_ttx_d7_t8500": TTX, "ee_uu_nlo_d7_t8500": UUV, "ee_uu_nlo_d8_t8500": UUV},
                    {"mu-only": "muonlymse"}),
         # adamw = the one-change Muon-off test (training.optimizer AdamW, muon_weight_decay out of the space)
         "adamw": ({"epem_ttx_d7_t8500": TTX}, {"AdamW": "adamw"})}
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
    r = site_python(REMOTE, names, "TOVF", timeout=1800)
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
        r.update(quality(r["val"], r.get("train"), hp["warmup_frac"] or 0.0))
        out.append(r)
    return out


def summary(g):
    b = min(g, key=lambda r: r["val_min"])
    return {"n": len(g), "n_well": sum(r["well"] for r in g), "n_spike_ok": sum(r["spike"] <= BAR["spike"] for r in g),
            "median_jitter": float(np.median([r["jitter"] for r in g])), "best_min": b["val_min"],
            "best": {k: b[k] for k in ("run_dir", "peak_lr", "steady", "spike", "jitter", "gap", "well")}}


def verdict(s, c):
    level = s["best_min"] <= 2 * c["best_min"]
    if s["best"]["well"] and s["n_well"] >= 3 and level:
        return "WELL"
    if s["best"]["well"] or s["n_well"] >= 3:
        if not level:
            return "CLEAN BUT WORSE"
    if (s["n_spike_ok"] >= 3 > c["n_spike_ok"]) or s["median_jitter"] < c["median_jitter"] / 3:
        return "BETTER, NOT WELL"
    return "NO CHANGE"


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
                S[centre]["failed"] = failed(name)
                best[(cell, arm, centre)] = min(g, key=lambda r: r["val_min"])
            for centre, s in S.items():
                s["verdict"] = None if centre == "control" or "control" not in S else verdict(s, S["control"])
                b = s["best"]
                print(f"{cell:20s} {arm:3s} {centre:7s} n={s['n']} well {s['n_well']} spike<=3 {s['n_spike_ok']} "
                      f"median jitter {s['median_jitter']:.2f}  best min {s['best_min']:.2e} (peak lr {b['peak_lr']:.2e}: "
                      f"steady {b['steady']:.2f} spike {b['spike']:.1f} jitter {b['jitter']:.2f} gap {b['gap'] or 0:.1f})"
                      + (f"  failed {s['failed']}" if s["failed"] else "") + (f"  -> {s['verdict']}" if s["verdict"] else ""))
            res[f"{cell}/{arm}"] = S
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
        name = "lr_centre_curves" if TEST == "lr" else f"{TEST}_curves_{cell.split('_t')[0]}"
        ps.save_panels(figs, os.path.join(HERE, "figs", name))


if __name__ == "__main__":
    main()
