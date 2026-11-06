#!/usr/bin/env python3
"""plan warm_start step 6 (D291, D298): the k=533 warm-start sweep ws_c1_k533_mse_wsd_hpo8 (objective and checkpoint
selection the validation MSE of the mean, schedule cosine or WSD) beside the first sweep ws_c1_k533_hpo8 (cosine
only, objective the het NLL), same warm start, data, seed 42 and 533 steps. Both from ws_het_split.py fetch:
    python3 analysis/catalog_v4/ws_het_split.py fetch --sweep ws_c1_k533_mse_wsd_hpo8 --out analysis/catalog_v4/data/ws_mse_hpo8/trial_curves.json
(a) each trial's validation MSE of the mean at its selected checkpoint against its lr scale: the MSE-objective trials
at their MSE-best checkpoint, the NLL-objective trials at their NLL-best one (their own selection, so the first sweep's
MSE is not its lowest; its lowest is printed beside); (b) the validation MSE of the mean over the steps for the best
trial of the MSE sweep and the first sweep's trial nearest its lr scale. figs/ws_mse_sweep_{a,b}.{png,pdf}; the last
line printed is one JSON record (per trial lr scale, schedule, cooldown, selected and lowest MSE, NLL there).
    python3 analysis/catalog_v4/plot_ws_mse_sweep.py"""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIGS = os.path.join(HERE, "figs")
SRC = {"mse_wsd": os.path.join(HERE, "data", "ws_mse_hpo8", "trial_curves.json"),
       "nll_cosine": os.path.join(HERE, "data", "ws_hpo8", "trial_curves.json")}
LABEL = {"mse_wsd": "MSE objective, WSD", "nll_cosine": "NLL objective, cosine"}


def rows(path):
    out = {}
    for k, t in json.load(open(path)).items():
        assert t["best_step"] >= t["validate_every"], f"{k}: best_step {t['best_step']} before the first validation"
        i = t["best_step"] // t["validate_every"] - 1
        h = t["hps"]
        out[k] = {"lr_scale": h["fine_tune.lr_scale"], "scheduler": h.get("training.scheduler") or "CosineAnnealingLR",
                  "cooldown": h.get("training.wsd_cooldown_frac"), "best_step": t["best_step"],
                  "mse_selected": t["val_mse_curve"][i], "mse_lowest": float(min(t["val_mse_curve"])),
                  "nll_at_selected": t["val_curve"][i],
                  "steps": [t["validate_every"] * (j + 1) for j in range(len(t["val_mse_curve"]))],
                  "mse_curve": t["val_mse_curve"]}
    return out


def main():
    sys.path.insert(0, ROOT)
    import plot_style as ps
    R = {s: rows(p) for s, p in SRC.items()}
    best = min(R["mse_wsd"], key=lambda k: R["mse_wsd"][k]["mse_selected"])
    lr_b = R["mse_wsd"][best]["lr_scale"]
    near = min(R["nll_cosine"], key=lambda k: abs(math.log(R["nll_cosine"][k]["lr_scale"] / lr_b)))
    (fa, aa), (fb, ab) = ps.panels(2)
    for j, s in enumerate(("mse_wsd", "nll_cosine")):
        r = R[s]; ks = sorted(r, key=lambda k: r[k]["lr_scale"])
        aa.plot([r[k]["lr_scale"] for k in ks], [r[k]["mse_selected"] for k in ks], "o", color=ps.CYCLE[j],
                label=LABEL[s])
    aa.plot([lr_b], [R["mse_wsd"][best]["mse_selected"]], "*", ms=14, color=ps.CYCLE[0], mec="k",
            label="best, MSE objective")
    aa.set_xscale("log"); aa.set_yscale("log")
    aa.set_xlabel("learning-rate scale of the warm start"); aa.set_ylabel("validation MSE of the mean")
    ps.legend(aa)
    for j, (s, k, lab) in enumerate((("mse_wsd", best, f"WSD, lr scale {lr_b:.2f}"),
                                     ("nll_cosine", near, f"cosine, lr scale {R['nll_cosine'][near]['lr_scale']:.2f}"))):
        r = R[s][k]
        ab.plot(r["steps"], r["mse_curve"], color=ps.CYCLE[j], label=lab)
    ab.set_yscale("log"); ab.set_xlabel("warm-start step"); ab.set_ylabel("validation MSE of the mean")
    ps.legend(ab)
    ps.save_panels([(fa, aa), (fb, ab)], os.path.join(FIGS, "ws_mse_sweep"))
    for s in R:
        for k in sorted(R[s], key=lambda k: R[s][k]["lr_scale"]):
            x = R[s][k]
            print(f"{s:10s} {k} lr_scale {x['lr_scale']:.3g} {x['scheduler']:17s} cooldown {x['cooldown']}"
                  f" MSE selected {x['mse_selected']:.4f} lowest {x['mse_lowest']:.4f} NLL there {x['nll_at_selected']:.4f}")
    out = {s: {k: {q: v for q, v in x.items() if q not in ("steps", "mse_curve")} for k, x in R[s].items()} for s in R}
    out["best"] = {"trial": best, **out["mse_wsd"][best]}
    out["n_wsd"] = {s: sum(x["scheduler"] == "WSD" for x in R[s].values()) for s in R}
    out["best_nll_cosine_mse"] = min(x["mse_selected"] for x in R["nll_cosine"].values())
    out["nearest_cosine"] = {"trial": near, **out["nll_cosine"][near]}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
