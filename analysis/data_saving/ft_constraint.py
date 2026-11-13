"""pretrain_data_saving, D414: what in the fine-tune loses the parent's head start on b b~ > Z Z (Joaquin 2026-11-13:
"the pretrained network's starting point is much better ... but ... not able at all to maintain that advantage during
training"; on V/B "the train loss in almost all of them ends up better for the pretrained model"). No new runs: the
stage-2 searches' logged curves (curve_quality.trials(), trial_overfit.json), validation and training MSE every
`every` steps, smoothed by a 3-check running median (curve_quality.smooth).
Per cell, the search bests of both arms: v_ft(t) / v_scr(t) along the run, the step-0 ratio (the parent's zero-shot
start against the random init, both standardised with the target's own statistics) and where it first reaches 1.
Per fine-tune trial: the warmup rise, max of v over the warmup / v at step 0 (the fit the warmup destroys), against
the peak learning rate training.lr * fine_tune.lr_scale. Per cell over all trials: median end train MSE ft / scr and
median end val MSE ft / scr.
    python analysis/data_saving/ft_constraint.py   -> data/ft_constraint.json, figs/ft_constraint
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from curve_quality import smooth, trials  # noqa: E402
import plot_style as ps  # noqa: E402

TG = {"bbx_zz": "tree", "bbx_zz_nlo": "one loop over tree"}
R = [r for r in trials() if r["stage"] == "s2" and r["target"] in TG and r["val"] and len(r["val"]) > 3]
out = {"cells": {}, "rise": {}}


def rise(r):
    v = smooth(r["val"])
    iw = max(1, int(round(r["hp"]["warmup_frac"] * (len(v) - 1))))
    return 10 ** (v[:iw + 1].max() - v[0])


fig, axs = ps.figure(ncols=2, nrows=2)
for j, (tg, lab) in enumerate(TG.items()):
    ax = axs[0, j]
    Ds = sorted({r["D"] for r in R if r["target"] == tg})
    cols = ps.sequence(len(Ds))
    for D, c in zip(Ds, cols):
        b = {a: [r for r in R if r["target"] == tg and r["D"] == D and r["arm"] == a and r.get("best")] for a in ("scr", "ft")}
        if not (b["scr"] and b["ft"]):
            continue
        s, f = smooth(b["scr"][0]["val"]), smooth(b["ft"][0]["val"])
        n = min(len(s), len(f))
        fr, ratio = np.arange(n) / (n - 1), 10 ** (f[:n] - s[:n])
        ax.plot(fr, ratio, color=c, label=f"$D={D:.0f}$")
        lost = next((float(fr[i]) for i in range(n) if ratio[i] >= 1), None)
        rr = lambda a, k: [r for r in R if r["target"] == tg and r["D"] == D and r["arm"] == a and r.get("train")]
        med = lambda a, k: float(np.median([np.log10(r[k][-1]) for r in rr(a, k)]))
        out["cells"][f"{tg}/{D:.0f}"] = {
            "start_ft_over_scr": float(ratio[0]), "head_start_lost_at_frac": lost,
            "ft_warmup_frac": b["ft"][0]["hp"]["warmup_frac"], "ft_best_warmup_rise": rise(b["ft"][0]),
            "median_end_train_ft_over_scr": 10 ** (med("ft", "train") - med("scr", "train")),
            "median_end_val_ft_over_scr": 10 ** (med("ft", "val") - med("scr", "val"))}
        axs[1, 1].scatter(out["cells"][f"{tg}/{D:.0f}"]["median_end_train_ft_over_scr"],
                          out["cells"][f"{tg}/{D:.0f}"]["median_end_val_ft_over_scr"],
                          color=ps.C.vermillion if tg == "bbx_zz" else ps.C.blue, marker="o" if tg == "bbx_zz" else "s",
                          label=None)
    ax.axhline(1, color="0.5", ls="--", label="equal")
    ax.set_yscale("log"); ax.set_xlabel("fraction of the run")
    ax.set_ylabel("val MSE fine-tune / scratch")
    ps.process_label(ax, r"$b\bar b\to ZZ$, " + lab, loc="lower right")
    ps.legend(ax, "upper right", ncol=2)
    F = [r for r in R if r["target"] == tg and r["arm"] == "ft"]
    x, y = [r["peak_lr"] for r in F], [rise(r) for r in F]
    axs[1, 0].scatter(x, y, color=ps.C.vermillion if tg == "bbx_zz" else ps.C.blue,
                      marker="o" if tg == "bbx_zz" else "s", label=lab)
    out["rise"][tg] = {"n": len(F), "share_above_3": float(np.mean(np.array(y) > 3)),
                       "spearman_peak_lr": float(spearmanr(x, y)[0]),
                       "spearman_min_val": float(spearmanr(y, [min(r["val"]) for r in F])[0])}
ax = axs[1, 0]
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"peak lr $=\mathrm{lr}\cdot\mathrm{lr\_scale}$"); ax.set_ylabel("warmup rise")
ps.legend(ax, "upper left")
ax = axs[1, 1]
ax.scatter([], [], color=ps.C.vermillion, marker="o", label="tree")
ax.scatter([], [], color=ps.C.blue, marker="s", label="one loop over tree")
ax.axvline(1, color="0.5", ls="--"); ax.axhline(1, color="0.5", ls="--", label="equal")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("end train MSE fine-tune / scratch"); ax.set_ylabel("end val MSE fine-tune / scratch")
ps.legend(ax, "upper left")
for a in axs.flat:
    ps.make_room(a)
for k, v in out["cells"].items():
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
print(json.dumps(out["rise"], indent=1))
json.dump(out, open(os.path.join(HERE, "data", "ft_constraint.json"), "w"), indent=1)
ps.save(fig, os.path.join(HERE, "figs", "ft_constraint"))
