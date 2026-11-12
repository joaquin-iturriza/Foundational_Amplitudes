#!/usr/bin/env python3
"""pretrain_data_saving: what a fine-tune starts from and how it trains against scratch, per run (D400 defect hunt,
Joaquin 2026-11-12: "something has to be wrong" on t t~). From CC, per run dir: the training-step record
(train_steps.npz: HETEROSC train loss, grad norm before clipping, lr), the validation HETEROSC loss at every
validation (out_0.log "Val loss" lines, step 0 = before any update for a fine-tune), the run's frozen data stats
(target mean/std, mass-column and momentum stats) and the parent's. Writes data/ft_curves.json and
figs/ft_curves.{png,pdf}: train (thin) and validation (markers) loss against step, fine-tune against scratch, per target.

    python analysis/data_saving/ft_curves.py <label>=<run dir on CC> ...
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from analysis.common.readers import site_python  # noqa: E402
import plot_style as ps  # noqa: E402

REMOTE = r'''
import json, os, re, sys
import numpy as np, yaml
out = {}
for a in sys.argv[1:]:
    lab, d = a.split("=", 1)
    r = {"run_dir": d}
    z = np.load(d + "/train_steps.npz")
    r["step"], r["loss"], r["grad_norm"], r["lr"] = (z[k].tolist() for k in ("step", "loss", "grad_norm", "lr"))
    r["val"] = [(int(m.group(2)), float(m.group(1))) for m in
                re.finditer(r"Val loss: (\S+) \| step (\d+)", open(d + "/out_0.log").read().split("sigma_stage")[0])]
    s = json.load(open(d + "/data_stats.json"))
    r["stats"] = {k: s.get(k) for k in ("prepd_mean", "prepd_std", "mom_mean", "mom_std", "dimless_mass_stats",
                                        "amp_trafos_pp", "target_physics_factor", "target_over_born")}
    c = yaml.safe_load(open(d + "/config.yaml"))
    pp = (c.get("fine_tune") or {}).get("pretrained_path")
    if pp:
        ps_ = json.load(open(os.path.join(os.path.dirname(os.path.dirname(pp)), "data_stats.json")))
        r["parent_stats"] = {k: ps_.get(k) for k in ("prepd_mean", "prepd_std", "mom_mean", "mom_std",
                                                     "dimless_mass_stats", "shared_standardization")}
        r["ft"] = {k: c["fine_tune"].get(k) for k in ("lr_scale", "layer_decay", "target_stats", "offshell_stats")}
    r["train_lr"] = c["training"]["lr"]
    out[lab] = r
print("FTC " + json.dumps(out))
'''


def main():
    runs = site_python(REMOTE, [f"'{a}'" for a in sys.argv[1:]], "FTC")
    json.dump(runs, open(os.path.join(REPO, "analysis/data_saving/data/ft_curves.json"), "w"))
    summ = {}
    for lab, r in runs.items():
        loss, gn = np.array(r["loss"]), np.array(r["grad_norm"])
        v = dict(r["val"])
        summ[lab] = {"val_step0": v.get(0), "val_first": r["val"][0 if 0 not in v else 1][1],
                     "val_min": min(x for _, x in r["val"]), "val_min_step": min(r["val"], key=lambda x: x[1])[0],
                     "val_end": r["val"][-1][1], "train_end": float(np.median(loss[-100:])),
                     "train_first100": float(np.median(loss[:100])), "grad_norm_step0": float(gn[0]),
                     "grad_norm_first100_median": float(np.median(gn[:100])), "lr_peak": float(max(r["lr"])),
                     "prepd": [r["stats"]["prepd_mean"][0], r["stats"]["prepd_std"][0]],
                     "parent_prepd": ([r["parent_stats"]["prepd_mean"][0], r["parent_stats"]["prepd_std"][0]]
                                      if "parent_stats" in r else None),
                     "mass_stats": r["stats"]["dimless_mass_stats"], "ft": r.get("ft")}
        print(lab, json.dumps(summ[lab]))
    targets = sorted({lab.rsplit("_", 1)[0] for lab in runs})
    fig, axs = ps.figure(ncols=2, nrows=(len(targets) + 1) // 2)
    axs = np.atleast_1d(axs).ravel()
    for ax, tg in zip(axs, targets):
        for arm, col in (("ft", ps.C.vermillion), ("scr", ps.C.blue)):
            r = runs.get(f"{tg}_{arm}")
            if r is None:
                continue
            st, lo = np.array(r["step"]), np.array(r["loss"])
            k = 50
            sm = np.convolve(lo, np.ones(k) / k, mode="valid")
            ax.plot(st[k - 1:], sm, color=col, alpha=0.4, lw=0.8)
            ax.plot([s for s, _ in r["val"]], [x for _, x in r["val"]], color=col, marker="o", ms=2, lw=1,
                    label={"ft": "fine-tune", "scr": "scratch"}[arm])
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("step")
        ax.set_ylabel("HETEROSC loss")
        ps.process_label(ax, tg)
    ps.legend(axs[0], "upper right")
    ps.save(fig, os.path.join(REPO, "analysis/data_saving/figs/ft_curves"))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
