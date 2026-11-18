#!/usr/bin/env python3
"""zng_uq_benchmark (2) stage 1: the new ladder + star parent (current setup, D487) against the old finale on the same
28 processes (recipes/transfer_finale.yaml).

    python analysis/zng_uq/parent_read.py read   (reads lxplus) -> analysis/zng_uq/data/parent_vs_finale.json
    python analysis/zng_uq/parent_read.py plot   -> analysis/zng_uq/figs/zng_parent_vs_finale.{png,pdf}

Per process, the validation MSE of ln|M|^2 at each run's best checkpoint:
  old finale (tp3_finale trial_0073, MSE loss, per-dataset standardisation): proc_val_losses_no_reg at the lowest
    combined val_loss_no_reg, times that dataset's prepd_std^2 (data_stats.json) -> ln|M|^2 units, on its val pools;
  new parent (HETEROSC; its logged losses are NLLs): the standardised residual^2 times prepd_std^2, averaged over sigma_stage/preds_val.npz
    (the best checkpoint's mu, kept by the sigma stage; D276), the events its evaluation scored per process.
Curve test (tools/kit_curves.py quality, on NLL in nats): the new parent's validation NLL against its training NLL
reduced to the median over each validation interval (train_steps.npz)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OLD, NEW = "runs/tp3_finale/trial_0073", "runs/data_saving/dsave_parent_finale_zng_uq"
SITE = r'''
import json, sys
import numpy as np
old, new = sys.argv[1], sys.argv[2]
m = json.load(open(old + "/plots_0/per_process_metrics.json")); st = json.load(open(old + "/data_stats.json"))
b = int(np.argmin(m["val_loss_no_reg"])); order = m["dataset_order"]
o = {p: float(m["proc_val_losses_no_reg"][p][b] * st["prepd_std"][order.index(p)] ** 2) for p in order}
z = np.load(new + "/sigma_stage/preds_val.npz"); names = [str(x) for x in z["names"]]; n = {}; cnt = {}
sn = json.load(open(new + "/data_stats.json"))["prepd_std"]
for i, p in enumerate(names):
    k = z["process_id"] == i
    # the standardised target's residual times its std: ln|M|^2 (signed log on signed pools), as the old finale's
    n[p] = float(np.mean((z["pred"][k, 0] - z["truth"][k, 0]) ** 2) * sn[i] ** 2); cnt[p] = int(k.sum())
q = json.load(open(new + "/plots_0/per_process_metrics.json"))
t = np.load(new + "/train_steps.npz")["loss"]; v = q["val_loss_no_reg"]; e = len(t) // len(v)
tr = [float(np.median(t[i * e:(i + 1) * e])) for i in range(len(v))]
print("PARENTREAD " + json.dumps({"old": o, "old_best_index": b, "old_n_vals": len(m["val_loss_no_reg"]), "new": n,
      "new_events": cnt, "new_val": v, "new_train": tr, "new_steps": len(t)}))
'''


def read(out):
    sys.path.insert(0, ROOT)
    from analysis.common.readers import site_python
    from tools.kit_curves import quality
    d = site_python(SITE, [OLD, NEW], "PARENTREAD", site="lxplus")
    d["curve"] = quality(d["new_val"], d["new_train"])
    common = sorted(set(d["old"]) & set(d["new"]))
    lr = np.log10([d["new"][p] / d["old"][p] for p in common])
    gm = lambda x: float(np.exp(np.mean(np.log(x))))
    d["summary"] = {"n_common": len(common), "gmean_old": gm([d["old"][p] for p in common]),
                    "gmean_new": gm([d["new"][p] for p in common]), "gmean_ratio_new_over_old": float(10 ** lr.mean()),
                    "n_new_lower": int((lr < 0).sum()), "ratio_min": float(10 ** lr.min()), "ratio_max": float(10 ** lr.max())}
    json.dump(d, open(out, "w"), indent=1)
    print(json.dumps({"summary": d["summary"], "curve": d["curve"]}))


def plot(path):
    sys.path.insert(0, ROOT)
    import plot_style as ps
    ps.use()
    d = json.load(open(path))
    procs = sorted(set(d["old"]) & set(d["new"]), key=lambda p: d["old"][p])
    fig, ax = ps.figure()
    x = np.arange(len(procs))
    ax.plot(x, [d["old"][p] for p in procs], "o", color=ps.C.grey, label="old finale")
    ax.plot(x, [d["new"][p] for p in procs], "s", color=ps.C.blue, label="new parent")
    ax.set_yscale("log"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ax.set_xlabel("process, ordered by the old finale's MSE"); ax.set_xticks(x[::3])
    ps.legend(ax, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "zng_parent_vs_finale"))


if __name__ == "__main__":
    data = os.path.join(HERE, "data", "parent_vs_finale.json")
    read(data) if sys.argv[1] == "read" else plot(data)
