#!/usr/bin/env python3
"""zng_uq_benchmark (2) stage 1: the new ladder + star parent (current setup, D487) against the old finale on the same
28 processes (recipes/transfer_finale.yaml).

    python analysis/zng_uq/parent_read.py read   (reads lxplus) -> analysis/zng_uq/data/parent_vs_finale.json
    python analysis/zng_uq/parent_read.py plot   -> analysis/zng_uq/figs/zng_parent_vs_finale.{png,pdf}
    python analysis/zng_uq/parent_read.py curves -> analysis/zng_uq/figs/zng_parent_curves.{png,pdf}
    python analysis/zng_uq/parent_read.py diag   (reads lxplus) -> analysis/zng_uq/data/parent_diag.json

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
      "new_events": cnt, "new_val": v, "new_train": tr, "new_steps": len(t),
      "new_proc_val": q["proc_val_losses_no_reg"], "new_every": q["validate_every_n_steps"]}))
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


TEX = {"ee_uu": r"e^+e^-\to u\bar u", "ee_mumu": r"e^+e^-\to\mu^+\mu^-", "ee_numu": r"e^+e^-\to\nu_\mu\bar\nu_\mu",
       "ee_bhabha": r"e^+e^-\to e^+e^-", "ee_aa": r"e^+e^-\to\gamma\gamma", "uubar_ddbar": r"u\bar u\to d\bar d",
       "uu_uu": r"uu\to uu", "uubar_uubar": r"u\bar u\to u\bar u", "dd_dd": r"dd\to dd", "ee_bbbar": r"e^+e^-\to b\bar b",
       "uubar_ttbar": r"u\bar u\to t\bar t", "ee_ZZ": r"e^+e^-\to ZZ", "ee_ZH": r"e^+e^-\to ZH", "udbar_Wg": r"u\bar d\to Wg",
       "udbar_Wgg": r"u\bar d\to Wgg", "udbar_Wggg": r"u\bar d\to Wggg", "ee_uu_nlo": r"e^+e^-\to u\bar u,\ V",
       "uubar_ddbar_nlo": r"u\bar u\to d\bar d,\ V", "uubar_ZZ_nlo": r"u\bar u\to ZZ,\ V", "ee_uug": r"e^+e^-\to u\bar ug",
       "ee_uugg": r"e^+e^-\to u\bar ugg", "ee_bbbarg": r"e^+e^-\to b\bar bg", "ee_bbbargg": r"e^+e^-\to b\bar bgg",
       "uubar_ttbar_nlo": r"u\bar u\to t\bar t,\ V", "udbar_enug": r"u\bar d\to e^+\nu_e g", "udbar_taunu": r"u\bar d\to\tau^+\nu_\tau",
       "ee_mumu_ew_nlo": r"e^+e^-\to\mu^+\mu^-,\ V_{\rm EW}", "ee_uu_ew_nlo": r"e^+e^-\to u\bar u,\ V_{\rm EW}"}


DIAG = r"""
import json, os, sys
import numpy as np
d = sys.argv[1]
POOL = {"udbar_enug": ("udbar_enug_25-1000GeV", 5), "ee_bhabha": ("ee_bhabha_25-1000GeV", 4),
        "uubar_ddbar": ("uubar_ddbar_25-1000GeV", 4), "ee_mumu": ("ee_mumu_25-1000GeV", 4), "udbar_taunu": ("udbar_taunu_25-1000GeV", 4)}
def mass(v): return np.sqrt(np.maximum(v[..., 0] ** 2 - np.sum(v[..., 1:] ** 2, -1), 0))
out = {}
for sp in ("val", "train"):
    z = np.load(d + "/sigma_stage/preds_%s.npz" % sp); names = [str(x) for x in z["names"]]
    for i, p in enumerate(names):
        k = z["process_id"] == i; r = (z["pred"][k, 0] - z["truth"][k, 0]) ** 2
        out.setdefault(p, {})[sp] = dict(mse=float(r.mean()), median=float(np.median(r)), top1_share=float(r.max() / r.sum()),
                                         corr=float(np.corrcoef(z["pred"][k, 0], z["truth"][k, 0])[0, 1]), n=int(k.sum()))
        if sp == "val" and p in POOL:
            # the validated events are the pool's first rows (eval_subsample), matched to the predictions by |M|^2
            f, n = POOL[p]; a = np.array(np.load(os.path.join(os.environ["DATA_DIR"], f + "_val_smix_amplitudes.npy"), mmap_mode="r")[:k.sum()])
            mom, amp, rt = a[:, :4 * n].reshape(-1, n, 4), a[:, -1], z["raw_truth"][k, 0]
            j = np.array([int(np.argmin(np.abs(amp - t) / abs(t))) for t in rt])
            assert np.max(np.abs(amp[j] - rt) / np.abs(rt)) < 1e-4, p
            sq, pair = mass(mom[j, 0] + mom[j, 1]), mass(mom[j, 2] + mom[j, 3])
            o = np.argsort(-r)[:8]
            out[p]["val_top8"] = [dict(r2=float(r[q]), sqrt_s=float(sq[q]), m_finals12=float(pair[q])) for q in o]
print("PARENTDIAG " + json.dumps(out))
"""


def diag(out):
    """Per process, the sigma stage's residuals on its validated and train events (standardised units): MSE, median,
    the largest event's share, corr(pred, truth); for five processes the 8 largest validation residuals with sqrt(s)
    and the first two finals' pair mass."""
    sys.path.insert(0, ROOT)
    from analysis.common.readers import site_python
    d = site_python(DIAG, [NEW], "PARENTDIAG", site="lxplus")
    json.dump(d, open(out, "w"), indent=1)
    for p, v in sorted(d.items()):
        print(p, {sp: {k: round(x, 8) for k, x in v[sp].items()} for sp in ("val", "train")})
        for t in v.get("val_top8", []):
            print("   ", {k: round(x, 6) for k, x in t.items()})


def plot_curves(path):
    """The new parent's validation NLL per process against step (its per_process_metrics.json, the events it validated on)."""
    sys.path.insert(0, ROOT)
    import plot_style as ps
    ps.use()
    d = json.load(open(path))
    procs = sorted(d["new_proc_val"], key=lambda p: d["old"][p])
    fig, axs = ps.figure(ncols=2, nrows=14, sharex=True)
    for ax, p in zip(axs.flat, procs):
        v = np.array(d["new_proc_val"][p])
        ax.plot((np.arange(len(v)) + 1) * d["new_every"], v, color=ps.C.blue)
        ps.process_label(ax, "$" + TEX[p] + "$", fontsize=6)
        ax.tick_params(labelsize=6)
    for ax in axs.flat[len(procs):]:
        ax.set_visible(False)
    for ax in axs.flat[len(procs) - 2:len(procs)]:
        ax.set_xlabel("step", fontsize=7)
        ax.xaxis.set_tick_params(labelbottom=True)
    for ax in axs[:, 0]:
        ax.set_ylabel("val NLL", fontsize=7)
    ps.save(fig, os.path.join(HERE, "figs", "zng_parent_curves"))


if __name__ == "__main__":
    data = os.path.join(HERE, "data", "parent_vs_finale.json")
    if sys.argv[1] == "diag":
        diag(os.path.join(HERE, "data", "parent_diag.json"))
    else:
        {"read": read, "plot": plot, "curves": plot_curves}[sys.argv[1]](data)
