#!/usr/bin/env python3
"""zng_uq_benchmark (2) stage 1: the new ladder + star parent (current setup, D487) against the old finale on the same
28 processes (recipes/transfer_finale.yaml).

    python analysis/zng_uq/parent_read.py read   (reads lxplus) -> analysis/zng_uq/data/parent_vs_finale.json
    python analysis/zng_uq/parent_read.py fullval (after read) adds "fullval": the new parent on its whole val pools, from
        tools/eval_best_val.py EVAL_FULL=1 on lxplus (FA-lxplus-44ab7dd7, data/parent_fullval_lxplus.json): proc_val_mse x
        prepd_std^2, same best weights; plot then draws it beside the 200-event read
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


def fullval(path):
    d = json.load(open(path))
    e = json.load(open(os.path.join(HERE, "data", "parent_fullval_lxplus.json")))
    d["fullval"] = {p: v * e["prepd_std"][0] ** 2 for p, v in e["proc_val_mse"].items()}
    assert len(set(e["prepd_std"])) == 1, "one shared target std expected"
    common = sorted(set(d["old"]) & set(d["fullval"]))
    lr = np.log10([d["fullval"][p] / d["old"][p] for p in common])
    gm = lambda x: float(np.exp(np.mean(np.log(x))))
    d["summary_fullval"] = {"n_common": len(common), "gmean_fullval": gm([d["fullval"][p] for p in common]),
                            "gmean_ratio_fullval_over_old": float(10 ** lr.mean()), "n_lower": int((lr < 0).sum()),
                            "ratio_min": float(10 ** lr.min()), "ratio_max": float(10 ** lr.max())}
    json.dump(d, open(path, "w"), indent=1)
    print(json.dumps(d["summary_fullval"]))


def plot(path):
    sys.path.insert(0, ROOT)
    import plot_style as ps
    ps.use()
    d = json.load(open(path))
    procs = sorted(set(d["old"]) & set(d["new"]), key=lambda p: d["old"][p])
    fig, ax = ps.figure()
    x = np.arange(len(procs))
    ax.plot(x, [d["old"][p] for p in procs], "o", color=ps.C.grey, label="old finale")
    ax.plot(x, [d["new"][p] for p in procs], "s", color=ps.C.blue, label="new parent, 200 val events")
    if "fullval" in d:
        ax.plot(x, [d["fullval"][p] for p in procs], "^", color=ps.C.vermillion, label="new parent, whole val pools")
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


EXPLAIN = r"""
import json, os, sys
import numpy as np
old, new = sys.argv[1], sys.argv[2]
PROCS = ("udbar_enug", "uubar_ttbar_nlo", "uubar_ttbar", "udbar_taunu")
POOL = {"udbar_enug": ("udbar_enug_25-1000GeV_val_smix", 5), "udbar_taunu": ("udbar_taunu_25-1000GeV_val_smix", 4),
        "uubar_ttbar_nlo": ("uubar_ttbar_nlo_363-450GeV_val", 4)}
def mass(v): return np.sqrt(np.maximum(v[..., 0] ** 2 - np.sum(v[..., 1:] ** 2, -1), 0))
sn = json.load(open(new + "/data_stats.json"))["prepd_std"]
out = {"new": {}, "old_curves": {}}
for sp in ("val", "train"):
    z = np.load(new + "/sigma_stage/preds_%s.npz" % sp); names = [str(x) for x in z["names"]]
    for p in PROCS:
        i = names.index(p); k = z["process_id"] == i; rt = z["raw_truth"][k, 0]; rp = z["raw_pred"][k, 0]
        e = dict(log_truth=np.log(np.abs(rt)).tolist(), log_pred=np.log(np.abs(rp)).tolist(),
                 # the residual of the trained target in ln units: (standardised pred - truth) x its std
                 resid=((z["pred"][k, 0] - z["truth"][k, 0]) * sn[i]).tolist(), sigma=(z["sigma"][k, 0] * sn[i]).tolist(),
                 n_neg_truth=int((rt < 0).sum()), n_neg_pred=int((rp < 0).sum()))
        if sp == "val" and p in POOL:
            # the validated events are the pool's first rows (eval_subsample), matched to the predictions by |M|^2
            f, n = POOL[p]; f = os.path.join(os.environ["DATA_DIR"], f + "_amplitudes")
            a = np.array(np.load(f + ".npy", mmap_mode="r")[:2 * k.sum()])
            mom, amp = a[:, :4 * n].reshape(-1, n, 4), a[:, -1]
            j = np.array([int(np.argmin(np.abs(amp - t) / abs(t))) for t in rt])
            assert np.max(np.abs(amp[j] - rt) / np.abs(rt)) < 1e-4, p
            e["sqrt_s"] = mass(mom[j, 0] + mom[j, 1]).tolist(); e["m_lnu"] = mass(mom[j, 2] + mom[j, 3]).tolist()
            e["m3"] = mass(mom[j, 2]).tolist()
            if os.path.exists(f + ".alphas.npy"):
                e["alphas"] = np.load(f + ".alphas.npy")[:2 * k.sum()][j].tolist()
        out["new"].setdefault(p, {})[sp] = e
m = json.load(open(old + "/plots_0/per_process_metrics.json")); st = json.load(open(old + "/data_stats.json")); order = m["dataset_order"]
for p in PROCS:
    out["old_curves"][p] = [float(v * st["prepd_std"][order.index(p)] ** 2) for v in m["proc_val_losses_no_reg"][p]]
out["old_every"] = m["validate_every_n_steps"]
out["prepd_std"] = dict(zip([str(x) for x in np.load(new + "/sigma_stage/preds_val.npz")["names"]], sn))
print("PARENTEXPLAIN " + json.dumps(out))
"""


def explain(out):
    """For u d~ -> e+ nu g, u u~ -> t t~ (one loop), u u~ -> t t~ and u d~ -> tau+ nu: the new parent's sigma-stage
    predictions on its validated and train events (ln|M|^2 true and predicted, residual of the trained target in ln
    units, sigma), m(l nu) and sqrt(s) of the validated e nu g and tau nu events, and the old finale's per-process
    validation MSE(ln|M|^2) curves -> analysis/zng_uq/data/parent_explain.json"""
    sys.path.insert(0, ROOT)
    from analysis.common.readers import site_python
    d = site_python(EXPLAIN, [OLD, NEW], "PARENTEXPLAIN", site="lxplus")
    json.dump(d, open(out, "w"))
    for p, v in d["new"].items():
        print(p, {sp: (len(e["resid"]), e["n_neg_truth"], e["n_neg_pred"]) for sp, e in v.items()})


SHOW = (("udbar_enug", "vermillion"), ("uubar_ttbar_nlo", "green"), ("uubar_ttbar", "blue"))


def plot_learning(path):
    """zng_parent_learning (2x2): (a) the new parent's validation NLL per process for u d~ -> e+ nu g, u u~ -> t t~ one
    loop and u u~ -> t t~ against step, with its training NLL over all 28 (the only training curve it logs); (b) the old
    finale's validation MSE(ln|M|^2) of the same three against step; (c) the new parent's final model, MSE(ln|M|^2) on
    train events against validation events per process (y = x); (d) the legend."""
    sys.path.insert(0, ROOT)
    import plot_style as ps
    ps.use()
    d = json.load(open(path)); x = json.load(open(os.path.join(HERE, "data", "parent_explain.json")))
    g = json.load(open(os.path.join(HERE, "data", "parent_diag.json")))
    assert len(set(x["prepd_std"].values())) == 1, "one shared target std expected"
    sd2 = next(iter(x["prepd_std"].values())) ** 2   # parent_diag's MSEs are standardised
    fig, axs = ps.figure(ncols=2, nrows=2)
    a, b, c, leg = axs[0][0], axs[0][1], axs[1][0], axs[1][1]
    st = (np.arange(len(d["new_val"])) + 1) * d["new_every"]
    for p, col in SHOW:
        a.plot(st, d["new_proc_val"][p], color=getattr(ps.C, col), label="$" + TEX[p] + "$")
        o = x["old_curves"][p]
        b.plot((np.arange(len(o)) + 1) * x["old_every"], o, color=getattr(ps.C, col))
    a.plot(st, d["new_train"], color="k", linestyle="--", label="training, all 28 processes")
    # symlog: t t~ V's last validations reach 100 nats, the rest sit between -9 and 0; every point drawn
    a.set_xlabel("step"); a.set_ylabel("NLL (nats)"); a.set_yscale("symlog", linthresh=1.0)
    b.set_yscale("log"); b.set_xlabel("step"); b.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(b, "old finale")
    ps.process_label(a, "new parent", loc="lower left")   # t t~ V's blow-up runs into the upper right
    others = [p for p in g if p not in dict(SHOW)]
    c.plot([g[p]["train"]["mse"] * sd2 for p in others], [g[p]["val"]["mse"] * sd2 for p in others], "o",
           color=ps.C.grey, label="the other 25 processes")
    for p, col in SHOW:
        c.plot(g[p]["train"]["mse"] * sd2, g[p]["val"]["mse"] * sd2, "o", color=getattr(ps.C, col))
    lim = (1e-7, 1e2)
    c.plot(lim, lim, color="k", linestyle=":", label="validation = training")
    c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(lim); c.set_ylim(lim); c._ps_equal_limits = True
    c.set_xlabel(r"MSE$(\log|\mathcal{M}|^2)$, training"); c.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$, validation")
    ps.legend_cell(leg, [a, c])
    ps.save(fig, os.path.join(HERE, "figs", "zng_parent_learning"))
    for p, _ in SHOW:
        v = np.array(d["new_proc_val"][p]); o = np.array(x["old_curves"][p])
        print(p, "new val NLL first %.2f min %.2f (val %d) last %.2f" % (v[0], v.min(), v.argmin() + 1, v[-1]),
              "| old MSE first %.2e min %.2e last %.2e" % (o[0], o.min(), o[-1]),
              "| final train %.2e val %.2e" % (g[p]["train"]["mse"] * sd2, g[p]["val"]["mse"] * sd2))


def plot_true_pred(path):
    """zng_parent_true_pred (3x2): per process (rows: u d~ -> e+ nu g, u u~ -> t t~ one loop, u u~ -> t t~), the new
    parent's predicted against true ln|M|^2 on its 2000 train and 200 validation events (left, y = x) and the
    residual ln|M|^2_pred - ln|M|^2_true against the true value (right)."""
    sys.path.insert(0, ROOT)
    import plot_style as ps
    ps.use()
    x = json.load(open(os.path.join(HERE, "data", "parent_explain.json")))["new"]
    fig, axs = ps.figure(ncols=2, nrows=3)
    for (p, _), (l, r) in zip(SHOW, axs):
        for sp, col, lab in (("train", ps.C.blue, "training events"), ("val", ps.C.vermillion, "validation events")):
            t, q, e = (np.array(x[p][sp][k]) for k in ("log_truth", "log_pred", "resid"))
            l.plot(t, q, ".", color=col, label=lab); r.plot(t, e, ".", color=col)
        lo = min(min(x[p][sp]["log_truth"] + x[p][sp]["log_pred"]) for sp in ("train", "val"))
        hi = max(max(x[p][sp]["log_truth"] + x[p][sp]["log_pred"]) for sp in ("train", "val"))
        l.plot((lo, hi), (lo, hi), color="k", linestyle=":", label="prediction = truth")
        l.set_xlim(lo, hi); l.set_ylim(lo, hi); l._ps_equal_limits = True
        r.axhline(0, color="k", linestyle=":")
        l.set_xlabel(r"true $\log|\mathcal{M}|^2$"); l.set_ylabel(r"predicted $\log|\mathcal{M}|^2$")
        r.set_xlabel(r"true $\log|\mathcal{M}|^2$"); r.set_ylabel(r"predicted $-$ true")
        ps.process_label(l, "$" + TEX[p] + "$", loc="lower right")
    ps.legend(axs[2][0], "upper left")   # the clean process's diagonal leaves that corner empty
    ps.save(fig, os.path.join(HERE, "figs", "zng_parent_true_pred"))
    for p, _ in SHOW:
        for sp in ("train", "val"):
            t, q, e = (np.array(x[p][sp][k]) for k in ("log_truth", "log_pred", "resid"))
            print(p, sp, "n %d  MSE %.2e  corr %.3f  true range %.2f..%.2f  pred range %.2f..%.2f" %
                  (len(t), (e ** 2).mean(), np.corrcoef(t, q)[0, 1], t.min(), t.max(), q.min(), q.max()))


def plot_wpole(path):
    """zng_parent_wpole (1x2): the new parent's residual ln|M|^2_pred - ln|M|^2_true on its 200 validated events against
    the lepton pair's mass m(l nu), for u d~ -> e+ nu g (massless legs, W internal) and u d~ -> tau+ nu (tau mass
    fed per sqrt(s_hat)), with m_W."""
    sys.path.insert(0, ROOT)
    import plot_style as ps
    ps.use()
    x = json.load(open(os.path.join(HERE, "data", "parent_explain.json")))["new"]
    fig, axs = ps.figure(ncols=2, nrows=1)
    for ax, p, lab in ((axs[0], "udbar_enug", r"$m(e^+\nu_e)$ (GeV)"), (axs[1], "udbar_taunu", r"$m(\tau^+\nu_\tau)$ (GeV)")):
        v = x[p]["val"]; m, e = np.array(v["m_lnu"]), np.array(v["resid"])
        ax.plot(m, e, ".", color=ps.C.vermillion, label="validation events")
        ax.axvline(80.419, color="k", linestyle=":", label="$m_W$")
        ax.axhline(0, color="k", linestyle="-", alpha=0.3, label="prediction = truth")
        ax.set_xscale("log"); ax.set_xlabel(lab); ax.set_ylabel(r"predicted $-$ true $\log|\mathcal{M}|^2$")
        ps.process_label(ax, "$" + TEX[p] + "$", loc="lower right")
        near = np.abs(m - 80.419) < 10
        print(p, "n %d  |resid| median %.3g; |m-mW|<10 GeV: n %d median |resid| %.3g; elsewhere %.3g; max |resid| %.3g at m %.1f"
              % (len(m), np.median(np.abs(e)), near.sum(), np.median(np.abs(e[near])) if near.any() else np.nan,
                 np.median(np.abs(e[~near])), np.abs(e).max(), m[np.abs(e).argmax()]))
        for lo, hi in ((0, 40), (40, 70), (70, 90), (90, 150), (150, 400), (400, 1000)):
            k = (m >= lo) & (m < hi)
            if k.any():
                print("   m in [%d,%d): n %d  rms resid %.3g  mean resid %+.3g" % (lo, hi, k.sum(), np.sqrt((e[k] ** 2).mean()), e[k].mean()))
    ps.legend(axs[0], "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "zng_parent_wpole"))


TTV = r"""
import json, os, sys
import numpy as np
import re
# the new parent's t t~ V pools, the files its out_0.log says it loaded (no path written here)
log = open(sys.argv[1] + "/out_0.log").read()
tr, va = (re.search(r"\[%s\] uubar_ttbar_nlo: \d+ events.*? from (\S+\.npy)" % r, log).group(1) for r in ("train", "val"))
def mass(v): return np.sqrt(np.maximum(v[..., 0] ** 2 - np.sum(v[..., 1:] ** 2, -1), 0))
o = {}
for k, f in (("train", tr), ("val", va)):
    a = np.array(np.load(f, mmap_mode="r")); s = mass(a[:, :16].reshape(-1, 4, 4)[:, :2].sum(1)); lm = np.log(np.abs(a[:, -1]))
    rec = json.load(open(f + ".recipe.json"))
    idx = np.digitize(s, [363, 370, 380, 400, 450]) - 1
    o[k] = dict(file=f, n=int(len(a)), created=rec["created"], commit=rec["provenance"]["git_commit"],
                pipeline=rec["provenance"]["pipeline_sha256"], seed=rec["effective_seed"],
                alphas=np.unique(np.load(f.replace(".npy", ".alphas.npy"))).tolist(),
                median_lnM2_by_sqrts=[float(np.median(lm[idx == i])) for i in range(4)], n_by_sqrts=[int((idx == i).sum()) for i in range(4)])
print("TTVPOOLS " + json.dumps(o))
"""


def ttv_pools(out):
    """The new parent's u u~ -> t t~ (one loop) train and val pools: generation date, commit, pipeline hash, alpha_s,
    and the median ln|M|^2 in four sqrt(s) bins [363,370,380,400,450] GeV -> analysis/zng_uq/data/parent_ttv_pools.json"""
    sys.path.insert(0, ROOT)
    from analysis.common.readers import site_python
    d = site_python(TTV, [NEW], "TTVPOOLS", site="lxplus")
    json.dump(d, open(out, "w"), indent=1)
    for k, v in d.items():
        print(k, v)
    print("train - val median ln|M|^2 per sqrt(s) bin:",
          [round(a - b, 4) for a, b in zip(d["train"]["median_lnM2_by_sqrts"], d["val"]["median_lnM2_by_sqrts"])])


POOLGEN = r"""
import json, os, re, sys
log = open(sys.argv[1] + "/out_0.log").read()
o = {}
for role, p, f in re.findall(r"\[(train|val)\] (\S+): \d+ events.*? from (\S+\.npy)", log):
    r = json.load(open(f + ".recipe.json")) if os.path.exists(f + ".recipe.json") else {}
    o.setdefault(p, {})[role] = dict(file=f, created=r.get("created"), pipeline=(r.get("provenance") or {}).get("pipeline_sha256"),
                                     commit=(r.get("provenance") or {}).get("git_commit"))
print("POOLGEN " + json.dumps(o))
"""


def pool_gen(out):
    """Per process of a parent run (its out_0.log's [train] and [val] lines): each pool's file, generation date, commit
    and pipeline hash (its .recipe.json) -> analysis/zng_uq/data/parent_pool_gen.json; prints the processes whose train
    pool was generated more than a week after its val pool (the hash differs for nearly every pool: separate jobs)."""
    sys.path.insert(0, ROOT)
    from analysis.common.readers import site_python
    d = {}
    for run in (NEW, NEW + "_w"):
        d[run] = site_python(POOLGEN, [run], "POOLGEN", site="lxplus")
    json.dump(d, open(out, "w"), indent=1)
    for run, v in d.items():
        from datetime import datetime   # the 7-day line: a shortcut that only separates the 2026-11-17 regenerations from the September/October pools
        age = lambda r: (datetime.fromisoformat(r["train"]["created"]) - datetime.fromisoformat(r["val"]["created"])).days
        diff = {p: (r["train"]["created"], r["val"]["created"]) for p, r in v.items() if r.get("train", {}).get("created") and r.get("val", {}).get("created") and age(r) > 7}
        print(run, len(v), "processes;", len(diff), "whose train pool was generated more than a week after its val pool:")
        for p, c in sorted(diff.items()):
            print("   ", p, "train", c[0], "val", c[1])


RELABEL = r"""
import json, os, sys
import numpy as np
o = {}
# FA-lxplus-e3eecae0 got VL_SAVE unexpanded and wrote into the checkout's literal "$SCRATCH" dir (job script fixed since)
d = next(x for x in (os.path.join(os.environ["SCRATCH"], "zz_vl_ttv"), os.path.join("$SCRATCH", "zz_vl_ttv")) if os.path.isdir(x))
for role in ("train", "val"):
    z = np.load(os.path.join(d, "uubar_ttbar_nlo_%s.npz" % role))
    o[role] = dict(pool=str(z["pool"]), sqrts=z["sqrts"].tolist(), ok=z["ok"].tolist(),
                   log_ratio=np.log(np.abs(z["relabelled"]) / np.abs(z["stored"])).tolist())
print("RELABEL " + json.dumps(o))
"""


def relabel(out):
    """Step 15: today's one-loop labeller on the first 200 rows of the u u~ -> t t~ one-loop train and val pools the
    parent loaded (scripts/job_virt_label_check.sh VL_FROM_RUN, npz in $SCRATCH/zz_vl_ttv on lxplus) ->
    analysis/zng_uq/data/parent_ttv_relabel.json and figure zng_ttv_relabel: ln(relabelled/stored) against sqrt(s)."""
    sys.path.insert(0, ROOT)
    from analysis.common.readers import site_python
    import plot_style as ps
    ps.use()
    d = site_python(RELABEL, [], "RELABEL", site="lxplus")
    json.dump(d, open(out, "w"))
    fig, ax = ps.figure()
    for role, col, lab in (("train", ps.C.blue, "training pool"), ("val", ps.C.vermillion, "validation pool")):
        r = d[role]; k = np.array(r["ok"]); x, y = np.array(r["sqrts"])[k], np.array(r["log_ratio"])[k]
        ax.plot(x, y, ".", color=col, label=lab)
        print(role, r["pool"], "n %d unusable %d  median ln ratio %.3g  max |ln ratio| %.3g" % (k.sum(), (~k).sum(), np.median(y), np.abs(y).max()))
    ax.axhline(0, color="k", linestyle=":", label="relabel = stored")
    ax.set_xlabel(r"$\sqrt{\hat s}$ (GeV)"); ax.set_ylabel(r"$\ln(|\mathcal{M}|^2_{\rm today}/|\mathcal{M}|^2_{\rm stored})$")
    ps.process_label(ax, r"$u\bar u\to t\bar t,\ V$")
    ps.legend(ax, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "zng_ttv_relabel"))


REBUILT = r"""
import glob, hashlib, json, os, re, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import datagen
site, run = sys.argv[1], sys.argv[2]
d = os.path.join(datagen.frozen_dir())
o = {"md5": {r: hashlib.md5(open(os.path.join(d, "uubar_ttbar_nlo_363-450GeV_%s_amplitudes.npy" % r), "rb").read()).hexdigest()
             for r in ("val", "test")}}
if site == "lxplus":
    paths = {}
    for l in open(os.path.join(run, "out_0.log")):
        m = re.search(r"\[(train|val|test)\] (\S+): \d+ events.*? from (\S+\.npy)", l)
        if m:
            paths.setdefault(m.group(1), {})[m.group(2)] = m.group(3)
    o["n_processes"] = len(paths.get("train", {}))
    o["flagged"] = datagen.mixed_generation(paths)               # the load check as it runs (|z| > 5)
    o["all_z"] = datagen.mixed_generation(paths, z_max=0)        # every version-mismatched pool's z, for the record
    s = os.path.join(os.environ["SCRATCH"], "zz_vl_ttv_rebuilt")
    for role in ("train", "val"):
        z = np.load(os.path.join(s, "uubar_ttbar_nlo_%s.npz" % role))
        o[role] = dict(pool=str(z["pool"]), sqrts=z["sqrts"].tolist(), ok=z["ok"].tolist(),
                       log_ratio=np.log(np.abs(z["relabelled"]) / np.abs(z["stored"])).tolist())
print("REBUILT " + json.dumps(o))
"""


def rebuilt(out):
    """Step 17 checks after the u u~ -> t t~ one-loop val/test rebuild (prebuild_recipes.py --rebuild, lxplus
    FA-lxplus-2ee4002c, CC FA-ccin2p3-33afc1ca): md5 of the new val and test pools on both sites, datagen.mixed_generation
    on the pools the retrain (NEW + "_w") loaded, and today's labeller against the rebuilt val (virt_label_check --save
    $SCRATCH/zz_vl_ttv_rebuilt) -> data/parent_ttv_rebuilt.json; figure zng_ttv_relabel v2: the step-15 points (training
    pool, old validation pool) with the rebuilt validation pool beside them."""
    sys.path.insert(0, ROOT)
    from analysis.common.readers import site_python
    import plot_style as ps
    ps.use()
    d = {"lxplus": site_python(REBUILT, ["lxplus", NEW + "_w"], "REBUILT", site="lxplus"),
         "ccin2p3": site_python(REBUILT, ["ccin2p3", NEW + "_w"], "REBUILT", site="ccin2p3")}
    d["md5_equal"] = d["lxplus"]["md5"] == d["ccin2p3"]["md5"]
    json.dump(d, open(out, "w"))
    old = json.load(open(os.path.join(HERE, "data", "parent_ttv_relabel.json")))
    fig, ax = ps.figure()
    for r, col, lab, mk in ((old["train"], ps.C.blue, "training pool", "."), (old["val"], ps.C.vermillion, "old validation pool", "."),
                            (d["lxplus"]["val"], ps.C.green, "rebuilt validation pool", "x")):
        k = np.array(r["ok"]); x, y = np.array(r["sqrts"])[k], np.array(r["log_ratio"])[k]
        ax.plot(x, y, mk, color=col, label=lab)
        print(lab, r["pool"], "n %d unusable %d  median ln ratio %.3g  max |ln ratio| %.3g" % (k.sum(), (~k).sum(), np.median(y), np.abs(y).max()))
    ax.axhline(0, color="k", linestyle=":", label="relabel = stored")
    ax.set_xlabel(r"$\sqrt{\hat s}$ (GeV)"); ax.set_ylabel(r"$\ln(|\mathcal{M}|^2_{\rm today}/|\mathcal{M}|^2_{\rm stored})$")
    ps.process_label(ax, r"$u\bar u\to t\bar t,\ V$")
    ps.legend(ax, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "zng_ttv_relabel"))
    print(json.dumps({s: {k: v for k, v in d[s].items() if k not in ("train", "val")} for s in ("lxplus", "ccin2p3")}),
          "md5_equal", d["md5_equal"])

SITEPOOL = r"""
import json, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import datagen
o = {}
for r in ("val", "test"):
    a = np.load(os.path.join(datagen.frozen_dir(), "uubar_ttbar_nlo_363-450GeV_%s_amplitudes.npy" % r))
    o[r] = dict(shape=list(a.shape), rows=a.tolist())
print("SITEPOOL " + json.dumps(o))
"""


def sitediff(out):
    """Step 17: the rebuilt u u~ -> t t~ one-loop val and test pools on lxplus and CC differ in md5
    (parent_ttv_rebuilt.json): compared event by event -> data/parent_ttv_sitediff.json (per role: max |delta| of the
    momenta columns, max and median |ln ratio| of the labels, rows whose label differs above 1e-6)."""
    sys.path.insert(0, ROOT)
    from analysis.common.readers import site_python
    a = {s: site_python(SITEPOOL, [], "SITEPOOL", site=s) for s in ("lxplus", "ccin2p3")}
    o = {}
    for r in ("val", "test"):
        x, y = np.array(a["lxplus"][r]["rows"]), np.array(a["ccin2p3"][r]["rows"])
        assert x.shape == y.shape, (x.shape, y.shape)
        lr = np.abs(np.log(np.abs(x[:, -1]) / np.abs(y[:, -1])))
        o[r] = dict(shape=list(x.shape), max_abs_momenta=float(np.abs(x[:, :-1] - y[:, :-1]).max()),
                    rows_momenta_differ=int((np.abs(x[:, :-1] - y[:, :-1]).max(1) > 0).sum()),
                    max_abs_ln_label=float(lr.max()), median_abs_ln_label=float(np.median(lr)),
                    rows_label_above_1e_6=int((lr > 1e-6).sum()), rows_label_differ=int((lr > 0).sum()))
    json.dump(o, open(out, "w"), indent=1)
    print(json.dumps(o))


W1262 = r"""
import json, sys
import numpy as np
# trial 1262's sigma stage (its best checkpoint's mu): per process the MSE of ln|M|^2 on the val events it scored
d = sys.argv[1] + "/sigma_stage"
z = np.load(d + "/preds_val.npz"); names = [str(x) for x in z["names"]]; sd = json.load(open(d + "/data_stats.json"))["prepd_std"]
o = {}
for i, p in enumerate(names):
    k = z["process_id"] == i
    r = (z["pred"][k, 0] - z["truth"][k, 0]) * sd[i]
    o[p] = dict(mse=float(np.mean(r ** 2)), med_abs=float(np.median(np.abs(r))), n=int(k.sum()))
print("W1262 " + json.dumps(o))
"""
TOK = sorted("tap tam ta mup mum ep em vex ve vmx vm vtx vt wp wm z h a g tx t bx b cx c sx s ux u dx d".split(), key=len, reverse=True)
LIGHT = set("g a u ux d dx s sx c cx b bx".split())


def legs(name):
    """A catalog_v4 tree's legs from its name (v4_<initial>_<finals>): the two tokenised strings, None if unparsed."""
    out = []
    for part in name.split("_")[1:3]:
        t = []
        while part:
            m = next((x for x in TOK if part.startswith(x)), None)
            if m is None:
                return None
            t.append(m); part = part[len(m):]
        out.append(t)
    return out


def w_group(name):
    """The class of a catalog_v4 tree with a hadronic initial state and one lepton pair besides light partons and
    photons (b massless in v4): 'W, e or mu' (charged lepton e/mu + neutrino: the W internal, every outside leg
    massless), 'W, tau' (tau + neutrino: the tau mass fixes a scale), 'Z, massless' (a same-flavour e/mu or neutrino
    pair: the Z internal and in internal_mass_pdgs); None for anything else (nlo, variants, other legs)."""
    if name.count("_") != 2:
        return None
    lg = legs(name)
    if lg is None or not all(x in LIGHT for x in lg[0]):
        return None
    lep = [x for x in lg[1] if x not in LIGHT]
    if len(lep) != 2:
        return None
    nu = [x for x in lep if x.startswith("v")]; ch = [x for x in lep if not x.startswith("v")]
    if len(nu) == 1 and ch[0] in ("ep", "em", "mup", "mum") and nu[0][:2] == {"e": "ve", "m": "vm"}[ch[0][0]]:
        return "W, e or mu"
    if len(nu) == 1 and ch[0] in ("tap", "tam") and nu[0][:2] == "vt":
        return "W, tau"
    if sorted(lep) in (["em", "ep"], ["mum", "mup"], ["ve", "vex"], ["vm", "vmx"]):
        return "Z, massless"
    return None


def w1262(out):
    """Trial 1262 (catalog_v4 long run, dimensionless inputs, internal_mass_pdgs [23, 6, 25]): per-class val MSE of
    ln|M|^2 on its 200 scored val events per class, grouped by w_group."""
    sys.path.insert(0, ROOT)
    from analysis.common.readers import site_python
    d = site_python(W1262, ["runs/catalog_v4_long/best_cov_v4_long_hpo8_cc_68224_trial_1262"], "W1262")
    for p in d:
        d[p]["group"] = w_group(p); lg = legs(p) if d[p]["group"] else None
        d[p]["n_final"] = len(lg[1]) if lg else None
    json.dump(d, open(out, "w"), indent=1)
    allm = np.array([v["mse"] for v in d.values()])
    print("all %d classes: median MSE %.3g" % (len(allm), np.median(allm)))
    for g in ("W, e or mu", "W, tau", "Z, massless"):
        r = sorted((v["n_final"], p, v["mse"]) for p, v in d.items() if v["group"] == g)
        print(g, len(r), "median MSE %.3g" % np.median([x[2] for x in r]), "rank-percentile median %.2f"
              % np.median([(allm < x[2]).mean() for x in r]))
        for x in r:
            print("   %d %-22s %.3g" % x)


def plot_w1262(path):
    """zng_w1262 (1x1): trial 1262's val MSE(ln|M|^2) per class against the number of final legs, the three groups of
    w_group, beside the median over all its classes."""
    sys.path.insert(0, ROOT)
    import plot_style as ps
    ps.use()
    d = json.load(open(path))
    fig, ax = ps.figure()
    off = {"W, e or mu": 0.12, "W, tau": 0.0, "Z, massless": -0.12}
    lab = {"W, e or mu": r"$W$, $e/\mu$ + $\nu$ (all massless)", "W, tau": r"$W$, $\tau$ + $\nu$",
           "Z, massless": r"$Z$, $\ell^+\ell^-$ or $\nu\bar\nu$"}
    col = {"W, e or mu": ps.C.vermillion, "W, tau": ps.C.blue, "Z, massless": ps.C.green}
    for g in off:
        r = [(v["n_final"], v["mse"]) for v in d.values() if v["group"] == g]
        ax.plot([x[0] + off[g] for x in r], [x[1] for x in r], "o", color=col[g], label=lab[g])
    ax.axhline(np.median([v["mse"] for v in d.values()]), color="k", linestyle=":", label="median, all classes")
    ax.set_xticks([2, 3, 4]); ax.set_yscale("log"); ax.set_xlabel("final-state legs"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.legend(ax, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "zng_w1262"))


if __name__ == "__main__":
    data = os.path.join(HERE, "data", "parent_vs_finale.json")
    if sys.argv[1] == "diag":
        diag(os.path.join(HERE, "data", "parent_diag.json"))
    elif sys.argv[1] == "poolgen":
        pool_gen(os.path.join(HERE, "data", "parent_pool_gen.json"))
    elif sys.argv[1] == "relabel":
        relabel(os.path.join(HERE, "data", "parent_ttv_relabel.json"))
    elif sys.argv[1] == "sitediff":
        sitediff(os.path.join(HERE, "data", "parent_ttv_sitediff.json"))
    elif sys.argv[1] == "rebuilt":
        rebuilt(os.path.join(HERE, "data", "parent_ttv_rebuilt.json"))
    elif sys.argv[1] == "ttvpools":
        ttv_pools(os.path.join(HERE, "data", "parent_ttv_pools.json"))
    elif sys.argv[1] == "w1262":
        w1262(os.path.join(HERE, "data", "w1262.json"))
    elif sys.argv[1] == "w1262plot":
        plot_w1262(os.path.join(HERE, "data", "w1262.json"))
    elif sys.argv[1] == "explain":
        explain(os.path.join(HERE, "data", "parent_explain.json"))
    else:
        {"read": read, "fullval": fullval, "plot": plot, "curves": plot_curves, "learning": plot_learning,
         "truepred": plot_true_pred, "wpole": plot_wpole}[sys.argv[1]](data)
