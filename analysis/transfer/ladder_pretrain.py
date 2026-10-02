"""Transfer study, the ladder's pretraining searches (tp3_ladder_r1..9; one single-fidelity DyHPO per rung, 64k steps,
both target factors off). Data: analysis/transfer/ladder_pretrain.json (collect_ladder.py on Jean Zay and lxplus).
  <base>_curves_a..i  per rung, each finished trial's validation curve (val_loss_no_reg, the rung's geometric mean
                      over its processes), its best checkpoint marked; a dotted line where a trial diverged
  <base>_hpo          each trial's best loss over its rung's best, against the searched HPs (open: EMA off);
                      a cross: diverged, at its best checkpoint before the blow-up (tools/eval_best_val.py,
                      ladder_eval_best.json); a diverged trial not rescored yet is not drawn
  <base>_per_dataset  each process at its rung's best checkpoint (best trial so far), MSE of log|M|^2, against the rung
    python analysis/transfer/ladder_pretrain.py
"""
import json, os, sys
import numpy as np
import matplotlib.cm as cm, matplotlib.colors as mc
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

L = json.load(open(os.path.join(ROOT, "analysis", "transfer", "ladder_pretrain.json")))
RUNGS = sorted(L, key=lambda n: int(n.split("_r")[1]))
FIG = os.path.join(ROOT, "analysis", "transfer", "figs", "ladder_pretrain")
EV = json.load(open(os.path.join(ROOT, "analysis", "transfer", "ladder_eval_best.json")))
for n, ts_ in L.items():                   # a diverged trial's value: its best checkpoint, rescored
    for t in ts_:
        if t["state"] == "diverged" and str(t["hp"]) in EV.get(n, {}):
            t["val_loss"] = EV[n][str(t["hp"])]["val_loss"]
done = lambda n: [t for t in L[n] if t["state"] == "done" or (t["state"] == "diverged" and "val_loss" in t)]
best = lambda n: min(done(n), key=lambda t: t["val_loss"])

norm = mc.LogNorm(1e-4, 1e-2)
figs = ps.panels(len(RUNGS))
for (fig, ax), n in zip(figs, RUNGS):
    for t in sorted(L[n], key=lambda t: t["lr"]):
        col = cm.viridis(norm(t["lr"]))
        if t["state"] == "done":
            y = np.array(t["val_curve"]); x = (np.arange(len(y)) + 1) * t["validate_every"]
            ax.plot(x, y, "-", color=col, label=f"lr {t['lr']:.1e}")
            ax.plot(t["best_step"], t["val_loss"], "o", color=col)
        elif t["state"] == "diverged":
            ax.axvline(t["step"], color=col, ls=":", label=f"lr {t['lr']:.1e}, diverged")
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel(r"$\mathcal{L}_{\rm val}$")
    run = sum(t["state"] == "running" for t in L[n])
    ps.process_label(ax, f"rung {n.split('_r')[1]}" + (f", {run} running" if run else ""))
    ps.legend(ax, "lower left")
    ps.make_room(ax)
ps.save_panels(figs, FIG + "_curves")

knobs = [("lr", "lr", True), ("lambda", r"$\lambda$", True), ("warmup", "warm-up fraction", False),
         ("ema_decay", "EMA decay", False)]
fig, axes = ps.figure(ncols=2, nrows=2)
for ax, (k, lab, lg) in zip(axes.flat, knobs):
    for n, col in zip(RUNGS, ps.sequence(len(RUNGS))):
        b = best(n)["val_loss"]
        for t in done(n):
            ax.plot(t[k], t["val_loss"] / b, "x" if t["state"] == "diverged" else "o", color=col,
                    mfc=col if (k != "ema_decay" or t["ema"]) else "none")
        ax.plot([], [], "o", color=col, label=f"rung {n.split('_r')[1]}")
    if lg:
        ax.set_xscale("log")
    ax.set_yscale("log"); ax.set_xlabel(lab); ax.set_ylabel(r"$\mathcal{L}_{\rm val}/\mathcal{L}_{\rm val}^{\rm best}$")
axes.flat[0].plot([], [], "x", color="k", label="diverged")
ps.legend(axes.flat[0], "upper left", ncol=2)
ps.save(fig, FIG + "_hpo")

LAB = {"ee_uu": r"$ee\to u\bar u$", "ee_mumu": r"$ee\to\mu\mu$", "ee_numu": r"$ee\to\nu_\mu\bar\nu_\mu$",
       "ee_bhabha": "Bhabha", "ee_aa": r"$ee\to\gamma\gamma$", "uubar_ddbar": r"$u\bar u\to d\bar d$",
       "uu_uu": r"$uu\to uu$", "uubar_uubar": r"$u\bar u\to u\bar u$", "dd_dd": r"$dd\to dd$",
       "ee_bbbar": r"$ee\to b\bar b$", "uubar_ttbar": r"$u\bar u\to t\bar t$", "ee_ZZ": r"$ee\to ZZ$",
       "ee_ZH": r"$ee\to ZH$", "udbar_Wg": r"$u\bar d\to Wg$", "udbar_Wgg": r"$u\bar d\to Wgg$",
       "udbar_Wggg": r"$u\bar d\to Wggg$", "ee_uu_nlo": r"$ee\to u\bar u$ (1L)",
       "uubar_ddbar_nlo": r"$u\bar u\to d\bar d$ (1L)", "uubar_ZZ_nlo": r"$u\bar u\to ZZ$ (1L)"}
G = [["ee_uu", "ee_mumu", "ee_numu", "ee_bhabha", "ee_aa"], ["uubar_ddbar", "uu_uu", "uubar_uubar", "dd_dd"],
     ["ee_bbbar", "uubar_ttbar", "ee_ZZ", "ee_ZH", "udbar_Wg"],
     ["udbar_Wgg", "udbar_Wggg", "ee_uu_nlo", "uubar_ddbar_nlo", "uubar_ZZ_nlo"]]
fig, axes = ps.figure(ncols=2, nrows=2)
for ax, g in zip(axes.flat, G):
    for p, col in zip(g, ps.CYCLE):
        R, Y = [], []
        for n in RUNGS:
            b = best(n)
            if p in b.get("proc_curves", {}):
                i = int(np.argmin(b["val_curve"]))
                R.append(int(n.split("_r")[1])); Y.append(b["proc_curves"][p][i])
        ax.plot(R, Y, "o-", color=col, label=LAB[p])
    ax.set_yscale("log"); ax.set_xlabel("rung"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ax.set_xticks(range(1, 10)); ps.legend(ax, "upper left"); ps.make_room(ax)
ps.save(fig, FIG + "_per_dataset")
for n in RUNGS:
    b = best(n)
    print(n, f"best hp{b['hp']} lr {b['lr']:.2e} ema {b['ema']} val {b['val_loss']:.2g} step {b['best_step']}",
          f"done {len(done(n))} div {sum(t['state'] == 'diverged' for t in L[n])} run {sum(t['state'] == 'running' for t in L[n])}")
