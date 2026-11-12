#!/usr/bin/env python3
"""pretrain_data_saving, the figures of Joaquin's 2026-11-12 review, from the stage-2 best trials (stage2_readout.json and
the curves each run logged itself, data/stage2_mse_curves.json from stage2_mse_curves.py):
  stage2_mse_curves_<target>_<a..>  per D of the study: val and train mu-MSE (the run's own val_mse_curve and
                                    train_eval_mse_curve, every validation), ln units, scratch and fine-tune;
  stage2_mismatch                   fine-tune over scratch MSE (val ln, test in stage2_LD's units) at every D against the
                                    head mismatch sigma_parent/sigma_own of the target's ln standardization;
  stage2_d316                       the D=316 outliers: test MSE with and without its worst event, and the horizon T.
    python analysis/data_saving/review_figs.py
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

DATA = os.path.join(REPO, "analysis/data_saving/data")
FIGS = os.path.join(REPO, "analysis/data_saving/figs/")
T = {"ee_uu": (r"$e^+e^-\to u\bar u$", r"$\ln|\mathcal{M}|^2$"), "epem_ttx": (r"$e^+e^-\to t\bar t$", r"$\ln|\mathcal{M}|^2$"),
     "ee_uu_nlo": (r"$e^+e^-\to u\bar u$, $V$", r"$\ln(V/B)$"), "ee_ttbar_nlo": (r"$e^+e^-\to t\bar t$, $V$", r"$\ln(V/B)$")}
ARM = {"scr": ("scratch", ps.C.blue), "ft": ("fine-tune", ps.C.vermillion)}
CELLS = json.load(open(os.path.join(DATA, "stage2_readout.json")))["cells"]
CURVES = {(r["target"], r["arm"], round(r["D"])): r for r in json.load(open(os.path.join(DATA, "stage2_mse_curves.json")))}


def curves():
    for tg, (lab, q) in T.items():
        Ds = sorted({D for (t, a, D) in CURVES if t == tg})
        figs = ps.panels(len(Ds))
        for (fig, ax), D in zip(figs, Ds):
            for a, (name, col) in ARM.items():
                r = CURVES.get((tg, a, D), {})
                if not r.get("val_mse_curve") or not r.get("prepd_std"):
                    ax.plot([], [], color=col, label=f"{name}: pruned")
                    continue
                s2, e = r["prepd_std"] ** 2, r["every"]
                x = [e * (i + 1) for i in range(len(r["val_mse_curve"]))]
                ax.plot(x, [v * s2 for v in r["val_mse_curve"]], color=col, label=f"{name}, val")
                if r.get("train_eval_mse_curve"):
                    ax.plot(x[:len(r["train_eval_mse_curve"])], [v * s2 for v in r["train_eval_mse_curve"]], color=col,
                            linestyle="--", label=f"{name}, train")
            ax.set_yscale("log")
            ax.set_xlabel("step")
            ax.set_ylabel("MSE(" + q + ")")
            ps.process_label(ax, lab + rf", $D={D}$")
            ps.legend(ax, "lower left")
        ps.save_panels(figs, FIGS + f"stage2_mse_curves_{tg}")


def mismatch():
    fig, axs = ps.figure(ncols=2)
    marks = dict(zip(T, "osD^"))
    for ax, (f, ylab) in zip(axs, (("val_ln", "val"), ("test", "test"))):
        for tg, (lab, q) in T.items():
            by = {(c["arm"], round(c["D"])): c for c in CELLS if c["target"] == tg}
            xs, ys, cs = [], [], []
            for (a, D), c in by.items():
                r = CURVES.get((tg, "ft", D), {})
                if a != "ft" or ("scr", D) not in by or not r.get("parent_prepd_std") or not c.get(f) or not by[("scr", D)].get(f):
                    continue
                xs.append(r["parent_prepd_std"] / r["prepd_std"])
                ys.append(c[f] / by[("scr", D)][f])
                cs.append(np.log10(D))
            sc = ax.scatter(xs, ys, c=cs, cmap="viridis", vmin=1, vmax=5, marker=marks[tg],
                            label=lab)
        ax.axhline(1, color=ps.C.grey, linestyle=":", label="equal")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(__import__("matplotlib.ticker").ticker.NullFormatter())
        ax.set_xlabel(r"$\sigma_{\rm parent}/\sigma_{\rm own}$ of the log target")
        ax.set_ylabel(ylab + r" MSE$_{\rm fine-tune}$/MSE$_{\rm scratch}$")
        ps.colorbar(ax, sc, r"$\log_{10} D$")
        ps.legend(ax, "upper left")
    ps.save(fig, FIGS + "stage2_mismatch")


def d316():
    figs = ps.panels(3)
    for (fig, ax), tg in zip(figs, ("epem_ttx", "ee_uu_nlo")):
        for a, (name, col) in ARM.items():
            cc = sorted((c for c in CELLS if c["target"] == tg and c["arm"] == a and c.get("test")), key=lambda c: c["D"])
            ax.plot([c["D"] for c in cc], [c["test"] for c in cc], marker="o", color=col, label=name)
            ax.plot([c["D"] for c in cc], [c["test"] * (1 - c["top1"]) for c in cc], marker="x", linestyle="--",
                    color=col, label=f"{name}, w/o worst event")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$D$")
        ax.set_ylabel("test MSE(" + {"epem_ttx": r"$\ln|\mathcal{M}|^2$", "ee_uu_nlo": r"$V/B$"}[tg] + ")")
        ps.process_label(ax, T[tg][0])
        ps.legend(ax, "lower left")
    fig, ax = figs[2]
    for tg, col in (("epem_ttx", ps.C.green), ("ee_uu_nlo", ps.C.purple)):
        cc = sorted((c for c in CELLS if c["target"] == tg and c["arm"] == "scr"), key=lambda c: c["D"])
        ax.plot([c["D"] for c in cc], [c["T"] for c in cc], marker="o", color=col, label=T[tg][0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D$")
    ax.set_ylabel("training steps $T$")
    ps.legend(ax, "upper left")
    ps.save_panels(figs, FIGS + "stage2_d316")


if __name__ == "__main__":
    curves()
    mismatch()
    d316()
