"""pretrain_data_saving stage-2 figures from analysis/data_saving/data/stage2_readout.json (stage2_readout.py):

    stage2_LD     per target the best test MSE per D, scratch and fine-tune, with the joint shared-exponent fit and
                  D* at the readout level (trees: MSE(ln|M|^2), one loops: MSE(V/B))
    stage2_gpuh   the same best test MSE against the GPU-hours of that cell's 8-trial search
    stage2_lr     every trial's sampled lr in window units against D (scratch lr/c, fine-tune lr_scale): trained,
                  diverged, best of the search

    python analysis/data_saving/plot_stage2.py
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

R = json.load(open(os.path.join(REPO, "analysis/data_saving/data/stage2_readout.json")))
FIG = os.path.join(REPO, "analysis/data_saving/figs/")
T = [("ee_uu", r"$e^+e^-\to u\bar u$", r"MSE$(\ln|\mathcal{M}|^2)$"),
     ("epem_ttx", r"$e^+e^-\to t\bar t$", r"MSE$(\ln|\mathcal{M}|^2)$"),
     ("ee_uu_nlo", r"$e^+e^-\to u\bar u$, $V$", r"MSE$(V/B)$"),
     ("ee_ttbar_nlo", r"$e^+e^-\to t\bar t$, $V$", r"MSE$(V/B)$")]
ARM = {"scr": ("scratch", ps.C.blue, "o"), "ft": ("fine-tune", ps.C.vermillion, "s")}


def cells(tg, a):
    return sorted((c for c in R["cells"] if c["target"] == tg and c["arm"] == a and c.get("test")), key=lambda c: c["D"])


def ld(gpu):
    fig, axs = ps.figure(ncols=2, nrows=2)
    for n, (ax, (tg, lab, ylab)) in enumerate(zip(axs.ravel(), T)):
        f = R["fits"]["test"].get(tg)
        for a, (name, col, m) in ARM.items():
            cc = cells(tg, a)
            x = [c["search_gpu_h"] for c in cc] if gpu else [c["D"] for c in cc]
            ax.plot(x, [c["test"] for c in cc], marker=m, linestyle="none" if not gpu else "-", color=col, label=name)
            if f and not gpu:
                A, Linf = f["fit"][a]
                D = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
                ax.plot(D, A * D ** -f["fit"]["alpha"] + Linf, linestyle="--", color=col,
                        label="joint fit" if a == "scr" else None)
                ax.plot([10 ** f["log10_Dstar"][a]], [f["level"]], marker="v", linestyle="none", color=col,
                        label=r"$D^*$" if a == "scr" else None)
        if f:
            ax.axhline(f["level"], color=ps.C.grey, linestyle=":", label=r"$10^{-4}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("search GPU-h" if gpu else r"$D$")
        ax.set_ylabel(ylab)
        ps.process_label(ax, lab)
        if n == 0:
            ps.legend(ax, "lower left")
    ps.save(fig, FIG + ("stage2_gpuh" if gpu else "stage2_LD"))


def lr():
    fig, axs = ps.figure(ncols=2)
    cols = dict(zip([t[0] for t in T], ps.CYCLE))
    for ax, a in zip(axs, ("scr", "ft")):
        lo, hi = (1 / 3, 3) if a == "scr" else (0.1, 10)
        for i, (tg, lab, _) in enumerate(T):
            sh = 10 ** (0.06 * (i - 1.5))             # targets side by side at each D
            tr = [t for t in R["trials"] if t["target"] == tg and t["arm"] == a]
            ok = [t for t in tr if t["status"] == "observed"]
            bad = [t for t in tr if t["status"] in ("failed", "lost")]
            best = [c for c in R["cells"] if c["target"] == tg and c["arm"] == a and "x_lr_best" in c]
            ax.plot([t["D"] * sh for t in ok], [t["x_lr"] for t in ok], marker=".", linestyle="none",
                    color=cols[tg], label=lab if a == "scr" else None)
            ax.plot([t["D"] * sh for t in bad], [t["x_lr"] for t in bad], marker="x", linestyle="none", color=cols[tg])
            ax.plot([c["D"] * sh for c in best], [c["x_lr_best"] for c in best], marker="o", linestyle="none",
                    markerfacecolor="none", color=cols[tg])
        for y in (lo, hi):
            ax.axhline(y, color=ps.C.grey, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$D$")
        ax.set_ylabel(r"$\eta/c$" if a == "scr" else "lr_scale")
    # marker key on the fine-tune panel: one legend per canvas, so the target colours join it
    axs[1].plot([], [], marker=".", linestyle="none", color=ps.C.grey, label="trained")
    axs[1].plot([], [], marker="x", linestyle="none", color=ps.C.grey, label="diverged")
    axs[1].plot([], [], marker="o", markerfacecolor="none", linestyle="none", color=ps.C.grey, label="best")
    axs[1].plot([], [], linestyle=":", color=ps.C.grey, label="window")
    for tg, lab, _ in T:
        axs[1].plot([], [], marker=".", linestyle="none", color=cols[tg], label=lab)
    ps.legend(axs[1], "upper left", ncol=2)
    ps.save(fig, FIG + "stage2_lr")


ld(False)
ld(True)
lr()
