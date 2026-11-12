"""pretrain_data_saving stage-2 figures from analysis/data_saving/data/stage2_readout.json (stage2_readout.py):

    stage2_LD     per target the best test MSE per D, scratch and fine-tune, with the joint shared-exponent fit and
                  D* at the readout level (trees: MSE(ln|M|^2), one loops: MSE(V/B))
    stage2_gpuh   the same best test MSE against the GPU-hours of that cell's 8-trial search
    stage2_lr     every trial's sampled lr in window units against D (scratch lr/c, fine-tune lr_scale): trained,
                  diverged, best of the search

    stage2_LD_bbx_zz  D413's second example (b b~ > Z Z, LO and one-loop V/B) on test_ln, alone with --second

    python analysis/data_saving/plot_stage2.py [--second [--only bbx_zz_nlo]]
"""
import json
import os
import sys

import matplotlib.ticker
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
ARM = {"scr": ("scratch, best of its 8-trial search", ps.C.blue, "o"),
       "ft": ("fine-tune, best of its 8-trial search", ps.C.vermillion, "s")}


# D413's second example, b b~ > Z Z at LO and one-loop V/B, read on test_ln (MSE in ln units at both orders)
SECOND = [("bbx_zz", r"$b\bar b\to ZZ$, LO", r"MSE$(\ln|\mathcal{M}|^2)$"),
          ("bbx_zz_nlo", r"$b\bar b\to ZZ$, $V/B$", r"MSE$(\ln(V/B))$")]


def cells(tg, a, key="test"):
    return sorted((c for c in R["cells"] if c["target"] == tg and c["arm"] == a and c.get(key)), key=lambda c: c["D"])


def tex_level(x):
    m, e = f"{x:.1e}".split("e")
    return rf"$10^{{{int(e)}}}$" if m == "1.0" else rf"${m}\times10^{{{int(e)}}}$"


def ld(gpu, panels=T, key="test", name="stage2_LD"):
    fig, axs = ps.figure(ncols=min(2, len(panels)), nrows=max(1, len(panels) // 2),
                         width="full" if len(panels) > 1 else "half")
    axs = np.atleast_1d(axs)
    for n, (ax, (tg, lab, ylab)) in enumerate(zip(axs.ravel(), panels)):
        f = R["fits"].get(key, {}).get(tg)
        lv = tex_level(f["level"]) if f else None    # the readout level: max(1e-4, the lowest both arms reach), D373
        for a, (name_, col, m) in ARM.items():
            cc = cells(tg, a, key)
            x = [c["search_gpu_h"] for c in cc] if gpu else [c["D"] for c in cc]
            ax.plot(x, [c[key] for c in cc], marker=m, linestyle="none" if not gpu else "-", color=col, label=name_)
            # curve_quality.py --cells: a search best whose validation leaves its training curve (separation > 3 or end > 2x best) is drawn open
            cq = os.path.join(REPO, f"analysis/data_saving/data/curve_quality_cells_{tg}.json")
            if not gpu and os.path.exists(cq):
                up = json.load(open(cq))[a]["per_cell"]
                xo = [xx for xx, c in zip(x, cc) if up.get(f"{c['D']:.0f}") is False]
                ax.plot(xo, [c[key] for c in cc if up.get(f"{c['D']:.0f}") is False], marker=m, linestyle="none",
                        color="white", markeredgecolor=col, label="val leaves train" if a == "ft" else None)
            if f and not gpu:
                A, Linf = f["fit"][a]
                D = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
                ax.plot(D, A * D ** -f["fit"]["alpha"] + Linf, linestyle="--", color=col,
                        label=r"fit $A D^{-\alpha}+L_\infty$" if a == "scr" else None)
                if np.isfinite(f["log10_Dstar"][a]):
                    ax.plot([10 ** f["log10_Dstar"][a]], [f["level"]], marker="v", linestyle="none", color=col,
                            label=f"where the fit crosses {lv}" if a == "scr" else None)
        if f:
            ax.axhline(f["level"], color=ps.C.grey, linestyle=":", label=lv)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("search GPU-h" if gpu else r"$D$")
        ax.set_ylabel(ylab)
        ps.process_label(ax, lab)
    # one legend above the panels: in a panel's corner it stretched the y axis far past the data
    if gpu:
        ps.legend(axs.ravel()[0], "lower left")
    else:
        ps.shared_legend(fig, axs.ravel()[0], ncol=2 if len(panels) > 1 else 1)
    ps.save(fig, FIG + ("stage2_gpuh" if gpu else name))


def lr():
    fig, axs = ps.figure(ncols=2)
    cols = dict(zip([t[0] for t in T], ps.CYCLE))
    for ax, a in zip(axs, ("scr", "ft")):
        lo, hi = (1 / 3, 3) if a == "scr" else (0.1, 10)
        for i, (tg, lab, _) in enumerate(T):
            sh = 10 ** (0.06 * (i - 1.5))             # targets side by side at each D
            tr = [t for t in R["trials"] if t["target"] == tg and t["arm"] == a]
            ok = [t for t in tr if t["status"] == "observed"]
            bad = [t for t in tr if t["diverged"]]   # read in the log; other failures are listed by the readout
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
        ax.set_ylabel(r"$\eta/c$" if a == "scr" else "lr scale")
    # marker key on the fine-tune panel: one legend per canvas, so the target colours join it
    axs[1].plot([], [], marker=".", linestyle="none", color=ps.C.grey, label="trained")
    axs[1].plot([], [], marker="x", linestyle="none", color=ps.C.grey, label="diverged")
    axs[1].plot([], [], marker="o", markerfacecolor="none", linestyle="none", color=ps.C.grey, label="best")
    axs[1].plot([], [], linestyle=":", color=ps.C.grey, label="window")
    for tg, lab, _ in T:
        axs[1].plot([], [], marker=".", linestyle="none", color=cols[tg], label=lab)
    ps.legend(axs[1], "upper left", ncol=2)
    ps.save(fig, FIG + "stage2_lr")


def ratio():
    """best val mu-MSE fine-tune / scratch per D (same target statistics in both arms of a cell)"""
    fig, ax = ps.figure()
    for (tg, lab, _), col in zip(T, ps.CYCLE):
        r = {a: {c["k"]: c["val"] for c in R["cells"] if c["target"] == tg and c["arm"] == a and c.get("val")}
             for a in ARM}
        ks = sorted(set(r["scr"]) & set(r["ft"]))
        ax.plot([10 ** (k / 2) for k in ks], [r["ft"][k] / r["scr"][k] for k in ks], marker="o", color=col, label=lab)
    ax.axhline(1, color=ps.C.grey, linestyle=":", label="equal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"$L_{\rm ft}/L_{\rm scr}$")
    ps.legend(ax, "upper left")
    ps.save(fig, FIG + "stage2_ratio")


def lr_val(tg, lab):
    """every trial's best val mu-MSE against its sampled lr (scratch lr/c, fine-tune lr_scale), one colour per D"""
    fig, axs = ps.figure(ncols=2)
    ks = sorted({t["k"] for t in R["trials"] if t["target"] == tg})
    cols = dict(zip(ks, ps.sequence(len(ks))))
    for ax, a in zip(axs, ("scr", "ft")):
        for k in ks:
            tr = [t for t in R["trials"] if t["target"] == tg and t["arm"] == a and t["k"] == k and t.get("val")]
            ax.plot([t["x_lr"] for t in tr], [t["val"] for t in tr], marker="o", linestyle="none", color=cols[k],
                    label=rf"$D=10^{{{k / 2:g}}}$" if a == "ft" else None)
            bad = [t for t in R["trials"] if t["target"] == tg and t["arm"] == a and t["k"] == k and t["diverged"]]
            top = 3 * max(t["val"] for t in R["trials"] if t["target"] == tg and t.get("val"))
            ax.plot([t["x_lr"] for t in bad], [top] * len(bad), marker="x", linestyle="none", color=cols[k],
                    label="diverged" if a == "ft" and k == ks[-1] and bad else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\eta/c$" if a == "scr" else "lr scale")
        ax.set_ylabel(r"best val $\mu$-MSE")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ps.process_label(axs[0], lab)   # one process per canvas; the right panel's corner holds the legend
    ps.legend(axs[1], "upper left", ncol=3)
    ps.make_room(axs[1])
    ps.save(fig, FIG + "stage2_lrval_" + tg)


if "--second" in sys.argv:
    # --only <target>: one half alone (the one-loop half was read while the LO searches still ran)
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    ld(False, [p for p in SECOND if only in (None, p[0])], "test_ln", "stage2_LD_bbx_zz" + (f"_{only}" if only else ""))
    sys.exit()
ld(False)
ld(True)
lr()
ratio()
for tg, lab, _ in T[1::2]:
    lr_val(tg, lab)
