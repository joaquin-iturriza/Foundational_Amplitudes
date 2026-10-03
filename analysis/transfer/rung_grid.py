"""Transfer study, the fine-tune grid: each probe's loss against D, from scratch and fine-tuned from each pretraining
in the grid's setup (tp3_<parent>fte: training.lr searched over [1e-3, 1e-2], lr_scale = layer_decay = 1, 5 trials;
docs/results.tex sec:ladder), with the ee->uu fine-tune of Table tab:ladder_transfer_all (cells.py, the earlier
setup) and the preliminary fine-tunes from rungs 4 and 9 (tp3_r4ftp, tp3_r9ftp: the earlier setup, from those rungs'
best trial at the time, hp15, at D = 10, 1e2, 1e3, 1e4) dashed, for reference. A cell's value is its search's best so far (MSE of log|M|^2 at the best checkpoint); cells still
running are drawn as they stand. ee->WW on the mixture pool throughout.
  <base>_ee_a..f, <base>_qcd_a..f
    python analysis/transfer/rung_grid.py      -> figs/rung_grid
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, ARM, best, scratch, finetune  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

LAB = {"ee_ddbar": r"$e^+e^-\to d\bar d$", "ee_nnbar": r"$e^+e^-\to\nu_e\bar\nu_e$", "ee_ttbar": r"$e^+e^-\to t\bar t$",
       "ee_WW": r"$e^+e^-\to W^+W^-$", "ee_dd_nlo": r"$e^+e^-\to d\bar d$ (1-loop)",
       "ee_bb_nlo": r"$e^+e^-\to b\bar b$ (1-loop)", "ee_Za": r"$e^+e^-\to Z\gamma$", "ud_ud": r"$ud\to ud$",
       "uubar_gg": r"$u\bar u\to gg$", "uubar_Zg": r"$u\bar u\to Zg$", "uubar_Zgg": r"$u\bar u\to Zgg$",
       "uubar_Zggg": r"$u\bar u\to Zggg$",
       "ee_ddbarg": r"$e^+e^-\to d\bar dg$", "ee_ttbarg": r"$e^+e^-\to t\bar tg$",
       "ee_ttbar_nlo_thr": r"$e^+e^-\to t\bar t$ (1-loop, thr.)", "ee_dd_nlo_hi": r"$e^+e^-\to d\bar d$ (1-loop, high)"}
P = ["ee_ddbar", "ee_nnbar", "ee_ttbar", "ee_WW", "ee_dd_nlo", "ee_bb_nlo", "ee_Za", "ud_ud", "uubar_gg",
     "uubar_Zg", "uubar_Zgg", "uubar_Zggg"]
# what each ladder rung adds (recipes/transfer_ladder_r*.yaml headers); the rungs are cumulative
RUNG_ADDS = {1: r"massless $s$-channel ($\gamma/Z$)", 2: r"+ EW $t$-channel", 3: "+ external photons",
             4: "+ QCD exchange, colour", 5: "+ masses", 6: r"+ $W$+jet, $2\to2$", 7: r"+ $2\to3$",
             8: r"+ $2\to4$", 9: "+ one loop"}
NEW = ["ee_ddbarg", "ee_ttbarg", "ee_ttbar_nlo_thr", "ee_dd_nlo_hi"]   # the star arms' probes
S = __import__("cells").S
RUNGS = sorted({r for r in range(1, 10) for n in S if n.startswith(f"tp3_r{r}fte_") or n.startswith(f"tp3_r{r}ftp_")})
cols = dict(zip(RUNGS, ps.sequence(len(RUNGS))))


def curve(f):
    D, L = [], []
    for k in range(2, 9):
        v = f(k)
        if v is not None:
            D.append(10 ** (k / 2)); L.append(v)
    return D, L


# scratch past the grid: one 64k-step run per cell at D = 10^4.5, 10^5 (analysis/transfer/large_d_64k.json), not a
# 5-trial search like the grid's cells, so drawn apart (open markers, dashed) and labelled as such
_LD = json.load(open(os.path.join(ROOT, "analysis", "transfer", "large_d_64k.json")))["mse_log_m2"]
LARGE_D_LABEL = "from scratch, 64k steps, one run"


def large_d(ax, p):
    """The probe's large-D scratch points, joined to its grid's last scratch cell; True if it has any."""
    pts = sorted((int(k), v) for k, v in _LD.get(p, {}).items())
    if not pts:
        return False
    last = scratch(p, 8, steered=False)[0]
    D = ([10 ** 4] if last else []) + [10 ** (k / 2) for k, _ in pts]
    L = ([last] if last else []) + [v for _, v in pts]
    ax.plot(D, L, "o--", color="k", mfc="none", label=LARGE_D_LABEL)
    return True


def legend_proxy(ax):
    """The shared legend is read off panel (a): give it the large-D entry when (a) has no such points."""
    if LARGE_D_LABEL not in ax.get_legend_handles_labels()[1]:
        ax.plot([], [], "o--", color="k", mfc="none", label=LARGE_D_LABEL)


def draw(ax, f, *a, **kw):
    """One series, left out (and out of the legend) where it has no cell yet."""
    D, L = curve(f)
    if D:
        ax.plot(D, L, *a, **kw)


if __name__ == "__main__":
    for part, order in (("ee", P[:6]), ("qcd", P[6:]), ("new", NEW)):
        figs = ps.panels(len(order))
        for (fig, ax), p in zip(figs, order):
            draw(ax, lambda k: scratch(p, k, steered=False)[0], "o-", color="k", label="from scratch")
            large_d(ax, p)
            draw(ax, lambda k: finetune(p, k, steered=False)[0], "o--", color=ps.C.blue, label=r"$ee\to u\bar u$, earlier setup")
            draw(ax, lambda k: best(f"tp3_uufte_{p}_d{k}")[0], "s-", color=ps.C.blue, mfc="none", label=r"$ee\to u\bar u$")
            if any(n.startswith("tp3_uu64fte_") for n in S):
                draw(ax, lambda k: best(f"tp3_uu64fte_{p}_d{k}")[0], "s-", color=ps.C.vermillion, mfc="none",
                        label=r"$ee\to u\bar u$, 64k steps")
            for r in RUNGS:
                if any(n.startswith(f"tp3_r{r}fte_") for n in S):
                    draw(ax, lambda k: best(f"tp3_r{r}fte_{p}_d{k}")[0], "o-", color=cols[r], label=f"rung {r}: {RUNG_ADDS[r]}")
                if any(n.startswith(f"tp3_r{r}ftp_") for n in S):
                    draw(ax, lambda k: best(f"tp3_r{r}ftp_{p}_d{k}")[0], "o--", color=cols[r],
                            label=f"rung {r}, earlier setup")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
            ps.process_label(ax, LAB[p])
            ps.make_room(ax)
        legend_proxy(figs[0][1])
        ps.shared_legend(figs[0][0], figs[0][1], ncol=2)     # seven series: no panel has a clear corner for them
        ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", f"rung_grid_{part}"))
