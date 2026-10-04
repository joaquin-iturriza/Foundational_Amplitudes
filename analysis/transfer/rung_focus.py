"""Transfer study, the fine-tune grid read per probe against the structure it needs: for each probe, from scratch,
the first ladder rung that adds the structure the probe carries, and the rung just below it (the same pretraining
without that structure; ee_uu alone below rung 1), with every other pretraining's fine-tunes in grey. The pairing is
the ladder's design (recipes/transfer_ladder_r*.yaml headers): s-channel r1, EW t-channel r2 (the W exchange in
ee -> nu_e nu_e), external photons r3, QCD exchange r4, masses r5, the first external W and gluon r6 (ud -> Wg; uu -> gg
and uu -> Zg cross it), 2->3 r7, 2->4 r8, one loop r9. Same cells and values as rung_grid.py. The 32k-step searches
at D = 10^3.5, 10^4 (tp3_scr32k, tp3_<parent>fte32k; preliminary) are open diamonds in their series' colour.
  <base>_ee_a..f, <base>_qcd_a..f
    python analysis/transfer/rung_focus.py      -> figs/rung_focus
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, best, scratch, finetune  # noqa: E402
from rung_grid import LAB, P, S, RUNG_ADDS, curve  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

# probe -> the first rung adding its structure (the rung below is one less; 0 = ee_uu alone)
FOCUS = {"ee_ddbar": 1, "ee_nnbar": 2, "ee_ttbar": 5, "ee_WW": 6, "ee_dd_nlo": 9, "ee_bb_nlo": 9, "ee_Za": 3,
         "ud_ud": 4, "uubar_gg": 6, "uubar_Zg": 6, "uubar_Zgg": 7, "uubar_Zggg": 8}
GREY = dict(color="0.6", alpha=0.35)
TIE = 1.2      # a lead under this factor is a tie: below what one seed's 5-trial search resolves (provisional)


def overall(p, fams):
    """The pretraining that does best over the D cells: the mean over cells of log10(scratch / fine-tune), equal
    weight per cell (each half-decade of D counts the same), on every cell scratch has. Only grids with a value in
    each of those cells are ranked; the others are returned as unranked (named on the panel), so a grid still
    filling in is never silently left out. Returns (leaders within TIE of the best, best first; unranked)."""
    K = [k for k in range(2, 9) if scratch(p, k, steered=False)[0]]
    full = [f for f in fams if all(best(f"{f}_{p}_d{k}")[0] for k in K)]
    score = {f: np.mean([np.log10(scratch(p, k, steered=False)[0] / best(f"{f}_{p}_d{k}")[0]) for k in K])
             for f in full}
    unranked = [f for f in fams if f not in full]
    if not score:
        return [], unranked
    top = max(score.values())
    return sorted((f for f in score if top - score[f] < np.log10(TIE)), key=lambda f: -score[f]), unranked


def short(f):
    return {"tp3_uufte": r"$u\bar u$", "tp3_uu64fte": r"$u\bar u$ 64k"}.get(f, "r" + f[len("tp3_r"):-len("fte")])


def fam(r):
    return "tp3_uufte" if r == 0 else f"tp3_r{r}fte"


def series():
    """Every pretraining's fine-tune family in the grid's setup, plus the earlier-setup references."""
    out = {fam(r): None for r in range(0, 10)}
    out["tp3_uu64fte"] = None
    out.update({f"tp3_r{r}ftp": None for r in (4, 9)})
    return [f for f in out if any(n.startswith(f + "_") for n in S)]


for part, order in (("ee", P[:6]), ("qcd", P[6:])):
    figs = ps.panels(len(order))
    for (fig, ax), p in zip(figs, order):
        hi, lo = fam(FOCUS[p]), fam(FOCUS[p] - 1)
        lead, unranked = overall(p, [f for f in series() if f.endswith("fte")])
        first = True
        for f in series():
            if f in (hi, lo) or (lead and f == lead[0]):
                continue
            D, L = curve(lambda k: best(f"{f}_{p}_d{k}")[0])
            if D:
                ax.plot(D, L, "o-", label="other pretrainings" if first else None, **GREY)
                first = False
        D, L = curve(lambda k: finetune(p, k, steered=False)[0])
        if D:
            ax.plot(D, L, "o--", **GREY)
        ax.plot(*curve(lambda k: scratch(p, k, steered=False)[0]), "o-", color="k", label="from scratch")
        ax.plot(*curve(lambda k: best(f"{lo}_{p}_d{k}")[0]), "o-", color=ps.C.blue, label="the rung below")
        ax.plot(*curve(lambda k: best(f"{hi}_{p}_d{k}")[0]), "o-", color=ps.C.vermillion,
                label="first rung with the structure")
        if lead and lead[0] not in (hi, lo):
            ax.plot(*curve(lambda k: best(f"{lead[0]}_{p}_d{k}")[0]), "o-", color=ps.C.green, label="best overall")
        # the 32k-step searches at D = 10^3.5, 10^4 (horizon32k.py; preliminary: 2-4 trials, two of them chosen points),
        # open diamonds in their 8k series' colour: scratch, and each parent that has one
        D32 = [10 ** 3.5, 10 ** 4]
        def pts32(name):
            v = [best(f"{name}_d{k}")[0] for k in (7, 8)]
            return [d for d, x in zip(D32, v) if x], [x for x in v if x]
        ax.plot(*pts32(f"tp3_scr32k_{p}"), "D", color="k", mfc="none")
        for f in sorted({n.split("32k_")[0] for n in S if "fte32k_" in n and f"_{p}_d" in n}):
            col = (ps.C.vermillion if f == hi else ps.C.blue if f == lo else
                   ps.C.green if lead and f == lead[0] else "0.6")
            ax.plot(*pts32(f"{f}32k_{p}"), "D", color=col, mfc="none")
        below = r"$ee\to u\bar u$" if FOCUS[p] == 1 else f"r{FOCUS[p] - 1}"
        ps.process_label(ax, LAB[p] + "\n" + f"r{FOCUS[p]} vs " + below
                         # the leader and its first tie by name, any further ties as a count (the full list ran
                         # off the panel once every grid was ranked)
                         + (", best " + r"$\approx$".join(short(f) for f in lead[:2])
                            + (f" (+{len(lead) - 2})" if len(lead) > 2 else "") if lead else "")
                         + ("\n" + "unranked " + ", ".join(short(f) for f in unranked) if unranked else ""))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.make_room(ax)
    a0 = figs[0][1]
    if "best overall" not in a0.get_legend_handles_labels()[1]:     # the legend is read off panel (a)
        a0.plot([], [], "o-", color=ps.C.green, label="best overall")
    a0.plot([], [], "D", color="k", mfc="none", label="open diamonds: 32k steps, same colours")
    for r, what in RUNG_ADDS.items():                           # the key to the panels' "r6 vs r5" labels
        a0.plot([], [], " ", label=f"r{r}: {what}")
    ps.shared_legend(figs[0][0], a0, ncol=2)
    ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", f"rung_focus_{part}"))
