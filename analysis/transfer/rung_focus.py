"""Transfer study, the fine-tune grid read per probe against the structure it needs: for each probe, from scratch,
the first ladder rung that adds the structure the probe carries, and the rung just below it (the same pretraining
without that structure; ee_uu alone, rung 0, below rung 1), with every other pretraining's fine-tunes in grey. The pairing is
the ladder's design (recipes/transfer_ladder_r*.yaml headers): s-channel r1, EW t-channel r2 (the W exchange in
ee -> nu_e nu_e), external photons r3, QCD exchange r4, masses r5, the first external W and gluon r6 (ud -> Wg; uu -> gg
and uu -> Zg cross it), 2->3 r7, 2->4 r8, one loop r9. Same cells and values as rung_grid.py: the 8k grid while
cells.USE_32K is off (now), the 32k cells at D = 10^3.5, 10^4 once it is on (cells.final; a cell still at 8k drawn open). The best pretraining overall is
ranked on the 8k grid, where every pretraining has every cell at the same compute.
  <base>_ee_a..f, <base>_qcd_a..f
    python analysis/transfer/rung_focus.py      -> figs/rung_focus
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, USE_32K, best, scratch  # noqa: E402
from rung_grid import LAB, P, S, RUNG_ADDS, draw_final, fam  # noqa: E402
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
    return {"tp3_uu64fte": "r0"}.get(f, "r" + f[len("tp3_r"):-len("fte")])


def series():
    """Every pretraining's fine-tune family: ee->uu (rung 0) and rungs 1-9."""
    return [f for f in (fam(r) for r in range(0, 10)) if any(n.startswith(f + "_") for n in S)]


for part, order in (("ee", P[:6]), ("qcd", P[6:])):
    figs = ps.panels(len(order))
    for (fig, ax), p in zip(figs, order):
        hi, lo = fam(FOCUS[p]), fam(FOCUS[p] - 1)
        lead, unranked = overall(p, [f for f in series() if f.endswith("fte")])
        first = True
        for f in series():
            if f in (hi, lo) or (lead and f == lead[0]):
                continue
            if draw_final(ax, f, p, "o-", label="other pretrainings" if first else None, **GREY) is not None:
                first = False
        draw_final(ax, "scr", p, "o-", color="k", label="from scratch")
        draw_final(ax, lo, p, "o-", color=ps.C.blue, label="the rung below")
        draw_final(ax, hi, p, "o-", color=ps.C.vermillion, label="first rung with the structure")
        if lead and lead[0] not in (hi, lo):
            draw_final(ax, lead[0], p, "o-", color=ps.C.green, label="best overall")
        below = f"r{FOCUS[p] - 1}"
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
    if USE_32K:
        a0.plot([], [], "o", color="k", mfc="white", label="open: 8k steps where the study runs 32k (for now)")
    # the key to the panels' "r6 vs r5" labels is the ladder's table (rung -> structure) in the text, not the legend: nine
    # extra legend rows pushed the six-panel figure past a page
    ps.shared_legend(figs[0][0], a0, ncol=1)
    ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", f"rung_focus_{part}"))
