"""Data saving in the old (transfer-ladder) setup, four examples for the draft (D433): from scratch against fine-tuned,
MSE of log|M|^2 against the fine-tune events D, on the ladder's 8k-step grid (the draft's ladder figures' cells).
Per panel: scratch, the pretraining that adds the probe's structure, and the pretraining with every structure (rung 9
plus every star arm's processes, called "finale" in analysis/transfer). Cells are read exactly as
analysis/transfer/finale_synth.py reads them (cell(): the best of a search's first 5 trials, a fixed run at the
reference point, or the two chosen points of the W-pole arm); scratch is the 8-trial search (cells.scratch).
    python analysis/data_saving/old_setup_examples.py   -> figs/old_setup_examples; per-cell ratios on stdout
"""
import io, contextlib, os, sys
os.environ["TP_USE_32K"] = "0"      # the 8k grid, as the draft's ladder figures
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "analysis", "transfer"))
sys.path.insert(0, ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    from finale_synth import cell, ARMLAB  # noqa: E402
    from rung_grid import LAB  # noqa: E402
    from cells import scratch  # noqa: E402
import plot_style as ps  # noqa: E402

# probe -> (the pretraining adding its structure, its legend name)
EX = {"uubar_Zggg": ("tp3_r8fte", r"ladder up to $2\to4$"),
      "ee_ddbarg": ("tp3_ssoftfte", "+ soft/collinear"),
      "udbar_enu": ("tp3_swpolefte", r"+ $W$ pole"),
      "ee_bb_nlo": ("tp3_r9fte", "ladder up to one loop")}
K = range(2, 9)
D = [10 ** (k / 2) for k in K]

fig, axs = ps.figure(ncols=2, nrows=2)
for ax, (p, (fam, lab)) in zip(axs.flat, EX.items()):
    sc = [scratch(p, k, steered=False)[0] for k in K]
    for vals, kw in ((sc, dict(color="k", label="scratch")),
                     ([cell(fam, p, k) for k in K], dict(color=ps.C.vermillion, label="structure's pretraining")),
                     ([cell("tp3_finfte", p, k) for k in K], dict(color=ps.C.blue, label="every structure"))):
        pts = [(x, y) for x, y in zip(D, vals) if y]
        ax.plot([q[0] for q in pts], [q[1] for q in pts], "o-", **kw)
        if kw["color"] != "k":
            print(p, kw["label"], "L_scratch/L_ft per D:",
                  " ".join(f"{s / y:.1f}" if (s and y) else "-" for s, y in zip(sc, vals)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(ax, {**LAB, **ARMLAB}[p] + "\n" + lab)
ps.legend(axs.flat[0], "lower left")
for ax in axs.flat:
    ps.make_room(ax)
ps.save(fig, os.path.join(ROOT, "analysis", "data_saving", "figs", "old_setup_examples"))
