"""Transfer study, the star arms: rung 1 plus one structure each (recipes/transfer_star_<arm>.yaml), fine-tuned with
the grid's protocol on the arm's own probe and on ee -> dd~ (tp3_s<arm>fte). Panels (a)-(f): each arm on its own probe,
against scratch and against rung 1 (the arm without its structure). Panel (g): ee -> dd~, rung 1's own probe, from
every arm, to show what adding a structure costs where rung 1 already does well. Same cells and values as rung_grid.py.
The study's six arms as finally designed (the first resonance and Sudakov arms tested neither structure and were
replaced by the W-pole and electroweak-Sudakov arms, docs/results.tex; they are not drawn). The W-pole and EW-Sudakov
arms run two chosen points per cell (sweep/pick_points.py), and on their own probes rung 1 is read on the same two
points only (r1_chosen.json; the design approved 2026-10-04: arm and rung 1 at the same points), not on its search.
Panel (g), ee -> dd~ from every arm, draws rung 1's full search as its one step-1 curve.
    python analysis/transfer/star_arms.py      -> figs/star_arms (the six arms, 3x2), figs/star_arms2 (ee -> dd~ from every arm, one panel)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, S, best, scratch  # noqa: E402
from rung_grid import LAB, curve  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

R1_CHOSEN = json.load(open(os.path.join(ROOT, "analysis", "transfer", "r1_chosen.json")))
CHOSEN_ONLY = {"wpole", "sudakovew"}      # the arms run at two chosen points only


def r1(p, k, arm):
    """Rung 1 on an arm's probe: on the arm's own trial set (the two chosen points) where the arm ran only those."""
    n = f"tp3_r1fte_{p}_d{k}"
    if arm not in CHOSEN_ONLY:
        return best(n)[0]
    tr = [t for t in S.get(n, []) + S.get(n + "_002", [])
          if t.get("val_loss") and t.get("prepd_std") and t["hp"] in R1_CHOSEN.get(n, [])]
    return min(t["val_loss"] * t["prepd_std"] ** 2 for t in tr) if tr else None

# arm -> (its probe, what it adds to rung 1); docs/results.tex sec:ladder-open, the star arms item
ARMS = {"soft": ("ee_ddbarg", "soft/collinear emission"), "isr": ("uubar_Zg", "initial-state collinear"),
        "deadcone": ("ee_ttbarg", "dead cone (massive collinear)"), "threshold": ("ee_ttbar_nlo_thr", r"$t\bar t$ threshold"),
        "wpole": ("udbar_enu", r"$W$ pole"), "sudakovew": ("ee_dd_ew_nlo", "EW Sudakov logarithms")}
# calm colours (Joaquin 2026-11-17: the six arm colours were too distinct, jarring): six shades of one warm scale, clear
# of step 1's blue and scratch's black; each arm also has its own marker and a dashed line
import matplotlib as _mpl  # noqa: E402
ARM_C = dict(zip(ARMS, [_mpl.colormaps["YlOrBr"](v) for v in (0.95, 0.85, 0.75, 0.62, 0.5, 0.38)]))
ARM_M = dict(zip(ARMS, ("s", "^", "v", "D", "P", "X")))

# One file per figure, legends inside (Joaquin 2026-11-17); 'ladder step 1', not rung 1.
FIGDIR = os.path.join(ROOT, "analysis", "transfer", "figs")


def axes_style(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D_{\rm train}$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")


def own_probe(ax, arm):
    p, what = ARMS[arm]
    ax.plot(*curve(lambda k: scratch(p, k, steered=False)[0]), "o-", color="k", label="from scratch")
    ax.plot(*curve(lambda k: r1(p, k, arm)), "o-", color=ps.C.blue, label="ladder step 1")
    ax.plot(*curve(lambda k: best(f"tp3_s{arm}fte_{p}_d{k}")[0]), ARM_M[arm] + "--", color=ARM_C[arm], label=f"step 1 + {what}")
    ps.process_label(ax, LAB[p])
    axes_style(ax)
    ps.legend(ax, "lower left", fontsize=ps.LEGEND_MIN_PT)


# Joaquin 2026-11-17: no legend outside the panels. Seven panels do not fit one figure (3 across runs past the printable
# width, 4 rows past the page height): the six arms on their own probes in a 3x2 (star_arms), ee -> dd~ from every arm
# as one full-width panel with its legend inside (star_arms2)
fig, axs = ps.figure(ncols=2, nrows=3)
for ax, arm in zip(axs.flat, ARMS):
    own_probe(ax, arm)
ps.save(fig, os.path.join(FIGDIR, "star_arms"))

fig, ax = ps.figure()
p = "ee_ddbar"
ax.plot(*curve(lambda k: scratch(p, k, steered=False)[0]), "o-", color="k", label="from scratch")
# one step-1 curve, unnamed points (Joaquin 2026-11-17: 'I don't really think we need to specify that at all'): step 1's
# own search, the reference of the four searched arms (repo-reviewer 2026-11-17: the two-point curve can only sit higher
# than it, overstating those arms' gain); the W-pole and Sudakov arms ran two points, a question put to Joaquin
ax.plot(*curve(lambda k: best(f"tp3_r1fte_{p}_d{k}")[0]), "o-", color=ps.C.blue, label="ladder step 1")
for arm, (_, what) in ARMS.items():
    ax.plot(*curve(lambda k: best(f"tp3_s{arm}fte_{p}_d{k}")[0]), ARM_M[arm] + "--", color=ARM_C[arm],
            label=f"step 1 + {what}")
ps.process_label(ax, LAB[p])
axes_style(ax)
ps.legend(ax, "lower left", fontsize=ps.LEGEND_MIN_PT)   # the curves fall from upper left: lower left is empty
ps.save(fig, os.path.join(FIGDIR, "star_arms2"))
