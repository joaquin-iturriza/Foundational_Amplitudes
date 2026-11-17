"""Transfer study, the star arms: rung 1 plus one structure each (recipes/transfer_star_<arm>.yaml), fine-tuned with
the grid's protocol on the arm's own probe and on ee -> dd~ (tp3_s<arm>fte). Panels (a)-(f): each arm on its own probe,
against scratch and against rung 1 (the arm without its structure). Panel (g): ee -> dd~, rung 1's own probe, from
every arm, to show what adding a structure costs where rung 1 already does well. Same cells and values as rung_grid.py.
The study's six arms as finally designed (the first resonance and Sudakov arms tested neither structure and were
replaced by the W-pole and electroweak-Sudakov arms, docs/results.tex; they are not drawn). The W-pole and EW-Sudakov
arms run two chosen points per cell (sweep/pick_points.py), and on their own probes rung 1 is read on the same two
points only (r1_chosen.json; the design approved 2026-10-04: arm and rung 1 at the same points), not on its search.
Panel (g) draws rung 1's full search (the four searched arms' reference) and, dashed, rung 1 at the two chosen
points (the reference of the two chosen-only arms).
    python analysis/transfer/star_arms.py      -> figs/star_arms (four arms), figs/star_arms2 (two arms, ee -> dd~ from every arm)
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
ARM_C = dict(zip(ARMS, (ps.C.vermillion, ps.C.orange, ps.C.green, ps.C.purple, ps.C.sky, ps.C.yellow)))
# each arm its own marker and a dashed line besides its colour: step 1 (blue circles, solid) and the arms are told apart
# in every panel, and the six arms on ee -> dd~ (cold figure reader 2026-11-17: close colours bunched at large D)
ARM_M = dict(zip(ARMS, ("s", "^", "v", "D", "P", "X")))

# One file per figure (Joaquin 2026-11-17): the four searched arms in a 2x2 (star_arms), the two chosen-point arms and
# ee -> dd~ from every arm in a 2x2 whose fourth cell holds the legend of (g) (star_arms2); 'ladder step 1', not rung 1.
FIGDIR = os.path.join(ROOT, "analysis", "transfer", "figs")


def axes_style(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D_{\rm train}$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")


def own_probe(ax, arm):
    p, what = ARMS[arm]
    ax.plot(*curve(lambda k: scratch(p, k, steered=False)[0]), "o-", color="k", label="from scratch")
    ax.plot(*curve(lambda k: r1(p, k, arm)), "o-", color=ps.C.blue,
            label="ladder step 1" + (", same two\nhyperparameter points" if arm in CHOSEN_ONLY else ""))
    ax.plot(*curve(lambda k: best(f"tp3_s{arm}fte_{p}_d{k}")[0]), ARM_M[arm] + "--", color=ARM_C[arm], label=f"step 1 + {what}")
    ps.process_label(ax, LAB[p])
    axes_style(ax)
    ps.legend(ax, "lower left", fontsize=ps.LEGEND_MIN_PT)


fig, axs = ps.figure(ncols=2, nrows=2)
for ax, arm in zip(axs.flat, ("soft", "isr", "deadcone", "threshold")):
    own_probe(ax, arm)
ps.save(fig, os.path.join(FIGDIR, "star_arms"))

fig, axs = ps.figure(ncols=2, nrows=2)
own_probe(axs[0][0], "wpole")
own_probe(axs[0][1], "sudakovew")
ax = axs[1][0]
p = "ee_ddbar"
ax.plot(*curve(lambda k: scratch(p, k, steered=False)[0]), "o-", color="k", label="from scratch")
ax.plot(*curve(lambda k: best(f"tp3_r1fte_{p}_d{k}")[0]), "o-", color=ps.C.blue, label="ladder step 1")
# the W-pole and EW-Sudakov arms ran two chosen points here too: step 1 at the same two points, their reference
ax.plot(*curve(lambda k: r1(p, k, "wpole")), "o--", color=ps.C.blue,
        label="ladder step 1 at the hyperparameter\npoints of the $W$-pole and Sudakov arms")
for arm, (_, what) in ARMS.items():
    ax.plot(*curve(lambda k: best(f"tp3_s{arm}fte_{p}_d{k}")[0]), ARM_M[arm] + "--", color=ARM_C[arm], label=f"step 1 + {what}")
ps.process_label(ax, LAB[p])
axes_style(ax)
ps.legend_cell(axs[1][1], ax, fontsize=ps.LEGEND_MIN_PT)
ps.save(fig, os.path.join(FIGDIR, "star_arms2"))
