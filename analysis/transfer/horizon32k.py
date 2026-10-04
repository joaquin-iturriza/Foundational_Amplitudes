"""Transfer study, the 32k-step searches at D = 10^3.5 and 10^4 (tp3_scr32k, tp3_<parent>fte32k; docs/results.tex
sec:ladder hand-off, longer horizons), against the grid's horizons (8k steps there), for each of the twelve probes.
Every value: the best trial of the cell's search so far, MSE of log|M|^2 at its best checkpoint (cells.best). The 32k
searches were cut on 2026-10-04 (the user's call): their 1-2 random start-up trials plus two chosen points
(horizon32k_chosen.json, sweep/pick_points.py), so a 32k value is the best of 2-4 trials against the 8k search's 5
(fine-tune) or 8 (scratch).
  <base>_a  each cell's 32k best against its 8k best (scratch and fine-tune; a point below the diagonal gained)
  <base>_b  per 32k cell, its best chosen point against its best random start-up trial
  <base>_c  the fine-tune's gain over scratch (scratch / fine-tune) at 32k against the same at 8k
    python analysis/transfer/horizon32k.py
"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, S, best, scratch  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

CH = json.load(open(os.path.join(ROOT, "analysis", "transfer", "horizon32k_chosen.json")))
FIG = os.path.join(ROOT, "analysis", "transfer", "figs", "horizon32k")
COL = {7: ps.C.blue, 8: ps.C.vermillion}
LAB = {7: r"$D=10^{3.5}$", 8: r"$D=10^4$"}
names = sorted({re.sub(r"_002$", "", n) for n in S if "32k_" in n})
cell = lambda n: (re.search(r"32k_(.+)_d(\d)$", n).group(1), int(re.search(r"_d(\d)$", n).group(1)))

rows = []                                         # (name, probe, k, scratch?, 32k best, 8k best)
for n in names:
    p, k = cell(n)
    v32 = best(n)[0]
    v8 = scratch(p, k, steered=False)[0] if "_scr32k_" in n else best(n.replace("32k", ""))[0]
    if v32 is not None and v8 is not None:
        rows.append((n, p, k, "_scr32k_" in n, v32, v8))

figs = ps.panels(3)
(_, a), (_, b), (_, c) = figs
for k in (7, 8):
    for scr, m in ((True, "o"), (False, "s")):
        r = [x for x in rows if x[2] == k and x[3] == scr]
        a.plot([x[5] for x in r], [x[4] for x in r], m, color=COL[k], mfc=COL[k] if scr else "none")
lo, hi = min(min(x[4], x[5]) for x in rows), max(max(x[4], x[5]) for x in rows)
a.plot([lo, hi], [lo, hi], "-", color=ps.C.grey, label="32k = 8k")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"best at 8k steps, MSE$(\log|\mathcal{M}|^2)$"); a.set_ylabel("best at 32k steps")
for k in (7, 8):
    a.plot([], [], "o", color=COL[k], label=LAB[k])
a.plot([], [], "o", color="k", label="scratch"); a.plot([], [], "s", color="k", mfc="none", label="fine-tune")
ps.legend(a, "upper left"); ps.make_room(a)

pts = []
for n, ps_ in CH.items():
    tr = [t for t in S.get(n, []) + S.get(n + "_002", []) if t.get("val_loss") is not None]
    ch = [t["val_loss"] * t["prepd_std"] ** 2 for t in tr if t["hp"] in ps_]
    rd = [t["val_loss"] * t["prepd_std"] ** 2 for t in tr if t["hp"] not in ps_]
    if ch and rd:
        pts.append((cell(n)[1], "_scr32k_" in n, min(rd), min(ch)))
for k in (7, 8):
    for scr, m in ((True, "o"), (False, "s")):
        r = [x for x in pts if x[0] == k and x[1] == scr]
        b.plot([x[2] for x in r], [x[3] for x in r], m, color=COL[k], mfc=COL[k] if scr else "none")
lo, hi = min(min(x[2], x[3]) for x in pts), max(max(x[2], x[3]) for x in pts)
b.plot([lo, hi], [lo, hi], "-", color=ps.C.grey, label="chosen = random")
ps.legend(b, "upper left"); ps.make_room(b)
b.set_xscale("log"); b.set_yscale("log")
b.set_xlabel("best random start-up trial"); b.set_ylabel("best chosen point")

gain = []
for p in sorted({x[1] for x in rows}):
    for k in (7, 8):
        s32 = best(f"tp3_scr32k_{p}_d{k}")[0]
        s8 = scratch(p, k, steered=False)[0]
        for x in rows:
            if x[1] == p and x[2] == k and not x[3] and s32 and s8:
                gain.append((k, s8 / x[5], s32 / x[4]))
for k in (7, 8):
    r = [x for x in gain if x[0] == k]
    c.plot([x[1] for x in r], [x[2] for x in r], "s", color=COL[k], mfc="none")
lo, hi = min(min(x[1], x[2]) for x in gain), max(max(x[1], x[2]) for x in gain)
c.plot([lo, hi], [lo, hi], "-", color=ps.C.grey, label="same gain")
ps.legend(c, "upper left"); ps.make_room(c)
c.set_xscale("log"); c.set_yscale("log")
c.set_xlabel("gain at 8k steps (scratch / fine-tune)"); c.set_ylabel("gain at 32k steps")
ps.save_panels(figs, FIG)

print(f"cells with both horizons: {len(rows)} ({sum(x[3] for x in rows)} scratch); 32k below 8k in "
      f"{sum(x[4] < x[5] for x in rows)}; median 32k/8k scratch {np.median([x[4] / x[5] for x in rows if x[3]]):.2f}, "
      f"fine-tune {np.median([x[4] / x[5] for x in rows if not x[3]]):.2f}")
print(f"chosen vs random: {len(pts)} cells, chosen better in {sum(x[3] < x[2] for x in pts)}, median random/chosen "
      f"{np.median([x[2] / x[3] for x in pts]):.2f}")
print(f"gain: {len(gain)} (cell, parent) pairs, median gain 8k {np.median([x[1] for x in gain]):.2f}, "
      f"32k {np.median([x[2] for x in gain]):.2f}")
