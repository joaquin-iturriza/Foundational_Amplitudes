"""Transfer study, the synthetic-amplitude pretraining fine-tuned at 32k (tp3_synfte32k, 48 cells: the twelve ladder
probes at D = 10^3.5 .. 10^5, ud -> ud also at 10^2.5, 10^3), each cell the best of its two chosen points
(cells.best_chosen; MSE of log|M|^2 at the best checkpoint). Against scratch at 32k (tp3_scr32k, its two chosen points)
and rung 9 at 32k (tp3_r9fte32k, where it ran), per cell log10(L_reference / L_synthetic). The cells of a probe are not
independent: the test that counts is over probes (each probe's median, Wilcoxon), the per-cell one descriptive.
    python analysis/transfer/synth32k.py     -> figs/synth32k_a, _b; numbers on stdout
"""
import os, re, sys
import numpy as np
from scipy.stats import wilcoxon, binomtest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, S, CHOSEN32, best_chosen  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

base = os.path.join(ROOT, "analysis", "transfer", "figs", "synth32k")
rows = []
for n in sorted(x for x in S if x.startswith("tp3_synfte32k_")):
    p, k = re.match(r"tp3_synfte32k_(.+)_d(\d+)$", n).groups()
    k = int(k)
    rows.append((p, k, best_chosen(n), best_chosen(f"tp3_scr32k_{p}_d{k}"), best_chosen(f"tp3_r9fte32k_{p}_d{k}")))
print(f"{len(rows)} synthetic 32k cells, {sum(r[2] is not None for r in rows)} with a value")
# scratch cells that ran other points than their chosen two are reported apart, never mixed in
for p, k, s, sc, r9 in rows:
    nm = f"tp3_scr32k_{p}_d{k}"
    if sc is None and S.get(nm):
        alt = min(t["val_loss"] * t["prepd_std"] ** 2 for t in S[nm] if t.get("prepd_std"))
        print(f"  {nm}: none of its chosen points {CHOSEN32.get(nm)} ran; best of its {len(S[nm])} other trials "
              f"{alt:.3g}, synthetic {s:.3g} ({np.log10(alt / s):+.2f} dex); left out of the tests")


def report(lab, g):
    d = np.array([x[2] for x in g])
    print(f"vs {lab}: {len(d)} cells, median {np.median(d):+.2f} dex, synthetic better in {np.sum(d > 0)} "
          f"(sign p={binomtest(int(np.sum(d > 0)), len(d)).pvalue:.2g}, Wilcoxon p={wilcoxon(d).pvalue:.2g})")
    for k in sorted({x[1] for x in g}):
        dk = np.array([x[2] for x in g if x[1] == k])
        print(f"    D=10^{k / 2:g}: {len(dk)} cells, median {np.median(dk):+.2f}, better in {np.sum(dk > 0)}")
    per = {p: np.median([x[2] for x in g if x[0] == p]) for p in sorted({x[0] for x in g})}
    v = np.array(list(per.values()))
    print(f"  over probes: median {np.median(v):+.2f}, better on {np.sum(v > 0)} of {len(v)}, "
          f"Wilcoxon p={wilcoxon(v).pvalue:.2g}; " + ", ".join(f"{p} {x:+.2f}" for p, x in per.items()))


gs = [(p, k, np.log10(sc / s)) for p, k, s, sc, r9 in rows if s and sc]
g9 = [(p, k, np.log10(r9 / s)) for p, k, s, sc, r9 in rows if s and r9]
report("scratch at 32k", gs)
report("rung 9 at 32k", g9)

figs = ps.panels(2)
for (fig, ax), (g, lab) in zip(figs, [(gs, "scratch"), (g9, "rung 9")]):
    x = np.array([k / 2 for _, k, _ in g])
    ax.scatter(x, [v for *_, v in g], color=ps.C.green, label="cell")
    ks = [kk for kk in sorted(set(x)) if np.sum(x == kk) >= 3]      # a median of one or two cells is the cells
    ax.plot(ks, [np.median([v for (_, k, v) in g if k / 2 == kk]) for kk in ks], color=ps.C.vermillion,
            marker="s", label="median at each D")
    ax.axhline(0, color="0.5", ls="--", label="equal")
    ax.set_xlabel(r"$\log_{10} D$")
    ax.set_ylabel(rf"$\log_{{10}}(L_\mathrm{{{lab.replace(' ', '~')}}}/L_\mathrm{{synthetic}})$")
    ps.legend(ax, "upper left")
    ps.make_room(ax)
ps.save_panels(figs, base)
