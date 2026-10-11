"""High-multiplicity test (D20, D22, D25, D28): u ubar -> Z+4g (2->5) and Z+5g (2->6), D = 10^1 .. 10^3 at 8k
(tp3_*_d2..d6) and 10^3.5 .. 10^5 at 32k (tp3_*32k_*_d7..d10). Four arms per cell: scratch (an 8-trial search,
tp3_hmscr[32k]; its value the best of every trial it ran), and the finale, finale+W+ng and finale+synthetic
pretrainings fine-tuned at their cell's two fixed points (expect_tp3.json; tp3_finfte, tp3_fwngfte, tp3_fsynfte), each
the best of the two. MSE of log|M|^2 at the best checkpoint (val_loss * prepd_std^2). Data: scratch_sweeps.json.
    python analysis/transfer/highmult.py     -> figs/highmult (MSE against D), figs/highmult_gain; table on stdout
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, S  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

E = json.load(open(os.path.join(ROOT, "analysis", "transfer", "expect_tp3.json")))
PROBES = [("uubar_Zgggg", r"$u\bar u\to Z+4g$"), ("uubar_Zggggg", r"$u\bar u\to Z+5g$")]
ARMS = [("scr", "scratch", ps.C.blue), ("finfte", "finale", ps.C.vermillion),
        ("fwngfte", "finale$+W$", ps.C.green), ("fsynfte", "finale$+$synthetic", ps.C.purple)]
KS = range(2, 11)


def name(arm, p, k):
    pre = "tp3_hmscr" if arm == "scr" else f"tp3_{arm}"
    return f"{pre}{'32k' if k >= 7 else ''}_{p}_d{k}"


def value(arm, p, k):
    n = name(arm, p, k)
    tr = [t for t in S.get(n, []) if t.get("val_loss") is not None and t.get("prepd_std")]
    if arm != "scr":
        tr = [t for t in tr if t["hp"] in E[n]]
        assert len(tr) == 2, (n, [t["hp"] for t in tr])
    elif len(tr) < 8:
        return None, len(tr)
    return min(t["val_loss"] * t["prepd_std"] ** 2 for t in tr), len(tr)


V = {(a, p, k): value(a, p, k) for a, _, _ in ARMS for p, _ in PROBES for k in KS}
for p, lab in PROBES:
    print(p)
    print("  D       " + "".join(f"{l:>22s}" for _, l, _ in ARMS))
    for k in KS:
        s = V[("scr", p, k)][0]
        row = f"  10^{k/2:<4}"
        for a, _, _ in ARMS:
            v, n = V[(a, p, k)]
            g = "" if a == "scr" or v is None or s is None else f" {np.log10(s / v):+.2f}"
            row += f"{(f'{v:.2e}' if v is not None else '-') + (f' ({n})' if a == 'scr' else '') + g:>22s}"
        print(row)

base = os.path.join(ROOT, "analysis", "transfer", "figs", "highmult")
fig, axs = ps.figure(ncols=2)
for ax, (p, lab) in zip(axs, PROBES):
    for a, l, c in ARMS:
        x = [k / 2 for k in KS if V[(a, p, k)][0] is not None]
        y = [V[(a, p, k)][0] for k in KS if V[(a, p, k)][0] is not None]
        ax.plot(x, y, "o-", color=c, label=l)
    ax.axvline(3.25, color="0.6", ls=":", label=r"$8$k $\to$ $32$k")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10} D$")
    ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(ax, lab)
ps.legend(axs[0], "lower left")
ps.save(fig, base)

fig, axs = ps.figure(ncols=2)
for ax, (p, lab) in zip(axs, PROBES):
    for a, l, c in ARMS[1:]:
        ks = [k for k in KS if V[(a, p, k)][0] is not None and V[("scr", p, k)][0] is not None]
        ax.plot([k / 2 for k in ks], [np.log10(V[("scr", p, k)][0] / V[(a, p, k)][0]) for k in ks], "o-", color=c, label=l)
    ax.axhline(0, color=ps.C.blue, label="scratch")
    ax.axvline(3.25, color="0.6", ls=":", label=r"$8$k $\to$ $32$k")
    ax.set_xlabel(r"$\log_{10} D$")
    ax.set_ylabel(r"$\log_{10}(L_\mathrm{scratch}/L)$")
    ps.process_label(ax, lab)
ps.legend(axs[0], "lower left")
ps.save(fig, base + "_gain")
