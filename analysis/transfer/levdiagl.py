"""Transfer study, the diagram arm with the arXiv:2606.23791 graph encoder (width 64, tp3_levdiagl) next to the first
diagram arm (our encoder, tp3_levdiag): rung 9's pretraining (hp73, 64k steps) with Feynman-diagram inputs, fine-tuned on
the twelve probes. (a) the pretrainings' validation curves, val_loss_no_reg against step, with rung 9's. (b)-(d) per
cell log10(L_rung9 / L_arm), loss = MSE of log|M|^2 at the best checkpoint, > 0 means the arm fine-tunes better:
D = 10^2 and 10^3 on the 8k grid (5-trial searches sharing rung 9's DyHPO seed, finale_synth.cell), D = 10^4 at 32k
(each cell the best of its chosen points in expect_tp3.json, all of them in). A cell is drawn once both sides are in.
With `accum` the same four panels for the per-process accumulation arms instead: the first one (tp3_accum_r9) and the
one redone as in arXiv:2606.23791 (tp3_accrr_r9, D4).
    python analysis/transfer/levdiagl.py [accum]   -> figs/levdiagl_a..d (figs/accrr_a..d), _legend; medians on stdout
"""
import io, contextlib, json, os, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, S  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    from finale_synth import LAB, P, cell  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

EXP = json.load(open(os.path.join(ROOT, "analysis", "transfer", "expect_tp3.json")))
# (pretraining, fine-tune family, label, colour, marker)
ARMS = [("tp3_levdiag", "tp3_levdiag", "diagrams, our encoder", ps.C.blue, "o"),
        ("tp3_levdiagl", "tp3_levdiagl", "diagrams, arXiv:2606.23791 encoder", ps.C.vermillion, "s")]
NAME = "levdiagl"
if sys.argv[1:] == ["accum"]:
    ARMS = [("tp3_accum_r9", "tp3_accum", "accumulation, first", ps.C.blue, "o"),
            ("tp3_accrr_r9", "tp3_accrr", "accumulation as in arXiv:2606.23791", ps.C.vermillion, "s")]
    NAME = "accrr"


def cell32(fam, p, k):
    """A 32k cell: the best of its chosen points (expect_tp3.json), once every one of them has a result."""
    name = f"{fam}32k_{p}_d{k}"
    want = EXP.get(name)
    got = {t["hp"]: t["val_loss"] * t["prepd_std"] ** 2 for t in S.get(name, [])
           if t.get("T") == 32000 and t.get("val_loss") is not None and t.get("prepd_std")}
    if not isinstance(want, list) or not all(h in got for h in want):
        return None
    return min(got[h] for h in want)


def curve(name):
    t = next(t for t in S[name] if t["hp"] == 73)
    v = np.array(t["val_curve"], float)
    return t["validate_every"] * np.arange(1, len(v) + 1), v


base = os.path.join(ROOT, "analysis", "transfer", "figs", NAME)
figs = ps.panels(4)
ax = figs[0][1]
x, y = curve("tp3_ladder_r9")
ax.plot(x, y, color="0.4", label="rung 9")
for pre, _, lab, col, _ in ARMS:
    x, y = curve(pre)
    ax.plot(x, y, color=col, label=lab)
ax.set_yscale("log")
ax.set_xlabel("step")
ax.set_ylabel(r"val. MSE$(\log|\mathcal{M}|^2)$")
ps.process_label(ax, "pretraining")

PANELS = [(4, r"$D=10^2$, $8$k", lambda f, p: cell(f + "fte", p, 4), lambda p: cell("tp3_r9fte", p, 4)),
          (6, r"$D=10^3$, $8$k", lambda f, p: cell(f + "fte", p, 6), lambda p: cell("tp3_r9fte", p, 6)),
          (8, r"$D=10^4$, $32$k", lambda f, p: cell32(f + "fte", p, 8), lambda p: cell32("tp3_r9fte", p, 8))]
for (fig, ax), (k, dlab, arm, ref) in zip(figs[1:], PANELS):
    for _, fam, lab, col, m in ARMS:
        xs, ys = [], []
        for i, p in enumerate(P):
            a, r = arm(fam, p), ref(p)
            if a and r:
                xs.append(np.log10(r / a)); ys.append(i)
        ax.scatter(xs, ys, color=col, marker=m)
        v = np.array(xs)
        if len(v):
            print(f"{dlab:18s} {lab:36s} {len(v):2d} probes  median {np.median(v):+.2f} dex  better on {sum(v > 0)}"
                  + (f"  Wilcoxon p={wilcoxon(v).pvalue:.3g}" if len(v) > 5 else ""))
    ax.axvline(0, color="0.4", ls="--")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.45 * (hi - lo))  # the D label sits upper right, clear of the top probe's points
    ax.set_yticks(range(len(P))); ax.set_yticklabels([LAB[p] for p in P]); ax.invert_yaxis()
    ax.set_xlabel(r"$\log_{10}(L_\mathrm{rung\,9}/L_\mathrm{arm})$")
    ps.process_label(ax, dlab)
ps.legend_strip(figs[0][1], base + "_legend", ncol=3)
ps.save_panels(figs, base)
