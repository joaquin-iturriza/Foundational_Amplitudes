"""Synthetic generator versions, fine-tuned (D36, D38, D41): each revision-2 pretraining (the finale plus that version's
eight 2->5, 2->6 processes, hp73, 64k; tp3_finale_synhm_v<V>r2) fine-tuned on Z+4g, Z+5g at D = 10^2, 10^3 (8k) and 10^4
(32k), at the v1 synthetic cells' own two points (expect_tp3.json), so only the pretraining differs. Per cell the best of
the two, as log10(L_finale / L): the gain over the finale's own child; v1 (tp3_fsynfte) and finale+W (tp3_fwngfte) for
reference. MSE of log|M|^2 at the best checkpoint. Data: scratch_sweeps.json (a cell run twice keeps its first result
per point).
    python analysis/transfer/synth_versions.py   -> figs/synth_versions (gain over the finale against D); table on stdout
"""
import io, contextlib, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    from highmult import value, PROBES  # noqa: E402
    from replay_children import pts  # noqa: E402  (its module run redraws fig:replay_children, unchanged)
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

KS = [4, 6, 8]
VERS = [("v1", "fsynfte", ps.C.purple), ("v2", "fsv2r2fte", ps.C.sky), ("v3", "fsv3r2fte", ps.C.orange),
        ("v4", "fsv4r2fte", ps.C.yellow), ("v5", "fsv5r2fte", ps.C.vermillion)]


def arm(a, p, k):
    if a.startswith("fsv"):
        return pts(f"tp3_{a}{'32k' if k == 8 else ''}_{p}_d{k}")
    return value(a, p, k)[0]


fig, axs = ps.figure(ncols=2)
for ax, (p, lab) in zip(axs, PROBES):
    print(p)
    for v, a, c in VERS + [(r"finale$+W$", "fwngfte", ps.C.green)]:
        g = [math.log10(arm("finfte", p, k) / arm(a, p, k)) for k in KS]
        print(f"  {v:12s}" + "  ".join(f"10^{k / 2:g} {x:+.2f}" for k, x in zip(KS, g)))
        ax.plot([k / 2 for k in KS], g, "s--" if a == "fwngfte" else "o-", color=c, label=v)
    ax.axhline(0, color=ps.C.blue, label="finale")
    ax.set_xlabel(r"$\log_{10} D$")
    ax.set_ylabel(r"$\log_{10}(L_\mathrm{finale}/L)$")
    ps.process_label(ax, lab)
ps.shared_legend(fig, axs[0], ncol=7)
ps.save(fig, os.path.join(ROOT, "analysis", "transfer", "figs", "synth_versions"))
