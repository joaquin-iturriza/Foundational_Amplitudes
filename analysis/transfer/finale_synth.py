"""Transfer study, the finale (rung 9 + every star arm's processes, tp3_finfte) and the synthetic-amplitude pretraining
(tp3_synfte) against the ladder, on the 8k grid. Per probe, the mean over its D cells of the gain over scratch,
log10(L_scratch / L_fine-tune) (gain_heatmap's measure), over the cells finished by every family compared: a search
with its 5 trials in, or the cell's fixed run at the reference point (<cell>_lad: rung 9's best, or the star arm's on
its own probe; the user's call, 2026-10-07). Left: the twelve ladder probes, rung 0 (ee->uu), rung 9, synthetic,
finale. Right: the six star-arm probes, the arm itself and the finale.
    python analysis/transfer/finale_synth.py     -> figs/finale_synth; per-probe numbers and cell counts on stdout
"""
import io, contextlib, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, S, scratch  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    from rung_grid import LAB, P  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

ARMP = {"ee_ddbarg": ("tp3_ssoftfte", "soft"), "ee_ttbarg": ("tp3_sdeadconefte", "dead cone"),
        "ee_ttbar_nlo_thr": ("tp3_sthresholdfte", "threshold"), "ee_dd_nlo_hi": ("tp3_ssudakovfte", "Sudakov"),
        "udbar_enu": ("tp3_swpolefte", "W pole"), "ee_dd_ew_nlo": ("tp3_ssudakovewfte", "Sudakov EW")}
ARMLAB = {"ee_ddbarg": r"$e^+e^-\to d\bar dg$", "ee_ttbarg": r"$e^+e^-\to t\bar tg$",
          "ee_ttbar_nlo_thr": r"$e^+e^-\to t\bar t$ (thr.)", "ee_dd_nlo_hi": r"$e^+e^-\to d\bar d$ (1-loop, hi)",
          "udbar_enu": r"$u\bar d\to e^+\nu_e$", "ee_dd_ew_nlo": r"$e^+e^-\to d\bar d$ (EW 1-loop)"}


def cell(fam, p, k):
    """Loss of a finished cell (MSE of log|M|^2 at the best checkpoint), else None: the best of a search with >= 5
    results, or the cell's fixed run at the reference point."""
    lad = [t for t in S.get(f"{fam}_{p}_d{k}_lad", []) if t.get("val_loss") is not None and t.get("prepd_std")]
    if lad:
        return lad[0]["val_loss"] * lad[0]["prepd_std"] ** 2
    tr = [t for t in S.get(f"{fam}_{p}_d{k}", []) + S.get(f"{fam}_{p}_d{k}_002", [])
          if t.get("val_loss") is not None and t.get("prepd_std") and t.get("hp", 0) >= 0]
    # the two redesigned arms (W pole, Sudakov EW) ran two chosen points per cell, not a 5-trial search (results.tex)
    need = 2 if fam in ("tp3_swpolefte", "tp3_ssudakovewfte") else 5
    return min(t["val_loss"] * t["prepd_std"] ** 2 for t in tr) if len(tr) >= need else None


def gains(p, fams):
    ks = [k for k in range(2, 9) if scratch(p, k, steered=False)[0] and all(cell(f, p, k) for f in fams)]
    return {f: (np.mean([np.log10(scratch(p, k, steered=False)[0] / cell(f, p, k)) for k in ks]) if ks else np.nan)
            for f in fams}, ks


base = os.path.join(ROOT, "analysis", "transfer", "figs", "finale_synth")


def pairs(fam, ref, probes):
    """Per cell finished by both and by scratch: (gain of ref, gain of fam), gain = log10(L_scratch / L)."""
    out = []
    for p in probes:
        r = ref(p) if callable(ref) else ref
        for k in range(2, 9):
            sc, a, b = scratch(p, k, steered=False)[0], cell(r, p, k), cell(fam, p, k)
            if sc and a and b:
                out.append((p, k, np.log10(sc / a), np.log10(sc / b)))
    return out


figs = ps.panels(2)
for (fig, ax), (fam, lab, col) in zip(figs, [("tp3_finfte", "finale", ps.C.vermillion),
                                              ("tp3_synfte", "synthetic", ps.C.green)]):
    lad = pairs(fam, "tp3_r9fte", P)
    ax.scatter([x[2] for x in lad], [x[3] for x in lad], color=col, label="vs rung 9")
    d = np.array([x[3] - x[2] for x in lad])
    print(f"{lab} vs rung 9: {len(lad)} cells over {len({x[0] for x in lad})} probes; "
          f"log10(L_r9 / L_{lab}) median {np.median(d):+.2f}, IQR [{np.percentile(d, 25):+.2f}, {np.percentile(d, 75):+.2f}], "
          f"{lab} better in {np.mean(d > 0):.0%}")
    for p in P:
        dd = [x[3] - x[2] for x in lad if x[0] == p]
        if dd:
            print(f"    {p:12s} {len(dd)} cells, median {np.median(dd):+.2f}")
    if fam == "tp3_finfte":
        arm = pairs(fam, lambda p: ARMP[p][0], list(ARMP))
        ax.scatter([x[2] for x in arm], [x[3] for x in arm], color=ps.C.orange, marker="D",
                   label="vs its star arm")
        if arm:
            d = np.array([x[3] - x[2] for x in arm])
            print(f"finale vs the star arm on its probe: {len(arm)} cells over {len({x[0] for x in arm})} probes; "
                  f"median {np.median(d):+.2f}, finale better in {np.mean(d > 0):.0%}")
    lim = [-1.0, 2.5]
    ax.plot(lim, lim, color="0.5", ls="--", label="equal")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$\log_{10}(L_\mathrm{scratch}/L_\mathrm{reference})$")
    ax.set_ylabel(rf"$\log_{{10}}(L_\mathrm{{scratch}}/L_\mathrm{{{lab}}})$")
    ps.legend(ax, "upper left")
ps.save_panels(figs, base)
