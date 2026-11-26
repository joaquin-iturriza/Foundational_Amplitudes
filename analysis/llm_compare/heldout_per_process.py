"""llm_compare part (a): MSE of log|M|^2 per held-out process in the layout of Suprio's held-out figure
(analysis/llm_compare/suprio_llm_heldout_fig.png): two rows, interpolation (96 relabels) and extrapolation (43 new),
one tick per process in his order and families, 2->2 processes shaded.
His points: the "pID emb. only (baseline)" runs, seeds 0-2, from his own per-process test metrics (suprio.py,
data/suprio_cond/): the median over the seeds, the bar from the lowest to the highest seed; MSE = rmse^2, |bias| =
|level|, MSE - bias^2 = shape^2. His order of the processes: the x positions read off his figure
(talks/progress_oct2026b/data/draft_pid_by_process.json, {"interp"|"extrap": {name: [x, MSE]}}; not committed). Ours: a run's
data/<tag>_heldout.json from tools/llm_score.py (first 1e4 test events per process, shared affine statistics).
Process families and labels: analysis/catalog_daniel/classes_default.json (test_group, process).
    python analysis/llm_compare/heldout_per_process.py [--tag a_trial5 ...] [--out NAME]
Default: every arm read so far (each tag of ARMS with a score file), one fixed name, colour and marker per arm, one fixed
y range, output figs/all_arms_heldout_per_process.
This is the layout Joaquin asked for (his model figure); the per-set pooled numbers come from control_score.py.
Output: figs/<out>_heldout_per_process.{png,pdf}.
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DAN = os.path.join(ROOT, "analysis", "catalog_daniel")
sys.path.insert(0, ROOT)
sys.path.insert(0, DAN)
import plot_style as ps  # noqa: E402
from heldout_delta import proc_label  # noqa: E402
import suprio  # noqa: E402

GROUPS = {"interp": [("ss~ annihilation processes (flavor-degenerate with dd~ and uu~)", r"$s\bar s$ annihilation"),
                     ("Muon final states from bb~ annihilation", r"$b\bar b\to\mu,\ \nu_\mu$"),
                     ("g+s processes from gluon + light-quark families", r"$gs$"),
                     ("b+s(~) scattering from b+light-quark families", r"$b\bar s$, $sb$"),
                     ("c+u~ scattering from c+light-quark families", r"$c\bar u$"),
                     ("ds and s~s~ from same-sign light quark families", r"$ds$, $\bar s\bar s$"),
                     ("Same-sign charm-bottom scattering (cb, c~b~)", r"$cb$, $\bar c\bar b$")],
          "extrap": [("Multi-Higgs production (hh, hhh, Xhh)", r"multi-$H$"),
                     ("Higgs + tau lepton processes", r"$H+\tau$"),
                     ("Triple-W production (W+W+W-, W+W-W-)", r"$WWW$"), ("Triple-Z production (ZZZ)", r"$ZZZ$"),
                     ("gg -> W + qq' (W production from gluon fusion)", r"$gg\to Wq\bar q'$"),
                     ("gg -> tt~ + boson (ttX via gluon fusion)", r"$gg\to t\bar tX$")]}
TITLE = {"interp": "held-out: interpolation", "extrap": "held-out: extrapolation"}


# One fixed name, colour and marker per arm, used in every figure (Joaquin 2026-11-25: the same run under two names and
# changing arms made two figures look inconsistent). "his setup": his target z, AdamW at constant lr, his inputs (part
# (a)/(b)); "our setup": the emission-scale target, our inputs, MuonAdamW, WSD (part (c)). Order = legend order.
ARMS = {"a_trial5": ("his setup, 8 heads", ps.C.blue, "o"),
        "control": ("his setup, 8 heads, untuned", ps.C.grey, "x"),
        "b_h16": ("his setup, 16 heads", ps.C.sky, "D"),
        "b_h32": ("his setup, 32 heads", ps.C.purple, "v"),
        "b_h64": ("his setup, 64 heads", ps.C.vermillion, "s"),
        "b_h120": ("his setup, 120 heads", ps.C.yellow, "P"),
        "level_file": ("his setup, 8 heads, level from diagrams", "#17BECF", "d"),
        "c_control": ("our setup, control", ps.C.green, "<"),
        "c_best": ("our setup, searched", ps.C.orange, "^"),
        # --part bias only (level_source.py): a small network's level error from the process alone, no events
        "lvl_feat": ("level from process", "#999999", "h"),
        "lvl_diag": ("level from process, diagrams", "#8c564b", "p")}
YLIM = (1e-7, 1e3)   # one range for every figure: his 1e-3..1e3 extended below our lowest point; never data-dependent
# --part bias (step 15): the level offset |bias| = |mean(pred - true)| in log|M|^2 (ours: tools/llm_score.py's split,
# data/split/<tag>_heldout.json; his: |level|); its own fixed range
PART = {"mse": ("data", YLIM, r"MSE$(\log|\mathcal{M}|^2)$", "mse"),
        "bias": (os.path.join("data", "split"), (1e-5, 1e2), r"$|$bias$|$ of $\log|\mathcal{M}|^2$", "bias"),
        # --part shape (D567): the error with the level removed, MSE - bias^2 = scale + floor (same split files; his:
        # shape^2); the MSE figure's range
        "shape": (os.path.join("data", "split"), YLIM, r"MSE$-$bias$^2$ of $\log|\mathcal{M}|^2$", "shape")}
LLM_LABEL = "pID emb. only (Suprio)"
LEVEL_ONLY = ("lvl_feat", "lvl_diag")   # no events, no shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=None, help="default: every arm of ARMS whose data/<tag>_heldout.json exists")
    ap.add_argument("--out", default=None, help="default all_arms (mse) or all_arms_<part>")
    ap.add_argument("--part", choices=list(PART), default="mse")
    a = ap.parse_args()
    ddir, ylim, ylabel, llm_stat = PART[a.part]
    a.out = a.out or ("all_arms" if a.part == "mse" else "all_arms_" + a.part)
    tags = a.tag or [t for t in ARMS if os.path.exists(os.path.join(HERE, ddir, f"{t}_heldout.json"))
                     and not (a.part == "shape" and t in LEVEL_ONLY)]
    assert all(t in ARMS for t in tags), f"no fixed name for {set(tags) - set(ARMS)}: add it to ARMS"
    draft = json.load(open(os.path.join(ROOT, "talks", "progress_oct2026b", "data", "draft_pid_by_process.json")))
    val = {"mse": lambda q: q["mse_logm2"], "bias": lambda q: abs(q["bias_logm2"]),
           "shape": lambda q: q["mse_logm2"] - q["bias_logm2"] ** 2}[a.part]
    arms = [(ARMS[t][0], {n: {"y": val(q)} for n, q in
                          json.load(open(os.path.join(HERE, ddir, f"{t}_heldout.json")))["processes"].items()},
             ARMS[t][1], ARMS[t][2]) for t in sorted(tags, key=list(ARMS).index)]
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(DAN, "classes_default.json")))["processes"]}
    sp_his = suprio.split()
    assert all(set(draft[sp]) == set(sp_his[sp]) for sp in draft), "his split differs from his figure's"
    for _, ours, _, _ in arms:
        assert set(ours) == {n for g in draft.values() for n in g}, "held-out sets differ"
    llm = {n: np.abs(v) for n, v in suprio.llm(llm_stat).items()}   # per seed; |level| for the bias
    w0, h0 = ps.PLOT_W_IN, ps.PLOT_H_IN
    ps.PLOT_W_IN, ps.PLOT_H_IN = 7.5, 1.9      # his wide rows: room for one tick per process
    fig, axes = ps.figure(ncols=1, nrows=2)
    for ax, sp in zip(axes, ("interp", "extrap")):
        names = sorted(draft[sp], key=lambda n: draft[sp][n][0])     # his order
        x = np.arange(len(names))
        for i, n in enumerate(names):
            if meta[n]["n_out"] == 2:
                ax.axvspan(i - 0.5, i + 0.5, color="0.88", lw=0, zorder=0)
        med, lo_, hi_ = (np.array([f(llm[n]) for n in names]) for f in (np.median, np.min, np.max))
        ax.errorbar(x, med, yerr=[med - lo_, hi_ - med], fmt="*", color="k", ms=4, elinewidth=0.6, capsize=0, zorder=3,
                    label=LLM_LABEL)
        for lab, ours, c, mk in arms:
            ax.plot(x, [ours[n]["y"] for n in names], mk, color=c, ls="none", zorder=2, alpha=0.85,
                    ms=3, label=lab)
        g_of = [meta[n]["test_group"] for n in names]
        assert set(g_of) <= {g for g, _ in GROUPS[sp]}, set(g_of) - {g for g, _ in GROUPS[sp]}
        for g, short in GROUPS[sp]:
            idx = [i for i, gg in enumerate(g_of) if gg == g]
            lo, hi = min(idx), max(idx)
            assert idx == list(range(lo, hi + 1)), f"{g} not contiguous in his order"
            if lo > 0:
                ax.axvline(lo - 0.5, color="0.4", lw=0.7)
            # his family names above the panel, as in his figure
            ax.text((lo + hi) / 2, 1.02, short, transform=ax.get_xaxis_transform(), ha="center", va="bottom")
        ax.text(0.5, 1.2, TITLE[sp], transform=ax.transAxes, ha="center", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([proc_label(meta[n]["process"]) for n in names], rotation=90, fontsize=3.0)  # his tick labels: one per process, only fits small
        ax.tick_params(axis="x", length=1.5, pad=1)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_yscale("log")
        lo = min([v["y"] for _, ours, _, _ in arms for v in ours.values()] + [lo_.min()])
        assert lo >= ylim[0], f"a point at {lo:.2g} is below the fixed range: lower it in PART (never clip)"
        ax.set_ylim(*ylim)
        ax.set_yticks(10.0 ** np.arange(np.log10(ylim[0]), np.log10(ylim[1]) + 1))
        ax.set_ylabel(ylabel)
        ax.grid(False)
        ax._ps_equal_limits = True    # his fixed y range; the legend sits in the free band above the data
    # |bias| (9 entries): under the fixed range every corner of the top panel covers 97-137 points (ps.covered),
    # the bottom panel's lower left 3 at the default row spacing (ps.save refits 4 columns back to 3)
    lax, loc = (axes[1], "lower left") if a.part == "bias" else (axes[0], "upper left")
    lax.fill_between([], [], color="0.88", label=r"$2\to2$ processes (others $2\to3$)")
    ps.legend(lax, loc, ncol=3, **({"labelspacing": 0.1} if a.part == "bias" else {}))
    ps.save(fig, os.path.join(HERE, "figs", f"{a.out}_heldout_per_process"))
    ps.PLOT_W_IN, ps.PLOT_H_IN = w0, h0


if __name__ == "__main__":
    main()
