"""llm_compare part (a): MSE of log|M|^2 per held-out process in the layout of Suprio's held-out figure
(analysis/llm_compare/suprio_llm_heldout_fig.png): two rows, interpolation (96 relabels) and extrapolation (43 new),
one tick per process in his order and families, 2->2 processes shaded.
His points: the "pID emb. only (baseline)" run, seed 0, 20 epochs (talks/progress_oct2026b/data/draft_pid_by_process.json,
{"interp"|"extrap": {name: [x in his figure, MSE]}}, read off his figure's vector paths; not committed). Ours: a run's
data/<tag>_heldout.json from tools/llm_score.py (first 1e4 test events per process, shared affine statistics).
Process families and labels: analysis/catalog_daniel/classes_default.json (test_group, process).
    python analysis/llm_compare/heldout_per_process.py [--tag control ...] [--label "..." ...] [--out NAME]
Several --tag/--label pairs draw several of our arms, one marker each (output named by --out, default the first tag).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["control"])
    ap.add_argument("--label", nargs="+", default=["our LLoCa, his optimizer settings"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    assert len(a.tag) == len(a.label), (a.tag, a.label)
    draft = json.load(open(os.path.join(ROOT, "talks", "progress_oct2026b", "data", "draft_pid_by_process.json")))
    arms = [(lab, json.load(open(os.path.join(HERE, "data", f"{t}_heldout.json")))["processes"], c, mk)
            for t, lab, c, mk in zip(a.tag, a.label, (ps.C.blue, ps.C.green, ps.C.orange), ("o", "D", "^"))]
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(DAN, "classes_default.json")))["processes"]}
    for _, ours, _, _ in arms:
        assert set(ours) == {n for g in draft.values() for n in g}, "held-out sets differ"
    w0, h0 = ps.PLOT_W_IN, ps.PLOT_H_IN
    ps.PLOT_W_IN, ps.PLOT_H_IN = 7.5, 1.9      # his wide rows: room for one tick per process
    fig, axes = ps.figure(ncols=1, nrows=2)
    for ax, sp in zip(axes, ("interp", "extrap")):
        names = sorted(draft[sp], key=lambda n: draft[sp][n][0])     # his order
        x = np.arange(len(names))
        for i, n in enumerate(names):
            if meta[n]["n_out"] == 2:
                ax.axvspan(i - 0.5, i + 0.5, color="0.88", lw=0, zorder=0)
        ax.plot(x, [draft[sp][n][1] for n in names], "*", color="k", ls="none", zorder=3,
                label="pID emb. only (Suprio)")
        for lab, ours, c, mk in arms:
            ax.plot(x, [ours[n]["mse_logm2"] for n in names], mk, color=c, ls="none", zorder=2, alpha=0.85,
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
        ax.set_ylim(1e-3, 1e3)
        ax.set_yticks(10.0 ** np.arange(-3, 4))
        ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ax.grid(False)
        ax._ps_equal_limits = True    # his fixed y range; the legend sits in the free band above the data
    axes[0].fill_between([], [], color="0.88", label=r"$2\to2$ processes (others $2\to3$)")
    ps.legend(axes[0], "upper left", ncol=3)
    ps.save(fig, os.path.join(HERE, "figs", f"{a.out or a.tag[0]}_heldout_per_process"))
    ps.PLOT_W_IN, ps.PLOT_H_IN = w0, h0


if __name__ == "__main__":
    main()
