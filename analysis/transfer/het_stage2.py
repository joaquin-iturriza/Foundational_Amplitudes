"""Stage 2 of the het fine-tune benchmark (D94): the big-catalog pretraining hp58 (runs/dancat_het_pre/trial_0058) fine-tuned
with the het loss on the seven finale probes that have no twin in its catalog, at D = 10^1, 10^2, 10^3, 10^4 (k = 2, 4, 6, 8),
against the finale fine-tuned with the het loss (stage 1 and 2, k = 4, 8) and the MSE cells of the ladder: scratch, rung 9,
synthetic, finale (finale_synth_scaling.py). Every cell is its DyHPO sweep's best trial, the loss its validation MSE of
log|M|^2 at the best checkpoint (het: the mu MSE, the het cells select on it), in the target's raw units (the standardized
value times prepd_std^2; every arm of a probe and D shares one standardization, het_stage2_stds.json). The two one-loop
probes are positive on every event, so the MSE cells' sign head is inert and their targets equal the het ones.
Inputs: sweeps_local/tp3_{hp58,fin}fte_*_het/results/*.json, het_stage2_stds.json (data_stats.json of every trial, from
CC), scratch_sweeps.json through cells.py / finale_synth.py. Outputs: the table printed, het_stage2.json,
figs/het_stage2_{a..g}.{png,pdf} and figs/het_stage2_legend.{png,pdf}.
    .venv-driver/bin/python analysis/transfer/het_stage2.py
"""
import contextlib, glob, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cells import ROOT, scratch  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    from finale_synth import cell  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

PROBES = {"ee_ddbar": r"$e^+e^-\to d\bar d$", "ee_WW": r"$e^+e^-\to W^+W^-$", "ee_nnbar": r"$e^+e^-\to\nu\bar\nu$",
          "ee_ttbar": r"$e^+e^-\to t\bar t$", "uubar_Zggg": r"$u\bar u\to Zggg$",
          "ee_dd_nlo": r"$e^+e^-\to d\bar d$, one loop", "ee_bb_nlo": r"$e^+e^-\to b\bar b$, one loop"}
STD = {k: v[0] for k, v in json.load(open(os.path.join(HERE, "het_stage2_stds.json"))).items()}


def het(sweep):
    """The sweep's best trial: (raw-unit mu MSE, number of trials), or None"""
    fs = glob.glob(os.path.join(ROOT, "sweeps_local", sweep, "results", "*.json"))
    rows = []
    for f in fs:
        r = json.load(open(f))
        rd = r["run_dir"].split("Foundational_Amplitudes/", 1)[1]
        rows.append(r["val_loss"] * STD[rd] ** 2)
    return (min(rows), len(rows)) if rows else None


def main():
    K = (2, 4, 6, 8)
    res = {}
    figs = ps.panels(len(PROBES))
    for (fig, ax), (p, lab) in zip(figs, PROBES.items()):
        D = {k: 10 ** (k / 2) for k in range(2, 9)}
        arms = {"scratch (MSE)": {k: scratch(p, k, steered=False)[0] for k in range(2, 9)},
                "rung 9 (MSE)": {k: cell("tp3_r9fte", p, k) for k in range(2, 9)},
                "synthetic (MSE)": {k: cell("tp3_synfte", p, k) for k in range(2, 9)},
                "finale (MSE)": {k: cell("tp3_finfte", p, k) for k in range(2, 9)},
                "finale (het)": {k: (het(f"tp3_finfte_{p}_d{k}_het") or (None,))[0] for k in (4, 8)},
                "hp58 (het)": {k: (het(f"tp3_hp58fte_{p}_d{k}_het") or (None,))[0] for k in K}}
        style = {"scratch (MSE)": dict(color="k", marker="o", ls="--"), "rung 9 (MSE)": dict(color=ps.C.blue, marker="s"),
                 "synthetic (MSE)": dict(color=ps.C.green, marker="^"), "finale (MSE)": dict(color=ps.C.vermillion, marker="D"),
                 "finale (het)": dict(color=ps.C.vermillion, marker="D", mfc="none", ls=":"),
                 "hp58 (het)": dict(color=ps.C.orange, marker="*", ls="-")}
        res[p] = {a: {str(k): v for k, v in c.items()} for a, c in arms.items()}
        for a, c in arms.items():
            pts = [(D[k], v) for k, v in sorted(c.items()) if v]
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], label=a, **style[a])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, lab)
        ps.make_room(ax, max_growth=3.0)
        f = lambda v: f"{v:.3g}" if v else "-"
        print(f"{p:11s} " + "  ".join(f"D=1e{k // 2}: hp58 {f(arms['hp58 (het)'].get(k))} finale het {f(arms['finale (het)'].get(k))} "
                                      f"finale {f(arms['finale (MSE)'].get(k))} r9 {f(arms['rung 9 (MSE)'].get(k))} "
                                      f"scr {f(arms['scratch (MSE)'].get(k))}" for k in K))
    a0 = max((ax for _, ax in figs), key=lambda ax: len(ax.get_legend_handles_labels()[1]))
    ps.legend_strip(a0, os.path.join(HERE, "figs", "het_stage2_legend"), ncol=3)
    ps.save_panels(figs, os.path.join(HERE, "figs", "het_stage2"))
    json.dump(res, open(os.path.join(HERE, "het_stage2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
