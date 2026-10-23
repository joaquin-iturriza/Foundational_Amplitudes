"""The dimensionless round (D90, D92, D93) in one figure: each short design run's median per-process MSE of log|M|^2 on
the held-out sets that change one axis at a time, read from analysis/catalog_daniel/data/catalog_2to4_heldout.json
(catalog_2to4_heldout.py) and catalog_short_heldout.json (catalog_short_heldout.py). Rows: the arms; markers: the sets.
Left: the catalog's 43 new held-out processes and the 2->3 parents at 100 TeV (energy alone). Right: the 2->4 probes
(one more gluon or photon than a training 2->3 process) at 13 TeV with pT >= 10 and 30 GeV and at 100 TeV.
Output: figs/dimless_round_{a,b}.{png,pdf}.
    .venv-driver/bin/python analysis/catalog_daniel/dimless_round.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

# (key in catalog_2to4_heldout.json, key in catalog_short_heldout.json, row label)
ARMS = [("Adim", "Adim (short)", r"$\hat s$ only"),
        ("Dless", "Dless (short)", r"dimensionless, $\Lambda^2=\hat s$"),
        ("Dpair_seed1", "Dpair seed 1 (short)", r"dimensionless, $\Lambda^2$ pairs"),
        ("DlessIn", "DlessIn (short)", "dimensionless inputs"),
        ("DlessInVirt", "DlessIn + virtuality pooling (short)", "inputs + virtuality pooling"),
        ("DlessInNoDiag", "DlessIn, no diagrams (short)", "inputs, no diagrams"),
        ("DlessInOrd", "DlessIn + coupling orders (short)", "inputs + coupling orders")]


def main():
    r4 = json.load(open(os.path.join(HERE, "data", "catalog_2to4_heldout.json")))
    rc = json.load(open(os.path.join(HERE, "data", "catalog_short_heldout.json")))
    figs = ps.panels(2)
    axL, axR = figs[0][1], figs[1][1]
    y = list(range(len(ARMS)))[::-1]
    for (k4, kc, _), yy in zip(ARMS, y):
        axL.plot(rc[f"{kc}|43 new"]["median"], yy, "o", color=ps.C.blue, label="catalog, 43 new")
        e = r4.get(f"{k4}|2to4E100|parent")
        if e and e["n"] == 40:
            axL.plot(e["mse"], yy, "s", color=ps.C.vermillion, label=r"$2\to3$ at 100 TeV")
        for st, m, c, lab in (("2to4", "o", ps.C.blue, r"$p_T\geq10$ GeV"), ("2to4c30", "^", ps.C.green, r"$p_T\geq30$ GeV"),
                              ("2to4E100p", "s", ps.C.vermillion, "100 TeV")):
            for kind, fill in (("gluon", c), ("photon", "none")):
                v = r4.get(f"{k4}|{st}|{kind}")
                if v:
                    axR.plot(v["mse"], yy + (0.12 if kind == "gluon" else -0.12), m, color=c, mfc=fill,
                             label=lab if kind == "gluon" and k4 == ARMS[0][0] else None)
    for ax in (axL, axR):
        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels([a[2] for a in ARMS])
        ax.set_ylim(-0.7, len(ARMS) + (-0.3 if ax is axR else 0.4))
    axL.set_xlabel(r"median MSE$(\log|\mathcal{M}|^2)$")
    axR.set_xlabel(r"median MSE$(\log|\mathcal{M}|^2)$, $2\to4$")
    # colour and marker give the set, filled = one more gluon, open = one more photon
    from matplotlib.lines import Line2D
    axR.plot([], [], "o", color="k", mfc="none", label=r"open: $+\gamma$, filled: $+g$")
    from matplotlib.ticker import FixedLocator, NullFormatter, FixedFormatter
    axR.xaxis.set_major_locator(FixedLocator([3, 10, 30]))
    axR.xaxis.set_major_formatter(FixedFormatter(["3", "10", "30"]))
    axR.xaxis.set_minor_formatter(NullFormatter())
    h, l = axL.get_legend_handles_labels()
    u = dict(zip(l, h))
    ps.legend(axL, "upper right", handles=list(u.values()), labels=list(u.keys()))
    ps.make_room(axL)
    # no corner of the 2->4 panel is clear of its rows: the legend goes in a strip above it
    ps.shared_legend(figs[1][0], axR, ncol=2)  # one entry per set: only the first row is labelled
    ps.save_panels(figs, os.path.join(HERE, "figs", "dimless_round"))


if __name__ == "__main__":
    main()
