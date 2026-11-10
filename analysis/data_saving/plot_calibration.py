"""Best validation MSE vs training-set size D, per target: the tree arm's calibration (MSE of ln|M|^2), or with
--arm loops the one-loop arm's (trained on V/B, MSE of ln|V/B| = MSE of ln|V|, the Born exact).

    python analysis/data_saving/plot_calibration.py [--arm trees|loops]
"""
import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import plot_style as ps

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["trees", "loops"], default="trees")
arm = ap.parse_args().arm
TAG = {"trees": "cal2", "loops": "cal_vb"}[arm]
DATA = os.path.join(REPO, f"analysis/data_saving/data/horizons_{TAG}.json")
rows = json.load(open(DATA))
targets = {"trees": [
    ("epem_ttx", r"$t\bar t$", ps.C.blue),
    ("epem_ccxg", r"$c\bar c g$", ps.C.vermillion),
    ("epem_bbxgg", r"$b\bar b gg$", ps.C.green),
], "loops": [
    ("bbx_zz_nlo", r"$b\bar b\to ZZ$", ps.C.blue),
    ("uux_epemz_nlo", r"$u\bar u\to e^+e^-Z$", ps.C.vermillion),
    ("bbx_gg_nlo", r"$b\bar b\to gg$", ps.C.green),
    ("dg_uwm_nlo", r"$dg\to uW^-$", ps.C.green),
    ("bbx_avevex_nlo", r"$b\bar b\to\gamma\nu\bar\nu$", ps.C.green),
    ("db_bdz_nlo", r"$db\to bdZ$", ps.C.green),
    ("uc_cuz_nlo", r"$uc\to cuZ$", ps.C.green),
]}[arm]
targets = [t for t in targets if any(r["target"] == t[0] for r in rows)]
fig, ax = ps.figure()
for t, lab, col in targets:
    r = sorted((r for r in rows if r["target"] == t and "best" in r), key=lambda r: r["D"])
    ax.plot([x["D"] for x in r], [x["best"] for x in r], marker="o", color=col, label=lab)
    # a run that blew up is shown apart at its D, at the top edge of the axis (never dropped; the caption says so); missing or non-ln|V| runs are printed
    bl = [x for x in rows if x["target"] == t and "blowup_step" in x]
    if bl:
        ax.plot([x["D"] for x in bl], [ax.get_ylim()[1]] * len(bl), marker="x", linestyle="none", color=col)
    for x in rows:
        if x["target"] == t and "best" not in x and "blowup_step" not in x:
            print("not plotted:", t, x.get("k"), "missing" if x.get("missing") else "not ln|V|" if x.get("not_ln_v") else "?")
    for x in bl:
        print("blowup:", t, "k", x["k"], "step", x["blowup_step"])
ax.axhline(1e-4, color="gray", linestyle="--", label=r"$10^{-4}$")
ax.axhline(1e-2, color="gray", linestyle=":", label=r"$10^{-2}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$D$")
ax.set_ylabel(r"best val MSE$(\ln|\mathcal{M}|^2)$" if arm == "trees" else r"best val MSE$(\ln|V|)$")
ps.legend(ax, "lower left", ncol=2)
ps.save(fig, os.path.join(REPO, "analysis/data_saving/figs/calibration_best_mse" + ("" if arm == "trees" else "_loops")))
print("ylim", ax.get_ylim())
