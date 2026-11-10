"""pretrain_data_saving, one-loop arm stage 1: the common measure against the training-set size D, scratch runs on V/B.

Left: MSE_VB = mean(((V/B)_pred - (V/B)_true)^2) on the 5e3 test pool of the final model (after the sigma stage, D276),
per target, with the stage-2 threshold L0 of the design (2026-11-10, cal_vb2): the largest power of ten that the scratch
curves of b b~ > Z Z and u u~ > e+e- Z both cross inside k=2..9 (above it at the smallest D, at or below it at some D).
Right: MSE ln|V| = mean((ln|(V/B)_pred| - ln|(V/B)_true|)^2) for the one-signed targets (B exact).
Reads analysis/data_saving/data/vb_test_mse_cal_vb_cal_vb2.json (vb_test_mse.py); a 2->3 run trained with the physics
factor on (the superseded u u~ > e+e- Z cal_vb runs) is listed apart, never drawn.

    python analysis/data_saving/plot_vb_test.py
"""
import json
import math
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import plot_style as ps

rows = json.load(open(os.path.join(REPO, "analysis/data_saving/data/vb_test_mse_cal_vb_cal_vb2.json")))
TARGETS = [("bbx_zz_nlo", r"$b\bar b\to ZZ$", ps.C.blue),
           ("uux_epemz_nlo", r"$u\bar u\to e^+e^-Z$", ps.C.vermillion),
           ("epem_bbx_nlo", r"$e^+e^-\to b\bar b$", ps.C.green)]
TWO_TO_TWO = ("bbx_zz_nlo", "epem_bbx_nlo")   # the physics factor is a constant there


def usable(r):
    return r.get("mode") == "scratch" and "mse_vb" in r and not (r.get("phys_factor") and r["target"] not in TWO_TO_TWO)


curves = {}
for t, _, _ in TARGETS:
    c = sorted((r for r in rows if r["target"] == t and usable(r)), key=lambda r: r["k"])
    curves[t] = c
    for r in rows:
        if r["target"] == t and not usable(r):
            print("listed apart:", t, "k", r.get("k"), r.get("error") or ("physics factor on" if r.get("phys_factor") else "?"))
    for r in c:
        print(f"{t:14s} k={r['k']} D={10 ** (r['k'] / 2):9.0f}  MSE_VB {r['mse_vb']:.3e}  var {r['var_vb']:.3g}"
              + (f"  MSE ln|V| {r['mse_lnv']:.3e} (wrong sign {r['wrong_sign']})" if "mse_lnv" in r else ""))


def crosses(c, L):
    return len(c) >= 2 and c[0]["mse_vb"] > L and min(r["mse_vb"] for r in c) <= L


L0 = None
both = [curves["bbx_zz_nlo"], curves["uux_epemz_nlo"]]
if all(both):
    hi = math.floor(math.log10(min(c[0]["mse_vb"] for c in both)))
    for p in range(hi, hi - 30, -1):
        if all(crosses(c, 10.0 ** p) for c in both):
            L0 = 10.0 ** p
            break
print("L0:", L0)
for t, _, _ in TARGETS:
    c = curves[t]
    if L0 and c:
        below = [r for r in c if r["mse_vb"] <= L0]
        print(f"{t}: first D at or below L0: {10 ** (below[0]['k'] / 2):.0f}" if below else f"{t}: above L0 at every D",
              "| below L0 at every D" if below and len(below) == len(c) else "")

fig, (axL, axR) = ps.figure(ncols=2)
for t, lab, col in TARGETS:
    c = curves[t]
    if not c:
        continue
    D = [10 ** (r["k"] / 2) for r in c]
    axL.plot(D, [r["mse_vb"] for r in c], marker="o", color=col, label=lab)
    ln = [(d, r["mse_lnv"]) for d, r in zip(D, c) if "mse_lnv" in r]
    if ln:
        axR.plot(*zip(*ln), marker="o", color=col, label=lab)
if L0:
    axL.axhline(L0, color="gray", linestyle="--", label=r"$L_0$")
for ax in (axL, axR):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D$")
axL.set_ylabel(r"test MSE$(V/B)$")
axR.set_ylabel(r"test MSE$(\ln|V|)$")
ps.legend(axL, "lower left")
ps.save(fig, os.path.join(REPO, "analysis/data_saving/figs/vb_test_mse_loops"))
json.dump({"L0": L0}, open(os.path.join(REPO, "analysis/data_saving/data/vb_L0.json"), "w"))
