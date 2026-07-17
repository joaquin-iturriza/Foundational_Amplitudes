#!/usr/bin/env python
"""Q2 dense plot: trained deep-IR MSE vs sigma-quality rho (single synthetic axis).

Combines the dense synthetic sweep (q2rho_eval_summary.json, rho=0.1..0.9) with the
4-point arms (q2rw_eval_summary.json: baseQ=rho0, oracle=rho1) into one MSE-vs-rho curve,
then overlays the REAL sigma head point (rho~0.457, arm 'sigma') to test whether real-sigma
error is as harmful as synthetic degradation of the same rank quality. Shows the SHAPE:
smooth ramp vs a knee. Right panel = gain fraction of oracle vs rho.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
DIV = os.path.join(REPO, "analysis/divergences")

# achieved rho per synthetic tag (from build_rho_sweep_pools.py stdout)
RHO_ACH = {"rho10": 0.105, "rho20": 0.232, "rho30": 0.302, "rho40": 0.373,
           "rho50": 0.533, "rho60": 0.618, "rho70": 0.700, "rho80": 0.774, "rho90": 0.886}


def load(fn):
    p = os.path.join(DIV, fn)
    return {d["tag"]: d for d in json.load(open(p))} if os.path.exists(p) else {}


def main():
    dense = load("q2rho_eval_summary.json")
    four = load("q2rw_eval_summary.json")

    pts = []  # (rho, mse, tag)
    for tag, r in RHO_ACH.items():
        if tag in dense:
            pts.append((r, dense[tag]["mse"], tag))
    if "baseQ" in four:
        pts.append((0.0, four["baseQ"]["mse"], "baseQ"))
    if "oracle" in four:
        pts.append((1.0, four["oracle"]["mse"], "oracle"))
    pts.sort()
    rho = np.array([p[0] for p in pts]); mse = np.array([p[1] for p in pts])

    base = four["baseQ"]["mse"] if "baseQ" in four else mse.max()
    orc = four["oracle"]["mse"] if "oracle" in four else mse.min()
    gain = (base - mse) / (base - orc)

    real_rho, real_mse = (0.457, four["sigma"]["mse"]) if "sigma" in four else (None, None)
    deg_rho, deg_mse = (0.299, four["deg029"]["mse"]) if "deg029" in four else (None, None)

    print(f"{'rho':>6} {'deepIR MSE':>12} {'gain/oracle':>12}")
    for r, m in zip(rho, mse):
        print(f"{r:6.3f} {m:12.4e} {100*(base-m)/(base-orc):11.1f}%")
    if real_rho:
        print(f"REAL sigma head: rho={real_rho:.3f} mse={real_mse:.4e} "
              f"gain/oracle={100*(base-real_mse)/(base-orc):.1f}%")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    a0.plot(rho, mse, "-o", color="C0", label="synthetic σ_ρ sweep")
    a0.axhline(base, ls="--", color="0.5", lw=1, label="baseQ (no σ)")
    a0.axhline(orc, ls="--", color="C0", lw=1, label="oracle (ρ=1)")
    if real_rho:
        a0.plot(real_rho, real_mse, "*", ms=16, color="C2", label="REAL σ head (ρ0.46)")
    if deg_rho:
        a0.plot(deg_rho, deg_mse, "s", ms=9, color="C3", label="deg029 (4-pt synthetic)")
    a0.set_xlabel("σ ranking quality  ρ = Spearman(σ, |r|)")
    a0.set_ylabel("deep-IR MSE (held-out, log|M|²)")
    a0.set_title("trained deep-IR MSE vs σ quality (dense)")
    a0.grid(alpha=0.3); a0.legend(fontsize=8)

    a1.plot(rho, gain, "-o", color="C0", label="synthetic sweep")
    if real_rho:
        a1.plot(real_rho, (base - real_mse) / (base - orc), "*", ms=16, color="C2",
                label="REAL σ head")
    tol = os.path.join(DIV, "q2_sigma_tolerance.json")
    if os.path.exists(tol):
        p = json.load(open(tol))
        a1.plot(p["rho_achieved"], p["efficiency"], ":", color="grey",
                label="offline targeting prediction")
    a1.axhline(0, color="k", lw=0.7); a1.axhline(1, color="C0", lw=0.7, ls="--")
    a1.set_xlabel("σ ranking quality  ρ"); a1.set_ylabel("fraction of oracle gain")
    a1.set_ylim(-0.15, 1.15); a1.set_title("realized gain vs ρ  (vs offline prediction)")
    a1.grid(alpha=0.3); a1.legend(fontsize=8)
    fig.tight_layout()
    b = os.path.join(DIV, "figs", "q2_dense")
    os.makedirs(os.path.dirname(b), exist_ok=True)
    fig.savefig(b + ".png", dpi=140); fig.savefig(b + ".pdf")
    print(f"wrote {b}.png/.pdf")


if __name__ == "__main__":
    main()
