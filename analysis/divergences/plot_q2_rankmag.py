#!/usr/bin/env python
"""Q2 rank-vs-magnitude verdict plot.

Combines:
  magnitude arms (pi ∝ Q*value):  baseQ, sigma(=mag_real), oracle(=mag |r|)   [q2rw_eval_summary.json]
  rank arms (pi ∝ Q*Phi[rank], magnitude-controlled): rank_real, rank_synth30/46/70  [q2rank_eval_summary.json]

Panel A: gain-fraction-of-oracle for each arm, grouped, so the decisive contrasts are visible:
  rank_real vs rank_synth46  (same rho, same magnitude -> pure rank quality)
  rank_real vs mag_real      (sigma's own magnitude vs true profile by its rank)
Panel B: the magnitude-controlled pure-rank tolerance curve (rank_synth vs rho) + rank_real,
  the honest 'how good must the ordering be' with magnitude held fixed.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
DIV = os.path.join(REPO, "analysis/divergences")
RHO = {"rank_synth30": 0.302, "rank_synth46": 0.452, "rank_synth70": 0.700}


def load(fn):
    p = os.path.join(DIV, fn)
    return {d["tag"]: d for d in json.load(open(p))} if os.path.exists(p) else {}


def main():
    mag = load("q2rw_eval_summary.json")
    rk = load("q2rank_eval_summary.json")
    base = mag["baseQ"]["mse"]; orc = mag["oracle"]["mse"]
    def gf(m): return (base - m) / (base - orc)

    print(f"{'arm':>14} {'kind':>10} {'MSE':>12} {'gain/oracle':>12}")
    rowspec = [
        ("baseQ", "magnitude", mag.get("baseQ")),
        ("mag_real(σ)", "magnitude", mag.get("sigma")),
        ("oracle", "magnitude", mag.get("oracle")),
        ("rank_real", "rank-only", rk.get("rank_real")),
        ("rank_synth30", "rank-only", rk.get("rank_synth30")),
        ("rank_synth46", "rank-only", rk.get("rank_synth46")),
        ("rank_synth70", "rank-only", rk.get("rank_synth70")),
    ]
    for name, kind, d in rowspec:
        if d: print(f"{name:>14} {kind:>10} {d['mse']:12.4e} {100*gf(d['mse']):11.1f}%")

    # decisive contrasts
    if rk.get("rank_real") and rk.get("rank_synth46"):
        print(f"\nDECISIVE  rank_real {100*gf(rk['rank_real']['mse']):.0f}%  vs  "
              f"rank_synth46 {100*gf(rk['rank_synth46']['mse']):.0f}%  "
              f"(same ρ0.46, same magnitude -> gap = rank quality of real σ)")
    if rk.get("rank_real") and mag.get("sigma"):
        print(f"          mag_real(σ) {100*gf(mag['sigma']['mse']):.0f}%  vs  "
              f"rank_real {100*gf(rk['rank_real']['mse']):.0f}%  "
              f"(σ own magnitude vs true-profile-by-rank -> gap = σ magnitude value)")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.8))
    names = [r[0] for r in rowspec if r[2]]
    vals = [gf(r[2]["mse"]) for r in rowspec if r[2]]
    kinds = [r[1] for r in rowspec if r[2]]
    cols = ["0.5" if k == "magnitude" else "C1" for k in kinds]
    cols[names.index("mag_real(σ)")] = "C2" if "mag_real(σ)" in names else cols[0]
    cols[names.index("oracle")] = "C0"
    a0.bar(range(len(names)), vals, color=cols)
    a0.set_xticks(range(len(names))); a0.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    a0.axhline(1, ls="--", color="C0", lw=0.8); a0.axhline(0, color="k", lw=0.7)
    a0.set_ylabel("fraction of oracle gain"); a0.set_title("rank-only (orange) vs magnitude (grey/green/blue)")
    a0.grid(alpha=0.3, axis="y")

    # panel B: pure-rank tolerance (magnitude controlled)
    rr = sorted((RHO[t], gf(rk[t]["mse"])) for t in RHO if t in rk)
    a1.plot([0]+[x for x, _ in rr]+[1.0],
            [0]+[y for _, y in rr]+[1.0], "-o", color="C1", label="rank-only synthetic (magnitude fixed)")
    if rk.get("rank_real"):
        a1.plot(0.457, gf(rk["rank_real"]["mse"]), "*", ms=16, color="C2", label="rank_real (σ ordering)")
    if mag.get("sigma"):
        a1.plot(0.457, gf(mag["sigma"]["mse"]), "P", ms=12, color="darkgreen", label="mag_real (σ magnitude)")
    a1.set_xlabel("σ ranking quality  ρ"); a1.set_ylabel("fraction of oracle gain")
    a1.axhline(0, color="k", lw=0.7); a1.set_ylim(-0.2, 1.15)
    a1.set_title("pure-rank tolerance (magnitude held fixed)")
    a1.grid(alpha=0.3); a1.legend(fontsize=8)
    fig.tight_layout()
    b = os.path.join(DIV, "figs", "q2_rankmag")
    os.makedirs(os.path.dirname(b), exist_ok=True)
    fig.savefig(b + ".png", dpi=140); fig.savefig(b + ".pdf")
    print(f"wrote {b}.png/.pdf")


if __name__ == "__main__":
    main()
