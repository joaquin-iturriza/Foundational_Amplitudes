#!/usr/bin/env python
"""Q2 confirmation plot: trained deep-IR MSE vs reweighting-signal quality.

Reads q2rw_eval_summary.json (overall + per-decade deep-IR MSE for the 4 reweighting
arms) and shows whether sigma-guided reweighting actually cuts the metric, and how the
gain tracks sigma QUALITY:
  baseQ (no sigma) -> oracle (rho=1) brackets the achievable gain; sigma (rho~0.46) and
  deg029 (rho~0.29) sit in between. The "confirmation" is: sigma recovers a large share
  of oracle's gain over baseQ, i.e. imperfect sigma already helps.
Also overlays the OFFLINE targeting-efficiency prediction (q2_sigma_tolerance.json) for a
predicted-vs-realized check.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
DIV = os.path.join(REPO, "analysis/divergences")
WT = os.path.join(REPO, "worktrees/wt-heterosc")

ORDER = ["baseQ", "deg029", "sigma", "oracle"]
RHO = {"baseQ": 0.0, "deg029": 0.29, "sigma": 0.46, "oracle": 1.0}
LABEL = {"baseQ": "baseQ\n(no σ, ρ=0)", "deg029": "deg029\n(ρ~0.29)",
         "sigma": "real σ\n(ρ~0.46)", "oracle": "oracle\n(ρ=1)"}


def main():
    summ = json.load(open(os.path.join(DIV, "q2rw_eval_summary.json")))
    by = {d["tag"]: d for d in summ}
    tags = [t for t in ORDER if t in by]
    mse = np.array([by[t]["mse"] for t in tags])
    base = by["baseQ"]["mse"] if "baseQ" in by else mse.max()
    oracle = by["oracle"]["mse"] if "oracle" in by else mse.min()

    # fraction of oracle's gain over baseQ that each arm realizes
    def gain_frac(m):
        return (base - m) / (base - oracle) if base > oracle else np.nan
    print(f"{'arm':>8} {'rho':>5} {'deepIR MSE':>12} {'gain vs baseQ':>14} {'frac of oracle':>15}")
    for t in tags:
        m = by[t]["mse"]
        print(f"{t:>8} {RHO[t]:5.2f} {m:12.4e} {100*(base-m)/base:13.1f}% {100*gain_frac(m):14.1f}%")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(tags))
    colors = {"baseQ": "0.5", "deg029": "C3", "sigma": "C2", "oracle": "C0"}
    a0.bar(x, mse, color=[colors[t] for t in tags])
    a0.axhline(base, ls="--", color="0.5", lw=1, label="baseQ (no σ)")
    a0.axhline(oracle, ls="--", color="C0", lw=1, label="oracle (ρ=1)")
    a0.set_xticks(x); a0.set_xticklabels([LABEL[t] for t in tags], fontsize=8)
    a0.set_ylabel("deep-IR MSE (held-out, log|M|²)")
    a0.set_title("Q2 confirmation: trained deep-IR MSE vs σ quality")
    a0.legend(fontsize=8); a0.grid(alpha=0.3, axis="y")
    for xi, t in zip(x, tags):
        a0.annotate(f"{100*gain_frac(by[t]['mse']):.0f}%", (xi, by[t]["mse"]),
                    ha="center", va="bottom", fontsize=8)

    # realized gain-fraction vs offline predicted efficiency
    a1.plot([RHO[t] for t in tags], [gain_frac(by[t]["mse"]) for t in tags],
            "-o", color="C2", label="realized (trained) gain fraction")
    tol = os.path.join(DIV, "q2_sigma_tolerance.json")
    if os.path.exists(tol):
        p = json.load(open(tol))
        a1.plot(p["rho_achieved"], p["efficiency"], "--", color="grey",
                label="offline targeting prediction")
    a1.set_xlabel("σ ranking quality  ρ")
    a1.set_ylabel("fraction of oracle reweighting gain")
    a1.axhline(0, color="k", lw=0.7); a1.set_ylim(-0.1, 1.1)
    a1.set_title("realized vs predicted")
    a1.grid(alpha=0.3); a1.legend(fontsize=8)
    fig.tight_layout()
    base_out = os.path.join(DIV, "figs", "q2_confirm")
    os.makedirs(os.path.dirname(base_out), exist_ok=True)
    fig.savefig(base_out + ".png", dpi=140); fig.savefig(base_out + ".pdf")
    print(f"wrote {base_out}.png/.pdf")


if __name__ == "__main__":
    main()
