#!/usr/bin/env python
"""Generated mixture fraction sweep for the ee->uu Z resonance: MSE vs f (fraction of training
events drawn flat-log|M|^2, rest uniform-sqrt(s)). Left: the equal-per-decade (logflat) and
resonance-region metrics vs f -> the interior optimum. Right: per-sqrt(s)-bin MSE, colored by f,
showing the bulk<->pole tradeoff. Saves BOTH .png and .pdf."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AN = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/analysis/divergences"
TAG2F = {"raw": 0.0, "mix025": 0.25, "mix050": 0.5, "mix075": 0.75, "genflat": 1.0}


def main():
    summ = json.load(open(os.path.join(AN, "eeuu_reson_sweep_summary.json")))
    by = {d["tag"]: d for d in summ if d["tag"] in TAG2F}
    tags = sorted(by, key=lambda t: TAG2F[t])
    f = np.array([TAG2F[t] for t in tags])

    logflat = np.array([by[t]["logflat"] for t in tags])
    overall = np.array([by[t]["overall"] for t in tags])
    zpeak = np.array([by[t]["zpeak"] for t in tags])
    # resonance-region MSE: mean over sqrt(s)<100 bins
    reson = []
    for t in tags:
        b = np.array(by[t]["binmse"], float)
        reson.append(np.nanmean(b[b[:, 0] < 100][:, 3]))
    reson = np.array(reson)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    axL.plot(f, logflat, "o-", color="#4c72b0", lw=2, label="logflat (equal wt / decade)")
    axL.plot(f, reson, "s-", color="#c44e52", lw=2, label=r"resonance ($\sqrt{s}<100$)")
    axL.plot(f, overall, "^--", color="#7f7f7f", lw=1.6, label="overall (event-wtd, bulk-heavy)")
    fstar = f[int(np.argmin(logflat))]
    axL.axvline(fstar, color="0.6", ls=":", lw=1)
    axL.set_yscale("log")
    axL.set_xlabel(r"$f$ = fraction flat-$\log|\mathcal{M}|^2$  (rest uniform-$\sqrt{s}$)")
    axL.set_ylabel(r"MSE $\Delta\log|\mathcal{M}|^2$")
    axL.set_title(f"Fraction sweep: interior optimum (logflat min at f={fstar:g})", fontsize=10)
    axL.legend(frameon=False, fontsize=8.5)
    axL.grid(True, which="both", alpha=0.2)

    cmap = plt.cm.viridis
    for t in tags:
        b = np.array(by[t]["binmse"], float)
        ctr = np.sqrt(b[:, 0] * b[:, 1])
        axR.plot(ctr, b[:, 3], "o-", color=cmap(TAG2F[t]), lw=1.8, ms=4, label=f"f={TAG2F[t]:g}")
    axR.axvline(91.19, color="0.5", ls=":", lw=1)
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel(r"$\sqrt{s}$ [GeV]")
    axR.set_ylabel(r"MSE $\Delta\log|\mathcal{M}|^2$")
    axR.set_title("Per-region tradeoff (bulk <-> pole)", fontsize=10)
    axR.legend(frameon=False, fontsize=8, ncol=2)
    axR.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    base = os.path.join(AN, "eeuu_genmix_fraction_sweep")
    fig.savefig(base + ".png", dpi=150)
    fig.savefig(base + ".pdf")
    print("wrote", base + ".png/.pdf")
    print("f      logflat      reson       overall")
    for i, t in enumerate(tags):
        print(f"{f[i]:.2f}  {logflat[i]:.4e}  {reson[i]:.4e}  {overall[i]:.4e}")


if __name__ == "__main__":
    main()
