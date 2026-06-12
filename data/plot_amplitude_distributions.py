"""
Plot amplitude/weight distributions for the e+e- -> u ubar datasets.

Produces three separate figures:
  1. LO |M|^2 distribution  (from MadGraph matrix2py, dimensionless)
  2. NLO event weights for 4-particle (Born+Virtual) events  (XWGTUP in pb)
  3. NLO event weights for 5-particle (Real emission) events  (XWGTUP in pb)

The NLO weights are Sherpa XWGTUP values — cross-section contributions per
phase-space point in pb, NOT bare matrix elements.  Negative weights appear
in the Born+Virtual sample (virtual corrections) and in the Real sample (IR
subtraction terms).  This is the issue being investigated.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import plot_histogram

DATA_DIR = os.path.dirname(__file__)
OUT_DIR  = os.path.dirname(__file__)

LO_FILE   = os.path.join(DATA_DIR, "ee_uu_91GeV_amplitudes.npy")
NLO4_FILE = os.path.join(DATA_DIR, "eeuu_nlo_10M_2in2out.npy")
NLO5_FILE = os.path.join(DATA_DIR, "eeuu_nlo_10M_2in3out.npy")

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

lo   = np.load(LO_FILE)
nlo4 = np.load(NLO4_FILE)
nlo5 = np.load(NLO5_FILE)

lo_weights   = lo[:, -1]       # |M|^2, dimensionless
nlo4_weights = nlo4[:, -1]     # XWGTUP in pb, can be negative
nlo5_weights = nlo5[:, -1]     # XWGTUP in pb, can be negative

print(f"LO   : {len(lo_weights):,} events  |M|^2 in [{lo_weights.min():.3f}, {lo_weights.max():.3f}]")
print(f"NLO4 : {len(nlo4_weights):,} events  weight in [{nlo4_weights.min():.3e}, {nlo4_weights.max():.3e}] pb")
print(f"NLO5 : {len(nlo5_weights):,} events  weight in [{nlo5_weights.min():.3e}, {nlo5_weights.max():.3e}] pb")
print(f"NLO4 negative fraction: {(nlo4_weights < 0).mean()*100:.1f}%")
print(f"NLO5 negative fraction: {(nlo5_weights < 0).mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# Figure 1: LO |M|^2
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4.5))

plot_histogram(lo_weights, ax=ax, n_bins=80, logx=False, logy=True,
               color="steelblue", label=r"LO $|M|^2$  ($N={:,}$)".format(len(lo_weights)))

ax.set_xlabel(r"$|M|^2$  (dimensionless)")
ax.set_ylabel("Events")
ax.set_title(r"LO amplitude: $e^+e^- \to u\bar{u}$ at $\sqrt{s}=91.2$ GeV")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.4)

fig.tight_layout()
out = os.path.join(OUT_DIR, "dist_lo_amplitude.pdf")
fig.savefig(out)
print(f"Saved: {out}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: NLO 4-particle (Born + Virtual) weights
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4.5))

plot_histogram(nlo4_weights, ax=ax, n_bins=70, logx=True, logy=True,
               color="darkorange", alpha=0.75,
               label=r"Born+Virtual  ($N={:,}$)".format(len(nlo4_weights)))
ax.set_xlabel("Event weight ?? [pb]")
ax.set_ylabel("Events")
ax.set_title(r"NLO $e^+e^- \to u\bar{u}$: Born+Virtual weights  ($\sqrt{s}=91.2$ GeV)")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.4)

fig.tight_layout()
out = os.path.join(OUT_DIR, "dist_nlo4_weights.pdf")
fig.savefig(out)
print(f"Saved: {out}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: NLO 5-particle (Real emission) weights
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4.5))

plot_histogram(nlo5_weights, ax=ax, n_bins=70, logx=True, logy=True,
               color="seagreen", alpha=0.75,
               label=r"Real emission  ($N={:,}$)".format(len(nlo5_weights)))
ax.set_xlabel("Event weight ?? [pb]")
ax.set_ylabel("Events")
ax.set_title(r"NLO $e^+e^- \to u\bar{u}g$: Real emission weights  ($\sqrt{s}=91.2$ GeV)")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.4)

fig.tight_layout()
out = os.path.join(OUT_DIR, "dist_nlo5_weights.pdf")
fig.savefig(out)
print(f"Saved: {out}")
plt.close(fig)

print("\nDone.")
