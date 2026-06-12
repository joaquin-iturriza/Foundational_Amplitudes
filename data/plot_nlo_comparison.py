"""
Compare Born, NLO-corrected (Born+Virtual), and MadGraph LO amplitude
distributions for the same variable-√s phase space.

Produces two figures:
  1. ee_uu_nlo_comparison.pdf   — e+e- -> uu,  √s ∈ [91, 1000] GeV
  2. ee_ttbar_nlo_comparison.pdf — e+e- -> tt,  √s ∈ [350, 1000] GeV

Normalization:  the .dat files store |M|² with the EW coupling factored out
(e=1 convention), while MadGraph uses physical couplings.  The conversion
factor is  e⁴ = (4π α_ew)²  where α_ew = 1/aEWM1 = 1/132.507.

Each figure: two panels
  Left:  amplitude distributions (density-normalised, log x)
         Born × e⁴,  (Born+Virt) × e⁴,  MG LO — should overlap
  Right: bin-median ratios vs √s, all relative to MG LO
         (Born × e⁴) / MG  →  normalization check, should be ≈ 1
         (Born+Virt) × e⁴ / MG  →  (1 + δ_NLO)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import plot_histogram

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# .dat column indices (after skipping the comment line):
# id(0) muR2(1) p1E(2) ... p4pz(17) born(18) virt_fin(19)
DAT_BORN = 18
DAT_VIRT = 19
DAT_P1E  = 2

# EW coupling: aEWM1 = 1/α_ew from param_card.dat
AEW_INV = 132.507
E4 = (4.0 * np.pi / AEW_INV) ** 2   # e⁴ = (4π α_ew)²  ≈ 8.99e-3 ≈ 1/111.2


def alphas_1loop(sqrts, alphas_mz=0.118, mz=91.1876, nf=5):
    b0 = (11 * 3 - 2 * nf) / (12 * np.pi)
    return alphas_mz / (1.0 + alphas_mz * b0 * 2.0 * np.log(sqrts / mz))


def load_dat(name):
    """Return (born_scaled, corrected_scaled, sqrts) from a raw .dat file.
    born_scaled    = born_dat × e⁴   (in MadGraph coupling convention)
    corrected_scaled = (born + (αs/2π)·virt) × e⁴
    """
    raw   = np.loadtxt(os.path.join(DATA_DIR, name), comments="#", dtype=np.float64)
    born  = raw[:, DAT_BORN] * E4
    virt  = raw[:, DAT_VIRT] * E4
    sqrts = 2.0 * raw[:, DAT_P1E]
    alphas = alphas_1loop(sqrts)
    corrected = born + (alphas / (2.0 * np.pi)) * virt
    return born, corrected, sqrts


def load_npy(name, sqrts_min=None, sqrts_max=None):
    """Return (amp, sqrts) from a standard .npy amplitude file, optionally filtered."""
    arr   = np.load(os.path.join(DATA_DIR, name))
    amp   = arr[:, -1]
    sqrts = 2.0 * arr[:, 0]
    if sqrts_min is not None:
        mask = (sqrts >= sqrts_min)
        amp, sqrts = amp[mask], sqrts[mask]
    if sqrts_max is not None:
        mask = (sqrts <= sqrts_max)
        amp, sqrts = amp[mask], sqrts[mask]
    return amp, sqrts


def bin_median_ratio(num, sqrts_num, den, sqrts_den, edges):
    """Median(num in bin) / Median(den in bin) for each √s bin."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    ratio   = np.full(len(centres), np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m_den = np.median(den[(sqrts_den >= lo) & (sqrts_den < hi)])
        m_num = np.median(num[(sqrts_num >= lo) & (sqrts_num < hi)])
        if m_den > 0:
            ratio[i] = m_num / m_den
    return centres, ratio


def make_figure(title, dat_file, lo_npy, sqrts_min, sqrts_max, out_name,
                mg_sqrts_min=None):
    """
    mg_sqrts_min: lower √s cut applied to the MG dataset (use when the MG
                  dataset starts below sqrts_min of the .dat file).
    """
    born_s, corr_s, sqrts_dat = load_dat(dat_file)
    amp_lo, sqrts_lo = load_npy(lo_npy,
                                 sqrts_min=mg_sqrts_min or sqrts_min,
                                 sqrts_max=sqrts_max)

    print(f"  Born × e⁴:      [{born_s.min():.3e}, {born_s.max():.3e}]")
    print(f"  (Born+Virt)×e⁴: [{corr_s.min():.3e}, {corr_s.max():.3e}]")
    print(f"  MG LO:          [{amp_lo.min():.3e}, {amp_lo.max():.3e}]")
    print(f"  e⁴ = {E4:.6f}  (1/e⁴ = {1/E4:.3f})")

    fig, (ax_amp, ax_ratio) = plt.subplots(1, 2, figsize=(13, 4.8))

    # ── amplitude distributions ─────────────────────────────────────────
    datasets = [
        (born_s,  sqrts_dat, "steelblue",   r"Born $\times e^4$"),
        (corr_s,  sqrts_dat, "darkorange",   r"(Born + Virtual) $\times e^4$"),
        (amp_lo,  sqrts_lo,  "seagreen",     r"LO MadGraph"),
    ]
    for amp, _, color, label in datasets:
        plot_histogram(amp, ax=ax_amp, n_bins=80, logx=True, logy=True,
                       density=True, color=color, alpha=0.45,
                       histtype="stepfilled", label=label, linewidth=1.3)
        plot_histogram(amp, ax=ax_amp, n_bins=80, logx=True, logy=True,
                       density=True, color=color, alpha=0.9,
                       histtype="step", label="_nolegend_", linewidth=1.3)

    ax_amp.set_xlabel(r"$|M|^2$")
    ax_amp.set_ylabel("Density")
    ax_amp.set_title("Amplitude distribution")
    ax_amp.legend(fontsize=9)
    ax_amp.grid(True, which="both", ls="--", alpha=0.35)

    # ── bin-median ratios vs √s ─────────────────────────────────────────
    edges = np.linspace(sqrts_min, sqrts_max, 26)

    # Ratio relative to MG LO
    c1, r1 = bin_median_ratio(born_s, sqrts_dat, amp_lo, sqrts_lo, edges)
    c2, r2 = bin_median_ratio(corr_s, sqrts_dat, amp_lo, sqrts_lo, edges)

    ax_ratio.plot(c1, r1, "o-", color="steelblue",  lw=1.5, ms=4,
                  label=r"Born $\times e^4$ / MG LO")
    ax_ratio.plot(c2, r2, "s-", color="darkorange", lw=1.5, ms=4,
                  label=r"(Born + Virtual) $\times e^4$ / MG LO")
    if "uu" in dat_file:

        alpha_ref = alphas_1loop(c1)

        analytic_virtual = (
            1.0
            + (alpha_ref / (2.0 * np.pi))
            * (4.0 / 3.0)
            * (np.pi**2 - 8.0)
        )

        ax_ratio.plot(
            c1,
            analytic_virtual,
            "--",
            color="black",
            lw=2.0,
            label=r"$1 + \frac{\alpha_s}{2\pi}\frac{4}{3}(\pi^2-8)$",
        )
    ax_ratio.axhline(1.0, color="gray", lw=1, ls="--")
    ax_ratio.set_xlabel(r"$\sqrt{s}$  [GeV]")
    ax_ratio.set_ylabel("Median ratio (relative to MG LO)")
    ax_ratio.set_title(r"Ratios vs $\sqrt{s}$")
    ax_ratio.legend()
    ax_ratio.grid(True, ls="--", alpha=0.35)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    out = os.path.join(DATA_DIR, out_name)
    fig.savefig(out, bbox_inches="tight")
    print(f"  → saved {out}")
    plt.close(fig)

def make_raw_figure(title, dat_file, lo_npy,
                    sqrts_min, sqrts_max,
                    out_name, mg_sqrts_min=None):
    """
    Raw comparison without:
      - e^4 rescaling
      - alpha_s/(2pi) virtual prefactor

    This reproduces the original normalization mismatch.
    """

    raw = np.loadtxt(os.path.join(DATA_DIR, dat_file),
                     comments="#", dtype=np.float64)

    born_raw = raw[:, DAT_BORN]
    virt_raw = raw[:, DAT_VIRT]

    # intentionally WRONG / naive combination
    corr_raw = born_raw + virt_raw

    sqrts_dat = 2.0 * raw[:, DAT_P1E]

    amp_lo, sqrts_lo = load_npy(
        lo_npy,
        sqrts_min=mg_sqrts_min or sqrts_min,
        sqrts_max=sqrts_max,
    )

    print(f"  Raw Born:         [{born_raw.min():.3e}, {born_raw.max():.3e}]")
    print(f"  Raw Born+Virt:    [{corr_raw.min():.3e}, {corr_raw.max():.3e}]")
    print(f"  MG LO:            [{amp_lo.min():.3e}, {amp_lo.max():.3e}]")

    fig, (ax_amp, ax_ratio) = plt.subplots(1, 2, figsize=(13, 4.8))

    # ── amplitude distributions ───────────────────────────────────────
    datasets = [
        (born_raw, sqrts_dat, "steelblue",  r"Raw Born"),
        (corr_raw, sqrts_dat, "darkorange", r"Raw Born + Virt"),
        (amp_lo,   sqrts_lo,  "seagreen",   r"LO MadGraph"),
    ]

    for amp, _, color, label in datasets:
        plot_histogram(
            amp,
            ax=ax_amp,
            n_bins=80,
            logx=True,
            logy=True,
            density=True,
            color=color,
            alpha=0.45,
            histtype="stepfilled",
            label=label,
            linewidth=1.3,
        )

        plot_histogram(
            amp,
            ax=ax_amp,
            n_bins=80,
            logx=True,
            logy=True,
            density=True,
            color=color,
            alpha=0.9,
            histtype="step",
            label="_nolegend_",
            linewidth=1.3,
        )

    ax_amp.set_xlabel(r"$|M|^2$")
    ax_amp.set_ylabel("Density")
    ax_amp.set_title("Raw amplitudes (uncorrected normalization)")
    ax_amp.legend(fontsize=9)
    ax_amp.grid(True, which="both", ls="--", alpha=0.35)

    # ── ratios ────────────────────────────────────────────────────────
    edges = np.linspace(sqrts_min, sqrts_max, 26)

    c1, r1 = bin_median_ratio(
        born_raw, sqrts_dat,
        amp_lo, sqrts_lo,
        edges
    )

    c2, r2 = bin_median_ratio(
        corr_raw, sqrts_dat,
        amp_lo, sqrts_lo,
        edges
    )

    ax_ratio.plot(
        c1, r1,
        "o-",
        color="steelblue",
        lw=1.5,
        ms=4,
        label="Raw Born / MG LO",
    )

    ax_ratio.plot(
        c2, r2,
        "s-",
        color="darkorange",
        lw=1.5,
        ms=4,
        label="Raw (Born+Virt) / MG LO",
    )

    ax_ratio.axhline(1.0, color="gray", lw=1, ls="--")

    ax_ratio.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_ratio.set_ylabel("Median ratio")
    ax_ratio.set_title("Raw ratios vs MG")
    ax_ratio.legend()
    ax_ratio.grid(True, ls="--", alpha=0.35)

    fig.suptitle(title + " (raw normalization)", fontsize=14, y=1.01)

    fig.tight_layout()

    out = os.path.join(DATA_DIR, out_name)
    fig.savefig(out, bbox_inches="tight")

    print(f"  → saved {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
print(f"\ne⁴ = {E4:.6e}  (1/e⁴ = {1/E4:.3f})")

print("\ne+e- -> uu")
make_figure(
    title=r"$e^+e^- \to u\bar{u}$,  $\sqrt{s} \in [91,\,1000]$ GeV",
    dat_file="eeuu.dat",
    lo_npy="ee_uu_91-1000GeV_amplitudes.npy",
    sqrts_min=91.0, sqrts_max=1000.0,
    out_name="ee_uu_nlo_comparison.pdf",
)

print("\ne+e- -> ttbar")
make_figure(
    title=r"$e^+e^- \to t\bar{t}$,  $\sqrt{s} \in [350,\,1000]$ GeV",
    dat_file="eett.dat",
    lo_npy="ee_ttbar_346-1000GeV_amplitudes.npy",
    sqrts_min=350.0, sqrts_max=1000.0,
    mg_sqrts_min=350.0,   # filter MG dataset to same range as .dat file
    out_name="ee_ttbar_nlo_comparison.pdf",
)

print("\nRAW e+e- -> uu")
make_raw_figure(
    title=r"$e^+e^- \to u\bar{u}$, raw comparison",
    dat_file="eeuu.dat",
    lo_npy="ee_uu_91-1000GeV_amplitudes.npy",
    sqrts_min=91.0,
    sqrts_max=1000.0,
    out_name="ee_uu_raw_comparison.pdf",
)

print("\nRAW e+e- -> ttbar")
make_raw_figure(
    title=r"$e^+e^- \to t\bar{t}$, raw comparison",
    dat_file="eett.dat",
    lo_npy="ee_ttbar_346-1000GeV_amplitudes.npy",
    sqrts_min=350.0,
    sqrts_max=1000.0,
    mg_sqrts_min=350.0,
    out_name="ee_ttbar_raw_comparison.pdf",
)

print("\nDone.")
