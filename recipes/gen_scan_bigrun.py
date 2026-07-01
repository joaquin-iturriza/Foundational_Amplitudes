#!/usr/bin/env python
"""Generate the big-run physics-scan recipe for the coupling+mass A/B.

LO: the 25 pretrain25 processes, each scanned over ~POINTS_PER_PROC points in
(alpha_s, alpha_ew, masses) -> ~500 LO datasets. Compile-aware: a coupling-only
point reuses the base backend (free), a mass point needs its own LO standalone, so
masses use a small set of MASS_FACTORS (few compiles) while couplings vary freely.

NLO: the certified virtual bases (see --nlo-bases) scanned over alpha_s (the
strong-coupling reference; the physical-alpha_s weighting is auto-enabled for virt
scans) -> ~100 NLO datasets. Top mass is additionally scanned for ee_ttbar_nlo.
Masses reach the model via the generated momenta (data.mass_from_momenta); the
single `physics` block is the source of truth for both generation and the model
coupling feature.

Usage:
  python recipes/gen_scan_bigrun.py --out recipes/scan_bigrun.yaml \
      --nlo-bases ee_uu_nlo,ee_dd_nlo,ee_ss_nlo,ee_cc_nlo,ee_bb_nlo,ee_ttbar_nlo
"""
import argparse, os, sys
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mg5_pipeline_final as mg
from particle_ids import PARTICLE_PROPERTIES, PARTICLE_FEATURE_NAMES, _MASSLESS

_MCOL = PARTICLE_FEATURE_NAMES.index("log10_mass_gev")
# EXTERNAL-mass scanning was DROPPED: a final-state mass shifts the whole |M|² by a
# per-dataset amount that per-dataset standardization removes — no learnable per-event
# signal (same conclusion as α_ew below). The internal-propagator mass IS scanned
# (Z block below): being s-channel/internal, it is hidden from the momenta, so its
# off-shellness is genuine per-event info.
# Widened from (0.09, 0.15): the coupling capability tests showed the α feature
# earns its keep ∝ (α range) × (running steepness). A wide α range makes the
# per-dataset baseline change the per-event *shape* (via running) enough to survive
# per-dataset standardization instead of being a removed offset.
ALPHA_S_RANGE  = (0.05, 0.25)
# Single COMMON √s floor for every LO dataset (anchors AND coupling scans): span from
# a shared minimum (clamped up only by the final-state kinematic threshold) to 1000, so
# the LO set is kinematically uniform. Low enough to cover the steep α_s-running region
# for QCD and to keep the light-EW anchors wide. Threshold-limited finals (dibosons,
# etc.) start just above 2·m as before.
LO_SQRTS_MIN = 25.0
LO_SQRTS_MAX = 1000.0
# α_ew is NOT scanned: it is a per-dataset CONSTANT multiplier on |M|^2 (it does not
# run per event), so in log-amplitude it is a constant offset that per-dataset
# preprocessing removes — no learnable per-event signal. Only the per-event knobs are
# scanned: α_s(√s) (runs with energy) and the INTERNAL Z mass (moves the s-channel pole).
N_ALPHAS = 48                      # α_s points per QCD process (free: shares backend)

# --- Internal-propagator (Z) mass scan + resonance-dense sampling ------------------
# The s-channel Z is INTERNAL in ee→ffbar neutral-current: its mass is HIDDEN from the
# external momenta (unlike an external final), so the model can't read it off the
# kinematics. Scanning M_Z therefore makes the propagator OFF-SHELLNESS s−M_Z² genuinely
# NEW info (data.offshell_per_event) — the capability the internal-mass tests proved.
# Each point also uses a NARROW √s window bracketing its (scanned) pole, so the sharp
# Breit-Wigner is densely sampled instead of the ~0.3% sliver it gets in a [10,1000]
# scan — the resonance-dense half of the fold. Clean 2→2 neutral-current finals only
# (light enough that the window clears threshold; Z-final processes are external-mass,
# a different lever). A mass patch gets its own standalone (mg.standalone_name), so
# MadGraph re-derives the EW sector from the patched card — the scan is consistent.
ZSCAN_PROCS = ["ee_mumu", "ee_tautau", "ee_ddbar", "ee_bbbar", "ee_nnbar"]
Z_PDG       = 23
N_MZ        = 12
MZ_FACTORS  = np.linspace(0.75, 1.25, N_MZ)   # ~[68,114] GeV around M_Z=91.19 (incl. ≈physical)
ZWIN_LO, ZWIN_HI = 0.78, 1.28                 # √s window = [lo,hi]×M_Z_scanned (dense pole bracket)


def zscan_points():
    """(mass, [√s_min,√s_max]) per scanned M_Z: a resonance-dense window centred on
    the scanned pole so the Breit-Wigner is well sampled at every mass."""
    mz0 = phys_mass(Z_PDG)
    return [(round(mz0 * float(f), 4),
             [round(ZWIN_LO * mz0 * f), round(ZWIN_HI * mz0 * f)]) for f in MZ_FACTORS]


# --- Resonant-INTERMEDIATE mass scans (exotic propagators) -------------------------
# The scanned particle is an INTERNAL intermediate that decays to the final state, so
# its pole lives in a SUB-INVARIANT (M(Wb), M(WW), a μμ pair) — sampled naturally by
# phase space, so NO narrow √s window (unlike the s-channel Z above): just √s above the
# production threshold. All three masses (MT, MH, MZ) are FREE param_card inputs, so
# each scan is a genuinely new, hidden per-event lever for data.offshell_per_event.
#   (process, pdg, n_pts, factor_lo, factor_hi)
RESON_SCANS = [
    ("ee_wwbb",     6,  10, 0.80, 1.20),   # top   : t→Wb resonance (MT)
    ("ee_wwbb",     25, 10, 0.80, 1.20),   # Higgs : H→WW* resonance (MH, gauge coupling)
    ("ee_mumumumu", 23, 12, 0.75, 1.25),   # Z     : μμ-pair resonance in 4-lepton (MZ)
]
RESON_TAG = {6: "mt", 25: "mh", 23: "mz4l"}


def reson_sqrts(pdg, m):
    """√s range for a resonant-intermediate scan point: low enough to clear the
    mediator's PRODUCTION threshold (so the pole is reachable), up to LO_SQRTS_MAX."""
    if pdg == 6:      smin = 1.05 * 2.0 * m               # tt̄ pair production
    elif pdg == 25:   smin = 1.05 * (phys_mass(23) + m)   # ZH associated production
    elif pdg == 23:   smin = 1.60 * m                     # on-shell Z + recoil in 4-lepton
    else:             smin = 2.10 * m
    return [round(max(smin, LO_SQRTS_MIN)), round(LO_SQRTS_MAX)]


def phys_mass(pdg):
    c = PARTICLE_PROPERTIES.get(int(pdg)) or PARTICLE_PROPERTIES.get(-int(pdg))
    if not c:
        return 0.0
    lm = c[_MCOL]
    return 0.0 if abs(lm - _MASSLESS) < 1e-6 else float(10.0 ** lm)


def lo_points(name, rng):
    """Scan points (list of physics dicts) for one LO process. Empty list ⇒ the
    process has no per-event scan axis (pure-EW) → emit a single anchor.

    α_s axis (QCD processes only): N_ALPHAS samples, backend shared (no recompile).
    External-mass scanning was dropped (per-dataset offset removed by standardization)."""
    cfg = mg.PROCESSES[name]
    if cfg.get("alphas_power", 0) <= 0:
        return []
    return [{"alpha_s": round(float(rng.uniform(*ALPHA_S_RANGE)), 4)} for _ in range(N_ALPHAS)]


def lo_sqrts(base):
    """Common LO √s range for EVERY LO process (anchors AND coupling scans): span from
    the shared floor LO_SQRTS_MIN (raised only to clear the final-state kinematic
    threshold, 1.05·Σm) up to LO_SQRTS_MAX, so the LO set is kinematically uniform."""
    cfg = mg.PROCESSES.get(base, {})
    finals = list(cfg.get("pdg_ids", []))[2:]
    thr = sum(phys_mass(p) for p in finals)
    return [round(max(1.05 * thr, LO_SQRTS_MIN)), round(LO_SQRTS_MAX)]


def entry(name, base, sqrts, physics, n):
    e = {"name": name, "sqrts": list(sqrts),
         "n_train": n[0], "n_val": n[1], "n_test": n[2]}
    if base != name:
        e["base"] = base
    if physics:
        e["physics"] = physics
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="recipes/scan_bigrun.yaml")
    ap.add_argument("--lo-recipe", default="recipes/pretrain25.yaml",
                    help="source of the LO process list + sqrts ranges")
    ap.add_argument("--nlo-bases", default="ee_ss_nlo,ee_ttbar_nlo,ee_dd_nlo,ee_cc_nlo,ee_uug_nlo,ee_ddg_nlo",
                    help="comma list of CERTIFIED NLO bases to scan (uu/bb failed cert)")
    ap.add_argument("--nlo-points", type=int, default=16, help="alpha_s points per NLO base")
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--lo-counts", default="10000,2000,2000")
    ap.add_argument("--nlo-counts", default="5000,1000,1000")
    ap.add_argument("--nlo3-counts", default="2000,500,500", help="counts for slow 2->3 NLO")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lo_n   = tuple(int(x) for x in args.lo_counts.split(","))
    nlo_n  = tuple(int(x) for x in args.nlo_counts.split(","))
    nlo3_n = tuple(int(x) for x in args.nlo3_counts.split(","))

    procs = []
    # --- LO scan -----------------------------------------------------------
    # Every LO process spans the SAME √s range (lo_sqrts, clamped up by threshold):
    # anchors and coupling scans alike. QCD procs additionally get an α_s scan.
    lo = yaml.safe_load(open(args.lo_recipe))["processes"]
    n_lo = n_anchor = 0
    for p in lo:
        base = p["name"]
        sq = lo_sqrts(base)
        pts = lo_points(base, rng)
        if not pts:                       # pure-EW → single anchor (no per-event scan)
            procs.append(entry(base, base, sq, None, lo_n))
            n_lo += 1; n_anchor += 1
            continue
        for i, phys in enumerate(pts):    # QCD: α_s scan, all points share lo_sqrts
            procs.append(entry(f"{base}__s{i:02d}", base, sq, phys, lo_n))
            n_lo += 1

    # --- Internal Z-mass scan + resonance-dense block ----------------------
    # New-info internal-propagator lever for data.offshell_per_event (pdg 23).
    n_zscan = 0
    for base in ZSCAN_PROCS:
        if base not in mg.PROCESSES:
            print(f"  [skip] Z-scan base {base} not registered", file=sys.stderr)
            continue
        for i, (mz, sq) in enumerate(zscan_points()):
            procs.append(entry(f"{base}__mz{i:02d}", base, sq, {"masses": {Z_PDG: mz}}, lo_n))
            n_zscan += 1

    # --- Resonant-intermediate (exotic) mass scans: top, Higgs, Z-in-4lepton ---
    # 2→4 processes are slower to generate → smaller counts for ee_wwbb.
    reson_counts = {"ee_wwbb": nlo_n, "ee_mumumumu": lo_n}
    n_reson = 0
    for base, pdg, npts, flo, fhi in RESON_SCANS:
        if base not in mg.PROCESSES:
            print(f"  [skip] reson base {base} not registered", file=sys.stderr)
            continue
        m0 = phys_mass(pdg)
        tag = RESON_TAG.get(pdg, f"m{pdg}")
        cnt = reson_counts.get(base, lo_n)
        for i, f in enumerate(np.linspace(flo, fhi, npts)):
            m  = round(m0 * float(f), 4)
            sq = reson_sqrts(pdg, m)
            procs.append(entry(f"{base}__{tag}{i:02d}", base, sq, {"masses": {pdg: m}}, cnt))
            n_reson += 1

    # --- NLO scan ----------------------------------------------------------
    n_nlo = 0
    for nb in [b for b in args.nlo_bases.split(",") if b]:
        if nb not in mg.PROCESSES:
            print(f"  [skip] NLO base {nb} not registered", file=sys.stderr)
            continue
        cfg = mg.PROCESSES[nb]
        smin = 1.05 * sum(cfg["m_finals"]) if sum(cfg["m_finals"]) > 0 else 50.0
        sqrts = [round(smin), 1000]
        counts = nlo3_n if cfg.get("nfinal", 2) >= 3 else nlo_n
        # alpha_s-ONLY scan (fixed masses): every variant reweights from one stripped
        # MadLoop parent (datagen._ensure_virt_reweighted) — no per-point MadLoop re-run.
        # (NLO mass-scanning would need a MadLoop rebuild per mass; out of scope here.)
        for i in range(args.nlo_points):
            phys = {"alpha_s": round(float(rng.uniform(*ALPHA_S_RANGE)), 4)}
            procs.append(entry(f"{nb}__s{i:02d}", nb, sqrts, phys, counts))
            n_nlo += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    header = (f"# Auto-generated by recipes/gen_scan_bigrun.py (seed={args.seed}).\n"
              f"# {n_lo} LO ({n_anchor} anchors + {n_lo - n_anchor} α_s-scanned) + {n_zscan} Z-mass"
              f" + {n_reson} exotic-mass + {n_nlo} NLO.\n"
              f"# Scan axes: alpha_s(running) on QCD procs; INTERNAL masses (hidden, off-shellness):\n"
              f"#   • s-channel M_Z on {len(ZSCAN_PROCS)} neutral-current procs × {N_MZ} "
              f"(~[68,114] GeV), resonance-dense √s;\n"
              f"#   • EXOTIC resonant intermediates: top MT & Higgs MH in ee_wwbb, Z MZ in ee→4μ\n"
              f"#     (pole is a sub-invariant, phase-space sampled). pdgs [23,6,25].\n"
              f"# External-mass scan DROPPED (per-dataset offset, removed by standardization). All\n"
              f"# LO share one √s range [{int(LO_SQRTS_MIN)},{int(LO_SQRTS_MAX)}] (clamped up by threshold).\n"
              f"# Run with: data.mass_from_momenta=true data.coupling_scalars=true\n"
              f"#           data.offshell_per_event=true data.internal_mass_scalars=true\n"
              f"#           data.internal_mass_pdgs=[23,6,25]  (offshell arm)\n"
              f"#           model.use_diagrams=true (A/B toggles these).\n")
    with open(args.out, "w") as f:
        f.write(header)
        yaml.safe_dump({"processes": procs}, f, sort_keys=False, default_flow_style=True, width=200)
    print(f"Wrote {args.out}: {n_lo} LO + {n_zscan} Z-mass + {n_reson} exotic + {n_nlo} NLO = "
          f"{n_lo + n_zscan + n_reson + n_nlo} datasets")


if __name__ == "__main__":
    main()
