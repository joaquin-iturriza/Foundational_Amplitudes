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
# Scannable masses = direct SLHA param_card MASS-block INPUTS that also appear as
# EXTERNAL finals (so the data-derived mass reads the scanned value from momenta):
# b(5), t(6), tau(15), Z(23), H(25). The W mass (24) is a DEPENDENT parameter
# (derived from MZ/Gf/aEW), not a settable input, so it is excluded — W-final
# processes (WW, WWa) get no mass scan and become anchors.
# Scannable masses: only those that (a) genuinely enter the matrix element and
# (b) are tree-level "clean" — i.e. patching them in the param_card does NOT leave
# stale EW-dependent parameters in the ME. That is MT(6), MTA(15), MH(25): each is
# an external final whose mass enters the spinor/propagator while the params it
# would otherwise feed (Yukawas, Higgs self-coupling) don't appear at tree level.
# EXCLUDED: MZ(23) is the EW master input (changing it must recompute MW/couplings —
# a copied/patched standalone is inconsistent); MW is dependent; light quarks
# (u/d/s/c/b) are massless in the 5-flavour ME so their mass does nothing.
HEAVY = {6, 15, 25}
# Extra base processes (not in the LO recipe) added purely for a clean mass scan.
EXTRA_LO = [{"name": "ee_ttbar", "sqrts": [400, 1000]}]   # MT scan (top threshold)
# Widened from (0.09, 0.15): the coupling capability tests showed the α feature
# earns its keep ∝ (α range) × (running steepness). A wide α range makes the
# per-dataset baseline change the per-event *shape* (via running) enough to survive
# per-dataset standardization instead of being a removed offset.
ALPHA_S_RANGE  = (0.05, 0.25)
# Extend QCD (alphas_power>0) processes' √s_min down into the steep-running region
# (α_s runs hard at low μ) so the running-shape — the part the coupling feature can
# actually resolve under per-dataset standardization — is non-negligible. Massless
# QCD drops to here; massive finals keep their (higher) kinematic threshold.
RUNNING_SQRTS_MIN = 25.0
# α_ew is NOT scanned: it is a per-dataset CONSTANT multiplier on |M|^2 (it does not
# run per event), so in log-amplitude it is a constant offset that per-dataset
# preprocessing removes — no learnable per-event signal. Only the per-event knobs
# are scanned: α_s(√s) (runs with energy) and masses (move thresholds/resonances).
N_ALPHAS = 48                      # α_s points per QCD process (free: shares backend)
N_MASS   = 8                       # distinct mass values per massive process (each = 1 LO compile)
MASS_LO, MASS_HI = 0.85, 1.15      # multiplicative mass-scan range


def phys_mass(pdg):
    c = PARTICLE_PROPERTIES.get(int(pdg)) or PARTICLE_PROPERTIES.get(-int(pdg))
    if not c:
        return 0.0
    lm = c[_MCOL]
    return 0.0 if abs(lm - _MASSLESS) < 1e-6 else float(10.0 ** lm)


def lo_points(name, rng):
    """Scan points (list of physics dicts) for one LO process. Empty list ⇒ the
    process has no per-event scan axis (pure-EW massless) → emit a single anchor.

    α_s axis (QCD processes): N_ALPHAS samples, backend shared (no recompile).
    mass axis (heavy externals): N_MASS distinct mass factors, each its own LO
    compile; for a QCD+massive process α_s is resampled at each mass value."""
    cfg = mg.PROCESSES[name]
    has_qcd = cfg.get("alphas_power", 0) > 0
    finals = [abs(p) for p in cfg["pdg_ids"][2:]]
    massive = sorted({p for p in finals if p in HEAVY and phys_mass(p) > 0})

    pts = []
    if massive:
        factors = np.linspace(MASS_LO, MASS_HI, N_MASS)
        as_per = max(1, N_ALPHAS // N_MASS) if has_qcd else 1
        for f in factors:
            masses = {int(p): round(phys_mass(p) * float(f), 4) for p in massive}
            for _ in range(as_per):
                phys = {"masses": masses}
                if has_qcd:
                    phys["alpha_s"] = round(float(rng.uniform(*ALPHA_S_RANGE)), 4)
                pts.append(phys)
    elif has_qcd:
        for _ in range(N_ALPHAS):
            pts.append({"alpha_s": round(float(rng.uniform(*ALPHA_S_RANGE)), 4)})
    return pts


def scan_sqrts(base, base_sqrts, physics):
    """Raise sqrts_min so the (possibly mass-shifted) final state is above threshold:
    sum of final-state masses (scanned override or table value) × 1.05. A heavier
    scanned mass needs a higher minimum √s."""
    smin, smax = float(base_sqrts[0]), float(base_sqrts[1])
    masses = (physics or {}).get("masses", {})
    finals = mg.PROCESSES[base]["pdg_ids"][2:]
    thr = sum(masses.get(abs(int(p)), phys_mass(p)) for p in finals)
    return [max(smin, round(1.05 * thr)), smax]


def running_floor(base, sqrts):
    """For QCD processes, extend √s_min down to RUNNING_SQRTS_MIN (never below a
    massive final's threshold) so events cover the steep-running region. No-op for
    pure-EW processes (alphas_power==0) and for anything already below the floor."""
    cfg = mg.PROCESSES.get(base, {})
    if cfg.get("alphas_power", 0) <= 0:
        return sqrts
    finals = list(cfg.get("pdg_ids", []))[2:]
    thr = sum(phys_mass(p) for p in finals)
    new_min = max(1.05 * thr, min(float(sqrts[0]), RUNNING_SQRTS_MIN))
    return [round(new_min), float(sqrts[1])]


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
    lo = yaml.safe_load(open(args.lo_recipe))["processes"] + EXTRA_LO
    n_lo = n_anchor = 0
    for p in lo:
        base, sqrts = p["name"], p["sqrts"]
        pts = lo_points(base, rng)
        if not pts:                       # pure-EW massless → single anchor (no scan)
            procs.append(entry(base, base, sqrts, None, lo_n))
            n_lo += 1; n_anchor += 1
            continue
        for i, phys in enumerate(pts):
            if phys.get("masses"):
                # mass-scan points need a higher √s_min to clear the shifted threshold
                sq = scan_sqrts(base, sqrts, phys)
            else:
                # pure-coupling QCD points: extend √s_min DOWN into the steep-running
                # region so the α feature carries real per-event shape (no-op for EW)
                sq = running_floor(base, sqrts)
            procs.append(entry(f"{base}__s{i:02d}", base, sq, phys, lo_n))
            n_lo += 1

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
              f"# {n_lo} LO ({n_anchor} anchors + {n_lo - n_anchor} scanned) + {n_nlo} NLO.\n"
              f"# Scan axes: alpha_s(running) on QCD procs, masses on heavy externals.\n"
              f"# Run with: data.mass_from_momenta=true data.coupling_scalars=true\n"
              f"#           model.use_diagrams=true (3-way A/B toggles these).\n")
    with open(args.out, "w") as f:
        f.write(header)
        yaml.safe_dump({"processes": procs}, f, sort_keys=False, default_flow_style=True, width=200)
    print(f"Wrote {args.out}: {n_lo} LO + {n_nlo} NLO = {n_lo + n_nlo} datasets")


if __name__ == "__main__":
    main()
