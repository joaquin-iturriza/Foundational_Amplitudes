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
HEAVY = {5, 23, 24, 25}            # b, Z, W, H — masses that move 2->2/2->3 kinematics
ALPHA_S_RANGE  = (0.09, 0.15)
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
    ap.add_argument("--nlo-bases", default="ee_uu_nlo,ee_dd_nlo,ee_ss_nlo,ee_cc_nlo,ee_bb_nlo,ee_ttbar_nlo",
                    help="comma list of certified NLO bases to scan")
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
    lo = yaml.safe_load(open(args.lo_recipe))["processes"]
    n_lo = n_anchor = 0
    for p in lo:
        base, sqrts = p["name"], p["sqrts"]
        pts = lo_points(base, rng)
        if not pts:                       # pure-EW massless → single anchor (no scan)
            procs.append(entry(base, base, sqrts, None, lo_n))
            n_lo += 1; n_anchor += 1
            continue
        for i, phys in enumerate(pts):
            procs.append(entry(f"{base}__s{i:02d}", base, sqrts, phys, lo_n))
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
