"""
MadGraph5 Amplitude Dataset Pipeline
=====================================
Unified pipeline that auto-selects the backend per subprocess:
  - matrix2py  (preferred): MadGraph's Python interface, imported directly
  - C++ driver (fallback):  compiled standalone_cpp binary, used when
                             matrix2py is unavailable (e.g. e+e- > aa)

The C++ driver now supports any number of final-state particles and runs in
**pipe mode**: the driver process is kept alive for the entire dataset build,
avoiding the per-batch startup overhead.

Usage:
    # Generate from scratch
    python mg5_pipeline.py --process ee_aa --com_energy 200 --nevents 100000 --nbatches 10

    # Delete existing runs and regenerate
    python mg5_pipeline.py --process qqbar_Zg --com_energy 13000 --nevents 100000 --nbatches 10 --reset

    # Add more batches to an existing dataset
    python mg5_pipeline.py --process qqbar_Zg --com_energy 13000 --nevents 100000 --nbatches 5 --extend

    # Skip event generation and/or compilation
    python mg5_pipeline.py --process ee_aa --com_energy 200 --skip_gen
    python mg5_pipeline.py --process ee_aa --com_energy 200 --skip_gen --skip_compile

To add a new process, add an entry to the PROCESSES dictionary below.
"""

import os
import re
import sys
import gzip
import json
import time
import shutil
import hashlib
import argparse
import subprocess
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION — paths
# Overridable via environment so the same script runs on the laptop and on
# Jean Zay without edits:
#   MG5_BIN         path to the mg5_aMC launcher
#   MG5_WORK_DIR    where the *_events / *_standalone dirs live (compiled once)
#   MG5_OUTPUT_DIR  where datasets + recipe sidecars are written
# Defaults: if $WORK is set (Jean Zay) derive from it, else the laptop layout.
# =============================================================================

_WORK = os.environ.get("WORK")
_def_root = f"{_WORK}/mg5amcnlo" if _WORK else "/home/joaquin/mg5amcnlo"
_def_out  = f"{_WORK}/datasets"  if _WORK else "/home/joaquin/datasets"

MG5_BIN    = os.environ.get("MG5_BIN",        f"{_def_root}/bin/mg5_aMC")
WORK_DIR   = os.environ.get("MG5_WORK_DIR",   _def_root)
OUTPUT_DIR = os.environ.get("MG5_OUTPUT_DIR", _def_out)

# Role-dependent seed offsets so train/val/test draw disjoint event streams
# from the same base --seed (continuous phase space ⇒ different streams never
# collide). Keep these fixed forever; changing them re-rolls every split.
ROLE_SEED_OFFSET = {"train": 0, "val": 1_000_000, "test": 2_000_000}

# =============================================================================
# RUNNING COUPLING
# alphas is computed automatically from the renormalization scale.
# The scale defaults to com_energy but can be overridden with --ren_scale.
# =============================================================================

def compute_alphas(mu, alphas_mz=0.118, mz=91.1876, nf=5):
    """Run alphas from MZ to scale mu at 1-loop."""
    b0 = (11 * 3 - 2 * nf) / (12 * np.pi)
    return alphas_mz / (1 + alphas_mz * b0 * 2 * np.log(mu / mz))

def read_alphas_from_param_card(param_card_path, default=0.118):
    """The α_s value baked into the standalone (Block SMINPUTS, '# aS').

    The C++ driver evaluates |M|² at this fixed α_s; we divide it out when
    rescaling to a per-event α_s(√s). Falls back to `default` if not found.
    """
    try:
        with open(param_card_path) as f:
            for line in f:
                if "# aS" in line:
                    toks = line.split("#")[0].split()
                    if len(toks) >= 2:
                        return float(toks[1])
    except Exception:
        pass
    return default


# =============================================================================
# Per-dataset PHYSICS SCAN — vary reference couplings + masses across datasets
# =============================================================================
# A recipe entry may carry a single ``physics`` block specifying the reference
# couplings and masses to use for that dataset:
#
#     physics: { alpha_s: 0.13, alpha_ew: 0.0078125, masses: {6: 173.0, 5: 4.2} }
#
# That ONE block is the single source of truth. ``physics_to_generation`` derives
# the MadGraph generation inputs (param_card_patches + final-state masses +
# reference alpha_s for the per-event running), and ``physics_to_couplings``
# derives the coupling VALUES the model conditions on (vertex factors + global
# scalar). Masses do NOT need to be passed to the model separately: the
# data-derived on-shell mass (data.mass_from_momenta) reads them straight from the
# generated 4-momenta. ``register_scan_process`` merges all of this onto a base
# PROCESSES entry under the (decorated) dataset name so the whole existing
# generation pipeline — backend compile, chunking, recipe identity, caching —
# works unchanged, with each (process, physics) variant getting its own dataset.

# PDG id → MadGraph SM param_card mass symbol (Block MASS).
PDG_TO_MASS_SYM = {
    6: "MT", 5: "MB", 4: "MC", 3: "MS", 2: "MU", 1: "MD",
    15: "MTA", 13: "MM", 11: "ME",
    23: "MZ", 24: "MW", 25: "MH",
}


def physics_to_generation(physics):
    """Recipe ``physics`` block → (param_card_patches, mass_by_pdg, alphas_mz).

    ``param_card_patches`` are MG symbol→value (aS, aEWM1=1/alpha_ew, MT, …).
    ``mass_by_pdg`` is {pdg: mass_GeV} for the masses the user overrode (used to
    rebuild m_finals). ``alphas_mz`` is the reference α_s(M_Z) threaded into the
    per-event running (``compute_alphas``); None ⇒ keep the 0.118 default."""
    physics = dict(physics or {})
    patches, mass_by_pdg = {}, {}
    alphas_mz = None
    if physics.get("alpha_s") is not None:
        alphas_mz = float(physics["alpha_s"])
        patches["aS"] = alphas_mz
    if physics.get("alpha_ew") is not None:
        patches["aEWM1"] = 1.0 / float(physics["alpha_ew"])   # MG input is 1/alpha_ew
    for pdg, m in (physics.get("masses") or {}).items():
        pdg, m = abs(int(pdg)), float(m)   # mass is identical for particle/antiparticle
        mass_by_pdg[pdg] = m
        sym = PDG_TO_MASS_SYM.get(pdg)
        if sym:
            patches[sym] = m
        else:
            print(f"  [WARN] physics.masses: no param_card symbol for PDG {pdg}; "
                  f"ME mass unchanged (phase-space mass still updated).")
    return patches, mass_by_pdg, alphas_mz


def physics_to_couplings(physics):
    """Recipe ``physics`` block → model coupling values {order_key: alpha}, keyed to
    diagram_graphs.DEFAULT_ORDER_KEYS (QED↔EW, QCD↔strong). None if neither set."""
    physics = dict(physics or {})
    c = {}
    if physics.get("alpha_ew") is not None:
        c["QED"] = float(physics["alpha_ew"])
    if physics.get("alpha_s") is not None:
        c["QCD"] = float(physics["alpha_s"])
    return c or None


def _table_mass(pdg):
    """Default on-shell mass (GeV) for a PDG id, from the shared property table
    (massless → 0). Used to rebuild m_finals so phase space matches the ME."""
    from particle_ids import PARTICLE_PROPERTIES, PARTICLE_FEATURE_NAMES, _MASSLESS
    col = PARTICLE_FEATURE_NAMES.index("log10_mass_gev")
    props = PARTICLE_PROPERTIES.get(int(pdg)) or PARTICLE_PROPERTIES.get(-int(pdg))
    if props is None:
        return 0.0
    lm = props[col]
    return 0.0 if abs(lm - _MASSLESS) < 1e-6 else float(10.0 ** lm)


def register_scan_process(name, base, physics):
    """Register ``PROCESSES[name]`` as ``base`` with the ``physics`` scan applied:
    merged param_card_patches, final-state m_finals rebuilt from the (possibly
    overridden) masses, and the reference alphas_mz recorded. Returns the new cfg.
    Idempotent. ``name`` may equal ``base`` (in-place physics override)."""
    if base not in PROCESSES:
        raise KeyError(f"register_scan_process: base process '{base}' not in PROCESSES")
    patches, mass_by_pdg, alphas_mz = physics_to_generation(physics)
    cfg = dict(PROCESSES[base])
    cfg["param_card_patches"] = {**cfg.get("param_card_patches", {}), **patches}
    if alphas_mz is not None:
        cfg["alphas_mz"] = alphas_mz
        # NLO-virtual: an α_s scan only bites if the physical α_s weighting is
        # restored (the stored virt is α_s-stripped by default). Turn it on for any
        # virt α_s scan; the base (non-scan) NLO datasets keep the legacy target.
        if cfg.get("kind") == "virt":
            cfg["alphas_prefactor"] = True
    # Final-state masses START from the base process (which already encodes its
    # intended choices, e.g. massless light quarks in ee_uug) and are overridden
    # ONLY where physics.masses changes a particle — so a pure coupling scan leaves
    # the phase space bit-identical to the base. Fall back to the table mass only
    # when the base specifies neither m_finals nor m_final.
    final_pdgs = list(cfg["pdg_ids"])[2:]
    if "m_finals" in cfg:
        base_mf = list(cfg["m_finals"])
    elif "m_final" in cfg:
        mf = cfg["m_final"]
        base_mf = list(mf) if isinstance(mf, (list, tuple)) else [float(mf)] * len(final_pdgs)
    else:
        base_mf = [_table_mass(p) for p in final_pdgs]
    m_finals = [mass_by_pdg.get(abs(int(p)), base_mf[i]) for i, p in enumerate(final_pdgs)]
    cfg["m_finals"] = m_finals
    if len(m_finals) == 2:
        cfg["m_final"] = m_finals          # (m3, m4) pair; 2→2 sampler accepts a list
    cfg["scan_base"] = base
    PROCESSES[name] = cfg
    return cfg


def standalone_name(process):
    """Backend-directory key for ``process``. A coupling-ONLY scan (same matrix-
    element inputs as its base — identical masses and EW params, differing only in
    the per-event α_s reference) reuses the BASE standalone: matrix2py is handed α_s
    explicitly per event, and the C++ driver's fixed α_s is divided back out in the
    analytic rescale — so the compiled amplitude is bit-identical. Mass/EW scans get
    their own backend. Saves recompiling one MG5 standalone per α_s point."""
    cfg = PROCESSES.get(process, {})
    base = cfg.get("scan_base")
    if not base or base not in PROCESSES:
        return process
    me = lambda c: {k: v for k, v in c.get("param_card_patches", {}).items() if k != "aS"}
    return base if me(cfg) == me(PROCESSES[base]) else process


def register_recipe_processes(specs):
    """Register every recipe spec carrying a ``physics`` scan (or a decorated
    ``base``) into PROCESSES, so generation can address them by dataset name.
    ``specs`` is the experiment's ``_recipe_specs`` list. No-op for plain entries
    (name already a PROCESSES key and no physics)."""
    for s in specs:
        name = s["name"]
        base = s.get("base", name)
        physics = s.get("physics")
        if physics or base != name:
            register_scan_process(name, base, physics or {})


# =============================================================================
# REPRODUCIBILITY — recipe sidecar + provenance
# A dataset is fully described by its *recipe* (the inputs that determine its
# contents). We write `<output_file>.recipe.json` next to every dataset and a
# stable `recipe_id` hash, so a dataset can be regenerated — or recognised as
# already-generated — without keeping the bytes around.
# =============================================================================

RECIPE_SCHEMA_VERSION = 1

def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)),
             "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None

def _file_sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def recipe_id(recipe):
    """Stable hash of the content-determining fields of a recipe.

    Volatile fields (timestamps, output paths, provenance) are excluded so the
    id depends only on *what the data is*, not when/where it was made — usable
    as a cache key on $SCRATCH and as a leak check across splits.
    """
    volatile = {"created", "output_file", "provenance", "schema_version",
                "backend", "effective_seed", "recipe_id",
                # `derived_from`: provenance for an NLO α_s dataset reweighted from
                # its stripped parent — traceability, not "what the data is" (the
                # content identity already lives in process/alphas_mz/prefactor), so
                # it must NOT change the id, or the trainer's lookup would miss it.
                "derived_from",
                # Traceability metadata, not part of "what the data is":
                #   - `chunked`    : whether the chunked path was used (a bool).
                #   - `chunk_size` : the nominal cost-aware events-per-chunk; the
                #                    BYTE-determining quantity is the realized
                #                    `n_chunks` (= ceil(n_events/chunk_size)), which
                #                    variable_energy_recipe folds INTO the id when it
                #                    differs from the legacy plan — so chunk_size
                #                    itself stays metadata here.
                #   - `gen_weight` : the cost estimate that derives chunk_size.
                # NOTE: `n_chunks` is intentionally NOT listed here — when present
                #       it IS identity. variable_energy_recipe only adds it for
                #       TRAIN (byte-strict $SCRATCH cache); val/test omit it so the
                #       frozen benchmark is chunk-policy-independent.
                "chunked", "chunk_size", "gen_weight"}
    core = {k: recipe[k] for k in sorted(recipe) if k not in volatile}
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]

def write_recipe(output_file, recipe):
    """Write `<output_file>.recipe.json` and upsert OUTPUT_DIR/manifest.json."""
    recipe = dict(recipe)
    recipe["schema_version"] = RECIPE_SCHEMA_VERSION
    recipe["created"]        = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    recipe["output_file"]    = os.path.abspath(output_file)
    recipe["provenance"]     = {
        "pipeline_sha256": _file_sha256(os.path.abspath(__file__)),
        "git_commit":      _git_commit(),
        "numpy":           np.__version__,
    }
    rid = recipe_id(recipe)
    recipe["recipe_id"] = rid

    sidecar = output_file + ".recipe.json"
    with open(sidecar, "w") as f:
        json.dump(recipe, f, indent=2, sort_keys=True)

    # Upsert into a flat manifest index keyed by recipe_id.
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        manifest = {}
    manifest[rid] = {k: recipe[k] for k in
                     ("process", "role", "seed", "n_events", "mode",
                      "output_file", "created")
                     if k in recipe}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[RECIPE] {sidecar}  (recipe_id={rid})")
    return rid

# =============================================================================
# PROCESS DEFINITIONS
# Each entry defines:
#   - mg5_generate:          list of MadGraph generate/add process commands
#   - nfinal:                number of final-state particles
#   - initial_state_filter:  optional — 'qqbar' or 'qg' (see parse_lhe_file)
#   - param_card_patches:    {parameter: value} overrides for param_card.dat
#   - run_card_patches:      {parameter: value} overrides beyond STANDARD_RUN_CARD
# =============================================================================

PROCESSES = {
    # ------------------------------------------------------------------
    # e+e- processes
    # ------------------------------------------------------------------
    "ee_qqbar": {
        "mg5_generate": [
            "generate e+ e- > u u~",
            "add process e+ e- > d d~",
            "add process e+ e- > s s~",
            "add process e+ e- > c c~",
            "add process e+ e- > b b~",
        ],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        # Variable-energy mode not supported for summed multi-flavor process;
        # use ee_uu for a single-flavor variable-energy dataset.
    },
    "ee_WW": {
        "mg5_generate": ["generate e+ e- > w+ w-"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 24, -24],
        "m_final": 80.419,   # W boson mass (GeV) — keep in sync with param_card
    },
    "ee_aa": {
        "mg5_generate": ["generate e+ e- > a a"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 22, 22],
        "m_final": 0.0,
    },
    "ee_aaa": {
        "mg5_generate": ["generate e+ e- > a a a"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 22, 22, 22],
        "m_finals": [0.0, 0.0, 0.0],
    },
    "ee_ttbar": {
        "mg5_generate": ["generate e+ e- > t t~"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 6, -6],
        "m_final": 173.0,   # top quark mass (GeV) — keep in sync with param_card
    },
    "ee_uu": {
        "mg5_generate": ["generate e+ e- > u u~"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 2, -2],
        "m_final": 0.0,
    },
    "ee_uug": {
        "mg5_generate": ["generate e+ e- > u u~ g"],
        "nfinal": 3,
        "alphas_power": 1,    # |M|² ∝ α_s¹ at LO (one gluon emission)
        "param_card_patches": {},
        "run_card_patches": {
	    "lpp1":        "0",
	    "lpp2":        "0",
	    "sde_strategy": "2",
	    "hard_survey":  "1",
	    "ptj":         "10.0",   # min pT on the gluon in GeV
	    "drjj":        "0.4",    # min deltaR between partons
        },
        "pdg_ids": [11, -11, 2, -2, 21],
        "m_finals": [0.0, 0.0, 0.0],
    },
    "ee_uugg": {
        "mg5_generate": ["generate e+ e- > u u~ g g"],
        "nfinal": 4,
        "alphas_power": 2,    # |M|² ∝ α_s² at LO (two gluon emissions)
        "param_card_patches": {},
        "run_card_patches": {
	    "lpp1":        "0",
	    "lpp2":        "0",
	    "sde_strategy": "2",
	    "hard_survey":  "1",
	    "ptj":         "10.0",   # min pT on the gluon in GeV
	    "drjj":        "0.4",    # min deltaR between partons
        },
        "pdg_ids": [11, -11, 2, -2, 21, 21],
        "m_finals": [0.0, 0.0, 0.0, 0.0],
    },
    "ee_wwz": {
        "mg5_generate": ["generate e+ e- > w+ w- z"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 24, -24, 23],
        "m_finals": [80.419, 80.419, 91.1880],  # mW, mW, mZ (GeV)
    },
    # Resonant-INTERNAL-propagator processes for the offshell mass lever: the scanned
    # particle is an intermediate that decays to the final state (hidden pole), so its
    # mass (a FREE param_card input) is genuinely new per-event info via s_prop − M².
    "ee_wwbb": {   # top: ee→tt̄→W⁺W⁻bb̄, internal t/t̄ propagators resonate at M(Wb)≈MT
        "mg5_generate": ["generate e+ e- > w+ w- b b~"],
        "nfinal": 4,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 24, -24, 5, -5],
        "m_finals": [80.419, 80.419, 0.0, 0.0],  # mW, mW, mb=0 (5F); top is internal
    },
    # ------------------------------------------------------------------
    # Foundation pretraining set (25 e+e- processes, tree-level LO).
    # Deliberately excludes ee_uu / ee_ttbar (their NLO versions are the
    # fine-tune transfer targets) and ee_ccbar (massless c ⇒ identical
    # amplitude to the excluded up-type ee_uu). Masses match the MG5 SM
    # param_card defaults so phase space and the matrix element agree.
    # ------------------------------------------------------------------
    # --- QED photon multiplicity ---
    "ee_aaaa": {
        "mg5_generate": ["generate e+ e- > a a a a"],
        "nfinal": 4,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 22, 22, 22, 22],
        "m_finals": [0.0, 0.0, 0.0, 0.0],
    },
    # --- leptons ---
    "ee_mumu": {
        "mg5_generate": ["generate e+ e- > mu+ mu-"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, -13, 13],
        "m_final": 0.0,
    },
    "ee_tautau": {
        "mg5_generate": ["generate e+ e- > ta+ ta-"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, -15, 15],
        "m_final": 1.777,
    },
    "ee_mumua": {
        "mg5_generate": ["generate e+ e- > mu+ mu- a"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, -13, 13, 22],
        "m_finals": [0.0, 0.0, 0.0],
    },
    "ee_tautaua": {
        "mg5_generate": ["generate e+ e- > ta+ ta- a"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, -15, 15, 22],
        "m_finals": [1.777, 1.777, 0.0],
    },
    "ee_mumumumu": {
        "mg5_generate": ["generate e+ e- > mu+ mu- mu+ mu-"],
        "nfinal": 4,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, -13, 13, -13, 13],
        "m_finals": [0.0, 0.0, 0.0, 0.0],
    },
    "ee_bhabha": {
        "mg5_generate": ["generate e+ e- > e+ e-"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, -11, 11],
        "m_final": 0.0,
    },
    "ee_nnbar": {
        "mg5_generate": ["generate e+ e- > ve ve~"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 12, -12],
        "m_final": 0.0,
    },
    # --- quarks (down-type light + heavy b) and QCD radiation ladders ---
    "ee_ddbar": {
        "mg5_generate": ["generate e+ e- > d d~"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 1, -1],
        "m_final": 0.0,
    },
    "ee_bbbar": {
        "mg5_generate": ["generate e+ e- > b b~"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 5, -5],
        "m_final": 4.7,
    },
    "ee_ddbarg": {
        "mg5_generate": ["generate e+ e- > d d~ g"],
        "nfinal": 3,
        "alphas_power": 1,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0", "sde_strategy": "2",
                             "hard_survey": "1", "ptj": "10.0", "drjj": "0.4"},
        "pdg_ids": [11, -11, 1, -1, 21],
        "m_finals": [0.0, 0.0, 0.0],
    },
    "ee_bbbarg": {
        "mg5_generate": ["generate e+ e- > b b~ g"],
        "nfinal": 3,
        "alphas_power": 1,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0", "sde_strategy": "2",
                             "hard_survey": "1", "ptj": "10.0", "drjj": "0.4"},
        "pdg_ids": [11, -11, 5, -5, 21],
        "m_finals": [4.7, 4.7, 0.0],
    },
    "ee_uuggg": {
        "mg5_generate": ["generate e+ e- > u u~ g g g"],
        "nfinal": 5,
        "alphas_power": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0", "sde_strategy": "2",
                             "hard_survey": "1", "ptj": "10.0", "drjj": "0.4"},
        "pdg_ids": [11, -11, 2, -2, 21, 21, 21],
        "m_finals": [0.0, 0.0, 0.0, 0.0, 0.0],
    },
    # --- electroweak di/tri-boson + Higgs (incl. unequal-mass 2→2) ---
    "ee_ZZ": {
        "mg5_generate": ["generate e+ e- > z z"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 23, 23],
        "m_final": 91.1880,
    },
    "ee_Za": {
        "mg5_generate": ["generate e+ e- > z a"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 23, 22],
        "m_finals": [91.1880, 0.0],   # unequal-mass 2→2 (mZ, mγ)
    },
    "ee_ZH": {
        "mg5_generate": ["generate e+ e- > z h"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 23, 25],
        "m_finals": [91.1880, 125.0],  # unequal-mass 2→2 (mZ, mH)
    },
    "ee_WWa": {
        "mg5_generate": ["generate e+ e- > w+ w- a"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 24, -24, 22],
        "m_finals": [80.419, 80.419, 0.0],
    },
    "ee_ZZa": {
        "mg5_generate": ["generate e+ e- > z z a"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 23, 23, 22],
        "m_finals": [91.1880, 91.1880, 0.0],
    },
    "ee_ZZZ": {
        "mg5_generate": ["generate e+ e- > z z z"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 23, 23, 23],
        "m_finals": [91.1880, 91.1880, 91.1880],
    },
    # ------------------------------------------------------------------
    # qq̄ processes
    # ------------------------------------------------------------------
    "qqbar_Zg": {
        "mg5_generate": [
            "generate u u~ > z g",
            "add process d d~ > z g",
            "add process s s~ > z g",
            "add process c c~ > z g",
            "add process b b~ > z g",
        ],
        "nfinal": 2,
        "initial_state_filter": "qqbar",
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
    },
    "qqbar_Zgg": {
        "mg5_generate": [
            "generate u u~ > z g g",
            "add process d d~ > z g g",
            "add process s s~ > z g g",
            "add process c c~ > z g g",
            "add process b b~ > z g g",
        ],
        "nfinal": 3,
        "initial_state_filter": "qqbar",
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
    },
    "qqbar_Zggg": {
        "mg5_generate": [
            "generate u u~ > z g g g",
            "add process d d~ > z g g g",
            "add process s s~ > z g g g",
            "add process c c~ > z g g g",
            "add process b b~ > z g g g",
        ],
        "nfinal": 4,
        "initial_state_filter": "qqbar",
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
    },
    "qqbar_Zgggg": {
        "mg5_generate": [
            "generate u u~ > z g g g g",
            "add process d d~ > z g g g g",
            "add process s s~ > z g g g g",
            "add process c c~ > z g g g g",
            "add process b b~ > z g g g g",
        ],
        "nfinal": 5,
        "initial_state_filter": "qqbar",
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
    },
    # ------------------------------------------------------------------
    # pp processes
    # ------------------------------------------------------------------
    "pp_Zj": {
        "mg5_generate": ["generate p p > z j"],
        "nfinal": 2,
        "initial_state_filter": "qqbar",
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "1", "lpp2": "1"},
    },
    # Template for new processes:
    # "ee_NEW": {
    #     "mg5_generate": ["generate e+ e- > X X~"],
    #     "nfinal": 2,
    #     "param_card_patches": {},
    #     "run_card_patches": {"lpp1": "0", "lpp2": "0"},
    # },
    # ------------------------------------------------------------------
    # NLO QCD *virtual* processes (kind="virt"): generated through MadLoop
    # (tools/nlo_virtual_pipeline.py), NOT the LO standalone path. datagen routes
    # them by `kind`: the backend is the [virt=QCD] standalone (`virt_base` keys
    # tools.VIRT_PROCESSES), and each chunk is generated in an isolated subprocess
    # because matrix2py.so chdir's and is a process-singleton. `n_loops=1` flows to
    # amp_orders=[1, alphas_power] (so the recipe/ id is distinct from LO same-name
    # processes); `virt=True` triggers the ×8 generation-cost weight. The stored
    # amplitude is virt_e4 (absolute one-loop finite part, no alpha_s prefactor).
    # Distinct names (…_nlo) keep recipes/outputs/cache separate from the LO entries.
    "ee_ss_nlo": {                       # easy: massless, fast loop
        "kind": "virt", "virt": True, "virt_base": "ee_ss",
        "nfinal": 2, "n_loops": 1, "alphas_power": 1,
        "pdg_ids": [11, -11, 3, -3], "m_finals": [0.0, 0.0],
        "param_card_patches": {},
    },
    "ee_ttbar_nlo": {                    # hard: massive top loop + stability checks
        "kind": "virt", "virt": True, "virt_base": "ee_ttbar",
        "nfinal": 2, "n_loops": 1, "alphas_power": 1,
        "pdg_ids": [11, -11, 6, -6], "m_finals": [172.5, 172.5],
        "param_card_patches": {},
    },
    # 2->2 qqbar QCD-virtual flavours (QED born; target ∝ α_s¹ → amp_orders [1,1]).
    "ee_uu_nlo": {"kind": "virt", "virt": True, "virt_base": "ee_uu", "nfinal": 2,
                  "n_loops": 1, "alphas_power": 1, "pdg_ids": [11, -11, 2, -2],
                  "m_finals": [0.0, 0.0], "param_card_patches": {}},
    "ee_dd_nlo": {"kind": "virt", "virt": True, "virt_base": "ee_dd", "nfinal": 2,
                  "n_loops": 1, "alphas_power": 1, "pdg_ids": [11, -11, 1, -1],
                  "m_finals": [0.0, 0.0], "param_card_patches": {}},
    "ee_cc_nlo": {"kind": "virt", "virt": True, "virt_base": "ee_cc", "nfinal": 2,
                  "n_loops": 1, "alphas_power": 1, "pdg_ids": [11, -11, 4, -4],
                  "m_finals": [0.0, 0.0], "param_card_patches": {}},
    "ee_bb_nlo": {"kind": "virt", "virt": True, "virt_base": "ee_bb", "nfinal": 2,
                  "n_loops": 1, "alphas_power": 1, "pdg_ids": [11, -11, 5, -5],
                  "m_finals": [4.18, 4.18], "param_card_patches": {}},
    # 2->3 qqg QCD-virtual (born ∝ α_s; target ∝ α_s² → amp_orders [1,2]).
    "ee_uug_nlo": {"kind": "virt", "virt": True, "virt_base": "ee_uug", "nfinal": 3,
                   "n_loops": 1, "alphas_power": 2, "pdg_ids": [11, -11, 2, -2, 21],
                   "m_finals": [0.0, 0.0, 0.0], "param_card_patches": {}},
    "ee_ddg_nlo": {"kind": "virt", "virt": True, "virt_base": "ee_ddg", "nfinal": 3,
                   "n_loops": 1, "alphas_power": 2, "pdg_ids": [11, -11, 1, -1, 21],
                   "m_finals": [0.0, 0.0, 0.0], "param_card_patches": {}},
}

# =============================================================================
# STANDARD RUN CARD SETTINGS applied to every process
# =============================================================================

STANDARD_RUN_CARD = {
    "lpp1":     "0",
    "lpp2":     "0",
    "polbeam1": "0.0",
    "polbeam2": "0.0",
    "ptj":      "0.0",
    "ptjmax":   "-1.0",
    "etaj":     "-1.0",
    "drjj":     "0.0",
    "nhel":     "0",
    "iseed":    None,   # set per batch
}

# =============================================================================
# STEP 1: Generate MadGraph process directories
# =============================================================================

def fortran_dir_for(standalone_dir):
    """Sibling dir holding the Fortran standalone (matrix2py backend)."""
    return standalone_dir + "_py"

def generate_mg5_process(process_name, config):
    """Run MadGraph to generate the events dir plus two standalone outputs:
       - <proc>_standalone     : standalone_cpp (C++ driver, fixed α_s) — fallback
       - <proc>_standalone_py  : Fortran standalone (matrix2py, per-event α_s) — preferred
    """
    events_dir    = f"{WORK_DIR}/{process_name}_events"
    standalone_dir = f"{WORK_DIR}/{process_name}_standalone"
    fortran_dir    = fortran_dir_for(standalone_dir)
    generate_cmds = "\n".join(config["mg5_generate"])

    if not os.path.exists(events_dir):
        print(f"\n[MG5] Generating process directories for {process_name}...")
        # The Fortran standalone (matrix2py) is only emitted when opted in — by
        # default the C++ driver + analytic α_s rescale covers everything.
        fortran_out = (f"output standalone {fortran_dir}\n"
                       if os.environ.get("MG5_USE_MATRIX2PY") else "")
        mg5_script = f"""
{generate_cmds}
output {events_dir}
output standalone_cpp {standalone_dir}
{fortran_out}"""
        result = subprocess.run(
            [MG5_BIN],
            input=mg5_script,
            capture_output=False,
            text=True
        )
        if result.returncode != 0:
            print("MadGraph error:", result.stderr[-2000:])
            sys.exit(1)
        print(f"[MG5] Process directories created.")
    else:
        print(f"[MG5] Process directory already exists, skipping generation.")

    # Apply per-dataset param_card overrides (masses, EW inputs) to the STANDALONE's
    # card — the one the C++ driver / matrix2py actually read at runtime (derived
    # params like MW and the couplings are recomputed from these inputs by the
    # model's Parameters class). Without this a mass/EW scan silently used defaults.
    # SLHA format ('<id> value # NAME'), so the SLHA patcher (not patch_card).
    param_patches = config.get("param_card_patches", {})
    if param_patches:
        for pc in (f"{standalone_dir}/Cards/param_card.dat",
                   f"{fortran_dir}/Cards/param_card.dat"):
            if os.path.exists(pc):
                print(f"  [CARD] Patching standalone param_card (SLHA) {param_patches} -> {pc}")
                patch_param_card_slha(pc, param_patches)

    return events_dir, standalone_dir

# =============================================================================
# STEP 2: Patch run_card and param_card
# =============================================================================

def patch_card(card_path, patches):
    """Apply key=value patches to a MadGraph card file.

    Handles both card formats:
      - "  value = key ! comment"   (original MadGraph format)
      - "  key = value ! comment"   (rewritten by MadGraph after first run)
    """
    with open(card_path, 'r') as f:
        lines = f.readlines()

    patched = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            new_lines.append(line)
            continue

        matched = False
        for key, value in patches.items():
            if key in patched:
                continue
            m1 = re.match(
                r'^(\s*)[^\s=!]+(\s*=\s*)(' + re.escape(key) + r')(\s*(?:!.*)?)$',
                line
            )
            m2 = re.match(
                r'^(\s*)(' + re.escape(key) + r')(\s*=\s*)[^\s!]+(\s*(?:!.*)?)$',
                line
            )
            if m1:
                new_lines.append(f"{m1.group(1)}{value}{m1.group(2)}{key}{m1.group(4)}\n")
                patched.add(key)
                matched = True
                break
            elif m2:
                new_lines.append(f"{m2.group(1)}{key}{m2.group(3)}{value}{m2.group(4)}\n")
                patched.add(key)
                matched = True
                break
        if not matched:
            new_lines.append(line)

    for key in patches:
        if key not in patched:
            print(f"  [WARN] Could not patch '{key}' in {card_path} — key not found")

    with open(card_path, 'w') as f:
        f.writelines(new_lines)


def patch_param_card_slha(card_path, patches):
    """Patch an SLHA param_card (Blocks MASS/SMINPUTS/…), whose lines are
    ``<id> <value> # NAME`` — a format ``patch_card`` (which expects ``value = key``)
    cannot touch. The value is the last token before the ``# NAME`` comment; NAME is
    matched case-insensitively against the patch keys (e.g. ``{"MZ": 90.0}``). Only
    real INPUT parameters are matched — a dependent line like ``24 ... # w+ : cmath…``
    has comment-name ``w+`` so ``MW`` won't (and can't) be set this way. Returns the
    set of keys that were NOT found."""
    want = {str(k).lower(): float(v) for k, v in patches.items()}
    out, done = [], set()
    for line in open(card_path):
        if "#" in line and line.split("#", 1)[0].strip():
            code, comment = line.split("#", 1)
            name = comment.strip().split()[0].lower() if comment.strip() else ""
            if name in want and name not in done:
                toks = code.rstrip("\n").rstrip().split()
                toks[-1] = f"{want[name]:.6e}"
                out.append(" " + " ".join(toks) + " #" + comment)
                done.add(name)
                continue
        out.append(line)
    with open(card_path, "w") as f:
        f.writelines(out)
    missing = set(want) - done
    if missing:
        print(f"  [WARN] patch_param_card_slha: keys not found in {card_path}: {sorted(missing)}")
    return missing


def get_existing_batch_count(events_dir):
    events_path = Path(events_dir) / "Events"
    if not events_path.exists():
        return 0
    return len(list(events_path.glob("run_*")))

def delete_existing_runs(events_dir):
    events_path = Path(events_dir) / "Events"
    if not events_path.exists():
        return
    run_dirs = list(events_path.glob("run_*"))
    for d in run_dirs:
        shutil.rmtree(d)
    print(f"  [RESET] Deleted {len(run_dirs)} existing run directories.")

def configure_cards(events_dir, config, com_energy, nevents, batch_idx):
    """Patch run_card and param_card for this batch."""
    run_card_path   = f"{events_dir}/Cards/run_card.dat"
    param_card_path = f"{events_dir}/Cards/param_card.dat"

    run_patches = dict(STANDARD_RUN_CARD)
    run_patches["ebeam1"]  = com_energy / 2
    run_patches["ebeam2"]  = com_energy / 2
    run_patches["nevents"] = nevents
    run_patches["iseed"]   = batch_idx + 1
    run_patches["run_tag"] = f"run_{batch_idx:03d}"
    run_patches.update(config.get("run_card_patches", {}))
    run_patches = {k: v for k, v in run_patches.items() if v is not None}

    print(f"  [CARD] Patching run_card...")
    patch_card(run_card_path, run_patches)

    param_patches = config.get("param_card_patches", {})
    if param_patches:
        print(f"  [CARD] Patching param_card (SLHA)...")
        # param_card is SLHA (`id value # NAME`), NOT the run_card `value = key`
        # format — must use the SLHA patcher or mass/coupling patches silently no-op.
        patch_param_card_slha(param_card_path, param_patches)

# =============================================================================
# STEP 3: Run MadGraph event generation
# =============================================================================

def run_event_generation(events_dir, batch_idx):
    print(f"  [MG5] Generating batch {batch_idx}...")
    mg5_script = f"launch {events_dir}\n0\n"
    result = subprocess.run(
        [MG5_BIN],
        input=mg5_script,
        capture_output=False,
        text=True
    )
    if result.returncode != 0:
        print(f"  [ERROR] MadGraph failed for batch {batch_idx}:")
        print(result.stderr[-1000:])
        return False
    print(f"  [MG5] Batch {batch_idx} done.")
    return True

# =============================================================================
# STEP 4: Backend detection and compilation
# =============================================================================

def get_subprocess_dirs(standalone_dir):
    subproc_base = f"{standalone_dir}/SubProcesses"
    # Absent dir => no backend built yet; report empty so the on-demand path
    # (generate_from_recipe) falls through to generate + compile instead of
    # crashing.
    if not os.path.isdir(subproc_base):
        return []
    return sorted([
        d for d in os.listdir(subproc_base)
        if d.startswith('P') and os.path.isdir(f"{subproc_base}/{d}")
    ])

def detect_backend(subproc_path):
    """Return 'matrix2py' or 'cpp' depending on what MadGraph generated."""
    # matrix2py is available if either the .so already exists or there's a
    # meson build system present (which means --python was used at output time)
    if (os.path.exists(os.path.join(subproc_path, "matrix2py.so")) or
            os.path.exists(os.path.join(subproc_path, "meson.build"))):
        return "matrix2py"
    elif os.path.exists(os.path.join(subproc_path, "CPPProcess.cc")):
        return "cpp"
    return "unknown"

# --- matrix2py compilation ---------------------------------------------------

def compile_matrix2py_subproc(subproc_path, subproc_dir):
    matrix2py_so = os.path.join(subproc_path, "matrix2py.so")
    if os.path.exists(matrix2py_so):
        print(f"  [CXX] {subproc_dir}/matrix2py.so already exists, skipping.")
        return True
    print(f"  [CXX] Compiling matrix2py for {subproc_dir}...")
    # f2py's default distutils backend imports distutils.msvccompiler, which the
    # setuptools-shimmed distutils on py3.11+ has dropped. Force the stdlib
    # distutils so f2py builds (meson backend isn't installed in this env).
    make_env = {**os.environ, "SETUPTOOLS_USE_DISTUTILS": "stdlib"}
    result = subprocess.run(
        ["make", "matrix2py.so"],
        capture_output=True,
        text=True,
        env=make_env,
        cwd=subproc_path
    )
    if result.returncode != 0:
        print(f"  [ERROR] matrix2py compilation failed:\n{result.stderr}")
        return False
    print(f"  [CXX] Done.")
    return True

# --- C++ driver (N-particle, pipe mode) --------------------------------------

def get_class_name(standalone_dir, subproc_dir):
    """Extract the C++ class name from CPPProcess.h."""
    header = f"{standalone_dir}/SubProcesses/{subproc_dir}/CPPProcess.h"
    with open(header, 'r') as f:
        for line in f:
            if line.strip().startswith('class CPPProcess'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    return parts[1].rstrip('{').strip()
    return "CPPProcess"

def write_wrapper(standalone_dir, subproc_dir, param_card_path, nparticles):
    """Write a wrapper .cc for one subprocess, accepting nparticles momenta."""
    suffix     = subproc_dir
    func_name  = f"get_ME2_{suffix}"
    init_name  = f"init_{suffix}"
    obj_name   = f"process_{suffix}"
    init_flag  = f"initialized_{suffix}"
    class_name = get_class_name(standalone_dir, subproc_dir)

    # Build the vector<double*> push_backs from the flat p[] array
    push_backs = "\n    ".join(
        f"p.push_back(&p_flat[{i * 4}]);"
        for i in range(nparticles)
    )

    wrapper = f"""#include <iostream>
#include <vector>
#include "SubProcesses/{subproc_dir}/CPPProcess.h"

static {class_name} {obj_name};
static bool {init_flag} = false;

void {init_name}() {{
    if (!{init_flag}) {{
        std::streambuf* old = std::cout.rdbuf(nullptr);
        {obj_name}.initProc("{param_card_path}");
        std::cout.rdbuf(old);
        {init_flag} = true;
    }}
}}

// p_flat: flat array of {nparticles} * 4 doubles, ordered (E,px,py,pz) per particle
double {func_name}(double* p_flat) {{
    {init_name}();
    std::vector<double*> p;
    {push_backs}
    {obj_name}.setMomenta(p);
    std::streambuf* old = std::cout.rdbuf(nullptr);
    {obj_name}.sigmaKin();
    std::cout.rdbuf(old);
    return {obj_name}.getMatrixElements()[0];
}}
"""
    wrapper_path = f"{standalone_dir}/wrapper_{suffix}.cc"
    with open(wrapper_path, 'w') as f:
        f.write(wrapper)
    return wrapper_path, func_name, suffix

def write_driver(standalone_dir, suffixes, nparticles):
    """Write driver.cpp that reads N-particle momenta lines from stdin (pipe mode)."""
    n_doubles = nparticles * 4

    # Forward declarations
    forward_decls = "\n".join(
        f"double get_ME2_{s}(double*);"
        for s in suffixes
    )

    # Accumulate sum
    sum_lines = "    double total = 0.0;\n"
    for s in suffixes:
        sum_lines += f"    total += get_ME2_{s}(p);\n"

    driver = f"""#include <iostream>
#include <cstdio>

{forward_decls}

int main() {{
    // Disable C stdio sync for maximum throughput in pipe mode
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    double p[{n_doubles}];
    while (true) {{
        for (int i = 0; i < {n_doubles}; ++i) {{
            if (!(std::cin >> p[i])) goto done;
        }}
{sum_lines}
        printf("%.15e\\n", total);
        fflush(stdout);
    }}
done:
    return 0;
}}
"""
    driver_path = f"{standalone_dir}/driver.cpp"
    with open(driver_path, 'w') as f:
        f.write(driver)
    return driver_path

def compile_cpp_driver(standalone_dir, subproc_dirs, nparticles):
    """Compile all C++ wrappers and link the N-particle pipe-mode driver."""
    param_card_path = f"{standalone_dir}/Cards/param_card.dat"
    suffixes = []

    print(f"  [CXX] Writing wrappers and driver ({nparticles} particles)...")
    for subproc_dir in subproc_dirs:
        _, _, suffix = write_wrapper(
            standalone_dir, subproc_dir, param_card_path, nparticles
        )
        suffixes.append(suffix)

    write_driver(standalone_dir, suffixes, nparticles)

    for subproc_dir, suffix in zip(subproc_dirs, suffixes):
        print(f"  [CXX] Compiling libME_{suffix}.so ...")
        cmd = [
            "g++", "-O2", "-fPIC", "-shared",
            "-I.", "-Isrc", f"-ISubProcesses/{subproc_dir}",
            f"wrapper_{suffix}.cc",
            f"SubProcesses/{subproc_dir}/CPPProcess.cc",
            "src/Parameters_sm.cc", "src/HelAmps_sm.cc", "src/read_slha.cc",
            "-o", f"libME_{suffix}.so"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=standalone_dir)
        if result.returncode != 0:
            print(f"  [ERROR] Compilation failed:\n{result.stderr}")
            sys.exit(1)

    print(f"  [CXX] Linking driver...")
    lib_flags = [f"-lME_{s}" for s in suffixes]
    cmd = ["g++", "-O2", "driver.cpp", "-L.", *lib_flags, "-Wl,-rpath,.", "-o", "driver"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=standalone_dir)
    if result.returncode != 0:
        print(f"  [ERROR] Driver linking failed:\n{result.stderr}")
        sys.exit(1)

    print(f"  [CXX] Driver compiled successfully.")
    return f"{standalone_dir}/driver"

def compile_backends(standalone_dir, nparticles):
    """
    Compile a backend, preferring matrix2py (per-event α_s) over the C++ driver
    (fixed α_s). The Fortran standalone lives in fortran_dir_for(standalone_dir);
    the C++ standalone lives in standalone_dir.

    Returns (backend, subproc_dirs, driver_bin_or_None, effective_dir), where
    effective_dir is the standalone dir the chosen backend lives in — pass it
    on to build_dataset* so the param card and SubProcesses resolve correctly.
    """
    fortran_dir = fortran_dir_for(standalone_dir)

    # --- matrix2py (opt-in): per-event α_s natively, but its Fortran setpara()
    # currently STOPs on init in this env, so it is OFF by default. The C++
    # driver + analytic α_s rescaling (below) is exact at LO and is preferred.
    # Set MG5_USE_MATRIX2PY=1 to attempt it once the setpara STOP is fixed. ---
    if os.environ.get("MG5_USE_MATRIX2PY") and os.path.isdir(fortran_dir):
        m2py_dirs = get_subprocess_dirs(fortran_dir)
        print(f"\n[CXX] matrix2py subprocesses: {m2py_dirs}")
        built = bool(m2py_dirs)
        for sd in m2py_dirs:
            if not compile_matrix2py_subproc(f"{fortran_dir}/SubProcesses/{sd}", sd):
                built = False
                break
        if built:
            return "matrix2py", m2py_dirs, None, fortran_dir
        print("  [WARN] matrix2py build failed — falling back to C++ driver.")

    # --- fallback: C++ driver from the standalone_cpp output ---
    cpp_dirs = [sd for sd in get_subprocess_dirs(standalone_dir)
                if detect_backend(f"{standalone_dir}/SubProcesses/{sd}") == "cpp"]
    print(f"[CXX] C++ subprocesses: {cpp_dirs}")
    if cpp_dirs:
        driver_bin = compile_cpp_driver(standalone_dir, cpp_dirs, nparticles)
        return "cpp", cpp_dirs, driver_bin, standalone_dir

    print("[ERROR] No compilable backend found for", standalone_dir)
    sys.exit(1)

def detect_compiled_backend(standalone_dir):
    """--skip_compile counterpart of compile_backends: locate an already-built
    backend without compiling, preferring matrix2py.
    Returns (backend, subproc_dirs, driver_bin_or_None, effective_dir).
    """
    fortran_dir = fortran_dir_for(standalone_dir)
    if os.environ.get("MG5_USE_MATRIX2PY") and os.path.isdir(fortran_dir):
        m2py_dirs = get_subprocess_dirs(fortran_dir)
        if m2py_dirs and all(
            os.path.exists(f"{fortran_dir}/SubProcesses/{sd}/matrix2py.so")
            for sd in m2py_dirs
        ):
            return "matrix2py", m2py_dirs, None, fortran_dir
    cpp_dirs   = get_subprocess_dirs(standalone_dir)
    driver_bin = f"{standalone_dir}/driver" \
        if os.path.exists(f"{standalone_dir}/driver") else None
    return "cpp", cpp_dirs, driver_bin, standalone_dir

# =============================================================================
# STEP 5: Parse LHE files
# =============================================================================

def parse_lhe_file(filepath, initial_state_filter=None):
    """Parse an LHE file and return a list of (momenta_array, pdg_ids_array).

    momenta_array shape: (nparticles, 4) — incoming first, then outgoing.
    """
    quarks    = {1, 2, 3, 4, 5, 6}
    antiquarks = {-1, -2, -3, -4, -5, -6}

    rows = []
    opener = gzip.open if str(filepath).endswith('.gz') else open

    with opener(filepath, 'rt') as f:
        content = f.read()

    for event in content.split('<event>')[1:]:
        event = event.split('</event>')[0].strip()
        lines = [l.strip() for l in event.split('\n')
                 if l.strip() and not l.strip().startswith('#')]

        particles     = []
        pdg_ids       = []
        incoming_pdgs = []

        for line in lines[1:]:
            cols = line.split()
            if len(cols) != 13:
                continue
            pdg_id = int(cols[0])
            status = int(cols[1])
            E  = float(cols[9])
            px = float(cols[6])
            py = float(cols[7])
            pz = float(cols[8])
            if status in (-1, 1):
                particles.append([E, px, py, pz])
                pdg_ids.append(pdg_id)
            if status == -1:
                incoming_pdgs.append(pdg_id)

        if initial_state_filter == 'qqbar':
            pdg_set = set(incoming_pdgs)
            is_qqbar = (any(p in quarks for p in pdg_set) and
                        any(p in antiquarks for p in pdg_set))
            if not is_qqbar:
                continue
        elif initial_state_filter == 'qg':
            pdg_set = set(incoming_pdgs)
            has_quark = any(p in quarks or p in antiquarks for p in pdg_set)
            has_gluon = 21 in pdg_set
            if not (has_quark and has_gluon):
                continue

        rows.append((np.array(particles), np.array(pdg_ids, dtype=int)))

    return rows

# =============================================================================
# STEP 6a: Compute amplitudes with matrix2py
# =============================================================================

def _invert_momenta(p):
    """Transpose (nparticles x 4) → (4 x nparticles) as required by matrix2py."""
    new_p = [[0.0] * len(p) for _ in range(len(p[0]))]
    for i, onep in enumerate(p):
        for j, x in enumerate(onep):
            new_p[j][i] = x
    return new_p

def compute_amplitudes_matrix2py(events, standalone_dir, subproc_dirs,
                                  param_card_path, alphas, nhel=-1):
    """Sum |M|² over all matrix2py subprocesses for a list of events."""
    n_events  = len(events)
    total_me2 = np.zeros(n_events)

    for subproc_dir in subproc_dirs:
        subproc_path = f"{standalone_dir}/SubProcesses/{subproc_dir}"

        if subproc_path in sys.path:
            sys.path.remove(subproc_path)
        sys.path.insert(0, subproc_path)
        if 'matrix2py' in sys.modules:
            del sys.modules['matrix2py']
        import matrix2py

        matrix2py.py_initialisemodel(param_card_path)

        me2 = np.empty(n_events)
        for j, (momenta, _) in enumerate(events):
            me2[j] = matrix2py.py_get_value(_invert_momenta(momenta), alphas, nhel)

        total_me2 += me2
        print(f"    [AMP] Subprocess {subproc_dir}: mean |M|² = {me2.mean():.4e}")

        sys.path.remove(subproc_path)
        del sys.modules['matrix2py']

    return total_me2

# =============================================================================
# STEP 6b: Compute amplitudes with C++ driver (pipe mode)
# =============================================================================

class CppDriverPipe:
    """
    Wraps the C++ driver in a persistent subprocess (pipe mode).
    The driver is started once and kept alive; momenta are written to its
    stdin line by line and amplitudes are read back from stdout.
    This avoids the per-batch process-startup overhead.
    """

    def __init__(self, driver_bin, standalone_dir):
        self.standalone_dir = standalone_dir
        env = {**os.environ, "LD_LIBRARY_PATH": standalone_dir}
        self._proc = subprocess.Popen(
            [driver_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env
        )
        print(f"  [PIPE] C++ driver started (pid {self._proc.pid})")

    def compute(self, events):
        """
        Send momenta for a list of events and return array of |M|² values.
        events: list of (momenta_array, pdg_ids) tuples.
                momenta_array shape: (nparticles, 4).
        """
        amps = []
        stdin  = self._proc.stdin
        stdout = self._proc.stdout

        for momenta, _ in events:
            # Flatten: E0 px0 py0 pz0  E1 px1 py1 pz1  ...
            line = " ".join(f"{x:.15e}" for x in momenta.flatten())
            stdin.write(line + "\n")
            stdin.flush()
            amps.append(float(stdout.readline()))

        return np.array(amps)

    def close(self):
        if self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait()
            print(f"  [PIPE] C++ driver closed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

# =============================================================================
# STEP 7: Build dataset
# =============================================================================

def _count_lhe_events(lhe_file, initial_state_filter):
    """Count how many events pass the filter in one LHE file without storing them."""
    return len(parse_lhe_file(lhe_file, initial_state_filter=initial_state_filter))


def build_dataset(events_dir, standalone_dir, backend, subproc_dirs,
                  driver_bin, config, output_file, existing_data=None):
    """
    Parse all LHE files, compute amplitudes, and save as .npy.

    Uses a memory-mapped output file so only one LHE file's worth of data
    is ever in RAM at once.  This allows arbitrarily large datasets to be
    built on memory-constrained machines.

    Output layout per row:  [momenta (nparticles*4) | pdg_ids (nparticles) | amplitude (1)]
    """
    param_card_path = f"{standalone_dir}/Cards/param_card.dat"
    alphas          = config["alphas"]
    nfinal          = config["nfinal"]
    nparticles      = nfinal + 2
    ncols           = nparticles * 5 + 1   # 4-mom + pdg + amp
    is_filter       = config.get("initial_state_filter")

    lhe_files = sorted(Path(events_dir).rglob("unweighted_events.lhe*"))

    # In extend mode, figure out how many files were already processed by
    # counting events in the first file and comparing to existing row count.
    n_existing = 0
    if existing_data is not None:
        n_existing = len(existing_data)
        # count by scanning the LHE file without building arrays
        nevents_per_file = sum(
            1 for line in (
                gzip.open(lhe_files[0], 'rt') if str(lhe_files[0]).endswith('.gz')
                else open(lhe_files[0], 'r')
            )
            if '<event>' in line
        )
        n_skip    = n_existing // nevents_per_file
        lhe_files = lhe_files[n_skip:]
        print(f"[DATA] Skipping {n_skip} already-processed LHE files ({n_existing} existing events)")

    print(f"\n[DATA] Found {len(lhe_files)} LHE files to process")
    print(f"[DATA] Backend: {backend}")

    if not lhe_files:
        print("[DATA] Nothing to do.")
        return existing_data

    # --- count total new events (one quick pass through each file) ---
    print("[DATA] Counting events in LHE files...")
    n_new_per_file = [_count_lhe_events(f, is_filter) for f in lhe_files]
    n_new_total    = sum(n_new_per_file)
    n_total        = n_existing + n_new_total
    print(f"[DATA] New events: {n_new_total}  |  Total after merge: {n_total}")

    # --- allocate memory-mapped output array ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    mmap_path = output_file + ".mmap"
    mmap = np.lib.format.open_memmap(
        mmap_path, mode="w+", dtype=np.float64, shape=(n_total, ncols)
    )

    # Copy existing data into the mmap if extending
    if existing_data is not None:
        mmap[:n_existing] = existing_data

    # --- process LHE files one at a time, writing directly into mmap ---
    write_cursor = n_existing

    pipe_ctx = CppDriverPipe(driver_bin, standalone_dir) if backend == "cpp" else None
    try:
        for lhe_file, n_expected in zip(lhe_files, n_new_per_file):
            print(f"  [DATA] Parsing {lhe_file.parent.name}...")
            parsed = parse_lhe_file(lhe_file, initial_state_filter=is_filter)
            n      = len(parsed)
            print(f"  [AMP]  Computing amplitudes for {n} events...")

            if backend == "matrix2py":
                amps = compute_amplitudes_matrix2py(
                    parsed, standalone_dir, subproc_dirs, param_card_path, alphas
                )
            else:
                amps = pipe_ctx.compute(parsed)

            # build row block for this file — only this file's data in RAM
            momenta  = np.array([e[0].flatten() for e in parsed], dtype=np.float64)
            pdg_ids  = np.array([e[1]           for e in parsed], dtype=np.float64)
            amps_col = np.asarray(amps, dtype=np.float64).reshape(-1, 1)
            block    = np.concatenate([momenta, pdg_ids, amps_col], axis=1)

            mmap[write_cursor : write_cursor + n] = block
            mmap.flush()
            write_cursor += n
            print(f"  [DATA] Done. Total so far: {write_cursor} events")

            # free this file's data immediately
            del parsed, momenta, pdg_ids, amps_col, block, amps

    finally:
        if pipe_ctx is not None:
            pipe_ctx.close()

    # --- convert mmap → final .npy (single rename, no extra copy) ---
    del mmap
    if os.path.exists(output_file):
        os.remove(output_file)
    os.rename(mmap_path, output_file)
    print(f"[DATA] Saved {write_cursor} events to {output_file}")

    # return a read-only mmap view so callers don't re-load the whole file
    return np.load(output_file, mmap_mode="r")

# =============================================================================
# VARIABLE-ENERGY MODE: direct phase-space sampling (bypasses MadGraph LHE gen)
# =============================================================================

# Fiducial cuts — RAMBO samples the FULL phase space (MadGraph's run-card cuts are
# bypassed), so soft/collinear/forward/low-mass-pair regions give an IR tail spanning
# ~10+ log units. These cuts define the fiducial region kept in the dataset. They
# apply ONLY to massless, VISIBLE final-state particles (m_final==0 and not a
# neutrino) — the IR-singular ones; massive finals (W/Z/t/H) and neutrinos are exempt.
# Toggle off with AMP_FIDUCIAL_CUTS=off (reproduces pre-cut data; cut params enter
# recipe_id so cut/no-cut datasets never share a cache entry).
FIDUCIAL_CUTS = {"pt_min": 10.0, "cos_max": 0.9, "dr_min": 0.4, "m_min": 10.0}
FIDUCIAL_CUTS_ENABLED = os.environ.get("AMP_FIDUCIAL_CUTS", "on").lower() != "off"
_NEUTRINOS = frozenset((12, 14, 16))


def _cut_key(cuts):
    """Identity-bearing view of the active cuts (for recipe_id)."""
    return {k: float(cuts[k]) for k in ("pt_min", "cos_max", "dr_min", "m_min")}


def fiducial_pass_mask(P, m_finals, pdg_ids, cuts, n_initial=2):
    """Boolean mask (N,) of events passing the fiducial cuts. P is (N, npart, 4).
    Cuts hit only massless visible finals (m_final==0, |pdg|∉{12,14,16}): single-
    particle pt>pt_min & |cosθ|<cos_max; pairwise (among those) ΔR>dr_min & m>m_min.
    A process with no such particle (all-massive / all-neutrino finals) passes all."""
    import itertools
    N = P.shape[0]
    m_finals = np.asarray(m_finals, float)
    fin = list(range(n_initial, n_initial + len(m_finals)))
    slots = [fin[i] for i in range(len(m_finals))
             if m_finals[i] == 0.0 and abs(int(pdg_ids[fin[i]])) not in _NEUTRINOS]
    keep = np.ones(N, dtype=bool)
    if not slots:
        return keep
    pt   = lambda p: np.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)
    cost = lambda p: np.abs(p[..., 3]) / np.clip(np.linalg.norm(p[..., 1:], axis=-1), 1e-12, None)
    eta  = lambda p: np.arcsinh(p[..., 3] / np.clip(pt(p), 1e-9, None))
    phi  = lambda p: np.arctan2(p[..., 2], p[..., 1])
    for s in slots:
        keep &= (pt(P[:, s]) > cuts["pt_min"]) & (cost(P[:, s]) < cuts["cos_max"])
    for i, j in itertools.combinations(slots, 2):
        pi, pj = P[:, i], P[:, j]
        dphi = np.abs(phi(pi) - phi(pj)); dphi = np.minimum(dphi, 2 * np.pi - dphi)
        dr   = np.sqrt((eta(pi) - eta(pj)) ** 2 + dphi ** 2)
        q    = pi + pj
        m    = np.sqrt(np.clip(q[..., 0] ** 2 - (q[..., 1:] ** 2).sum(-1), 0.0, None))
        keep &= (dr > cuts["dr_min"]) & (m > cuts["m_min"])
    return keep


def _collect_with_cuts(draw, n_events, m_finals, pdg_ids, cuts):
    """Oversample-and-reject: call draw(n)->(P (n,npart,4), sqrts (n,)) repeatedly,
    keep events passing fiducial_pass_mask, until n_events accumulate. Returns
    (P (n_events,npart,4), sqrts (n_events,)). cuts=None → single unfiltered draw.
    Deterministic given the draw's rng (adaptive batch size depends only on the
    running acceptance estimate, itself a function of the drawn — seeded — events)."""
    if cuts is None:
        return draw(n_events)
    accP, accS, got, acc = [], [], 0, 0.5
    while got < n_events:
        need  = n_events - got
        batch = int(min(max(need / max(acc, 0.03) * 1.3, need), need * 50)) + 32
        P, sq = draw(batch)
        m = fiducial_pass_mask(P, m_finals, pdg_ids, cuts)
        if m.any():
            accP.append(P[m]); accS.append(sq[m]); got += int(m.sum())
        acc = 0.85 * acc + 0.15 * (float(m.mean()) if m.size else acc)
    return np.concatenate(accP)[:n_events], np.concatenate(accS)[:n_events]


def sample_2to2_phase_space(n, sqrts_min, sqrts_max, m_final, pdg_ids, rng=None, cuts=None):
    """
    Sample uniform phase space for a 2→2 partonic process with a massless,
    back-to-back initial state (e+e-, gg, qq~, …).

    `m_final` gives the final-state mass(es): a scalar for an equal-mass pair
    (X X~), or a (m3, m4) pair for unequal masses (e.g. e+ e- > z h). The CM
    energies split by the exact two-body formula
        E3 = (s + m3² − m4²) / (2√s),   E4 = √s − E3,
        |p| = √λ(s, m3², m4²) / (2√s),
    which reduces to E3 = E4 = √s/2, |p| = √(E²−m²) when m3 = m4 — so the
    scalar path is unchanged.

    √s ~ U[sqrts_min, sqrts_max]; cos θ, φ uniform over the sphere. Returns a
    list of (momenta_array, pdg_ids_array) plus the per-event √s array.

    Column ordering matches the .npy convention and the NLO .dat files:
        p1 = beam-  (E_beam, 0, 0, +E_beam)
        p2 = beam+  (E_beam, 0, 0, -E_beam)
        p3 = X      (E3, px, py, pz)
        p4 = Xbar   (E4, -px, -py, -pz)
    """
    if rng is None:
        rng = np.random.default_rng()

    if np.isscalar(m_final):
        m3 = m4 = float(m_final)
    else:
        m3, m4 = float(m_final[0]), float(m_final[1])

    pdg_arr = np.array(pdg_ids, dtype=int)
    m_finals = [m3, m4]

    def draw(nb):
        sqrts  = rng.uniform(sqrts_min, sqrts_max, nb)
        cos_t  = rng.uniform(-1.0, 1.0, nb)
        phi    = rng.uniform(0.0, 2.0 * np.pi, nb)
        s      = sqrts ** 2
        E_beam = sqrts / 2.0
        E3     = (s + m3 ** 2 - m4 ** 2) / (2.0 * sqrts)
        E4     = sqrts - E3
        p_mag  = np.sqrt(np.maximum(E3 ** 2 - m3 ** 2, 0.0))
        sin_t  = np.sqrt(1.0 - cos_t ** 2)
        px = p_mag * sin_t * np.cos(phi)
        py = p_mag * sin_t * np.sin(phi)
        pz = p_mag * cos_t
        zero = np.zeros(nb)
        P = np.stack([
            np.stack([E_beam, zero,  zero,  E_beam], axis=1),   # beam-
            np.stack([E_beam, zero,  zero, -E_beam], axis=1),   # beam+
            np.stack([E3,     px,    py,    pz],     axis=1),    # X
            np.stack([E4,    -px,   -py,   -pz],     axis=1),    # Xbar
        ], axis=1)                                              # (nb, 4, 4)
        return P, sqrts

    P, sqrts = _collect_with_cuts(draw, n, m_finals, pdg_arr, cuts)
    events = [(P[i], pdg_arr) for i in range(n)]
    return events, sqrts


# =============================================================================
# RAMBO n-body phase-space sampler
# =============================================================================

def _rambo_massless_batch(sqrts_arr, n_final, rng):
    """
    Vectorized RAMBO for a batch of N events, each with its own √s.
    Reference: Kleiss & Stirling, Comput. Phys. Commun. 40 (1986) 359.

    sqrts_arr : (N,) CM energies
    n_final   : number of final-state particles (massless)
    Returns   : (N, n_final, 4) array of 4-momenta (E, px, py, pz)
    """
    N = len(sqrts_arr)
    u     = rng.random((N, n_final, 4))
    cos_t = 2.0 * u[..., 0] - 1.0
    sin_t = np.sqrt(1.0 - cos_t**2)
    phi   = 2.0 * np.pi * u[..., 1]
    E     = -np.log(u[..., 2] * u[..., 3])   # (N, n_final)

    q = np.empty((N, n_final, 4))
    q[..., 0] = E
    q[..., 1] = E * sin_t * np.cos(phi)
    q[..., 2] = E * sin_t * np.sin(phi)
    q[..., 3] = E * cos_t

    Q  = q.sum(axis=1)                                              # (N, 4)
    M  = np.sqrt(np.maximum(Q[:,0]**2 - np.sum(Q[:,1:]**2,axis=1), 1e-30))  # (N,)
    b  = -Q[:, 1:] / M[:, None]                                    # (N, 3)
    gm = Q[:, 0] / M                                               # (N,)
    a  = 1.0 / (1.0 + gm)                                         # (N,)
    x  = sqrts_arr / M                                             # (N,)

    bq = np.einsum('ni,nki->nk', b, q[:, :, 1:])                  # (N, n_final)

    p = np.empty((N, n_final, 4))
    p[..., 0]  = x[:, None] * (gm[:, None] * q[..., 0] + bq)
    p[..., 1:] = x[:, None, None] * (
        q[..., 1:]
        + b[:, None, :] * (q[..., 0:1] + a[:, None, None] * bq[..., None])
    )
    return p


def _rambo_massive_batch(sqrts_arr, masses, p_massless):
    """
    Vectorized massive-RAMBO upgrade.  Finds the isotropic rescaling ξ[N]
    such that Σ_i sqrt(mi² + (ξ|pi|)²) = √s for each event independently.
    Newton-Raphson converges in ≤10 steps for physical √s (above threshold).

    sqrts_arr  : (N,) CM energies
    masses     : (n_final,) particle masses
    p_massless : (N, n_final, 4) massless RAMBO momenta
    Returns    : (N, n_final, 4)
    """
    masses = np.asarray(masses, dtype=float)[None, :]               # (1, n_final)
    pmag   = np.sqrt(np.sum(p_massless[:, :, 1:]**2, axis=2))      # (N, n_final)
    xi     = np.ones(len(sqrts_arr))                                # (N,)

    for _ in range(50):
        Ei    = np.sqrt(masses**2 + (xi[:, None] * pmag)**2)        # (N, n_final)
        f     = Ei.sum(axis=1) - sqrts_arr                          # (N,)
        df    = np.sum(xi[:, None] * pmag**2 / Ei, axis=1)         # (N,)
        delta = f / df
        xi   -= delta
        if np.max(np.abs(delta)) < 1e-10 * np.max(np.abs(xi) + 1e-30):
            break

    Ei     = np.sqrt(masses**2 + (xi[:, None] * pmag)**2)
    p      = np.empty_like(p_massless)
    p[:, :, 0]  = Ei
    p[:, :, 1:] = xi[:, None, None] * p_massless[:, :, 1:]
    return p


def sample_nbody_phase_space(n_events, sqrts_min, sqrts_max, m_finals, pdg_ids, rng=None, cuts=None):
    """
    Sample n-body phase space using RAMBO for a range of CM energies.

    Uses fully vectorized batch RAMBO; no Python loop over events.
    The initial-state convention (e- row 0, e+ row 1) matches CPPProcess.
    With ``cuts`` set, oversamples and keeps only events in the fiducial region.

    Returns (events, sqrts_arr) where events is a list of
    (momenta_array (nparticles x 4), pdg_ids_array).
    """
    if rng is None:
        rng = np.random.default_rng()

    masses   = np.asarray(m_finals, dtype=float)
    nfinal   = len(masses)
    all_zero = np.all(masses == 0.0)
    pdg_arr  = np.array(pdg_ids, dtype=int)

    def draw(nb):
        sqrts   = rng.uniform(sqrts_min, sqrts_max, nb)
        E_beam  = sqrts / 2.0
        p_ml    = _rambo_massless_batch(sqrts, nfinal, rng)          # (nb, nfinal, 4)
        p_final = p_ml if all_zero else _rambo_massive_batch(sqrts, masses, p_ml)
        zero    = np.zeros(nb)
        p_init  = np.stack([
            np.stack([E_beam, zero, zero,  E_beam], axis=1),
            np.stack([E_beam, zero, zero, -E_beam], axis=1),
        ], axis=1)                                                  # (nb, 2, 4)
        return np.concatenate([p_init, p_final], axis=1), sqrts     # (nb, nparticles, 4)

    P, sqrts = _collect_with_cuts(draw, n_events, m_finals, pdg_arr, cuts)
    events = [(P[i], pdg_arr) for i in range(n_events)]
    return events, sqrts


def build_dataset_variable_energy(n_events, sqrts_min, sqrts_max,
                                   standalone_dir, backend, subproc_dirs,
                                   driver_bin, config, output_file, rng=None):
    """
    Build a variable-√s LO amplitude dataset by direct phase-space sampling.
    Bypasses MadGraph event generation entirely.

    2→2 processes: analytic uniform phase space (sample_2to2_phase_space).
    n-body (nfinal>2):  RAMBO (sample_nbody_phase_space).

    α_s is evaluated per event at μ = √s for matrix2py backend.
    C++ pipe driver uses fixed α_s at the geometric-mean scale.

    `rng` is a seeded np.random.Generator. The only randomness in this path is
    the phase-space sampling, so a fixed rng makes the dataset reproducible
    (given identical n_events, drawn in one shot, and matching backend/param
    cards). If None, a fresh unseeded generator is used (non-reproducible).
    """
    nfinal     = config["nfinal"]
    nparticles = nfinal + 2
    ncols      = nparticles * 5 + 1
    pdg_ids    = config["pdg_ids"]
    param_card = f"{standalone_dir}/Cards/param_card.dat"
    # Reference α_s(M_Z) for the per-event running μ=√s. Per-dataset scans set this
    # (register_scan_process); default 0.118 keeps existing datasets bit-identical.
    amz        = float(config.get("alphas_mz", 0.118))
    cuts       = FIDUCIAL_CUTS if FIDUCIAL_CUTS_ENABLED else None
    cut_msg    = f"  fiducial cuts {_cut_key(cuts)}" if cuts else "  (no cuts)"

    if nfinal == 2:
        # Prefer an explicit (m3, m4) pair for unequal-mass 2→2 (e.g. e+ e- > z h);
        # fall back to the scalar m_final for an equal-mass pair (X X~).
        m_final = config["m_finals"] if "m_finals" in config else config["m_final"]
        print(f"\n[DATA] Sampling {n_events:,} events  √s ∈ [{sqrts_min}, {sqrts_max}] GeV{cut_msg}")
        print(f"[DATA] m_final = {m_final} GeV   backend = {backend}")
        events, sqrts_arr = sample_2to2_phase_space(
            n_events, sqrts_min, sqrts_max, m_final, pdg_ids, rng=rng, cuts=cuts
        )
    else:
        m_finals = config["m_finals"]
        print(f"\n[DATA] Sampling {n_events:,} events  √s ∈ [{sqrts_min}, {sqrts_max}] GeV{cut_msg}")
        print(f"[DATA] nfinal = {nfinal}  m_finals = {m_finals}  backend = {backend}")
        events, sqrts_arr = sample_nbody_phase_space(
            n_events, sqrts_min, sqrts_max, m_finals, pdg_ids, rng=rng, cuts=cuts
        )

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    mmap_path = output_file + ".mmap"
    mmap = np.lib.format.open_memmap(
        mmap_path, mode="w+", dtype=np.float64, shape=(n_events, ncols)
    )

    # ----------------------------------------------------------------
    # Compute amplitudes
    # ----------------------------------------------------------------
    total_me2 = np.zeros(n_events)

    if backend == "matrix2py":
        for subproc_dir in subproc_dirs:
            subproc_path = f"{standalone_dir}/SubProcesses/{subproc_dir}"
            if subproc_path in sys.path:
                sys.path.remove(subproc_path)
            sys.path.insert(0, subproc_path)
            if "matrix2py" in sys.modules:
                del sys.modules["matrix2py"]
            import matrix2py
            matrix2py.py_initialisemodel(param_card)

            # Fortran matrix2py expects e+ in row 0, e- in row 1; our phase-space
            # convention stores e- in row 0, e+ in row 1. Swap the two incoming
            # rows and leave all final-state rows in place (works for any nfinal).
            swap_idx = [1, 0] + list(range(2, nparticles))
            me2 = np.empty(n_events)
            for i, (momenta, _) in enumerate(events):
                alphas = compute_alphas(sqrts_arr[i], alphas_mz=amz)
                p_mg = momenta[swap_idx]
                me2[i] = matrix2py.py_get_value(_invert_momenta(p_mg), alphas, -1)
                if (i + 1) % 100_000 == 0:
                    print(f"  [AMP] {i+1:,}/{n_events:,}", flush=True)

            total_me2 += me2
            print(f"    [AMP] subprocess {subproc_dir}: mean |M|² = {me2.mean():.4e}")
            sys.path.remove(subproc_path)
            del sys.modules["matrix2py"]

    elif backend == "cpp":
        # C++ pipe driver runs at the standalone's fixed α_s (param-card value).
        # At LO |M|² = K(kinematics)·α_sᵏ exactly (k = alphas_power), so we
        # recover the per-event α_s(√s) result by an exact analytic rescale:
        #     |M|²(√sᵢ) = |M|²_driver · (α_s(√sᵢ) / α_s_ref)ᵏ
        # (k=0 ⇒ factor 1, i.e. EW processes are untouched.)
        k = config.get("alphas_power", 0)
        with CppDriverPipe(driver_bin, standalone_dir) as pipe:
            total_me2 = pipe.compute(events)
        if k:
            alphas_ref = read_alphas_from_param_card(param_card)
            scale = (compute_alphas(sqrts_arr, alphas_mz=amz) / alphas_ref) ** k
            print(f"  [AMP] per-event α_s rescale: k={k}, α_s_ref={alphas_ref:.4f}, "
                  f"α_s(√s)∈[{compute_alphas(sqrts_max, alphas_mz=amz):.4f},"
                  f"{compute_alphas(sqrts_min, alphas_mz=amz):.4f}]")
            total_me2 = np.asarray(total_me2, dtype=np.float64) * scale

    # ----------------------------------------------------------------
    # Assemble and save
    # ----------------------------------------------------------------
    momenta_flat = np.array([e[0].flatten() for e in events], dtype=np.float64)
    pdg_block    = np.tile(np.array(pdg_ids, dtype=np.float64), (n_events, 1))
    amp_col      = total_me2.reshape(-1, 1)
    mmap[:] = np.concatenate([momenta_flat, pdg_block, amp_col], axis=1)
    mmap.flush()
    del mmap

    if os.path.exists(output_file):
        os.remove(output_file)
    os.rename(mmap_path, output_file)

    w = total_me2
    print(f"[DATA] Saved {n_events:,} events → {output_file}")
    print(f"  shape: ({n_events}, {ncols})  "
          f"|M|² ∈ [{w.min():.4e}, {w.max():.4e}]  neg: {(w < 0).sum()}")


# =============================================================================
# LIBRARY API: recipe -> dataset (importable by the training job)
# =============================================================================

# ── Cost-aware chunk policy ───────────────────────────────────────────────────
# Per-process generation cost drives how a dataset is sliced into work units for
# parallel generation. Lives here (next to PROCESSES + the recipe builders) because
# the resolved chunk_size is part of the recipe IDENTITY: the per-chunk RNG stream
# depends on the chunking, so two policies produce different bytes and must get
# different recipe_ids. See datagen.py for the full rationale.
MAX_CHUNK = 100_000          # cheap 2→2 chunk size (unchanged from the old policy)
MIN_CHUNK = 2_500            # floor so expensive processes don't over-fragment
TARGET_CHUNK_COST = 100_000  # one chunk's work in 2→2-equivalent events

def process_gen_weight(process):
    """Relative per-event generation cost of `process`, normalized to 2→2 LO = 1.
    Used only to size chunks/schedule load — never enters the physics. An explicit
    ``gen_weight`` in the PROCESSES entry wins; else a heuristic from final-state
    multiplicity (dominant cost driver) and loop order."""
    cfg = PROCESSES.get(process, {})
    explicit = cfg.get("gen_weight")
    if explicit is not None:
        return float(explicit)
    nfinal = int(cfg.get("nfinal", 2))
    w = 6.0 ** max(0, nfinal - 2)        # 2→2:1, 2→3:6, 2→4:36, 2→5:216
    if cfg.get("loop") or cfg.get("nlo") or cfg.get("virt"):
        w *= 8.0
    return w

def process_chunk_size(process):
    """Deterministic per-process events-per-chunk, sized for ≈ equal wall-time.
    Cheap → MAX_CHUNK (unchanged bytes); expensive → fewer events per chunk."""
    size = TARGET_CHUNK_COST / process_gen_weight(process)
    return int(min(MAX_CHUNK, max(MIN_CHUNK, round(size))))


def process_n_chunks(process, n_events):
    """Number of generation chunks for a dataset — the byte-determining split.

    NLO-virtual processes are generated as ONE unit (datagen runs them in an
    isolated subprocess: MadLoop's matrix2py.so is a process-singleton, and many
    concurrent fresh-interpreter per-chunk subprocesses race on numpy's C-extension
    import on Lustre). NLO is cheap per point (~0.2 ms), so a whole dataset in one
    process is fast enough and needs no intra-process parallelism. LO uses the
    cost-aware size."""
    if PROCESSES.get(process, {}).get("kind") == "virt":
        return 1
    return -(-int(n_events) // process_chunk_size(process))


def variable_energy_recipe(process, sqrts_min, sqrts_max, n_events,
                           role=None, seed=None):
    """Canonical content-determining recipe dict for a variable-energy dataset.
    recipe_id() of this uniquely identifies the dataset's contents — including the
    cost-aware chunking, since the per-chunk RNG streams (and therefore the bytes)
    depend on how the dataset is split.

    The byte-determining quantity is the REALIZED number of chunks, n_chunks =
    ceil(n_events / chunk_size) — not the nominal chunk_size. A dataset small
    enough to be one chunk under both the cost-aware and the legacy 100k policy is
    byte-identical regardless of the policy. So `n_chunks` is recorded in the
    identity ONLY when it differs from the legacy ceil(n_events / MAX_CHUNK): then
    cost-aware data that actually re-chunks gets a fresh id, while everything whose
    chunking is unchanged keeps its original id and existing cache. (The nominal
    chunk_size is kept as excluded traceability metadata, written in finalize.)

    SCOPE: this byte-strict chunk identity applies to TRAIN only, where recipe_id
    is the $SCRATCH cache key and we want exact-bytes reuse within a sweep. For the
    FROZEN val/test benchmark, chunking is a pure generation-parallelism detail:
    any valid sample of the physical spec is an equally good benchmark, so we do
    NOT want a different chunk policy to mint a second, physically-identical frozen
    dataset. Hence `n_chunks` is left out of the val/test identity entirely — one
    test set serves regardless of how it was sliced for generation."""
    cfg = PROCESSES[process]
    k   = cfg.get("alphas_power", 0)
    n   = int(n_events)
    recipe = {
        "process":            process,
        "mode":               "variable_energy",
        "sqrts_min":          float(sqrts_min),
        "sqrts_max":          float(sqrts_max),
        "n_events":           n,
        "role":               role,
        "seed":               seed,
        "pdg_ids":            list(cfg["pdg_ids"]),
        "param_card_patches": cfg.get("param_card_patches", {}),
        "alphas_power":       k,
        "amp_orders":         [int(cfg.get("n_loops", 0)), k],
        "per_event_alphas":   True,
    }
    # Per-dataset physics scan (register_scan_process): the reference α_s(M_Z)
    # drives the per-event running but is NOT in param_card_patches, so it must
    # enter the identity explicitly. Only recorded when non-default so existing
    # datasets keep their recipe_id (and cache). m_finals records the phase-space
    # masses (also identity-bearing when a mass is scanned).
    if abs(float(cfg.get("alphas_mz", 0.118)) - 0.118) > 1e-12:
        recipe["alphas_mz"] = float(cfg["alphas_mz"])
    if cfg.get("alphas_prefactor"):
        recipe["alphas_prefactor"] = True   # NLO target carries the physical α_s weight
    if "scan_base" in cfg:
        recipe["m_finals"] = [float(m) for m in cfg.get("m_finals", [])]
    # Fiducial cuts (when active) change the sampled events → part of the identity, so
    # cut and pre-cut (no-cut) datasets never collide in the cache. Omitted when off,
    # so existing pre-cut datasets keep their recipe_id.
    if FIDUCIAL_CUTS_ENABLED:
        recipe["fiducial_cuts"] = _cut_key(FIDUCIAL_CUTS)
    ceil_div    = lambda a, b: -(-a // b)
    n_chunks    = process_n_chunks(process, n)
    legacy_nch  = ceil_div(n, MAX_CHUNK)
    # Chunk identity is byte-strict for TRAIN only; frozen val/test stay
    # chunk-policy-independent (one physical sample per spec — see docstring).
    if role not in ("val", "test") and n_chunks != legacy_nch:
        recipe["n_chunks"] = n_chunks          # identity-bearing; legacy plan is implicit
    return recipe

def recipe_output_path(recipe, out_dir):
    role_tag = f"_{recipe['role']}" if recipe.get("role") else ""
    return (f"{out_dir}/{recipe['process']}"
            f"_{recipe['sqrts_min']:.0f}-{recipe['sqrts_max']:.0f}GeV"
            f"{role_tag}_amplitudes.npy")

def generate_from_recipe(recipe, out_dir=None, reuse=True, compile_if_needed=True):
    """Materialize the variable-energy dataset described by `recipe` into
    `out_dir` and return the .npy path.

    If `reuse` and a dataset with a matching recipe_id already sits at the
    target path, regeneration is skipped (cache hit) — this is what lets a
    training job (or every trial of a sweep) ask for the same train set and
    pay for generation at most once. Compiled backends are reused; a missing
    backend is built on demand unless `compile_if_needed` is False.
    """
    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    process     = recipe["process"]
    cfg         = dict(PROCESSES[process])
    output_file = recipe_output_path(recipe, out_dir)
    wanted_id   = recipe_id(recipe)

    if reuse and os.path.exists(output_file):
        try:
            with open(output_file + ".recipe.json") as f:
                if json.load(f).get("recipe_id") == wanted_id:
                    print(f"[CACHE] reuse {output_file} (recipe_id={wanted_id})")
                    return output_file
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    seed, role = recipe.get("seed"), recipe.get("role")
    if seed is None:
        print("[WARN] recipe has no seed: dataset will NOT be reproducible.")
        eff_seed, rng = None, np.random.default_rng()
    else:
        eff_seed = seed + ROLE_SEED_OFFSET.get(role, 0)
        rng = np.random.default_rng(eff_seed)

    nparticles     = cfg["nfinal"] + 2
    standalone_dir = f"{WORK_DIR}/{process}_standalone"
    backend, subproc_dirs, driver_bin, eff_dir = detect_compiled_backend(standalone_dir)
    if not subproc_dirs or (backend == "cpp" and driver_bin is None):
        if not compile_if_needed:
            raise RuntimeError(
                f"No compiled backend for {process} under {standalone_dir}. "
                f"Build it once with mg5_pipeline_final.py before on-demand runs.")
        generate_mg5_process(process, cfg)
        backend, subproc_dirs, driver_bin, eff_dir = compile_backends(
            standalone_dir, nparticles)

    build_dataset_variable_energy(
        recipe["n_events"], recipe["sqrts_min"], recipe["sqrts_max"],
        eff_dir, backend, subproc_dirs, driver_bin, cfg, output_file, rng=rng)

    full = dict(recipe)
    full["effective_seed"] = eff_seed
    full["backend"]        = backend
    write_recipe(output_file, full)
    return output_file


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MadGraph5 amplitude dataset pipeline")
    parser.add_argument("--process",      required=True, choices=list(PROCESSES.keys()),
                        help="Process to generate")
    parser.add_argument("--com_energy",   type=float, default=None,
                        help="Centre-of-mass energy in GeV (required unless --sqrts_min/max are set)")
    parser.add_argument("--nevents",      type=int, default=100000,
                        help="Events per batch")
    parser.add_argument("--nbatches",     type=int, default=10,
                        help="Number of batches to generate")
    parser.add_argument("--reset",        action="store_true",
                        help="Delete existing runs and regenerate from scratch")
    parser.add_argument("--extend",       action="store_true",
                        help="Add more batches to existing dataset (appends)")
    parser.add_argument("--ren_scale",    type=float, default=None,
                        help="Renormalization scale in GeV for running alphas. "
                             "Defaults to com_energy. Common alternatives: "
                             "com_energy/2, leading jet pT.")
    parser.add_argument("--skip_gen",     action="store_true",
                        help="Skip event generation (use existing LHE files)")
    parser.add_argument("--skip_compile", action="store_true",
                        help="Skip compilation (use existing driver / matrix2py)")
    parser.add_argument("--sqrts_min",    type=float, default=None,
                        help="Variable-energy mode: lower bound of uniform √s [GeV]. "
                             "Bypasses MadGraph event generation; phase space is sampled directly.")
    parser.add_argument("--sqrts_max",    type=float, default=None,
                        help="Variable-energy mode: upper bound of uniform √s [GeV].")
    parser.add_argument("--nevents_total", type=int, default=None,
                        help="Total events for variable-energy mode "
                             "(defaults to nevents when --sqrts_min/max are set).")
    parser.add_argument("--seed",          type=int, default=None,
                        help="Base RNG seed for variable-energy phase-space "
                             "sampling. Combined with --role offset to make the "
                             "dataset reproducible. If unset, generation is "
                             "non-reproducible (a warning is printed).")
    parser.add_argument("--role",          choices=["train", "val", "test"],
                        default=None,
                        help="Split role. Adds a fixed per-role offset to "
                             "--seed so train/val/test draw disjoint event "
                             "streams, and tags the output filename + recipe.")
    return parser.parse_args()

def main():
    args   = parse_args()
    config = dict(PROCESSES[args.process])

    variable_energy = args.sqrts_min is not None or args.sqrts_max is not None

    if variable_energy:
        if args.sqrts_min is None or args.sqrts_max is None:
            print("[ERROR] Both --sqrts_min and --sqrts_max must be provided together.")
            sys.exit(1)
        if args.reset or args.extend:
            print("[ERROR] --reset and --extend are not supported in variable-energy mode.")
            sys.exit(1)
        nfinal_check = config["nfinal"]
        # 2→2 accepts either a scalar m_final (equal-mass pair) or m_finals (m3, m4).
        has_mass = (("m_final" in config or "m_finals" in config)
                    if nfinal_check == 2 else "m_finals" in config)
        if "pdg_ids" not in config or not has_mass:
            key = "m_final" if nfinal_check == 2 else "m_finals"
            print(f"[ERROR] Process '{args.process}' needs 'pdg_ids' and '{key}' "
                  "for variable-energy mode — add them to the PROCESSES entry.")
            sys.exit(1)
    else:
        if args.reset and args.extend:
            print("[ERROR] --reset and --extend are mutually exclusive.")
            sys.exit(1)
        if args.com_energy is None:
            print("[ERROR] --com_energy is required in fixed-energy mode.")
            sys.exit(1)

    nfinal     = config["nfinal"]
    nparticles = nfinal + 2

    events_dir     = f"{WORK_DIR}/{args.process}_events"
    standalone_dir = f"{WORK_DIR}/{args.process}_standalone"

    # ----------------------------------------------------------------
    # Variable-energy mode: sample phase space directly, skip LHE gen
    # ----------------------------------------------------------------
    if variable_energy:
        n_total = args.nevents_total if args.nevents_total else args.nevents
        sqrts_min, sqrts_max = args.sqrts_min, args.sqrts_max

        recipe = variable_energy_recipe(
            args.process, sqrts_min, sqrts_max, n_total,
            role=args.role, seed=args.seed)

        m_info = (f"m_final = {config['m_final']} GeV" if config["nfinal"] == 2
                  else f"m_finals = {config['m_finals']} GeV")
        print(f"\n{'='*60}")
        print(f"  Process:     {args.process}")
        print(f"  Mode:        VARIABLE ENERGY  √s ~ Uniform[{sqrts_min}, {sqrts_max}] GeV")
        print(f"  {m_info}")
        print(f"  PDG IDs:     {config['pdg_ids']}")
        print(f"  Events:      {n_total:,}")
        print(f"  Role/seed:   {args.role or '—'} / base={args.seed}")
        print(f"{'='*60}\n")

        # CLI invocation always (re)generates; the library API caches via reuse=True.
        output_file = generate_from_recipe(
            recipe, out_dir=OUTPUT_DIR, reuse=False,
            compile_if_needed=not args.skip_compile)
        print(f"\nDone! Dataset saved to {output_file}")
        return

    # ----------------------------------------------------------------
    # Fixed-energy mode (original behaviour)
    # ----------------------------------------------------------------
    mu     = args.ren_scale if args.ren_scale else args.com_energy
    alphas = compute_alphas(mu)
    config["alphas"] = alphas

    print(f"\n{'='*60}")
    print(f"  Process:               {args.process}")
    print(f"  CoM energy:            {args.com_energy} GeV")
    print(f"  Renormalization scale: {mu} GeV")
    print(f"  αS(μ):                 {alphas:.4f}")
    print(f"  Particles/event:       {nparticles} ({2} in + {nfinal} out)")
    print(f"  Events:                {args.nevents} × {args.nbatches} = "
          f"{args.nevents * args.nbatches} total")
    if args.reset:
        print(f"  Mode:                  RESET")
    elif args.extend:
        print(f"  Mode:                  EXTEND")
    print(f"{'='*60}\n")

    output_file = (f"{OUTPUT_DIR}/{args.process}_{int(args.com_energy)}GeV"
                   f"_amplitudes.npy")

    # ------------------------------------------------------------------
    # Step 1–3: Generate process dirs and events
    # ------------------------------------------------------------------
    if not args.skip_gen:
        generate_mg5_process(args.process, config)

        if args.reset:
            print(f"\n[RESET] Deleting existing runs...")
            delete_existing_runs(events_dir)
            start_batch = 0
        elif args.extend:
            start_batch = get_existing_batch_count(events_dir)
            print(f"\n[EXTEND] Found {start_batch} existing runs, "
                  f"continuing from batch {start_batch + 1}")
        else:
            start_batch = 0

        for i, batch_idx in enumerate(range(start_batch,
                                            start_batch + args.nbatches)):
            print(f"\n[BATCH {i + 1}/{args.nbatches}]")
            configure_cards(events_dir, config, args.com_energy,
                            args.nevents, batch_idx)
            if not run_event_generation(events_dir, batch_idx):
                print(f"Stopping at batch {batch_idx}.")
                sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Compile backend
    # ------------------------------------------------------------------
    if not args.skip_compile:
        backend, subproc_dirs, driver_bin, eff_dir = compile_backends(
            standalone_dir, nparticles
        )
    else:
        backend, subproc_dirs, driver_bin, eff_dir = \
            detect_compiled_backend(standalone_dir)
        print(f"[CXX] Skipping compilation. Backend: {backend}, "
              f"subprocesses: {subproc_dirs}")

    # ------------------------------------------------------------------
    # Step 5: Build dataset
    # ------------------------------------------------------------------
    if args.extend and os.path.exists(output_file):
        print(f"\n[DATA] Extend mode: loading existing dataset...")
        existing_data = np.load(output_file)
        print(f"[DATA] Existing dataset: {len(existing_data)} events")
        build_dataset(events_dir, eff_dir, backend, subproc_dirs,
                      driver_bin, config, output_file,
                      existing_data=existing_data)
    else:
        build_dataset(events_dir, eff_dir, backend, subproc_dirs,
                      driver_bin, config, output_file)

    print(f"\nDone! Dataset saved to {output_file}")

if __name__ == "__main__":
    main()
