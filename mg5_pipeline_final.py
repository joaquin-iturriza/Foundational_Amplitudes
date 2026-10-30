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
import glob
import math
import re
import sys
import gzip
import tarfile
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
    """Run alphas from MZ to scale mu at 1-loop. `mz` is the fixed reference scale of the
    running (the SM Z mass), also for an M_Z scan point: alpha_s(M_Z^SM) is the reference
    value the coupling feature carries, not the pole of that parameter point."""
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
            if pdg == 6: patches["ymt"] = m      # the Yukawa follows the pole mass (ttH, tbW)
            if pdg == 5: patches["ymb"] = m
        else:
            print(f"  [WARN] physics.masses: no param_card symbol for PDG {pdg}; "
                  f"ME mass unchanged (phase-space mass still updated).")
    # physics.widths {pdg: GeV}: a width set with the mass it belongs to (catalog_v3's heavy-quark layer scales the top
    # width with m_t); opt-in, so a mass scan without it keeps the card's width
    for pdg, w in (physics.get("widths") or {}).items():
        sym = {6: "WT"}.get(abs(int(pdg)))
        if sym is None:
            raise KeyError(f"physics.widths: no param_card width symbol for PDG {pdg}")
        patches[sym] = float(w)
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
    # physics.cuts {pt_min, cos_max, dr_min, m_min}: this entry's fiducial cuts in place of the global ones (catalog_v3's
    # H7, pT >= 30 GeV); identity-bearing through recipe["fiducial_cuts"], so it never shares a pool with the default cuts
    if (physics or {}).get("cuts"):
        if not FIDUCIAL_CUTS_ENABLED:
            raise ValueError(f"{name}: physics.cuts set while AMP_FIDUCIAL_CUTS=off (the pool would be built uncut)")
        base_cuts = cfg.get("fiducial_cuts") or FIDUCIAL_CUTS       # a catalog_v4 entry carries its own
        assert set(physics["cuts"]) <= set(FIDUCIAL_CUTS), f"physics.cuts: unknown keys {physics['cuts']}"
        cfg["fiducial_cuts"] = {**base_cuts, **{k: float(v) for k, v in physics["cuts"].items()}}
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


def is_synthetic(process):
    """True for a synthetic-amplitude process name (``syn_<k>``, ``syn<n>_<k>``,
    ``syn[<n>]v<V>_<k>``): tools/synthetic_amplitudes.py, no MadGraph behind it. The name grammar is that module's."""
    if not str(process).startswith("syn"):
        return False
    from tools import synthetic_amplitudes as syn
    return syn.is_synthetic(process)


def register_synthetic(process, diagrams_dir=None):
    """Register a synthetic process into PROCESSES (from its name alone; idempotent) and
    write its diagram sidecar where data.offshell_per_event looks for it. Returns its cfg."""
    if process not in PROCESSES:
        from tools import synthetic_amplitudes as syn
        PROCESSES[process] = syn.catalog_entry(process)
        syn.write_sidecar(process, diagrams_dir)
    return PROCESSES[process]


def internal_mass(process, pdg):
    """Mass (GeV) of an INTERNAL propagator `pdg` of `process`: the entry's own
    ``internal_masses`` (a synthetic process's random scalar mass), else the table mass."""
    if is_synthetic(process):
        register_synthetic(process)
    im = PROCESSES.get(process, {}).get("internal_masses") or {}
    m = im.get(int(pdg), im.get(-int(pdg)))
    return float(m) if m is not None else float(_table_mass(pdg))


def register_recipe_processes(specs, default_sampling=None):
    """Register every recipe spec carrying a ``physics`` scan (or a decorated
    ``base``) into PROCESSES, so generation can address them by dataset name, and
    attach the sampling policy (per-entry ``sampling``, else the recipe-level
    ``default_sampling``, else the pipeline default) to the entry generation reads.
    Synthetic processes (``syn_<k>``) are registered from their names.
    ``specs`` is the experiment's ``_recipe_specs`` list."""
    for s in specs:
        name = s["name"]
        base = s.get("base", name)
        if is_synthetic(base):
            register_synthetic(base)
        physics = s.get("physics")
        if physics or base != name:
            register_scan_process(name, base, physics or {})
        samp = s.get("sampling") or default_sampling
        if samp is not None and name in PROCESSES:
            PROCESSES[name]["sampling"] = dict(samp)


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
                # `amp_orders`: the model's label for the target, fully derived from the
                # process; the bytes do not depend on it, so a relabelling must not
                # regenerate every pool.
                "amp_orders",
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
                "chunked", "chunk_size", "gen_weight",
                # `lhe_run`: what an lhe_pp MadEvent run reported (event count, cross section,
                # MG5 version, timings): a record of the run, not "what the data is".
                "lhe_run",
                # `physics`: the model, card values, coupling-order clause and alpha_s rule the labels were computed
                # with, read back from the backend (tools/pool_physics.py): a record of what the backend did, written
                # after the fact (and backfilled into older sidecars), so it must not move a pool's id.
                "physics"}
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
    # the physics the labels were computed with, read from the backend's own cards (tools/pool_physics.py): what a
    # probe read checks against its parent's training catalog. Never fails generation: an unreadable block is
    # recorded as {schema, error} and recomputed at read.
    if "physics" not in recipe:
        from tools import pool_physics
        recipe["physics"] = pool_physics.safe_block(recipe["process"], recipe)

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

# MadGraph particle names (sm / loop_sm) -> PDG id, for reading generate strings.
MG5_PDG = {"e-": 11, "e+": -11, "mu-": 13, "mu+": -13, "ta-": 15, "ta+": -15,
           "ve": 12, "ve~": -12, "vm": 14, "vm~": -14, "vt": 16, "vt~": -16,
           "u": 2, "u~": -2, "d": 1, "d~": -1, "s": 3, "s~": -3, "c": 4, "c~": -4,
           "b": 5, "b~": -5, "t": 6, "t~": -6, "g": 21, "a": 22, "z": 23,
           "w+": 24, "w-": -24, "h": 25,
           # non-SM models (NONSM below): DMsimp_s_spin1's vector mediator, the 2HDM's scalars (its h1 is PDG 25)
           "y1": 5000001, "h+": 37, "h-": -37, "h1": 25, "h2": 35, "h3": 36}


def generate_slot_pdgs(gen):
    """(initial, final) PDG lists of one `generate`/`add process` line, in MadGraph
    SLOT order: the order CPPProcess::setMomenta / matrix2py expect their momenta in.
    Coupling orders (QED<=2), [virt=QCD], / $ @ suffixes are ignored."""
    s = gen.strip()
    for pre in ("generate", "add process"):
        if s.startswith(pre):
            s = s[len(pre):]
    s = re.split(r"[\[/$@]", s)[0]
    s = re.sub(r"\b[A-Z]+(\^2)?\s*(<=|>=|==|=|<|>)\s*\d+", " ", s)   # QED<=2, and squared orders QED^2==4
    ini, fin = s.split(">", 1)
    ini, fin = ini.split(), fin.split()
    unknown = [t for t in ini + fin if t not in MG5_PDG]
    if unknown:
        raise ValueError(f"unknown MG5 particle name(s) {unknown} in '{gen}'")
    return [MG5_PDG[t] for t in ini], [MG5_PDG[t] for t in fin]


def row_to_slot_perm(pdg_ids, mg5_generate):
    """Row permutation taking a stored event (rows in `pdg_ids` order: the two beams
    first, beam- along +z in row 0) to MadGraph slot order, so `momenta[perm]` is
    what the matrix element must be fed. Every labelling path (C++ driver, matrix2py,
    MadLoop) goes through this: the stored label of a row IS the particle the ME saw.
    Identical particles are assigned in order (|M|^2 is symmetric under exchanging
    them). Raises if the catalog pdg_ids and the generate string disagree."""
    gen = mg5_generate[0] if isinstance(mg5_generate, (list, tuple)) else mg5_generate
    ini, fin = generate_slot_pdgs(gen)
    pdg = [int(x) for x in pdg_ids]
    if len(pdg) != len(ini) + len(fin):
        raise ValueError(f"pdg_ids {pdg} has {len(pdg)} legs, '{gen}' has {len(ini)+len(fin)}")
    perm = []
    for slots, rows in ((ini, range(len(ini))), (fin, range(len(ini), len(pdg)))):
        avail = list(rows)
        for want in slots:
            j = next((r for r in avail if pdg[r] == want), None)
            if j is None:
                raise ValueError(f"pdg_ids {pdg} do not match '{gen}' (no row for pdg {want})")
            avail.remove(j)
            perm.append(j)
    return perm


# Locked heavy-quark convention, identical for LO and NLO-virt (tools/nlo_conventions.py
# pins MT to the collaborators' ttbar reference; MB stays the model default 4.7):
# every entry with a top carries these param_card patches AND samples at LOCKED_MT.
LOCKED_MT = 172.5
TOP_PATCHES = {"MT": LOCKED_MT, "ymt": LOCKED_MT}

def order_vector(cfg):
    """Coupling-order vector of a catalog entry's STORED target:
        [L_QCD, L_EW, alpha_s_power_max, alpha_ew_power_max].
    Tree level: |M|^2 is homogeneous of total degree nfinal in (alpha_s, alpha_ew),
    so a pure entry is [0,0,k,nfinal-k]; a mixed entry ("QED<=N" in the generate
    string: four-quark, heavy-quark pair) has terms from alpha_s^k down to
    alpha_ew^N, recorded as the two maxima [0,0,k,N]. The maxima say WHICH monomials
    exist, not their size; where the EW term dominates (Z/W pole of the s-channel
    four-quark entries) the kinematics carry that. Yukawa and Higgs self-couplings
    count as electroweak powers (fixed functions of v and the masses in the SM).
    NLO virt (kind="virt", one QCD loop): the base dataset stores the alpha_s-STRIPPED
    coefficient (c0+shift)*born, so its alpha_s power is the born's, k-1; with
    alphas_prefactor (an alpha_s scan) the physical alpha_s/2pi is restored and the
    power is k. The loop itself is counted in L_QCD, which is what tells a stripped
    virt target apart from the LO of the same process at equal alpha_s power."""
    if cfg.get("order") is not None:          # explicit (loop-induced entries)
        return [int(x) for x in cfg["order"]]
    nfinal = int(cfg["nfinal"])
    k = int(cfg.get("alphas_power", 0))
    if cfg.get("kind") == "virt":
        born_k = max(k - 1, 0)
        a = k if cfg.get("alphas_prefactor") else born_k
        b = nfinal - born_k
        # mixed born: the EW maximum is on the one-loop table's generate string (QED<=N)
        from tools.nlo_virtual_pipeline import VIRT_PROCESSES
        vb = cfg.get("virt_base")
        if vb not in VIRT_PROCESSES:
            raise KeyError(f"virt entry's base '{vb}' is not in tools.nlo_virtual_pipeline.VIRT_PROCESSES")
        m = re.search(r"QED\s*<=\s*(\d+)", VIRT_PROCESSES[vb]["mg5"])
        if m:
            b = int(m.group(1))
        return [int(cfg.get("n_loops", 1)), 0, a, b]
    gen = cfg.get("mg5_generate", [""])
    gen = gen[0] if isinstance(gen, (list, tuple)) else gen
    m = re.search(r"QED\s*<=\s*(\d+)", gen)
    b = int(m.group(1)) if m else nfinal - k
    return [0, 0, k, b]


def normalize_order_vector(o, cfg=None):
    """Accept the legacy 2-vector [n_loops, alpha_s_power] (recipes and configs written
    before the 4-vector) and widen it: the EW power is taken from the catalog entry
    when one is given, else 0."""
    o = [int(x) for x in o]
    if len(o) == 4:
        return o
    if len(o) == 2:
        b = order_vector(cfg)[3] if cfg is not None else 0
        return [o[0], 0, o[1], b]
    raise ValueError(f"amp_orders entry must have 2 or 4 components, got {o}")


def sidecar_born_orders(path):
    """The (QCD, QED) orders of a process's Born amplitudes, sorted, from its diagram sidecar (tools/dump_diagrams.py):
    each diagram's sums over its vertices, the QCD one minus 2 for a one-loop diagram (the loop's two powers of g_s;
    the sum is then the Born's whatever the loop, so an electroweak loop lands on the same total). An amplitude of
    orders (a, b) carries g_s^a e^b, so |M|^2's alpha_s powers run between the least and the largest a, its alpha
    powers between the least and the largest b. A non-SM coupling (the Z' model's DMV) is neither: a fixed constant
    of the model, it enters no power. None without diagrams (tools/mixed_order_audit.py reads the QCD half)."""
    out = set()
    with open(path) as f:
        d = json.load(f)
    for sp in d["subprocesses"]:
        for dg in sp["diagrams"]:
            o = [n.get("orders", {}) for n in dg["nodes"] if n["kind"] == "vertex"]
            a = sum(int(x.get("QCD", 0)) for x in o)
            b = sum(int(x.get("QED", 0)) for x in o)
            out.add((a - 2 if dg.get("diagram_type") == "loop" else a, b))
    return sorted(out) or None


def order_vector_from_diagrams(cfg, born_orders, o=None):
    """The order vector of a catalog entry's stored target with its coupling powers read from the generated diagrams
    (sidecar_born_orders) instead of the catalog's hand-set alphas_power and QED<=N, in order_vector's convention:
    [L_QCD, L_EW, alpha_s power max, alpha power max]. A pure entry gets its one monomial, a mixed one (QCD-EW
    interference) both maxima, as order_vector records the four-quark entries; for the SM's trees this is
    [k, nfinal - k] and [k_max, nfinal - k_min], and a BSM coupling counts in neither (zp_ee_Y1a: alpha^1, not 2).
    With alphas_prefactor (an alpha_s scan of a stripped virtual) the restored alpha_s / 2 pi adds one alpha_s power.
    The loop counts are kept from `o` (default order_vector(cfg)); an entry with an explicit order (loop-induced) or
    without diagrams keeps `o`. Only the model's order features and the target read this: the pools' recipe_id leaves
    amp_orders out and keys on alphas_power, which stays the catalog's, so no stored pool or recipe id changes."""
    o = list(order_vector(cfg) if o is None else o)
    if cfg.get("order") is not None or not born_orders:
        return o
    pref = 1 if (cfg.get("kind") == "virt" and cfg.get("alphas_prefactor")) else 0
    return [int(o[0]), int(o[1]), max(a for a, _ in born_orders) + pref, max(b for _, b in born_orders)]


def target_monomial_from_diagrams(cfg, born_orders):
    """(alpha_s power, alpha power) the dimensionless target divides by: the monomial with the highest alpha_s power
    (catalog_v3's minimal-QED pool of the same process stores it alone) and, among its diagrams, the highest alpha
    power; alphas_prefactor adds one alpha_s power as above."""
    pref = 1 if (cfg.get("kind") == "virt" and cfg.get("alphas_prefactor")) else 0
    q = max(a for a, _ in born_orders)
    return q + pref, max(b for a, b in born_orders if a == q)


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
        "param_card_patches": dict(TOP_PATCHES),
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 6, -6],
        "m_final": LOCKED_MT,   # top quark mass (GeV) — keep in sync with param_card
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
        "m_finals": [80.419, 80.419, 4.7, 4.7],   # mW, mW, mb at the card value (was sampled massless); top is internal
    },
    "ee_mumutautau": {   # Higgs: ee→ZH, Z→μμ, H→ττ — internal H resonates at M(ττ)≈MH.
        # τ is massive (nonzero Yukawa), so H→ττ exists at the physical M_H≈125 (unlike
        # H→WW in ee_wwbb, bounded by 2·M_W). The μμ pair also carries the s-channel Z.
        "mg5_generate": ["generate e+ e- > mu+ mu- ta+ ta-"],
        "nfinal": 4,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, -13, 13, -15, 15],
        "m_finals": [0.0, 0.0, 1.777, 1.777],   # μμ massless, ττ massive; H/Z are internal
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
        "alphas_power": 1,
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
        "alphas_power": 2,
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
        "alphas_power": 3,
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
        "alphas_power": 4,
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
    # ------------------------------------------------------------------
    # catalog_v1: structured 2->2 / 2->3 catalog (recipes/catalog_v1_*.yaml).
    # One subprocess per entry with explicit pdg_ids (variable-energy mode does
    # not support flavour-summed processes). Quark-initiated entries are
    # parton-level, no PDFs: a single representative flavour where flavour only
    # rescales |M|^2 (per-dataset standardization removes that), separate u/d
    # entries where the gamma/Z interference changes the shape.
    # Four-quark and heavy-quark-pair entries carry "QED<=2": MG5's default
    # coupling-order rule (minimal QED) would keep only the gluon diagrams and
    # drop the gamma/Z exchange and the s-channel W, which is the structure those
    # entries are in the catalog for (and would make ud_ud, us_us, udbar_udbar
    # exact duplicates). Their |M|^2 is therefore MIXED in alpha_s and alpha_ew:
    # alphas_power (hence amp_orders) records the dominant power only, which the
    # s-channel entries violate at the Z/W pole where the resonant EW term wins.
    # The stored values are exact anyway: alpha_s is set per event inside the ME
    # (no reweighting), so coupling scans on these entries need nothing special.
    # ------------------------------------------------------------------
    "ee_numu": {   # Z only; pair with ee_nnbar (Z + t-channel W)
        "mg5_generate": ["generate e+ e- > vm vm~"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 14, -14],
        "m_final": 0.0,
    },
    "uubar_gg": {   # the only triple-gluon vertex reachable from these initial states
        "mg5_generate": ["generate u u~ > g g"],
        "nfinal": 2,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 21, 21],
        "m_final": 0.0,
    },
    "uubar_ga": {   # mixed QCD x QED, t/u-channel quark
        "mg5_generate": ["generate u u~ > g a"],
        "nfinal": 2,
        "alphas_power": 1,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 21, 22],
        "m_final": 0.0,
    },
    "uubar_Zg": {
        "mg5_generate": ["generate u u~ > z g"],
        "nfinal": 2,
        "alphas_power": 1,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 23, 21],
        "m_final": [91.188, 0.0],
    },
    "uubar_Za": {
        "mg5_generate": ["generate u u~ > z a"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 23, 22],
        "m_final": [91.188, 0.0],
    },
    "uubar_ZZ": {
        "mg5_generate": ["generate u u~ > z z"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 23, 23],
        "m_final": 91.188,
    },
    "uubar_WW": {   # t-channel d exchange + s-channel gamma/Z (WWV vertices)
        "mg5_generate": ["generate u u~ > w+ w-"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 24, -24],
        "m_final": 80.419,
    },
    "uubar_ZH": {
        "mg5_generate": ["generate u u~ > z h"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 23, 25],
        "m_final": [91.188, 125.0],
    },
    "uubar_ttbar": {   # s-channel gluon dominant (+ gamma/Z): mixed
        "mg5_generate": ["generate u u~ > t t~ QED<=2"],
        "nfinal": 2,
        "alphas_power": 2,
        "param_card_patches": dict(TOP_PATCHES),
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 6, -6],
        "m_final": LOCKED_MT,
    },
    "uubar_bbbar": {   # mixed
        "mg5_generate": ["generate u u~ > b b~ QED<=2"],
        "nfinal": 2,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 5, -5],
        "m_final": 4.7,
    },
    "uubar_uubar": {   # s + t channel gluon/gamma/Z, identical flavours: mixed
        "mg5_generate": ["generate u u~ > u u~ QED<=2"],
        "nfinal": 2,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 2, -2],
        "m_final": 0.0,
    },
    "uubar_ddbar": {   # s-channel only: mixed
        "mg5_generate": ["generate u u~ > d d~ QED<=2"],
        "nfinal": 2,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 1, -1],
        "m_final": 0.0,
    },
    "udbar_udbar": {   # t-channel g/gamma/Z AND s-channel W: mixed
        "mg5_generate": ["generate u d~ > u d~ QED<=2"],
        "nfinal": 2,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -1, 2, -1],
        "m_final": 0.0,
    },
    "udbar_enu": {   # charged Drell-Yan, s-channel W
        "mg5_generate": ["generate u d~ > e+ ve"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -1, -11, 12],
        "m_final": 0.0,
    },
    "udbar_taunu": {   # charged Drell-Yan to a massive lepton, s-channel W (star arm on the W pole; tau massive, so not
        "mg5_generate": ["generate u d~ > ta+ vt"],     # udbar_enu's dataset)
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -1, -15, 16],
        "m_final": [1.777, 0.0],
    },
    "udbar_tbbar": {   # s-channel single top
        "mg5_generate": ["generate u d~ > t b~"],
        "nfinal": 2,
        "param_card_patches": dict(TOP_PATCHES),
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -1, 6, -5],
        "m_final": [LOCKED_MT, 4.7],
    },
    "udbar_WZ": {   # WWZ vertex, charged current
        "mg5_generate": ["generate u d~ > w+ z"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -1, 24, 23],
        "m_final": [80.419, 91.188],
    },
    "udbar_Wa": {   # WWgamma vertex
        "mg5_generate": ["generate u d~ > w+ a"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -1, 24, 22],
        "m_final": [80.419, 0.0],
    },
    "udbar_WH": {   # WWH vertex
        "mg5_generate": ["generate u d~ > w+ h"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -1, 24, 25],
        "m_final": [80.419, 125.0],
    },
    "udbar_Wg": {
        "mg5_generate": ["generate u d~ > w+ g"],
        "nfinal": 2,
        "alphas_power": 1,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -1, 24, 21],
        "m_final": [80.419, 0.0],
    },
    "uu_uu": {   # pure t/u-channel, identical quarks (no annihilation): mixed
        "mg5_generate": ["generate u u > u u QED<=2"],
        "nfinal": 2,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, 2, 2, 2],
        "m_final": 0.0,
    },
    "ud_ud": {   # t-channel g/gamma/Z (+ W exchange to d u): mixed
        "mg5_generate": ["generate u d > u d QED<=2"],
        "nfinal": 2,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, 1, 2, 1],
        "m_final": 0.0,
    },
    "us_us": {   # t-channel, no W piece: mixed
        "mg5_generate": ["generate u s > u s QED<=2"],
        "nfinal": 2,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, 3, 2, 3],
        "m_final": 0.0,
    },
    "ee_ttbarg": {   # gluon radiation off a massive quark (dead cone)
        "mg5_generate": ["generate e+ e- > t t~ g"],
        "nfinal": 3,
        "alphas_power": 1,
        "param_card_patches": dict(TOP_PATCHES),
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 6, -6, 21],
        "m_finals": [LOCKED_MT, LOCKED_MT, 0.0],
    },
    "ee_uua": {   # ISR + FSR photon on a quark line
        "mg5_generate": ["generate e+ e- > u u~ a"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 2, -2, 22],
        "m_finals": [0.0, 0.0, 0.0],
    },
    "ee_nnbara": {   # mono-photon: ISR only
        "mg5_generate": ["generate e+ e- > ve ve~ a"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 12, -12, 22],
        "m_finals": [0.0, 0.0, 0.0],
    },
    "ee_Zaa": {
        "mg5_generate": ["generate e+ e- > z a a"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 23, 22, 22],
        "m_finals": [91.188, 0.0, 0.0],
    },
    "ee_ZHH": {   # HHH trilinear + ZZHH quartic: Higgs self-coupling
        "mg5_generate": ["generate e+ e- > z h h"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 23, 25, 25],
        "m_finals": [91.188, 125.0, 125.0],
    },
    "ee_nnbarH": {   # WW fusion (t-channel W) + ZH with Z > nu nu
        "mg5_generate": ["generate e+ e- > ve ve~ h"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 12, -12, 25],
        "m_finals": [0.0, 0.0, 125.0],
    },
    "ee_eeH": {   # ZZ fusion + ZH with Z > ee
        "mg5_generate": ["generate e+ e- > e+ e- h"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, -11, 11, 25],
        "m_finals": [0.0, 0.0, 125.0],
    },
    "ee_ttbarH": {   # top Yukawa
        "mg5_generate": ["generate e+ e- > t t~ h"],
        "nfinal": 3,
        "param_card_patches": dict(TOP_PATCHES),
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 6, -6, 25],
        "m_finals": [LOCKED_MT, LOCKED_MT, 125.0],
    },
    "ee_WWH": {   # WWH vertex, e+e- side
        "mg5_generate": ["generate e+ e- > w+ w- h"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 24, -24, 25],
        "m_finals": [80.419, 80.419, 125.0],
    },
    "ee_enuW": {   # single W: t-channel photon, forward
        "mg5_generate": ["generate e+ e- > e- ve~ w+"],
        "nfinal": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 11, -12, 24],
        "m_finals": [0.0, 0.0, 80.419],
    },
    "uubar_Zgg": {
        "mg5_generate": ["generate u u~ > z g g"],
        "nfinal": 3,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 23, 21, 21],
        "m_finals": [91.188, 0.0, 0.0],
    },
    "uubar_ggg": {   # triple AND quartic gluon vertices
        "mg5_generate": ["generate u u~ > g g g"],
        "nfinal": 3,
        "alphas_power": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 21, 21, 21],
        "m_finals": [0.0, 0.0, 0.0],
    },
    "uubar_agg": {
        "mg5_generate": ["generate u u~ > a g g"],
        "nfinal": 3,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 22, 21, 21],
        "m_finals": [0.0, 0.0, 0.0],
    },
    "uubar_uubarg": {   # five-parton QCD: mixed
        "mg5_generate": ["generate u u~ > u u~ g QED<=2"],
        "nfinal": 3,
        "alphas_power": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 2, -2, 21],
        "m_finals": [0.0, 0.0, 0.0],
    },
    "uubar_ttbarg": {   # mixed
        "mg5_generate": ["generate u u~ > t t~ g QED<=2"],
        "nfinal": 3,
        "alphas_power": 3,
        "param_card_patches": dict(TOP_PATCHES),
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 6, -6, 21],
        "m_finals": [LOCKED_MT, LOCKED_MT, 0.0],
    },
    "udbar_Wgg": {
        "mg5_generate": ["generate u d~ > w+ g g"],
        "nfinal": 3,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -1, 24, 21, 21],
        "m_finals": [80.419, 0.0, 0.0],
    },
    "udbar_enug": {   # W + jet
        "mg5_generate": ["generate u d~ > e+ ve g"],
        "nfinal": 3,
        "alphas_power": 1,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -1, -11, 12, 21],
        "m_finals": [0.0, 0.0, 0.0],
    },
    "ee_ss": {   # HOLD-OUT: d<->s relabel of ee_ddbar (s massless)
        "mg5_generate": ["generate e+ e- > s s~"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 3, -3],
        "m_final": 0.0,
    },
    "ee_cc": {   # HOLD-OUT: u<->c relabel of ee_uu (c massless in sm)
        "mg5_generate": ["generate e+ e- > c c~"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 4, -4],
        "m_final": 0.0,
    },
    "ee_nutau": {   # HOLD-OUT: relabel of ee_numu
        "mg5_generate": ["generate e+ e- > vt vt~"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 16, -16],
        "m_final": 0.0,
    },
    "uubar_mumu": {   # HOLD-OUT: crossing of ee_uu
        "mg5_generate": ["generate u u~ > mu+ mu-"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, -13, 13],
        "m_final": 0.0,
    },
    "uubar_aa": {   # HOLD-OUT: crossing of ee_aa
        "mg5_generate": ["generate u u~ > a a"],
        "nfinal": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 22, 22],
        "m_final": 0.0,
    },
    "dd_dd": {   # HOLD-OUT: relabel of uu_uu
        "mg5_generate": ["generate d d > d d QED<=2"],
        "nfinal": 2,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [1, 1, 1, 1],
        "m_final": 0.0,
    },
    "ds_ds": {   # HOLD-OUT: relabel of us_us
        "mg5_generate": ["generate d s > d s QED<=2"],
        "nfinal": 2,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [1, 3, 1, 3],
        "m_final": 0.0,
    },
    "cs_cs": {   # HOLD-OUT: relabel of us_us (u<->c)
        "mg5_generate": ["generate c s > c s QED<=2"],
        "nfinal": 2,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [4, 3, 4, 3],
        "m_final": 0.0,
    },
    "ee_ssbarg": {   # HOLD-OUT: relabel of ee_ddbarg
        "mg5_generate": ["generate e+ e- > s s~ g"],
        "nfinal": 3,
        "alphas_power": 1,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 3, -3, 21],
        "m_finals": [0.0, 0.0, 0.0],
    },
    "uubar_mumug": {   # HOLD-OUT: crossing of ee_uug
        "mg5_generate": ["generate u u~ > mu+ mu- g"],
        "nfinal": 3,
        "alphas_power": 1,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, -13, 13, 21],
        "m_finals": [0.0, 0.0, 0.0],
    },
    # ------------------------------------------------------------------
    # catalog_v1, layer 2->4 (multiplicity axis)
    # ------------------------------------------------------------------
    "ee_uuuu": {   # identical-flavour four-quark final
        "mg5_generate": ["generate e+ e- > u u~ u u~"],
        "nfinal": 4,
        "alphas_power": 2,   # MG5 default order QED=2 QCD=2: g -> q qbar splitting
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 2, -2, 2, -2],
        "m_finals": [0, 0, 0, 0],
    },
    "ee_uudd": {
        "mg5_generate": ["generate e+ e- > u u~ d d~"],
        "nfinal": 4,
        "alphas_power": 2,   # MG5 default order QED=2 QCD=2: g -> q qbar splitting
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 2, -2, 1, -1],
        "m_finals": [0, 0, 0, 0],
    },
    "ee_ttbargg": {
        "mg5_generate": ["generate e+ e- > t t~ g g"],
        "nfinal": 4,
        "alphas_power": 2,
        "param_card_patches": dict(TOP_PATCHES),
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 6, -6, 21, 21],
        "m_finals": [LOCKED_MT, LOCKED_MT, 0, 0],
    },
    "ee_ttbarbb": {   # heavy four-quark final: mixed
        "mg5_generate": ["generate e+ e- > t t~ b b~"],
        "nfinal": 4,
        "alphas_power": 2,
        "param_card_patches": dict(TOP_PATCHES),
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 6, -6, 5, -5],
        "m_finals": [LOCKED_MT, LOCKED_MT, 4.7, 4.7],
    },
    "ee_WWaa": {   # WWgammagamma quartic vertex
        "mg5_generate": ["generate e+ e- > w+ w- a a"],
        "nfinal": 4,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 24, -24, 22, 22],
        "m_finals": [80.419, 80.419, 0, 0],
    },
    "ee_WWZZ": {   # WWZZ quartic vertex
        "mg5_generate": ["generate e+ e- > w+ w- z z"],
        "nfinal": 4,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 24, -24, 23, 23],
        "m_finals": [80.419, 80.419, 91.188, 91.188],
    },
    "ee_WWWW": {   # WWWW quartic vertex
        "mg5_generate": ["generate e+ e- > w+ w- w+ w-"],
        "nfinal": 4,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 24, -24, 24, -24],
        "m_finals": [80.419, 80.419, 80.419, 80.419],
    },
    "ee_nnbarWW": {   # t-channel W fusion to WW
        "mg5_generate": ["generate e+ e- > ve ve~ w+ w-"],
        "nfinal": 4,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 12, -12, 24, -24],
        "m_finals": [0, 0, 80.419, 80.419],
    },
    "ee_nnbarHH": {   # WW fusion double Higgs: WWHH quartic + HHH
        "mg5_generate": ["generate e+ e- > ve ve~ h h"],
        "nfinal": 4,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 12, -12, 25, 25],
        "m_finals": [0, 0, 125.0, 125.0],
    },
    "ee_ddbargg": {   # HOLD-OUT: relabel of ee_uugg
        "mg5_generate": ["generate e+ e- > d d~ g g"],
        "nfinal": 4,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [11, -11, 1, -1, 21, 21],
        "m_finals": [0, 0, 0, 0],
    },
    "uubar_gggg": {   # quartic gluon vertex everywhere
        "mg5_generate": ["generate u u~ > g g g g"],
        "nfinal": 4,
        "alphas_power": 4,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 21, 21, 21, 21],
        "m_finals": [0, 0, 0, 0],
    },
    "uubar_uubargg": {   # mixed
        "mg5_generate": ["generate u u~ > u u~ g g QED<=2"],
        "nfinal": 4,
        "alphas_power": 4,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 2, -2, 21, 21],
        "m_finals": [0, 0, 0, 0],
    },
    "uubar_uubarddbar": {   # mixed
        "mg5_generate": ["generate u u~ > u u~ d d~ QED<=2"],
        "nfinal": 4,
        "alphas_power": 4,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 2, -2, 1, -1],
        "m_finals": [0, 0, 0, 0],
    },
    "uubar_Zggg": {
        "mg5_generate": ["generate u u~ > z g g g"],
        "nfinal": 4,
        "alphas_power": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 23, 21, 21, 21],
        "m_finals": [91.188, 0, 0, 0],
    },
    "udbar_Wggg": {
        "mg5_generate": ["generate u d~ > w+ g g g"],
        "nfinal": 4,
        "alphas_power": 3,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -1, 24, 21, 21, 21],
        "m_finals": [80.419, 0, 0, 0],
    },
    # high-multiplicity Z+ng / W+ng (2026-10-09, the user's high-multiplicity test, site decision D20)
    "uubar_Zgggg": {
        "mg5_generate": ["generate u u~ > z g g g g"],
        "nfinal": 5,
        "alphas_power": 4,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 23, 21, 21, 21, 21],
        "m_finals": [91.188, 0, 0, 0, 0],
    },
    "uubar_Zggggg": {
        "mg5_generate": ["generate u u~ > z g g g g g"],
        "nfinal": 6,
        "alphas_power": 5,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 23, 21, 21, 21, 21, 21],
        "m_finals": [91.188, 0, 0, 0, 0, 0],
    },
    "udbar_Wgggg": {
        "mg5_generate": ["generate u d~ > w+ g g g g"],
        "nfinal": 5,
        "alphas_power": 4,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -1, 24, 21, 21, 21, 21],
        "m_finals": [80.419, 0, 0, 0, 0],
    },
    "udbar_Wggggg": {
        "mg5_generate": ["generate u d~ > w+ g g g g g"],
        "nfinal": 6,
        "alphas_power": 5,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -1, 24, 21, 21, 21, 21, 21],
        "m_finals": [80.419, 0, 0, 0, 0, 0],
    },
    "uubar_ttbargg": {   # mixed
        "mg5_generate": ["generate u u~ > t t~ g g QED<=2"],
        "nfinal": 4,
        "alphas_power": 4,
        "param_card_patches": dict(TOP_PATCHES),
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 6, -6, 21, 21],
        "m_finals": [LOCKED_MT, LOCKED_MT, 0, 0],
    },
    "uubar_WWgg": {
        "mg5_generate": ["generate u u~ > w+ w- g g"],
        "nfinal": 4,
        "alphas_power": 2,
        "param_card_patches": {},
        "run_card_patches": {"lpp1": "0", "lpp2": "0"},
        "pdg_ids": [2, -2, 24, -24, 21, 21],
        "m_finals": [80.419, 80.419, 0, 0],
    },
    # ------------------------------------------------------------------
    # catalog_v1, layer NLO: one-loop QCD interference (alpha_s-stripped, like the
    # existing *_nlo entries) for trees of the catalog, and LOOP-INDUCED targets
    # (|M_1|^2, no tree; explicit `order`). Backends: tools/nlo_virtual_pipeline.py.
    # ------------------------------------------------------------------
    "ee_bbbarg_nlo": {
        "kind": "virt", "virt": True, "virt_base": "ee_bbg", "nfinal": 3,
        "n_loops": 1, "alphas_power": 2,
        "pdg_ids": [11, -11, 5, -5, 21], "m_finals": [4.7, 4.7, 0], "param_card_patches": {},
    },
    "ee_ttbarg_nlo": {
        "kind": "virt", "virt": True, "virt_base": "ee_ttbarg", "nfinal": 3,
        "n_loops": 1, "alphas_power": 2,
        "pdg_ids": [11, -11, 6, -6, 21], "m_finals": [LOCKED_MT, LOCKED_MT, 0], "param_card_patches": {},
    },
    "uubar_gg_nlo": {
        "kind": "virt", "virt": True, "virt_base": "uubar_gg", "nfinal": 2,
        "n_loops": 1, "alphas_power": 3,
        "pdg_ids": [2, -2, 21, 21], "m_finals": [0, 0], "param_card_patches": {},
    },
    "uubar_uubar_nlo": {   # mixed born (QED<=2)
        "kind": "virt", "virt": True, "virt_base": "uubar_uubar", "nfinal": 2,
        "n_loops": 1, "alphas_power": 3,
        "pdg_ids": [2, -2, 2, -2], "m_finals": [0, 0], "param_card_patches": {},
    },
    "uubar_ttbar_nlo": {
        "kind": "virt", "virt": True, "virt_base": "uubar_ttbar", "nfinal": 2,
        "n_loops": 1, "alphas_power": 3,
        "pdg_ids": [2, -2, 6, -6], "m_finals": [LOCKED_MT, LOCKED_MT], "param_card_patches": {},
    },
    "uubar_mumu_nlo": {   # HOLD-OUT: crossing of ee_uu_nlo
        "kind": "virt", "virt": True, "virt_base": "uubar_mumu", "nfinal": 2,
        "n_loops": 1, "alphas_power": 1,
        "pdg_ids": [2, -2, -13, 13], "m_finals": [0, 0], "param_card_patches": {},
    },
    "uubar_Zg_nlo": {
        "kind": "virt", "virt": True, "virt_base": "uubar_Zg", "nfinal": 2,
        "n_loops": 1, "alphas_power": 2,
        "pdg_ids": [2, -2, 23, 21], "m_finals": [91.188, 0], "param_card_patches": {},
    },
    "udbar_tbbar_nlo": {
        "kind": "virt", "virt": True, "virt_base": "udbar_tbbar", "nfinal": 2,
        "n_loops": 1, "alphas_power": 1,
        "pdg_ids": [2, -1, 6, -5], "m_finals": [LOCKED_MT, 4.7], "param_card_patches": {},
    },
    "ee_aH_loop": {   # loop-induced, EW loop -- NOT CERTIFIED (spurious MadLoop pole), not in any recipe
        "kind": "virt", "virt": True, "virt_base": "ee_aH", "nfinal": 2,
        "n_loops": 1, "alphas_power": 0, "loopind": True, "order": [0, 1, 0, 4],
        "pdg_ids": [11, -11, 25, 22], "m_finals": [125.0, 0], "param_card_patches": {},
    },
    "ee_HH_loop": {   # loop-induced, Higgs self-coupling at one loop
        "kind": "virt", "virt": True, "virt_base": "ee_HH", "nfinal": 2,
        "n_loops": 1, "alphas_power": 0, "loopind": True, "order": [0, 1, 0, 4],
        "pdg_ids": [11, -11, 25, 25], "m_finals": [125.0, 125.0], "param_card_patches": {},
    },
    "uubar_Hg_loop": {   # loop-induced, top-loop ggH
        "kind": "virt", "virt": True, "virt_base": "uubar_Hg", "nfinal": 2,
        "n_loops": 1, "alphas_power": 3, "loopind": True, "order": [1, 0, 3, 1],
        "pdg_ids": [2, -2, 25, 21], "m_finals": [125.0, 0], "param_card_patches": {},
    },
    "ee_gg_loop": {   # loop-induced, quark box -- VANISHES (Furry/Landau-Yang), |M|^2 ~ 1e-37 noise; not in any recipe
        "kind": "virt", "virt": True, "virt_base": "ee_gg", "nfinal": 2,
        "n_loops": 1, "alphas_power": 2, "loopind": True, "order": [1, 0, 2, 2],
        "pdg_ids": [11, -11, 21, 21], "m_finals": [0, 0], "param_card_patches": {},
    },
    # catalog_v2 entries (rule-enumerated, MadGraph-validated) are appended below from
    # recipes/catalog_v2_processes.yaml by _load_catalog_v2(); see recipes/gen_catalog_v2.py.
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
                  "m_finals": [4.7, 4.7], "param_card_patches": {}},
    # 2->2 ELECTROWEAK one loop ([virt=QED], tools/nlo_virtual_pipeline.py `ew`): the alpha-stripped finite part,
    # carrying the EW Sudakov log^2(s/M_W^2); the Sudakov star arm and its probe (2026-10-04). The stored target is
    # the born's alpha^2 with one EW loop: [L_QCD, L_EW, alpha_s, alpha_ew] = [0, 1, 0, 2].
    "ee_dd_ew_nlo": {"kind": "virt", "virt": True, "virt_base": "ee_dd_ew", "nfinal": 2, "n_loops": 1,
                     "alphas_power": 0, "order": [0, 1, 0, 2], "pdg_ids": [11, -11, 1, -1],
                     "m_finals": [0.0, 0.0], "param_card_patches": {}},
    "ee_uu_ew_nlo": {"kind": "virt", "virt": True, "virt_base": "ee_uu_ew", "nfinal": 2, "n_loops": 1,
                     "alphas_power": 0, "order": [0, 1, 0, 2], "pdg_ids": [11, -11, 2, -2],
                     "m_finals": [0.0, 0.0], "param_card_patches": {}},
    "ee_mumu_ew_nlo": {"kind": "virt", "virt": True, "virt_base": "ee_mumu_ew", "nfinal": 2, "n_loops": 1,
                       "alphas_power": 0, "order": [0, 1, 0, 2], "pdg_ids": [11, -11, -13, 13],
                       "m_finals": [0.0, 0.0], "param_card_patches": {}},
    # 2->3 qqg QCD-virtual (born ∝ α_s; target ∝ α_s² → amp_orders [1,2]).
    "ee_uug_nlo": {"kind": "virt", "virt": True, "virt_base": "ee_uug", "nfinal": 3,
                   "n_loops": 1, "alphas_power": 2, "pdg_ids": [11, -11, 2, -2, 21],
                   "m_finals": [0.0, 0.0, 0.0], "param_card_patches": {}},
    "ee_ddg_nlo": {"kind": "virt", "virt": True, "virt_base": "ee_ddg", "nfinal": 3,
                   "n_loops": 1, "alphas_power": 2, "pdg_ids": [11, -11, 1, -1, 21],
                   "m_finals": [0.0, 0.0, 0.0], "param_card_patches": {}},
}


# =============================================================================
# NON-SM LAGRANGIANS (D49; docs/results.tex sec:nonsm). Three UFO models from the MG5 model database
# (`import model` downloads them on first use; CC-IN2P3, MG5 3.7.0), each at ONE fixed parameter point:
#   DMsimp_s_spin1   a vector mediator Y1 (PDG 5000001, a Z'), M = 500 GeV, vector and axial couplings to
#                    every lepton (gVl = 0.3, gAl = 0.1), quarks at the model default gVq = 0.25, the dark
#                    matter decoupled (gVXd = 0); width 18.64129 GeV from MG5's compute_widths at this point
#   2HDM             the general 2HDM (Higgs basis), H+ 200, h2 250, h3 300 GeV, h1 = the SM-like 125 GeV;
#                    the second doublet's lepton Yukawas set to zero, so ee -> scalars is gauge exchange only
#   SMEFTatNLO-LO    dim-6 SMEFT at Lambda = 1 TeV, every Wilson coefficient zero except the process's own
# Every model's SM inputs are set to the sm card's (aEWM1, Gf, MZ, MH, the widths) and the locked top mass, so each
# model's SM part is the SM pool's amplitude (tools/nonsm_validate.py checks it with the BSM couplings at zero)
# and the particle table's masses are the generator's (particle_ids.NONSM_MASSES_GEV mirrors NONSM_POINT).
NONSM_POINT = {"MY1": 500.0, "WY1": 18.64129, "mhc": 200.0, "mh2": 250.0, "mh3": 300.0}
_SM_ALIGN = {"aEWM1": 132.507, "Gf": 1.16639e-05, "MZ": 91.188, "MH": 125.0, "MT": LOCKED_MT, "ymt": LOCKED_MT,
             "WZ": 2.441404, "WW": 2.0476, "WT": 1.4915, "WH": 6.38233934e-03}
_ZP = {**_SM_ALIGN, "MTA": 1.777, "ymtau": 1.777, "gVXd": 0.0, "MXd": 1000.0,
       "MY1": NONSM_POINT["MY1"], "WY1": NONSM_POINT["WY1"],
       **{f"gVl{i}{i}": 0.3 for i in (1, 2, 3)}, **{f"gAl{i}{i}": 0.1 for i in (1, 2, 3)}}
_ZP_OFF = {**_ZP, **{f"gVl{i}{i}": 0.0 for i in (1, 2, 3)}, **{f"gAl{i}{i}": 0.0 for i in (1, 2, 3)}}
_THDM = {**{k: v for k, v in _SM_ALIGN.items() if k not in ("MH", "WH")}, "mh1": 125.0, "Wh1": 6.38233934e-03,
         "mhc": NONSM_POINT["mhc"], "mh2": NONSM_POINT["mh2"], "mh3": NONSM_POINT["mh3"],
         **{f"GL{ri}{i}x{j}": 0.0 for ri in ("R", "I") for i in (1, 2, 3) for j in (1, 2, 3)}}
# the quark-beam 2HDM entries (catalog_v3): the second doublet's quark Yukawas zeroed too (the UFO defaults are O(1),
# GDR1x1 = 1), so u u~ -> scalars is gauge exchange only, as e+ e- -> scalars is
_THDM_Q = {**_THDM, **{f"G{q}{ri}{i}x{j}": 0.0 for q in ("U", "D") for ri in ("R", "I") for i in (1, 2, 3) for j in (1, 2, 3)}}
# SMEFTatNLO (restrict_LO) is an {MW, MZ, Gf} scheme: MW = the sm card's derived 80.419 gives the same alpha
SMEFT_WILSON = ("cpDC cpWB cdp cp cWWW cG cpG cpW cpBB cpl1 cpl2 cpl3 c3pl1 c3pl2 c3pl3 cpe cpmu cpta cpqMi cpq3i "
                "cpQ3 cpQM cpu cpt cpd ctp ctZ ctW ctG cQq83 cQq81 cQu8 ctq8 cQd8 ctu8 ctd8 cQq13 cQq11 cQu1 ctq1 "
                "cQd1 ctu1 ctd1 cQQ8 cQQ1 cQt1 ctt1 cQt8 cQlM1 cQlM2 cQl31 cQl32 cQe1 cQe2 ctl1 ctl2 cte1 cte2 "
                "cQlM3 cQl33 cQe3 ctl3 cte3 ctlS3 ctlT3 cblS3 cll1111 cll2222 cll3333 cll1122 cll1133 cll2233 "
                "cll1221 cll1331 cll2332").split()
_SMEFT0 = {**{k: v for k, v in _SM_ALIGN.items() if k != "aEWM1"}, "MW": 80.419, "Lambda": 1000.0,
           **{c: 0.0 for c in SMEFT_WILSON}}
_LEP = {"lpp1": "0", "lpp2": "0"}
NONSM = {
    # Y1 in the s-channel next to gamma and Z: a new resonance in a process every rung pretrains on (ee_mumu)
    "zp_ee_mumu": {"model": "DMsimp_s_spin1", "mg5_generate": ["generate e+ e- > mu+ mu- QED<=2 DMV<=2"],
                   "nfinal": 2, "param_card_patches": _ZP, "pdg_ids": [11, -11, -13, 13], "m_final": 0.0},
    "zp_ee_uu":   {"model": "DMsimp_s_spin1", "mg5_generate": ["generate e+ e- > u u~ QED<=2 DMV<=2"],
                   "nfinal": 2, "param_card_patches": _ZP, "pdg_ids": [11, -11, 2, -2], "m_final": 0.0},
    # Y1 as an external leg (a new particle in the encoding), radiated with a photon
    "zp_ee_Y1a":  {"model": "DMsimp_s_spin1", "mg5_generate": ["generate e+ e- > y1 a QED<=1 DMV<=1"],
                   "nfinal": 2, "param_card_patches": _ZP, "pdg_ids": [11, -11, 5000001, 22],
                   "m_finals": [NONSM_POINT["MY1"], 0.0]},
    # charged-scalar pairs: spin 0 with charge, a combination no SM particle has
    "thdm_ee_HpHm": {"model": "2HDM", "mg5_generate": ["generate e+ e- > h+ h-"], "nfinal": 2,
                     "param_card_patches": _THDM, "pdg_ids": [11, -11, 37, -37],
                     "m_finals": [NONSM_POINT["mhc"], NONSM_POINT["mhc"]]},
    "thdm_ee_h2h3": {"model": "2HDM", "mg5_generate": ["generate e+ e- > h2 h3"], "nfinal": 2,
                     "param_card_patches": _THDM, "pdg_ids": [11, -11, 35, 36],
                     "m_finals": [NONSM_POINT["mh2"], NONSM_POINT["mh3"]]},
    # dim-6 contact (ee)(mumu) and the top's Z dipole plus an (ee)(tt) contact: new vertex structures
    "smeft_ee_mumu":  {"model": "SMEFTatNLO-LO", "mg5_generate": ["generate e+ e- > mu+ mu- QED<=2 NP<=1"],
                       "nfinal": 2, "param_card_patches": {**_SMEFT0, "cll1122": 0.5},
                       "pdg_ids": [11, -11, -13, 13], "m_final": 0.0},
    "smeft_ee_ttbar": {"model": "SMEFTatNLO-LO", "mg5_generate": ["generate e+ e- > t t~ QED<=2 NP<=1"],
                       "nfinal": 2, "param_card_patches": {**_SMEFT0, "ctZ": 0.5, "cte1": 0.5},
                       "pdg_ids": [11, -11, 6, -6], "m_final": LOCKED_MT},
    # catalog_v3's BSM layer (D98: "at least one BSM process that can be encoded with our current particle embedding"),
    # beyond the five above: quark beams, an external Y1 with a gluon or a Z, the charged scalar with h2, and a 2->3
    "zp_uux_mumu": {"model": "DMsimp_s_spin1", "mg5_generate": ["generate u u~ > mu+ mu- QED<=2 DMV<=2"],
                    "nfinal": 2, "param_card_patches": _ZP, "pdg_ids": [2, -2, -13, 13], "m_final": 0.0},
    "zp_uux_Y1g":  {"model": "DMsimp_s_spin1", "mg5_generate": ["generate u u~ > y1 g DMV<=1"],
                    "nfinal": 2, "param_card_patches": _ZP, "pdg_ids": [2, -2, 5000001, 21],
                    "m_finals": [NONSM_POINT["MY1"], 0.0]},
    "zp_uux_Y1z":  {"model": "DMsimp_s_spin1", "mg5_generate": ["generate u u~ > y1 z QED<=1 DMV<=1"],
                    "nfinal": 2, "param_card_patches": _ZP, "pdg_ids": [2, -2, 5000001, 23],
                    "m_finals": [NONSM_POINT["MY1"], 91.188]},
    "thdm_uux_HpHm": {"model": "2HDM", "mg5_generate": ["generate u u~ > h+ h-"], "nfinal": 2,
                      "param_card_patches": _THDM_Q, "pdg_ids": [2, -2, 37, -37],
                      "m_finals": [NONSM_POINT["mhc"], NONSM_POINT["mhc"]]},
    "thdm_udx_Hph2": {"model": "2HDM", "mg5_generate": ["generate u d~ > h+ h2"], "nfinal": 2,
                      "param_card_patches": _THDM_Q, "pdg_ids": [2, -1, 37, 35],
                      "m_finals": [NONSM_POINT["mhc"], NONSM_POINT["mh2"]]},
    "thdm_ee_HpHma": {"model": "2HDM", "mg5_generate": ["generate e+ e- > h+ h- a"], "nfinal": 3,
                      "param_card_patches": _THDM, "pdg_ids": [11, -11, 37, -37, 22],
                      "m_finals": [NONSM_POINT["mhc"], NONSM_POINT["mhc"], 0.0]},
    "thdm_uux_h2h3g": {"model": "2HDM", "mg5_generate": ["generate u u~ > h2 h3 g"], "nfinal": 3,
                       "param_card_patches": _THDM_Q, "pdg_ids": [2, -2, 35, 36, 21],
                       "m_finals": [NONSM_POINT["mh2"], NONSM_POINT["mh3"], 0.0]},
    # validation only (tools/nonsm_validate.py): the same models with every non-SM coupling at zero
    "zp0_ee_mumu":    {"model": "DMsimp_s_spin1", "mg5_generate": ["generate e+ e- > mu+ mu- QED<=2 DMV<=2"],
                       "nfinal": 2, "param_card_patches": _ZP_OFF, "pdg_ids": [11, -11, -13, 13], "m_final": 0.0},
    "smeft0_ee_mumu": {"model": "SMEFTatNLO-LO", "mg5_generate": ["generate e+ e- > mu+ mu- QED<=2 NP<=1"],
                       "nfinal": 2, "param_card_patches": dict(_SMEFT0), "pdg_ids": [11, -11, -13, 13], "m_final": 0.0},
    "smeft0_ee_ttbar": {"model": "SMEFTatNLO-LO", "mg5_generate": ["generate e+ e- > t t~ QED<=2 NP<=1"],
                        "nfinal": 2, "param_card_patches": dict(_SMEFT0), "pdg_ids": [11, -11, 6, -6],
                        "m_final": LOCKED_MT},
}
for _n, _e in NONSM.items():
    assert _n not in PROCESSES, _n
    PROCESSES[_n] = dict(_e, run_card_patches=dict(_LEP))


def _load_catalog_v2():
    """Append the generated catalog_v2 tree and one-loop entries (recipes/gen_catalog_v2.py
    --write). Generated entries never override a hand-written one."""
    import yaml as _yaml
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes", "catalog_v2_processes.yaml")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        d = _yaml.safe_load(f) or {}
    n = 0
    for name, entry in (d.get("processes") or {}).items():
        if name in PROCESSES:
            hw = PROCESSES[name].get("mg5_generate", [None])[0]
            if entry.get("mg5_generate", [None])[0] != hw:
                raise KeyError(f"catalog_v2 entry {name} collides with a hand-written one ({hw!r} vs {entry.get('mg5_generate')})")
            continue
        PROCESSES[name] = dict(entry, _v2=True); n += 1   # tagged: the generator must not treat these as hand-written
    return n

CATALOG_V2_ENTRIES = _load_catalog_v2()


def _load_catalog_v3():
    """Append the catalog_v3 census trees (recipes/gen_catalog_v3.py --write; D98), tagged _v3. Names are v3_*, so none
    may collide with an existing entry."""
    import yaml as _yaml
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes", "catalog_v3_processes.yaml")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        d = _yaml.load(f, Loader=getattr(_yaml, "CSafeLoader", _yaml.SafeLoader)) or {}
    n = 0
    loops = {}
    # stage 2: the one-loop and loop-induced entries; the sign transfer's extension (gen_catalog_v3.py --write-loops-signext)
    for lf in ("catalog_v3_loops.yaml", "catalog_v3_loops_signext.yaml"):
        lp = os.path.join(os.path.dirname(path), lf)
        if os.path.exists(lp):
            loops.update((_yaml.load(open(lp), Loader=getattr(_yaml, "CSafeLoader", _yaml.SafeLoader)) or {}).get("processes", {}))
    for name, entry in list((d.get("processes") or {}).items()) + list(loops.items()):
        if name in PROCESSES:
            raise KeyError(f"catalog_v3 entry {name} collides with an existing process")
        PROCESSES[name] = dict(entry, _v3=True); n += 1
    return n

CATALOG_V3_ENTRIES = _load_catalog_v3()


def _load_catalog_v4():
    """Append the catalog_v4 entries (recipes/gen_catalog_v4.py --write; D130), tagged _v4: the trees of
    recipes/catalog_v4_processes.yaml and the one-loop and loop-induced entries of recipes/catalog_v4_loops.yaml. Names
    are v4_*, so none may collide with an existing entry."""
    import yaml as _yaml
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes")
    n = 0
    # the one-loop sign extension (gen_catalog_v4.py --write-loops-signext) too
    for f in ("catalog_v4_processes.yaml", "catalog_v4_loops.yaml", "catalog_v4_loops_signext.yaml"):
        path = os.path.join(here, f)
        if not os.path.exists(path):
            continue
        d = _yaml.load(open(path), Loader=getattr(_yaml, "CSafeLoader", _yaml.SafeLoader)) or {}
        for name, entry in (d.get("processes") or {}).items():
            if name in PROCESSES:
                raise KeyError(f"catalog_v4 entry {name} collides with an existing process")
            PROCESSES[name] = dict(entry, _v4=True); n += 1
    return n

CATALOG_V4_ENTRIES = _load_catalog_v4()


def _load_catalog_daniel():
    """Append the arXiv:2606.23791 (LLM4LHC) catalog (recipes/gen_catalog_daniel.py): pp processes
    whose events come from MadEvent with their cards (kind ``lhe_pp``; generation in the LHE_PP
    section below). Every entry is named dc_<their folder>, so none may collide with another."""
    import yaml as _yaml
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes", "catalog_daniel_processes.yaml")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        d = _yaml.load(f, Loader=getattr(_yaml, "CSafeLoader", _yaml.SafeLoader)) or {}
    lhe = dict(d.get("lhe") or {})
    n = 0
    for name, entry in (d.get("processes") or {}).items():
        if name in PROCESSES:
            raise KeyError(f"catalog_daniel entry {name} collides with an existing process")
        PROCESSES[name] = dict(entry, lhe=lhe); n += 1
    return n

CATALOG_DANIEL_ENTRIES = _load_catalog_daniel()


def _load_catalog_2to4():
    """Append the 2->4 probes (recipes/gen_catalog_2to4.py, D90): catalog 2->3 training processes plus one gluon or
    photon, generated with the catalog's MadEvent cards (kind lhe_pp), named d4_<parent folder>_<g|a>."""
    import yaml as _yaml
    import glob as _glob
    n = 0
    # catalog_2to4_processes.yaml and its harder-cut copies (catalog_2to4c<cut>_processes.yaml, gen_catalog_2to4.py --cut)
    for path in sorted(_glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes",
                                               "catalog_2to4*_processes.yaml"))
                       + _glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes",
                                                "catalog_v3_pp_processes.yaml"))
                       + _glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes",
                                                "catalog_v4_pp_processes.yaml"))
                       # catalog_v4's default-sampled share and e+e- MadEvent test pools (recipes/gen_catalog_v4_default.py)
                       + _glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes",
                                                "catalog_v4_ee_processes.yaml"))
                       + _glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes",
                                                "catalog_v4_default_processes.yaml"))):
        with open(path) as f:
            d = _yaml.load(f, Loader=getattr(_yaml, "CSafeLoader", _yaml.SafeLoader)) or {}
        lhe = dict(d.get("lhe") or {})
        for name, entry in (d.get("processes") or {}).items():
            if name in PROCESSES:
                raise KeyError(f"catalog_2to4 entry {name} collides with an existing process")
            # an entry's own lhe keys win (its param card)
            PROCESSES[name] = dict(entry, lhe={**lhe, **(entry.get("lhe") or {})}); n += 1
    return n

CATALOG_2TO4_ENTRIES = _load_catalog_2to4()


def is_lhe_pp(process):
    """True for a process whose events are MadEvent pp events (catalog_daniel), not our samplers."""
    return PROCESSES.get(process, {}).get("kind") == "lhe_pp"

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


def repatch_standalone_param_cards(standalone_dir, config):
    """Re-apply config's param_card_patches to an ALREADY-compiled standalone's cards.

    Standalone dirs are keyed by process name, compiled once, and the runtime driver
    reads the param_card at generation time. When a name-keyed standalone is reused
    across recipe versions with DIFFERENT physics — e.g. a mass scan whose M_Z/M_T
    values changed — the compiled backend is fine but its param_card still holds the
    FIRST-build masses. ensure_backend() calls this on every reuse so the card always
    reflects THIS dataset's physics (a mass patch that isn't re-applied silently
    generates at the stale mass — NaN couplings if it pushed M_Z below the on-shell
    root). No-op when there are no patches (base processes)."""
    patches = config.get("param_card_patches", {})
    if not patches:
        return
    for pc in (f"{standalone_dir}/Cards/param_card.dat",
               f"{fortran_dir_for(standalone_dir)}/Cards/param_card.dat"):
        if os.path.exists(pc):
            patch_param_card_slha(pc, patches)

def generate_mg5_process(process_name, config, events=None):
    """Run MadGraph to generate the standalone outputs, plus the MadEvent dir:
       - <proc>_standalone     : standalone_cpp (C++ driver, fixed α_s) — fallback
       - <proc>_standalone_py  : Fortran standalone (matrix2py, per-event α_s) — preferred
       - <proc>_events         : MadEvent output, read only by the fixed-energy LHE mode of main()
                                 (configure_cards / run_event_generation / the LHE reader; it passes
                                 events=True). The recipe path (datagen.ensure_backend,
                                 generate_from_recipe) reads the standalone alone, so a catalog_v4
                                 entry (`_v4`) builds no events dir (~51 MB each); events=None keeps
                                 every other entry's (catalog_v3's) behaviour, the events dir included.
    """
    if events is None:
        events = not config.get("_v4")
    events_dir    = f"{WORK_DIR}/{process_name}_events"
    standalone_dir = f"{WORK_DIR}/{process_name}_standalone"
    fortran_dir    = fortran_dir_for(standalone_dir)
    generate_cmds = "\n".join(config["mg5_generate"])
    if config.get("model"):     # a non-SM UFO model (NONSM); the sm default otherwise
        generate_cmds = f"import model {config['model']}\n" + generate_cmds

    # Generation is complete only when the standalone tree is: a run killed mid-output
    # (the 432-way parallel prebuild that died on fork) leaves a partial standalone that
    # would otherwise be "skipped" and then fail to compile forever. The partial dirs are
    # this function's own output; an events dir is removed with them only when it is
    # rebuilt here (events=True), else it is left as it is.
    complete = bool(os.path.exists(f"{standalone_dir}/Cards/param_card.dat")
                    and glob.glob(f"{standalone_dir}/SubProcesses/P*/CPPProcess.cc"))
    partial = [d for d in (standalone_dir, fortran_dir) + ((events_dir,) if events else ()) if os.path.isdir(d)]
    if not complete and partial:
        print(f"[MG5] {process_name}: incomplete output from an interrupted run, regenerating.")
        for d in partial:
            shutil.rmtree(d)
    need_events = events and not os.path.exists(events_dir)
    if not complete or need_events:
        print(f"\n[MG5] Generating process directories for {process_name}...")
        # The Fortran standalone (matrix2py) is only emitted when opted in — by
        # default the C++ driver + analytic α_s rescale covers everything.
        fortran_out = (f"output standalone {fortran_dir}\n"
                       if os.environ.get("MG5_USE_MATRIX2PY") and not complete else "")
        mg5_script = f"""
{generate_cmds}
{f"output {events_dir}" if need_events else ""}
{"" if complete else f"output standalone_cpp {standalone_dir}"}
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
                # MadGraph's Fortran reader (Source/MODEL/lha_read.f) takes a width only from a line whose first five
                # characters are DECAY: an indented DECAY line is skipped and the compiled default width used instead
                lead = "" if toks[0].upper() == "DECAY" else " "
                out.append(lead + " ".join(toks) + " #" + comment)
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

# Binary name of the pipe-mode C++ driver. Versioned because the stdin protocol is
# part of the ABI: v2 reads "<4n momenta> <aS>" per line (per-event alpha_s) and its
# wrappers take (double*, double). A pre-v2 "driver" left in a standalone must never
# be picked up by detect_compiled_backend, or it would misread every line after the
# first and label the pool with garbage.
CPP_DRIVER_NAME = "driver_v2"


def parameters_class_name(standalone_dir):
    """Model parameter class of a standalone_cpp output (Parameters_sm, ...)."""
    heads = sorted(glob.glob(f"{standalone_dir}/src/Parameters_*.h"))
    if not heads:
        raise FileNotFoundError(f"no src/Parameters_*.h under {standalone_dir}")
    return os.path.splitext(os.path.basename(heads[0]))[0]


def write_wrapper(standalone_dir, subproc_dir, param_card_path, nparticles):
    """Write a wrapper .cc for one subprocess, accepting nparticles momenta."""
    suffix     = subproc_dir
    func_name  = f"get_ME2_{suffix}"
    init_name  = f"init_{suffix}"
    obj_name   = f"process_{suffix}"
    init_flag  = f"initialized_{suffix}"
    class_name = get_class_name(standalone_dir, subproc_dir)
    params_class = parameters_class_name(standalone_dir)

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

// p_flat: flat array of {nparticles} * 4 doubles in MadGraph slot order, (E,px,py,pz)
// per particle; aS: the strong coupling for this event (sigmaKin recomputes the
// dependent couplings from it on every call).
double {func_name}(double* p_flat, double aS) {{
    {init_name}();
    {params_class}::getInstance()->aS = aS;
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
        f"double get_ME2_{s}(double*, double);"
        for s in suffixes
    )

    # Accumulate sum
    sum_lines = "    double total = 0.0;\n"
    for s in suffixes:
        sum_lines += f"    total += get_ME2_{s}(p, aS);\n"

    driver = f"""#include <iostream>
#include <cstdio>

{forward_decls}

int main() {{
    // Disable C stdio sync for maximum throughput in pipe mode
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    double p[{n_doubles}];
    double aS;
    while (true) {{
        for (int i = 0; i < {n_doubles}; ++i) {{
            if (!(std::cin >> p[i])) goto done;
        }}
        if (!(std::cin >> aS)) goto done;
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
    params_cls = parameters_class_name(standalone_dir)   # Parameters_sm, or the UFO model's
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
            f"src/{params_cls}.cc", f"src/{params_cls.replace('Parameters_', 'HelAmps_', 1)}.cc", "src/read_slha.cc",
            "-o", f"libME_{suffix}.so"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=standalone_dir)
        if result.returncode != 0:
            print(f"  [ERROR] Compilation failed:\n{result.stderr}")
            sys.exit(1)

    print(f"  [CXX] Linking driver...")
    lib_flags = [f"-lME_{s}" for s in suffixes]
    cmd = ["g++", "-O2", "driver.cpp", "-L.", *lib_flags, "-Wl,-rpath,.", "-o", CPP_DRIVER_NAME]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=standalone_dir)
    if result.returncode != 0:
        print(f"  [ERROR] Driver linking failed:\n{result.stderr}")
        sys.exit(1)

    print(f"  [CXX] Driver compiled successfully.")
    return f"{standalone_dir}/{CPP_DRIVER_NAME}"

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
    driver_bin = f"{standalone_dir}/{CPP_DRIVER_NAME}" \
        if os.path.exists(f"{standalone_dir}/{CPP_DRIVER_NAME}") else None
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

    def compute(self, events, perm=None, alphas=None):
        """
        Send momenta for a list of events and return array of |M|² values.
        events: list of (momenta_array, pdg_ids) tuples, momenta (nparticles, 4)
                in STORED row order.
        perm:   row -> MadGraph slot permutation (row_to_slot_perm); None = rows
                already in slot order.
        alphas: per-event alpha_s (array) or one value; None = the standalone's
                param-card value (fixed coupling, as the old driver did).
        """
        if alphas is None:
            alphas = read_alphas_from_param_card(f"{self.standalone_dir}/Cards/param_card.dat")
        alphas = np.broadcast_to(np.asarray(alphas, dtype=np.float64), (len(events),))
        amps = []
        stdin  = self._proc.stdin
        stdout = self._proc.stdout

        for (momenta, _), a_s in zip(events, alphas):
            if perm is not None:
                momenta = momenta[perm]
            # Flatten: E0 px0 py0 pz0  E1 px1 py1 pz1  ...  aS
            line = " ".join(f"{x:.15e}" for x in momenta.flatten()) + f" {a_s:.15e}"
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

# catalog_v4 (D130, the user 2026-10-25: "yes to 1 to 3, yes to electroweak production and the smaller fixes, no to the
# decayed final states"). Its entries (recipes/gen_catalog_v4.py) carry these as their own fields, so nothing above
# changes for any earlier catalog:
#   cuts        hadronic beams: pT > 10 GeV, |eta| < 5, Delta R > 0.4, no pair-mass cut (m_min < 0); e+e- beams:
#               |cos theta| < 0.99, the other cuts as FIDUCIAL_CUTS; the b and the tau cut like the other legs
#               (cut_pdgs; the b is massless in the five-flavour scheme, so only the tau needs it)
#   sqrt(s)     from the threshold the masses and the cuts give (V4_SQRTS_LO) to 6 TeV, log-uniform (sqrts_measure)
#   forward     f_fwd of each pool from the forward component (rotate_forward), fwd_cand of the flat part's
#               candidates turned forward: the coverage check's settings (analysis/catalog_v3/coverage_check.py
#               FWD_SHARE, FWD_CAND), open (no decision behind the values)
#   top         MT = ymt = 172.5 in every entry, internal tops too, WT 1.4915 (V4_WT): the width every catalog_v4 pool
#               built before the DECAY-column fix (7b00eb21) actually has, MadGraph having ignored the indented 1.480327
#               line (the user 2026-10-26, "do 1.4915 for all"); the heavy-quark layer keeps top_width_lo at its mass
#   b           massless, beams and finals: MadGraph's sm-no_b_mass (the five-flavour scheme, no b Yukawa)
V4_COS_ETA5 = float(np.tanh(5.0))      # |eta| < 5 is |cos theta| < tanh 5 for any particle
V4_CUTS = {"hadronic": {"pt_min": 10.0, "cos_max": V4_COS_ETA5, "dr_min": 0.4, "m_min": -1.0, "cut_pdgs": [5, 15]},
           "ee":       {"pt_min": 10.0, "cos_max": 0.99, "dr_min": 0.4, "m_min": 10.0, "cut_pdgs": [5, 15]}}
V4_SQRTS_MAX = 6000.0
V4_SQRTS_FLOOR = 10.0      # GeV: the lower edge of a class with neither a cut leg nor a mass (e+e- > nu nu~), open
V4_SAMPLING = {"mode": "mixture", "f_fwd": 0.15, "fwd_cand": 1.0 / 3.0}
V4_MODEL = "sm-no_b_mass"
V4_MT = 172.5


def top_width_lo(m, mw=80.419, gf=1.16639e-5):
    """LO Gamma(t -> b W+) with a massless b and V_tb = 1: G_F m^3 / (8 pi sqrt 2) (1 - x)^2 (1 + 2x), x = M_W^2 / m^2;
    the top's only LO decay with a diagonal CKM, so the total width; 0 at or below M_W (catalog_v4's width rule for the
    heavy-quark layer; the physical top takes V4_WT)"""
    if m <= mw:
        return 0.0
    x = mw ** 2 / m ** 2
    return float(gf * m ** 3 / (8.0 * np.pi * np.sqrt(2.0)) * (1.0 - x) ** 2 * (1.0 + 2.0 * x))


V4_WT = 1.4915         # the user 2026-10-26, "do 1.4915 for all" (top_width_lo(172.5) = 1.480327 is not used)
V4_TOP_PATCHES = {"MT": V4_MT, "ymt": V4_MT, "WT": V4_WT}


def v4_cut_kind(pdg_ids):
    return "ee" if abs(int(pdg_ids[0])) in (11, 13) else "hadronic"


def v4_sqrts_lo(pdg_ids, m_finals, cuts=None):
    """catalog_v4's lower sqrt(s) edge (D130: "from threshold", the 1.05 strip and the 25 GeV floor dropped, the IR
    regulators the cuts give kept): the least sqrt(s) at which the masses and the pT cut allow an event, rounded up
    to the GeV. Cut legs need E >= sqrt(m^2 + pT_min^2); with one cut leg the rest recoils against its pT, so it needs
    E >= sqrt(M_rest^2 + pT_min^2), M_rest the sum of the other masses; uncut legs need their mass. A class with
    neither a cut leg nor a mass takes V4_SQRTS_FLOOR."""
    cuts = cuts or V4_CUTS[v4_cut_kind(pdg_ids)]
    extra = {abs(int(p)) for p in cuts.get("cut_pdgs") or ()}
    pt = float(cuts["pt_min"])
    cut, rest = [], []
    for p, m in zip(pdg_ids[2:], m_finals):
        (cut if abs(int(p)) not in _NEUTRINOS and (float(m) == 0.0 or abs(int(p)) in extra) else rest).append(float(m))
    if len(cut) == 1:
        lo = np.hypot(cut[0], pt) + np.hypot(sum(rest), pt)
    else:
        lo = sum(np.hypot(m, pt) for m in cut) + sum(rest)
    if lo <= 0:
        lo = V4_SQRTS_FLOOR
    return float(np.ceil(lo - 1e-9))


def _cut_key(cuts):
    """Identity-bearing view of the active cuts (for recipe_id). `cut_pdgs` (catalog_v4: the massive finals cut like
    the massless ones) enters only when set, so every earlier recipe_id is unchanged."""
    key = {k: float(cuts[k]) for k in ("pt_min", "cos_max", "dr_min", "m_min")}
    if cuts.get("cut_pdgs"):
        key["cut_pdgs"] = sorted(int(abs(p)) for p in cuts["cut_pdgs"])
    return key


def fiducial_pass_mask(P, m_finals, pdg_ids, cuts, n_initial=2):
    """Boolean mask (N,) of events passing the fiducial cuts. P is (N, npart, 4).
    Cuts hit only massless visible finals (m_final==0, |pdg|∉{12,14,16}): single-
    particle pt>pt_min & |cosθ|<cos_max; pairwise (among those) ΔR>dr_min & m>m_min.
    A process with no such particle (all-massive / all-neutrino finals) passes all.
    cuts["cut_pdgs"] (catalog_v4: the tau, and the b where it is massive) adds the finals of those
    |pdg| to the cut legs whatever their mass; absent, the rule above is the whole rule."""
    import itertools
    N = P.shape[0]
    m_finals = np.asarray(m_finals, float)
    fin = list(range(n_initial, n_initial + len(m_finals)))
    extra = {abs(int(p)) for p in (cuts.get("cut_pdgs") or ())}
    slots = [fin[i] for i in range(len(m_finals))
             if abs(int(pdg_ids[fin[i]])) not in _NEUTRINOS
             and (m_finals[i] == 0.0 or abs(int(pdg_ids[fin[i]])) in extra)]
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


# Masses of the beams as the sm card has them: a b beam is on its 4.7 GeV shell, like a b in the final state (the only
# massive beam the catalogs generate here; light quarks, leptons and gluons are massless). Each beam carries
# E_i = (s + m_i^2 - m_j^2) / (2 sqrt s) and |p_z| = lambda^(1/2)(s, m_1^2, m_2^2) / (2 sqrt s), beam- along +z; with
# both beams massless the arrays are the old (E, 0, 0, +-E) ones bit for bit.
BEAM_MASS = {5: 4.7}


def _beams(sqrts, pdg_ids, beam_mass=None):
    """(n, 2, 4) beam momenta in the c.o.m. frame for beams pdg_ids[:2]; beam_mass {|pdg|: GeV} in place of BEAM_MASS
    (an entry's `beam_masses`: catalog_v4's five-flavour scheme has every beam massless)"""
    sqrts = np.asarray(sqrts, float); z = np.zeros(len(sqrts))
    bm = BEAM_MASS if beam_mass is None else beam_mass
    m1, m2 = (bm.get(abs(int(p)), 0.0) for p in pdg_ids[:2])
    if m1 == 0.0 and m2 == 0.0:
        E = sqrts / 2.0
        return np.stack([np.stack([E, z, z, E], 1), np.stack([E, z, z, -E], 1)], axis=1)
    s = sqrts ** 2
    E1 = (s + m1 ** 2 - m2 ** 2) / (2.0 * sqrts); E2 = sqrts - E1
    p = np.sqrt(np.maximum((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2), 0.0)) / (2.0 * sqrts)
    return np.stack([np.stack([E1, z, z, p], 1), np.stack([E2, z, z, -p], 1)], axis=1)


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
        p1 = beam-  (E_1, 0, 0, +p)   (_beams: on the beams' mass shells; E, +-E when massless)
        p2 = beam+  (E_2, 0, 0, -p)
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
        E3     = (s + m3 ** 2 - m4 ** 2) / (2.0 * sqrts)
        E4     = sqrts - E3
        p_mag  = np.sqrt(np.maximum(E3 ** 2 - m3 ** 2, 0.0))
        sin_t  = np.sqrt(1.0 - cos_t ** 2)
        px = p_mag * sin_t * np.cos(phi)
        py = p_mag * sin_t * np.sin(phi)
        pz = p_mag * cos_t
        P = np.concatenate([_beams(sqrts, pdg_arr), np.stack([
            np.stack([E3,     px,    py,    pz],     axis=1),    # X
            np.stack([E4,    -px,   -py,   -pz],     axis=1),    # Xbar
        ], axis=1)], axis=1)                                    # (nb, 4, 4): beam-, beam+, X, Xbar
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


def sample_nbody_phase_space(n_events, sqrts_min, sqrts_max, m_finals, pdg_ids, rng=None, cuts=None,
                             sqrts_measure="uniform", beam_mass=None):
    """
    Sample n-body phase space using RAMBO for a range of CM energies.

    Uses fully vectorized batch RAMBO; no Python loop over events.
    The initial-state convention (e- row 0, e+ row 1) matches CPPProcess.
    With ``cuts`` set, oversamples and keeps only events in the fiducial region.

    sqrts_measure "log" draws sqrt(s) log-uniform (catalog_v4); beam_mass as _beams.

    Returns (events, sqrts_arr) where events is a list of
    (momenta_array (nparticles x 4), pdg_ids_array).
    """
    if rng is None:
        rng = np.random.default_rng()

    masses   = np.asarray(m_finals, dtype=float)
    nfinal   = len(masses)
    all_zero = np.all(masses == 0.0)
    pdg_arr  = np.array(pdg_ids, dtype=int)

    sq_draw = _sqrts_sampler(sqrts_min, sqrts_max, sqrts_measure, rng)

    def draw(nb):
        sqrts   = sq_draw(nb)
        p_ml    = _rambo_massless_batch(sqrts, nfinal, rng)          # (nb, nfinal, 4)
        p_final = p_ml if all_zero else _rambo_massive_batch(sqrts, masses, p_ml)
        p_init  = _beams(sqrts, pdg_arr, beam_mass)                  # (nb, 2, 4)
        return np.concatenate([p_init, p_final], axis=1), sqrts     # (nb, nparticles, 4)

    P, sqrts = _collect_with_cuts(draw, n_events, m_finals, pdg_arr, cuts)
    events = [(P[i], pdg_arr) for i in range(n_events)]
    return events, sqrts


# =============================================================================
# SAMPLING POLICY (docs/results.tex, coverage vs. emphasis): a dataset is a MIXTURE of
#   (1-f)*N events drawn the legacy way (sqrt s uniform in the window x RAMBO / isotropic
#           angles under the fiducial cuts): the bulk;
#   f*N events kept FLAT IN log|M|^2 from an oversampled, labelled candidate pool, thinned
#           without replacement: equal training density per amplitude decade, so every
#           amplitude extreme (s-channel pole, soft/collinear corner, threshold) is covered
#           by generation, not by resampling what the bulk happens to contain.
# The candidate pool reaches the extremes through (a) the IR-democratic 1->2 splitter for
# multi-leg finals (log-uniform intermediate masses) mixed with RAMBO, and (b) an adaptive
# sqrt(s) proposal: a first pass of candidates shows where log|M|^2 varies steeply in sqrt(s)
# (a pole, a threshold) and the second pass concentrates candidates there. Nothing here
# knows which process has which resonance; only |M|^2 is read. f ~ 0.25-0.5 is the measured
# interior optimum (pure flat starves the bulk, pure uniform starves the pole).
# `mode: flat` is the legacy uniform-only sampling (keeps old recipe ids).
# =============================================================================
DEFAULT_SAMPLING = {
    "mode":         "flat",  # legacy uniform-sqrt(s) sampling; a recipe opts into "mixture"
    "f_flat":       0.35,    # fraction of the dataset kept flat in log|M|^2
    "oversample":   6,       # candidates labelled per flat event kept
    "democratic":   0.5,     # share of multi-leg candidates from the IR-democratic splitter
    "y_lo":         1e-6,    # deepest intermediate invariant (units of s) the splitter reaches
    "sqrts_adapt":  True,    # second candidate pass concentrated where log|M|^2 varies in sqrt(s)
    "n_sqrts_bins": 200,
}


def sampling_policy(cfg, role=None):
    """The effective sampling dict of a catalog entry (defaults filled in). Every role
    samples the same distribution: a shaped (mixture) entry shapes its val/test pools
    exactly like its train pool, so the loss that is measured is the loss that is
    trained (until this change val/test were forced flat, and the model was scored on
    a measure it never trained on). One-loop (MadLoop) entries are never shaped."""
    pol = dict(DEFAULT_SAMPLING); pol.update(cfg.get("sampling") or {})
    if cfg.get("kind") == "virt" or float(pol.get("f_flat", 0)) <= 0:
        pol["mode"] = "flat"
    if float(pol.get("f_default", 0)) > 0:
        # the default-sampled share (D178): that fraction of the pool is the generator's own unweighted events, read
        # from the MadEvent pools of the lhe entry `default_source` (default_share_momenta); only in the mixture
        src = pol.get("default_source")
        if pol["mode"] != "mixture":
            raise ValueError(f"f_default is part of the mixture policy (mode: mixture), got {pol['mode']}")
        if not src or PROCESSES.get(src, {}).get("kind") != "lhe_pp":
            raise ValueError(f"f_default {pol['f_default']} needs default_source, an lhe_pp entry (got {src!r})")
    return pol


# =============================================================================
# DEFAULT-SAMPLED SHARE (catalog_coverage, the user's priority of 2026-10-29, D178)
# =============================================================================
# A shaped entry with sampling.f_default > 0 takes that share of each pool from MadEvent unweighted events, as a user
# makes them, instead of the RAMBO bulk: the events of the lhe entry `default_source` (recipes/gen_catalog_v4_default.py,
# dsh_<x>: the class's generate string, model and cards, 13 TeV pp with Daniel's run card or e+e- at fixed energies),
# preceded by the events already generated for the class where `default_source` names a `reuse` entry (its pp13v4 /
# pp13 / ee4 pool of the same role), never repeating them (the dsh entry's own iseeds). The momenta are boosted to the
# partonic rest frame (every catalog pool's frame) and labelled by the entry's own backend with the rest of the pool,
# so the default share carries exactly the pool's |M|^2 function. Only events inside the pool's own sqrt(s_hat) window
# and fiducial cuts (in the partonic frame, as every other event of the pool) are kept, so no 13 TeV event above the
# 6 TeV training edge leaks the held-out energy probes into training; the kept events, in order, are what a pool reads.
# A role's pool reads the first rows: chunk c of a chunked pool reads [round(f N_<c), round(f N_<=c)) with N the
# cumulative chunk counts (datagen.chunk_tasks), so a chunked pool reads exactly [0, round(f n)) as an unchunked one,
# and a smaller f_default reads a prefix of a larger one's events and needs no generation of its own.

def to_partonic_frame(P):
    """P (n, k, 4) with the beams in rows 0 and 1 -> the same events in the beams' rest frame (a boost along the
    beams' total momentum; the identity for e+e- without ISR)."""
    P = np.asarray(P, dtype=np.float64)
    n, k = P.shape[:2]
    Q = P[:, 0] + P[:, 1]
    Qinv = np.concatenate([Q[:, :1], -Q[:, 1:]], axis=1)          # boosting from the rest of Qinv undoes Q's motion
    out = _boost_from_rest(P.reshape(n * k, 4), np.repeat(Qinv, k, axis=0)).reshape(n, k, 4)
    return out


def default_share_pools(source, role, seed):
    """[(process, recipe, path)] of the pools the default share of `role` reads, in order: the `reuse` entry's pool
    (if any) then the source's own (dsh) pool, each at its full block of the role."""
    import datagen
    out = []
    for name in ([PROCESSES[source]["reuse"]] if PROCESSES[source].get("reuse") else []) + [source]:
        lo, hi = lhe_layout(PROCESSES[name])[role]
        if hi - lo <= 0:
            continue
        rec = lhe_pp_recipe(name, 13000, 13000, hi - lo, role=role, seed=seed)
        out.append((name, rec, recipe_output_path(rec, datagen.dest_for_role(role))))
    return out


def default_share_identity(source, role, seed):
    """The default share's part of a mixture pool's identity: the recipe ids of the pools its rows come from."""
    return {"source": source, "frame": "partonic", "accept": "pool window and fiducial cuts",
            "pools": [{"process": n, "recipe_id": recipe_id(r)} for n, r, _ in default_share_pools(source, role, seed)]}


def default_share_momenta(source, role, seed, rows, accept=None):
    """Momenta (n, k, 4) in the partonic rest frame and sqrt(s_hat) (n,) of rows [a, b) of the default share of
    `role`: the reuse pool's rows then the source's, boosted to the partonic frame and, with `accept` = (sqrts_lo,
    sqrts_hi, m_finals, pdg_ids, cuts), only those inside the pool's window and cuts. Every pool must be built with its
    recipe id (a MadEvent CPU job: prebuild_recipes.py on recipes/catalog_v4_default_lhe*.yaml); a missing or stale one
    raises, so a mixture chunk never starts MadEvent."""
    a, b = int(rows[0]), int(rows[1])
    if b <= a:
        return np.zeros((0, 0, 4)), np.zeros(0)
    parts = []
    for name, rec, path in default_share_pools(source, role, seed):
        try:
            with open(path + ".recipe.json") as f:
                ok = json.load(f).get("recipe_id") == recipe_id(rec)
        except (FileNotFoundError, json.JSONDecodeError):
            ok = False
        if not ok or not os.path.exists(path):
            raise RuntimeError(f"default share of {source} ({role}): pool {path} missing or stale "
                               f"(recipe_id {recipe_id(rec)}); build the default-share lhe pools first")
        k = int(PROCESSES[name]["nfinal"]) + 2
        x = np.load(path, mmap_mode="r")
        parts.append(np.asarray(x[:, :4 * k], dtype=np.float64).reshape(len(x), k, 4))
    P = to_partonic_frame(np.concatenate(parts))
    Q = P[:, 0] + P[:, 1]
    sq = np.sqrt(np.clip(Q[:, 0] ** 2 - (Q[:, 1:] ** 2).sum(1), 0.0, None))
    n_all = len(P)
    if accept is not None:
        lo, hi, m_finals, pdg_ids, cuts = accept
        ok = (sq >= float(lo)) & (sq <= float(hi))
        if cuts:
            ok &= fiducial_pass_mask(P, np.asarray(m_finals, float), np.asarray(pdg_ids, int), cuts)
        P, sq = P[ok], sq[ok]
    if len(P) < b:
        raise RuntimeError(f"default share of {source} ({role}): rows [{a}, {b}) asked, its pools hold {len(P)} "
                           f"accepted of {n_all}; raise ACCEPT_MARGIN in recipes/gen_catalog_v4_default.py and rebuild")
    return P[a:b], sq[a:b]


def default_share_rows(f_default, counts):
    """[(a_c, b_c)] the default-share rows of each chunk of a pool split into `counts`: consecutive, cumulative
    rounding, so the chunks never read the same event and together read [0, round(f_default * n))."""
    cum = np.concatenate([[0], np.cumsum([int(c) for c in counts])])
    edge = [int(round(float(f_default) * int(x))) for x in cum]
    return [(edge[i], edge[i + 1]) for i in range(len(counts))]


def _sqrts_sampler(lo, hi, measure, rng):
    """n -> sqrt(s) draws on [lo, hi]: "uniform" (every catalog up to v3; the same random stream as before) or "log"
    (log-uniform, catalog_v4)"""
    lo, hi = float(lo), float(hi)
    if measure in (None, "uniform"):
        return lambda n: rng.uniform(lo, hi, n)
    if measure == "log":
        if lo <= 0:
            raise ValueError(f"log-uniform sqrt(s) needs a positive lower edge (got {lo})")
        a, b = np.log(lo), np.log(hi)
        return lambda n: np.exp(rng.uniform(a, b, n))
    raise ValueError(f"unknown sqrts_measure {measure!r}")


def rotate_forward(P, cos_cap, rng):
    """catalog_v4's forward component (D130; analysis/catalog_v3/coverage_check.py, which measured it): P (n, k, 4),
    beams in rows 0 and 1, the partonic rest frame. Per event one final particle at random is turned to an angle with
    1 - |cos theta| = exp(-u), u uniform in [0, log(1/(1 - cos_cap))], phi uniform, either hemisphere; the whole final
    state turns with it (one rotation), so momentum conservation and every mass shell hold. Nothing here knows which
    process has a t-channel pole."""
    n, k = P.shape[:2]
    j = rng.integers(2, k, n)
    p = P[np.arange(n), j, 1:]
    a = p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-300, None)
    u = rng.uniform(0.0, np.log(1.0 / (1.0 - cos_cap)), n)
    c = (1.0 - np.exp(-u)) * np.where(rng.random(n) < 0.5, 1.0, -1.0)
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    s = np.sqrt(np.clip(1.0 - c ** 2, 0.0, None))
    b = np.stack([s * np.cos(ph), s * np.sin(ph), c], 1)
    v = np.cross(a, b); sv = np.linalg.norm(v, axis=1); cc = (a * b).sum(1)
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = v[:, 2], -v[:, 1], v[:, 0]
    f = np.where(sv > 1e-12, (1.0 - cc) / np.clip(sv ** 2, 1e-300, None), 0.0)
    R = np.eye(3)[None] + K + f[:, None, None] * (K @ K)
    anti = (sv <= 1e-12) & (cc < 0)               # a already opposite b: a half turn about an axis normal to a
    if anti.any():
        e = np.where(np.abs(a[anti, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        w = np.cross(a[anti], e); w /= np.linalg.norm(w, axis=1, keepdims=True)
        R[anti] = 2.0 * w[:, :, None] * w[:, None, :] - np.eye(3)[None]
    Q = P.copy()
    Q[:, 2:, 1:] = np.einsum("nij,nkj->nki", R, P[:, 2:, 1:])
    return Q


def _random_dirs(n, rng):
    """n isotropic unit 3-vectors."""
    cos_t = rng.uniform(-1.0, 1.0, n); phi = rng.uniform(0.0, 2.0 * np.pi, n)
    sin_t = np.sqrt(np.clip(1.0 - cos_t ** 2, 0.0, None))
    return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)


def _boost_from_rest(qstar, Pblob):
    """Boost rest-frame 4-vectors qstar (n,4) into the lab, where the blob has lab
    4-momentum Pblob (n,4)."""
    E = Pblob[:, 0]; p3 = Pblob[:, 1:]
    M = np.sqrt(np.clip(E ** 2 - (p3 ** 2).sum(1), 1e-18, None)); gamma = E / M
    pmag = np.sqrt(np.clip((p3 ** 2).sum(1), 1e-30, None)); nhat = p3 / pmag[:, None]; beta = pmag / E
    qE = qstar[:, 0]; q3 = qstar[:, 1:]; ndotq = (nhat * q3).sum(1)
    lE = gamma * (qE + beta * ndotq)
    l3 = q3 + nhat * ((gamma - 1.0) * ndotq + gamma * beta * qE)[:, None]
    return np.concatenate([lE[:, None], l3], axis=1)


def _two_body(M, m_a, m_b, rng):
    """Two-body split of a blob of mass M (n,) into masses m_a, m_b in its rest frame."""
    n = len(M)
    Ea = (M ** 2 + m_a ** 2 - m_b ** 2) / (2.0 * M)
    lam = (M ** 2 - (m_a + m_b) ** 2) * (M ** 2 - (m_a - m_b) ** 2)
    pmag = np.sqrt(np.clip(lam, 0.0, None)) / (2.0 * M)
    d = _random_dirs(n, rng)
    qa = np.concatenate([Ea[:, None], pmag[:, None] * d], axis=1)
    Eb = np.sqrt(np.clip(pmag ** 2 + m_b ** 2, 0.0, None))
    qb = np.concatenate([Eb[:, None], -(pmag[:, None] * d)], axis=1)
    return qa, qb


def democratic_draw(nb, sqrts, masses, y_lo, rng):
    """IR-democratic N-body final state (analysis/divergences/gen_ir_democratic.py, the
    process-agnostic SARGE/HAAG core): recursive 1->2 splitting of the total invariant
    mass with the intermediate masses log-uniform, so soft and collinear corners are
    visited with equal weight per decade of invariant. Momentum conservation and
    on-shellness are exact by construction. Returns P (nb, N, 4) in the order of `masses`."""
    masses = np.asarray(masses, float); N = len(masses)
    Pout = np.zeros((nb, N, 4))
    blob = np.zeros((nb, 4)); blob[:, 0] = sqrts; M = sqrts.copy()
    suffix = np.concatenate([np.cumsum(masses[::-1])[::-1], [0.0]])
    Mfloor = sqrts * np.sqrt(y_lo)
    for k in range(N - 1):
        m_k = masses[k]; m_rest = suffix[k + 1]
        if k == N - 2:
            Mp = np.full(nb, masses[N - 1])          # the last blob IS the last particle
        else:
            Mhi = M - m_k
            Mlo = np.minimum(np.maximum(m_rest + Mfloor, m_rest + 1e-9), Mhi)
            lo2 = np.log(np.maximum(Mlo ** 2, 1e-18)); hi2 = np.log(np.maximum(Mhi ** 2, 1e-18))
            Mp = np.sqrt(np.exp(lo2 + rng.uniform(0.0, 1.0, nb) * (hi2 - lo2)))
        q_k, q_bp = _two_body(M, m_k, Mp, rng)
        Pout[:, k, :] = _boost_from_rest(q_k, blob)
        blob = _boost_from_rest(q_bp, blob); M = Mp
    Pout[:, N - 1, :] = blob
    return Pout


def _adaptive_bins(n, cap, per_bin=50, floor=20):
    """Bin count scaled with the sample size (>= per_bin entries per bin on average): a
    chunked pool with few candidates must not turn the adaptive proposal into noise."""
    return int(max(floor, min(cap, n // per_bin)))


def _sqrts_proposal_from_pass(sqrts_c, logm_c, lo, hi, n_bins, n_logm_bins=60):
    """Adaptive sqrt(s) proposal, targeting flatness directly: each first-pass candidate is
    weighted by 1 / (density of its log|M|^2 in the pass), i.e. by how RARE its amplitude
    value is, and the weights are accumulated per sqrt(s) bin. A resonance or a threshold
    is a narrow band of sqrt(s) holding rare (large or small) amplitudes and receives
    second-pass candidates in proportion; a floor keeps every bin populated. Returns
    (edges, probabilities)."""
    v_edges = np.linspace(logm_c.min(), logm_c.max() + 1e-12, n_logm_bins + 1)
    vb = np.clip(np.searchsorted(v_edges, logm_c, side="right") - 1, 0, n_logm_bins - 1)
    dens = np.bincount(vb, minlength=n_logm_bins).astype(float)
    w = 1.0 / dens[vb]
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, sqrts_c, side="right") - 1, 0, n_bins - 1)
    per_bin = np.bincount(idx, weights=w, minlength=n_bins)
    per_bin = per_bin + 0.1 * per_bin.mean()          # floor: no sqrt(s) bin starved
    return edges, per_bin / per_bin.sum()


def _draw_sqrts_from_bins(nb, edges, probs, rng):
    b = rng.choice(len(probs), size=nb, p=probs)
    return rng.uniform(edges[b], edges[b + 1])


def _flat_logm_select(logm, n_keep, rng, n_bins=60):
    """Indices of n_keep candidates thinned WITHOUT replacement toward a flat histogram in
    log|M|^2: weight ∝ 1 / (candidate density at that log|M|^2)."""
    idx_all = np.nonzero(np.isfinite(logm))[0]
    if len(idx_all) <= n_keep:
        return idx_all
    v = logm[idx_all]
    edges = np.linspace(v.min(), v.max() + 1e-12, n_bins + 1)
    b = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n_bins - 1)
    dens = np.bincount(b, minlength=n_bins).astype(float)
    w = 1.0 / dens[b]; w /= w.sum()
    return idx_all[rng.choice(len(idx_all), size=n_keep, replace=False, p=w)]


def _candidates(nb, sqrts_draw, m_finals, pdg_ids, rng, cuts, frac_dem, y_lo, frac_fwd=0.0, beam_mass=None):
    """nb candidate events with sqrt(s) from `sqrts_draw(n)`: 2->2 isotropic two-body; for
    n >= 3 a frac_dem share from the democratic splitter, the rest RAMBO. Fiducial cuts by
    rejection with top-up (the pipeline's _collect_with_cuts); the cuts bite the democratic
    draws hardest (they sit in the corners), so the realised democratic share of the
    survivors is below frac_dem. frac_fwd (catalog_v4) turns that share of the draws forward (rotate_forward, up to
    the cuts' |cos theta| edge); zero draws no extra random numbers, so every earlier pool keeps its stream."""
    m_finals = np.asarray(m_finals, float); N = len(m_finals); pdg = np.asarray(pdg_ids, int)
    all_zero = np.all(m_finals == 0.0)
    if frac_fwd > 0 and cuts is None:
        raise ValueError("the forward component needs the fiducial cuts (its edge is their |cos theta| bound)")
    def draw(n_draw):
        sq = np.asarray(sqrts_draw(n_draw), float)
        beams = _beams(sq, pdg, beam_mass)
        if N == 2:
            P = democratic_draw(n_draw, sq, m_finals, y_lo, rng)     # N=2: back-to-back, isotropic
        else:
            n_dem = int(round(frac_dem * n_draw)); parts = []
            if n_dem:
                parts.append(democratic_draw(n_dem, sq[:n_dem], m_finals, y_lo, rng))
            if n_draw - n_dem:
                p_ml = _rambo_massless_batch(sq[n_dem:], N, rng)
                parts.append(p_ml if all_zero else _rambo_massive_batch(sq[n_dem:], m_finals, p_ml))
            P = np.concatenate(parts, axis=0)
        P = np.concatenate([beams, P], axis=1)
        if frac_fwd > 0:
            r = rng.random(n_draw) < frac_fwd
            if r.any():
                P[r] = rotate_forward(P[r], float(cuts["cos_max"]), rng)
        return P, sq
    P, sq = _collect_with_cuts(draw, nb, m_finals, pdg, cuts)
    return [(P[i], pdg) for i in range(nb)], sq


def build_mixture_dataset(n_events, sqrts_min, sqrts_max, m_finals, pdg_ids, rng, cuts, pol, label,
                          sqrts_measure="uniform", beam_mass=None, default=None):
    """The mixture policy above. `label(events, sqrts) -> |M|^2` is the exact matrix
    element (the pipeline's backend). Returns (events, sqrts, me2) shuffled.
    catalog_v4 (D130) adds, all absent from every earlier policy (so their pools are unchanged): sqrts_measure "log"
    (the bulk, the forward part and the first candidate pass log-uniform in sqrt(s); the adaptive pass bins in
    log sqrt(s)), pol["f_fwd"] (that share of the dataset from the forward component, out of the bulk) and
    pol["fwd_cand"] (that share of the flat part's candidates turned forward, so the flat selection reaches the
    forward extremes too); beam_mass as _beams.
    `default` = (P (n_d, k, 4), sqrt(s_hat) (n_d,)): the default-sampled share (pol["f_default"], D178), MadEvent events
    already in the partonic frame (default_share_momenta), labelled here with the rest; they take n_d events out of the
    bulk. None (every earlier pool) draws exactly the random numbers it drew before."""
    lo, hi = float(sqrts_min), float(sqrts_max)
    f_fwd, fwd_cand = float(pol.get("f_fwd", 0.0)), float(pol.get("fwd_cand", 0.0))
    n_flat = int(round(pol["f_flat"] * n_events)); n_fwd = int(round(f_fwd * n_events))
    n_def = 0 if default is None else len(default[0])
    n_bulk = n_events - n_flat - n_fwd - n_def
    if default is not None:
        # the default share's rows round cumulatively over chunks (default_share_rows), the flat and forward shares
        # per chunk: when the shares fill the pool (f_default 0.50 with 0.35 + 0.15, no bulk left) or a chunk's
        # roundings overshoot it, the remainder goes to the forward share, so the shares sum to the chunk exactly.
        # The default=None path (every pool without a default share) is untouched.
        f_sum = float(pol["f_flat"]) + f_fwd + float(pol.get("f_default", 0.0))
        if n_bulk < 0 or abs(f_sum - 1.0) < 1e-9:
            n_fwd += n_bulk
            n_bulk = 0
            if n_fwd < 0:
                raise ValueError(f"mixture shares exceed the pool: flat {n_flat} + default {n_def} > {n_events}")
    if n_bulk < 0:
        raise ValueError(f"mixture shares exceed the pool: flat {n_flat} + forward {n_fwd} + default {n_def} > {n_events}")
    uni = _sqrts_sampler(lo, hi, sqrts_measure, rng)
    log_s = sqrts_measure == "log"
    if n_bulk:
        ev_b, sq_b = _candidates(n_bulk, uni, m_finals, pdg_ids, rng, cuts, 0.0, pol["y_lo"], 0.0, beam_mass)   # bulk
        me_b = np.asarray(label(ev_b, sq_b), float)
    else:   # no bulk left (f_default 0.50: flat + forward + default fill the pool); _candidates(0) cannot run
        ev_b, sq_b, me_b = [], np.zeros(0), np.zeros(0)
    if n_fwd:
        ev_w, sq_w = _candidates(n_fwd, uni, m_finals, pdg_ids, rng, cuts, 0.0, pol["y_lo"], 1.0, beam_mass)
        ev_b = ev_b + ev_w; sq_b = np.concatenate([sq_b, sq_w])
        me_b = np.concatenate([me_b, np.asarray(label(ev_w, sq_w), float)])
    if n_def:
        pdg = np.asarray(pdg_ids, int)
        ev_d = [(default[0][i], pdg) for i in range(n_def)]
        sq_d = np.asarray(default[1], float)
        ev_b = ev_b + ev_d; sq_b = np.concatenate([sq_b, sq_d])
        me_b = np.concatenate([me_b, np.asarray(label(ev_d, sq_d), float)])
    ev_f, sq_f, me_f = [], np.zeros(0), np.zeros(0)
    if n_flat:
        n_cand = max(int(pol["oversample"]) * n_flat, n_flat)
        n1 = n_cand // 2 if pol["sqrts_adapt"] else n_cand
        ev1, sq1 = _candidates(n1, uni, m_finals, pdg_ids, rng, cuts, pol["democratic"], pol["y_lo"], fwd_cand, beam_mass)
        me1 = np.asarray(label(ev1, sq1), float)
        cand, sq_c, me_c = ev1, sq1, me1
        if pol["sqrts_adapt"] and n_cand - n1 > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                lg1 = np.log(np.where(me1 > 0, me1, np.nan))
            ok = np.isfinite(lg1)
            nb_s = _adaptive_bins(int(ok.sum()), int(pol["n_sqrts_bins"]))
            # the proposal's variable: sqrt(s), or log sqrt(s) under the log measure (bins equal in log, as the draws)
            tr, itr = (np.log, np.exp) if log_s else ((lambda x: x), (lambda x: x))
            t_lo, t_hi = tr(lo), tr(hi)
            if ok.any():
                edges, probs = _sqrts_proposal_from_pass(tr(sq1[ok]), lg1[ok], t_lo, t_hi, nb_s, n_logm_bins=_adaptive_bins(int(ok.sum()), 60, 100))
            else:
                # no positive candidate (a signed target negative throughout, e.g. a synthetic one-loop pool): the
                # proposal stays uniform in sqrt(s); every pool that has one is unchanged
                edges, probs = np.linspace(t_lo, t_hi, nb_s + 1), np.full(nb_s, 1.0 / nb_s)
            ev2, sq2 = _candidates(n_cand - n1, lambda n: itr(_draw_sqrts_from_bins(n, edges, probs, rng)),
                                   m_finals, pdg_ids, rng, cuts, pol["democratic"], pol["y_lo"], fwd_cand, beam_mass)
            me2 = np.asarray(label(ev2, sq2), float)
            cand, sq_c, me_c = ev1 + ev2, np.concatenate([sq1, sq2]), np.concatenate([me1, me2])
        with np.errstate(divide="ignore", invalid="ignore"):
            logm = np.log(np.where(me_c > 0, me_c, np.nan))
        keep = _flat_logm_select(logm, n_flat, rng, n_bins=_adaptive_bins(len(cand), 60, 100))
        if len(keep) < n_flat:
            # too few finite candidates (non-positive |M|^2 is excluded from the flat part;
            # the bulk keeps whatever the measure gives): top up from the remaining candidates
            rest = np.setdiff1d(np.arange(len(cand)), keep)
            keep = np.concatenate([keep, rng.choice(rest, size=n_flat - len(keep), replace=False)])
        ev_f = [cand[i] for i in keep]; sq_f = sq_c[keep]; me_f = me_c[keep]
        print(f"  [SAMPLE] mixture: bulk {n_bulk:,} + flat-log|M|^2 {len(keep):,} kept from {len(cand):,} candidates"
              f" (adaptive sqrt s: {pol['sqrts_adapt']}, democratic share {pol['democratic']})")
    events = ev_b + ev_f
    sqrts = np.concatenate([sq_b, sq_f]); me2 = np.concatenate([me_b, me_f])
    assert len(events) == n_events, (len(events), n_events)
    order = rng.permutation(len(events))
    return [events[i] for i in order], sqrts[order], me2[order]


def build_dataset_variable_energy(n_events, sqrts_min, sqrts_max,
                                   standalone_dir, backend, subproc_dirs,
                                   driver_bin, config, output_file, rng=None, role=None,
                                   default_rows=None, default_seed=None):
    """
    Build a variable-√s LO amplitude dataset by direct phase-space sampling.
    Bypasses MadGraph event generation entirely.

    2→2 processes: analytic uniform phase space (sample_2to2_phase_space).
    n-body (nfinal>2):  RAMBO (sample_nbody_phase_space).

    α_s is evaluated per event at μ = √s for matrix2py backend.
    C++ pipe driver uses fixed α_s at the geometric-mean scale.

    `default_rows` = (a, b): the rows of the default-sampled share this pool (or chunk) reads (sampling f_default > 0;
    None reads [0, round(f_default * n_events))), from the default-share pools of seed `default_seed`.

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
    cuts       = dict(config.get("fiducial_cuts") or FIDUCIAL_CUTS) if FIDUCIAL_CUTS_ENABLED else None
    cut_msg    = f"  fiducial cuts {_cut_key(cuts)}" if cuts else "  (no cuts)"
    pol        = sampling_policy(config, role)
    # catalog_v4 (D130): the entry's sqrt(s) measure and beam masses; absent, uniform and BEAM_MASS as before
    measure    = config.get("sqrts_measure", "uniform")
    beam_mass  = config.get("beam_masses")
    if (pol.get("f_fwd") or pol.get("fwd_cand")) and pol["mode"] != "mixture":
        raise ValueError("the forward component is part of the mixture policy (mode: mixture)")
    synthetic  = backend == "synthetic"     # tools/synthetic_amplitudes.py labels the events
    if synthetic:
        from tools.synthetic_amplitudes import label_events
        syn_label = lambda ev, sq: label_events(config["syn_name"], ev)
    if pol["mode"] == "mixture" and backend not in ("cpp", "synthetic"):
        raise RuntimeError(f"sampling policy 'mixture' needs the C++ backend (got {backend}); "
                           f"unset MG5_USE_MATRIX2PY or set the recipe's sampling mode to flat")
    if pol["mode"] == "mixture":
        # shaped sampling (DEFAULT_SAMPLING): the backend labels candidates as we go
        m_list = list(config["m_finals"]) if "m_finals" in config else \
                 (list(config["m_final"]) if isinstance(config["m_final"], (list, tuple)) else [float(config["m_final"])] * nfinal)
        print(f"\n[DATA] Sampling {n_events:,} events  √s ∈ [{sqrts_min}, {sqrts_max}] GeV{cut_msg}"
              f"  policy=mixture f_flat={pol['f_flat']} oversample={pol['oversample']}")
        default = None
        if float(pol.get("f_default", 0)) > 0:
            rows = default_rows or (0, int(round(float(pol["f_default"]) * n_events)))
            default = default_share_momenta(pol["default_source"], role or "train", default_seed, rows,
                                            accept=(sqrts_min, sqrts_max, m_list, pdg_ids, cuts))
            print(f"  [SAMPLE] default share: rows [{rows[0]}, {rows[1]}) of {pol['default_source']} ({role})")
        if synthetic:
            events, sqrts_arr, total_me2 = build_mixture_dataset(
                n_events, sqrts_min, sqrts_max, m_list, pdg_ids, rng, cuts, pol, syn_label, measure, beam_mass, default)
        else:
            perm = row_to_slot_perm(pdg_ids, config["mg5_generate"])
            with CppDriverPipe(driver_bin, standalone_dir) as pipe:
                label = lambda ev, sq: pipe.compute(ev, perm=perm, alphas=compute_alphas(np.asarray(sq, float), alphas_mz=amz))
                events, sqrts_arr, total_me2 = build_mixture_dataset(
                    n_events, sqrts_min, sqrts_max, m_list, pdg_ids, rng, cuts, pol, label, measure, beam_mass, default)
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        mmap_path = output_file + ".mmap"
        mmap = np.lib.format.open_memmap(mmap_path, mode="w+", dtype=np.float64, shape=(n_events, ncols))
        momenta_flat = np.array([e[0].flatten() for e in events], dtype=np.float64)
        pdg_block    = np.tile(np.array(pdg_ids, dtype=np.float64), (n_events, 1))
        mmap[:] = np.concatenate([momenta_flat, pdg_block, np.asarray(total_me2, float).reshape(-1, 1)], axis=1)
        mmap.flush(); del mmap
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(mmap_path, output_file)
        w = np.asarray(total_me2, float)
        print(f"[DATA] Saved {n_events:,} events → {output_file}  |M|² ∈ [{w.min():.4e}, {w.max():.4e}]  neg: {(w < 0).sum()}")
        return

    if measure != "uniform" or beam_mass is not None:
        raise ValueError(f"{config.get('mg5_generate')}: sqrts_measure / beam_masses are implemented on the mixture "
                         f"path only (catalog_v4's policy)")
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
    perm = None if synthetic else row_to_slot_perm(pdg_ids, config["mg5_generate"])
    if synthetic:
        total_me2 = np.asarray(syn_label(events, sqrts_arr), dtype=np.float64)
    elif backend == "matrix2py":
        for subproc_dir in subproc_dirs:
            subproc_path = f"{standalone_dir}/SubProcesses/{subproc_dir}"
            if subproc_path in sys.path:
                sys.path.remove(subproc_path)
            sys.path.insert(0, subproc_path)
            if "matrix2py" in sys.modules:
                del sys.modules["matrix2py"]
            import matrix2py
            matrix2py.py_initialisemodel(param_card)

            # Stored rows -> MadGraph slot order (row_to_slot_perm); the stored
            # label of a row is the particle the ME sees, for any initial state.
            me2 = np.empty(n_events)
            for i, (momenta, _) in enumerate(events):
                alphas = compute_alphas(sqrts_arr[i], alphas_mz=amz)
                p_mg = momenta[perm]
                me2[i] = matrix2py.py_get_value(_invert_momenta(p_mg), alphas, -1)
                if (i + 1) % 100_000 == 0:
                    print(f"  [AMP] {i+1:,}/{n_events:,}", flush=True)

            total_me2 += me2
            print(f"    [AMP] subprocess {subproc_dir}: mean |M|² = {me2.mean():.4e}")
            sys.path.remove(subproc_path)
            del sys.modules["matrix2py"]

    elif backend == "cpp":
        # C++ pipe driver, per-event α_s(√s): the wrapper sets the model's aS before
        # sigmaKin, which recomputes the dependent couplings, so the result is exact
        # for any coupling structure -- pure α_sᵏ (where it equals the old analytic
        # rescale of a fixed-α_s evaluation) AND the mixed QCD+EW four-quark entries,
        # where a global rescale would have misweighted the EW/interference terms.
        alphas_ev = compute_alphas(sqrts_arr, alphas_mz=amz)
        print(f"  [AMP] per-event α_s(√s) ∈ [{alphas_ev.min():.4f}, {alphas_ev.max():.4f}]"
              f"  (rows -> MG5 slots via perm {perm})")
        with CppDriverPipe(driver_bin, standalone_dir) as pipe:
            total_me2 = np.asarray(pipe.compute(events, perm=perm, alphas=alphas_ev),
                                   dtype=np.float64)

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
    any valid sample of the spec is an equally good benchmark, so we do
    NOT want a different chunk policy to mint a second, physically-identical frozen
    dataset. Hence `n_chunks` is left out of the val/test identity entirely — one
    test set serves regardless of how it was sliced for generation."""
    cfg = register_synthetic(process) if is_synthetic(process) else PROCESSES[process]
    if cfg.get("kind") == "lhe_pp":
        # MadEvent pp events (catalog_daniel): a different generator, so a different identity
        return lhe_pp_recipe(process, sqrts_min, sqrts_max, n_events, role=role, seed=seed)
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
        "amp_orders":         order_vector(cfg),
        "per_event_alphas":   True,
        # Labelling convention v2: rows are mapped to MadGraph slots from the generate
        # string (row_to_slot_perm) and α_s is set per event inside the ME. Datasets
        # built before this carried the beams mirrored on the C++ path; the key keeps
        # them from being reused as cache hits.
        "label_convention":   "mg5_slot_v2",
    }
    if cfg.get("model"):
        recipe["model"] = cfg["model"]      # a non-SM UFO model; absent (sm) keeps every SM recipe_id
    if cfg.get("kind") == "synthetic":
        # the random structure (tools/synthetic_amplitudes.py) is what the data is
        recipe["synthetic"] = dict(cfg["synthetic"])
    if cfg.get("kind") == "virt" or cfg.get("virt"):
        # One-loop pools: the generator redraws events whose MadLoop point is exceptional
        # or carries a wrong double pole (v2), so the bytes differ from an unguarded pool
        # of the same recipe; the version keeps the two apart as cache keys.
        recipe["virt_generator"] = 2
    # Per-dataset physics scan (register_scan_process): the reference α_s(M_Z)
    # drives the per-event running but is NOT in param_card_patches, so it must
    # enter the identity explicitly. Only recorded when non-default so existing
    # datasets keep their recipe_id (and cache). m_finals records the phase-space
    # masses (also identity-bearing when a mass is scanned).
    if abs(float(cfg.get("alphas_mz", 0.118)) - 0.118) > 1e-12:
        recipe["alphas_mz"] = float(cfg["alphas_mz"])
    if cfg.get("alphas_prefactor"):
        recipe["alphas_prefactor"] = True   # NLO target carries the physical α_s weight
    # the sampling policy shapes the events (mixture vs flat); legacy flat keeps old ids.
    # Never for val/test (always flat) nor for virt datasets (MadLoop path, never shaped).
    pol = sampling_policy(cfg, role)
    if pol["mode"] != "flat":
        recipe["sampling"] = {k: pol[k] for k in sorted(pol)}
    if float(pol.get("f_default", 0)) > 0:
        # the MadEvent pools the default share reads (their recipe ids), so a rebuilt source mints a new pool id
        recipe["default_share"] = default_share_identity(pol["default_source"], role or "train", seed)
    # sampling masses are identity-bearing for every entry (a changed mass changes the
    # events), not only for mass scans
    recipe["m_finals"] = [float(m) for m in cfg.get("m_finals", cfg.get("m_final", []) if isinstance(cfg.get("m_final"), (list, tuple)) else [cfg.get("m_final", 0.0)] * int(cfg.get("nfinal", 0)))]
    # Fiducial cuts (when active) change the sampled events → part of the identity, so
    # cut and pre-cut (no-cut) datasets never collide in the cache. Omitted when off,
    # so existing pre-cut datasets keep their recipe_id.
    if FIDUCIAL_CUTS_ENABLED:
        recipe["fiducial_cuts"] = _cut_key(cfg.get("fiducial_cuts") or FIDUCIAL_CUTS)
    # catalog_v4 (D130): the sqrt(s) measure and the beam masses, recorded only where an entry sets them
    if cfg.get("sqrts_measure", "uniform") != "uniform":
        recipe["sqrts_measure"] = cfg["sqrts_measure"]
    if cfg.get("beam_masses") is not None:
        recipe["beam_masses"] = {str(k): float(v) for k, v in sorted(cfg["beam_masses"].items())}
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
    # a shaped (mixture) pool never shares a path with the flat pool of the same spec
    mode = (recipe.get("sampling") or {}).get("mode", "flat")
    # a steered pool (tools/steer_pool.py) has its own tag: it is built from a reference model's
    # sigma, never by this generator, and must not shadow the mixture pool of the same spec
    samp_tag = {"flat": "", "steered": "_ssteer"}.get(mode, "_smix")
    f_def = float((recipe.get("sampling") or {}).get("f_default", 0))
    if f_def > 0:
        samp_tag += f"d{int(round(100 * f_def)):02d}"     # a default-share pool never shadows its catalog's _smix pool
    if recipe.get("mode") == "lhe_pp":
        samp_tag = "_lhe"          # MadEvent pp events (catalog_daniel), never a sampler's pool
    return (f"{out_dir}/{recipe['process']}"
            f"_{recipe['sqrts_min']:.0f}-{recipe['sqrts_max']:.0f}GeV"
            f"{role_tag}{samp_tag}_amplitudes.npy")

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
    cfg         = dict(register_synthetic(process) if is_synthetic(process) else PROCESSES[process])
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

    if recipe.get("mode") == "lhe_pp":
        return build_lhe_pp(process, [(recipe, output_file)])[0][0]
    if (recipe.get("sampling") or {}).get("mode") == "steered":
        raise RuntimeError(
            f"{output_file}: a steered pool is not generated here; build it with tools/steer_pool.py "
            f"(recipe_id {wanted_id})")
    seed, role = recipe.get("seed"), recipe.get("role")
    if seed is None:
        print("[WARN] recipe has no seed: dataset will NOT be reproducible.")
        eff_seed, rng = None, np.random.default_rng()
    else:
        eff_seed = seed + ROLE_SEED_OFFSET.get(role, 0)
        rng = np.random.default_rng(eff_seed)

    nparticles     = cfg["nfinal"] + 2
    standalone_dir = f"{WORK_DIR}/{process}_standalone"
    if cfg.get("kind") == "synthetic":
        backend, subproc_dirs, driver_bin, eff_dir = "synthetic", ["synthetic"], None, None
    else:
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
        eff_dir, backend, subproc_dirs, driver_bin, cfg, output_file, rng=rng, role=role, default_seed=seed)

    full = dict(recipe)
    full["effective_seed"] = eff_seed
    full["backend"]        = backend
    write_recipe(output_file, full)
    return output_file


# =============================================================================
# LHE_PP: pp events from MadEvent (catalog_daniel, arXiv:2606.23791)
# =============================================================================
# Our samplers have no PDFs and do not unweight, so a catalog defined by a hadron-collider
# generation (theirs: MadEvent, pp at 2 x 6500 GeV, nn23lo1, dynamical scale, unweighted, pT/eta
# generation cuts) takes its momenta from MadEvent itself, run with THEIR run_card and param_card
# verbatim except nevents and iseed. Each LHE event is labelled with our standalone_cpp |M|^2
# (CppDriverPipe, rows -> MadGraph slots by row_to_slot_perm) at the event's own alpha_s (AQCDUP:
# the PDF set's running alpha_s at the event's dynamical scale, the value MadEvent used), on the
# event's own momenta: a massive incoming b keeps its LHE kinematics, no special treatment.
# One MadEvent run per process feeds all three roles: the events are shuffled (seeded) and cut into
# [val | test | train] blocks of the catalog's sizes (lhe_layout), so the roles are disjoint; a role
# asking fewer events than its block takes the block's first n. Bytes are reproducible given the
# MadEvent output; MadEvent itself is deterministic at fixed iseed on one core (run_mode 0), but the
# MG5 version is not part of the identity (3.7.0 at CC-IN2P3; they used 3.5.12), as for every pool.

LHE_MARGIN = float(os.environ.get("AMP_LHE_MARGIN", "0.02"))   # extra events asked of MadEvent


def lhe_layout(cfg):
    """role -> [start, stop) of its block in the shuffled events of the process's MadEvent run."""
    l = cfg["lhe"]
    nv, nt, ntr = int(l["n_val"]), int(l["n_test"]), int(l["n_train"])
    return {"val": (0, nv), "test": (nv, nv + nt), "train": (nv + nt, nv + nt + ntr)}


def lhe_n_requested(cfg):
    """Events asked of MadEvent: the layout plus a margin (an unweighted run can come back short)."""
    total = lhe_layout(cfg)["train"][1]
    # a process's own margin (lhe.margin: one whose MadEvent run came back short at the default) wins over the global
    # one, so the identity read back and the one generated agree without an environment variable
    return int(math.ceil(total * (1.0 + float(cfg["lhe"].get("margin", LHE_MARGIN)))))


def lhe_cards(cfg):
    """(run_card, param_card) paths of an lhe_pp entry (their cards, copied verbatim)."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg["lhe"]["cards_dir"])
    return os.path.join(d, cfg["lhe"]["run_card"]), os.path.join(d, cfg["lhe"]["param_card"])


def lhe_iseed(process, seed):
    """MadEvent iseed of a process, deterministic in (process, seed) and kept small: MadEvent derives
    its per-channel ranmar seeds from it."""
    salt = (PROCESSES.get(process, {}).get("lhe") or {}).get("iseed_salt")
    key = f"{process}|{seed}" + (f"|{salt}" if salt else "")    # lhe.iseed_salt: a pool reseeded (its runs gave no event)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 20000 + 1


def lhe_pp_recipe(process, sqrts_min, sqrts_max, n_events, role=None, seed=None):
    """Identity of an lhe_pp pool. sqrts_min/max only label the path (the hadronic 13 TeV); the
    events are fixed by the cards, the generate line, iseed, the events asked of MadEvent and the
    block of the shuffled run this role reads."""
    cfg = PROCESSES[process]
    seed = 0 if seed is None else int(seed)
    lo, hi = lhe_layout(cfg)[role or "train"]
    n = int(n_events)
    if n > hi - lo:
        raise ValueError(f"{process}: {n} {role} events asked, the catalog's block holds {hi - lo} "
                         f"(lhe n_val/n_test/n_train in recipes/catalog_daniel_processes.yaml)")
    run_card, param_card = lhe_cards(cfg)
    return {
        "process":          process,
        "mode":             "lhe_pp",
        "sqrts_min":        float(sqrts_min),
        "sqrts_max":        float(sqrts_max),
        "n_events":         n,
        "role":             role,
        "seed":             seed,
        "pdg_ids":          [int(p) for p in cfg["pdg_ids"]],
        "m_finals":         [float(m) for m in cfg["m_finals"]],
        "alphas_power":     int(cfg.get("alphas_power", 0)),
        "amp_orders":       order_vector(cfg),
        "mg5_generate":     list(cfg["mg5_generate"]),
        # the alpha_s the pool is labelled at: MadEvent's AQCDUP (Daniel's catalog), or the window pools' convention,
        # alpha_s(sqrt(s_hat)) run at one loop from 0.118 (lhe.label_alphas: running_sqrt_shat; the pp study, so a pp
        # pool is the same |M|^2 function as its catalog_v3 window parent)
        "per_event_alphas": ("lhe_aqcdup" if cfg["lhe"].get("label_alphas", "aqcdup") == "aqcdup"
                             else str(cfg["lhe"]["label_alphas"])),
        "label_convention": "mg5_slot_v2",
        "lhe": {
            "model":             cfg["lhe"].get("model", "sm"),
            "iseed":             lhe_iseed(process, seed),
            "n_requested":       lhe_n_requested(cfg),
            "block":             [lo, lo + n],
            "run_card_sha256":   _file_sha256(run_card),
            "param_card_sha256": _file_sha256(param_card),
            # several MadEvent runs only where set (lhe.n_runs > 1), so every other pool keeps its identity
            **({"n_runs": int(cfg["lhe"]["n_runs"])} if int(cfg["lhe"].get("n_runs", 1)) > 1 else {}),
            # the events asked of each run where set (lhe.run_events: a pool split into runs of one job each,
            # recipes/gen_catalog_v4_default.py); absent, each run asks n_requested (every pool built before)
            **({"run_events": int(cfg["lhe"]["run_events"])} if cfg["lhe"].get("run_events") else {}),
            # built from one-run part jobs (lhe.parts: tools/lhe_parts.py, lhe_assemble): its own identity, since a
            # part's fresh MadEvent directory is not shown to give the events a run in a reused directory gives
            **({"build": "parts"} if cfg["lhe"].get("parts") else {}),
            # an e+e- entry's fixed collider energies (lhe.energies), one MadEvent run each; absent for pp
            **({"energies": [float(e) for e in cfg["lhe"]["energies"]]} if cfg["lhe"].get("energies") else {}),
            **({"iseed_salt": int(cfg["lhe"]["iseed_salt"])} if cfg["lhe"].get("iseed_salt") else {}),
            **({"energy_runs": [int(x) for x in cfg["lhe"]["energy_runs"]]} if cfg["lhe"].get("energy_runs")
               else {}),
            # lhe.energy_run_events: an e+e- pool whose runs of run_events stop short at one energy takes smaller runs
            # there (one value per energy), drawn from the gridpacks the probe measured (lhe.grid_seed_runs: the run
            # count their iseeds were set from; lhe.gridpack_eid: the identity they were built and are kept under)
            **({"energy_run_events": [int(x) for x in cfg["lhe"]["energy_run_events"]]}
               if cfg["lhe"].get("energy_run_events") else {}),
            **({k: cfg["lhe"][k] for k in ("grid_seed_runs", "gridpack_eid") if cfg["lhe"].get(k)}),
            # lhe.complete_runs: only the runs that returned every event they asked are assembled (lhe_assemble). A run
            # that stops short of its target ("fail to reach target") returns a biased sample (catalog_coverage,
            # analysis/catalog_v4/short_run_bias.py); a run that completes does not, whatever its size. In the
            # identity: a pool once assembled from short runs never passes for one of complete runs
            **({"runs": "complete"} if cfg["lhe"].get("complete_runs") else {}),
            # lhe.gridpack = G: every run is drawn from one MadEvent integration made at the precision of a run of G
            # events (one gridpack per energy, lhe_gridpacks), not integrated on its own: a small run that integrates
            # on its own differs from a large complete run (docs/results.tex par:v4_short_runs). In the identity: a
            # pool of gridpack runs never passes for one of runs integrated each on their own, nor one of another G
            **({"gridpack": int(cfg["lhe"]["gridpack"])} if cfg["lhe"].get("gridpack") else {}),
        },
    }


def lhe_runs(process, iseed, n_req):
    """[(iseed, nevents, run-card patches, run name)] of an lhe entry's MadEvent runs. pp: lhe.n_runs runs of n_req at
    iseed + 7919 k (the runs every pp pool was built with). e+e- (lhe.energies): each energy E gets n_runs runs of
    ceil(n_req / len(energies)) with ebeam1 = ebeam2 = E / 2, so the energies have equal shares of the events."""
    l = PROCESSES[process]["lhe"]
    n_runs = int(l.get("n_runs", 1))
    seed_k = lambda k: (iseed + 7919 * k - 1) % 30000 + 1
    energies = [float(e) for e in (l.get("energies") or [])]
    # lhe.run_events (set only on pools split into one-run jobs): each run asks that many events instead of n_req
    r_ev = int(l["run_events"]) if l.get("run_events") else None
    if not energies:
        return [(seed_k(k), r_ev or n_req, {}, f"run_{k + 1:02d}") for k in range(n_runs)]
    n_e = r_ev or int(math.ceil(n_req / len(energies)))
    # lhe.energy_run_events: energy j's runs ask sizes[j] events (smaller runs where runs of n_e stop short)
    sizes = [int(x) for x in l["energy_run_events"]] if l.get("energy_run_events") else [n_e] * len(energies)
    if len(sizes) != len(energies):
        raise ValueError(f"{process}: lhe.energy_run_events has {len(sizes)} values for {len(energies)} energies")
    # lhe.grid_seed_runs = G pins the gridpacks' iseeds (lhe_gridpacks continues the sequence past G runs): runs from G
    # on skip the len(energies) seeds the gridpacks hold, so no run shares a gridpack's iseed
    G = l.get("grid_seed_runs")
    if G:
        seed_k = lambda k, s0=seed_k: s0(k if k < int(G) else k + len(energies))
    # lhe.energy_runs (a pool whose runs came back short at some energies): energy j gets energy_runs[j] runs of n_e,
    # sized from that energy's measured yield (recipes/gen_catalog_v4_default.py), so each energy still returns its
    # equal share; runs numbered and seeded in sequence over the energies. _lhe_finish refuses a pool where one does not
    if l.get("energy_runs"):
        nrs = [int(x) for x in l["energy_runs"]]
        if len(nrs) != len(energies):
            raise ValueError(f"{process}: lhe.energy_runs has {len(nrs)} values for {len(energies)} energies")
        i0 = [sum(nrs[:j]) for j in range(len(nrs))]
        return [(seed_k(i0[j] + k), sizes[j], {"ebeam1": E / 2.0, "ebeam2": E / 2.0}, f"run_{i0[j] + k + 1:02d}")
                for j, E in enumerate(energies) for k in range(nrs[j])]
    return [(seed_k(j * n_runs + k), sizes[j], {"ebeam1": E / 2.0, "ebeam2": E / 2.0}, f"run_{j * n_runs + k + 1:02d}")
            for j, E in enumerate(energies) for k in range(n_runs)]


def lhe_energy_of_runs(runs):
    """The collider energy of each run of lhe_runs (2 x its ebeam1 patch), None for a pp run."""
    return [2.0 * float(r[2]["ebeam1"]) if r[2].get("ebeam1") is not None else None for r in runs]


def lhe_equal_shares(process, moms, aqcds, runs, need):
    """An e+e- pool's runs' events cut to equal counts per energy (lhe.energies: equal shares of the events): each
    energy keeps the first q events of its runs in run order, q the fewest any energy returned, so a pool whose every
    run returned what it asked is unchanged. Refuses when q x the energies is below the blocks' need."""
    E = lhe_energy_of_runs(runs)
    if not E or E[0] is None:
        return moms, aqcds
    Es = list(dict.fromkeys(E))
    cnt = {e: sum(len(m) for m, x in zip(moms, E) if x == e) for e in Es}
    q = min(cnt.values())
    share = int(math.ceil(need / len(Es)))
    if q < share:
        short = {f"{e:g}": c for e, c in cnt.items() if c < share}
        raise RuntimeError(f"{process}: energies short of their equal share ({share} each, the blocks need {need}): "
                           f"{short} of {len(Es)} energies; raise lhe.energy_runs at those energies")
    left = dict.fromkeys(Es, q)
    m_out, a_out = [], []
    for m, a, e in zip(moms, aqcds, E):
        k = min(len(m), left[e])
        left[e] -= k
        m_out.append(m[:k]); a_out.append(a[:k])
    return m_out, a_out


def lhe_work_dir():
    """Where the per-process MadEvent directories live while they run (deleted afterwards)."""
    d = os.environ.get("AMP_LHE_WORK_DIR")
    if d:
        return d
    s = os.environ.get("SCRATCH")
    return os.path.join(s, "lhe_work") if s else os.path.join(OUTPUT_DIR, ".lhe_work")


def lhe_standalone_dir(process):
    # a default-share entry (dsh_<x>) labels with the standalone of the entry it names (same generate line, model and
    # card), so the 13 TeV pp standalones already compiled for pp13v4 are not built again
    return f"{WORK_DIR}/{PROCESSES.get(process, {}).get('standalone_of') or process}_standalone"


def _mg5_script(script, log_path):
    with open(log_path, "a") as log:
        r = subprocess.run([MG5_BIN], input=script, text=True, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"mg5_aMC exited {r.returncode}; log {log_path}")


def _set_me5_config(path, values):
    """Set `key = value` lines of a me5_configuration.txt (commented defaults included)."""
    lines, done = open(path).read().splitlines(True), set()
    for i, line in enumerate(lines):
        m = re.match(r"^\s*#?\s*(\w+)\s*=", line)
        if m and m.group(1) in values and m.group(1) not in done:
            lines[i] = f"{m.group(1)} = {values[m.group(1)]}\n"
            done.add(m.group(1))
    lines += [f"{k} = {v}\n" for k, v in values.items() if k not in done]
    with open(path, "w") as f:
        f.writelines(lines)


def lhe_prepare(process, work_root, log_path, madevent=True, standalone=True):
    """MG5 `output` of a fresh MadEvent directory (madevent) and, unless already compiled, of the
    standalone_cpp backend (standalone; their param card in its Cards/, C++ pipe driver). Returns (ev_dir, sa_dir);
    ev_dir is None without madevent. A one-run part job (lhe_build_part) asks for the MadEvent directory only, so
    parallel runs of one process never compile the same standalone; its assembly (lhe_assemble) builds that."""
    cfg = PROCESSES[process]
    sa = lhe_standalone_dir(process)
    _, sub, drv, _ = detect_compiled_backend(sa)
    need_sa = standalone and not (sub and drv and os.path.exists(f"{sa}/Cards/param_card.dat"))
    ev = os.path.join(work_root, f"{process}_madevent") if madevent else None
    if not (madevent or need_sa):
        return ev, sa
    for d in ([ev] if ev else []) + ([sa] if need_sa else []):
        if os.path.isdir(d):
            shutil.rmtree(d)          # a half-written output from an interrupted run
    os.makedirs(work_root, exist_ok=True)
    script = (f"import model {cfg['lhe'].get('model', 'sm')}\n" + "\n".join(cfg["mg5_generate"]) +
              (f"\noutput {ev}" if ev else "") + "\n" + (f"output standalone_cpp {sa}\n" if need_sa else "") + "quit\n")
    _mg5_script(script, log_path)
    if need_sa:
        shutil.copy2(lhe_cards(cfg)[1], f"{sa}/Cards/param_card.dat")
        compile_backends(sa, int(cfg["nfinal"]) + 2)
    return ev, sa


def lhe_second_refine():
    v = os.environ.get("AMP_LHE_SECOND_REFINE")
    return float(v) if v else None


def add_card_params(card_path, values):
    """Append `value = key` lines for keys a run card does not have (MadGraph's hidden parameters, e.g.
    second_refine_treshold, are read from any line of the card); keys already there are left to patch_card."""
    have = set(re.findall(r"=\s*(\w+)", "\n".join(l.split("!")[0] for l in open(card_path)
                                                 if l.strip() and not l.lstrip().startswith("#"))))
    new = {k: v for k, v in values.items() if k not in have}
    if new:
        with open(card_path, "a") as f:
            f.writelines(f"  {v} = {k}\n" for k, v in new.items())


def lhe_run_madevent(process, ev, nevents, iseed, log_path, mg_cores=1, run_name="run_01", patches=None):
    """generate_events in `ev` with the catalog's cards (nevents and iseed set). run_mode 0 (one core)
    unless mg_cores > 1. Returns (lhe_path, banner_path)."""
    cfg = PROCESSES[process]
    run_card, param_card = lhe_cards(cfg)
    shutil.copy2(param_card, f"{ev}/Cards/param_card.dat")
    shutil.copy2(run_card, f"{ev}/Cards/run_card.dat")
    patch_card(f"{ev}/Cards/run_card.dat", {"nevents": int(nevents), "iseed": int(iseed), **(patches or {})})
    # AMP_LHE_SECOND_REFINE=T: MadEvent's hidden second_refine_treshold (default 0.9). It runs its second refine unless
    # the first refine's cross section is within T of the survey's; above 1 it always does, so a run whose maximum
    # weight is poorly known no longer stops short of the events it asks ("fail to reach target"). An execution
    # setting like the cores (how MadEvent iterates, not which density it unweights): recorded in the part, not in the
    # pool's identity
    sr = lhe_second_refine()
    if sr is not None:
        add_card_params(f"{ev}/Cards/run_card.dat", {"second_refine_treshold": sr})
        patch_card(f"{ev}/Cards/run_card.dat", {"second_refine_treshold": sr})
    mg_cores = max(1, int(mg_cores))
    _set_me5_config(f"{ev}/Cards/me5_configuration.txt",
                    {"run_mode": 0 if mg_cores == 1 else 2, "nb_core": mg_cores,
                     "automatic_html_opening": "False"})
    with open(log_path, "a") as log:
        r = subprocess.run([f"{ev}/bin/generate_events", run_name, "-f"], cwd=ev, stdout=log,
                           stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS="1"))
    lhe = f"{ev}/Events/{run_name}/unweighted_events.lhe.gz"
    gridpack = str((patches or {}).get("gridpack", "")).lower() in ("true", ".true.", "t")
    if r.returncode != 0 or not (gridpack or os.path.exists(lhe)):
        raise RuntimeError(f"{process}: generate_events exited {r.returncode}, no {lhe}; log {log_path}")
    if gridpack:                # a gridpack run writes the packed grids, not events (lhe_make_gridpack)
        lhe = None
    banners = sorted(glob.glob(f"{ev}/Events/{run_name}/*_banner.txt"))
    return lhe, (banners[0] if banners else None)


def lhe_make_gridpack(process, ev, nevents, iseed, log_path, mg_cores=1, patches=None, run_name="grid"):
    """One MadEvent integration of `process` in `ev`, packed as a gridpack (run card gridpack = True): the survey and
    refine run at the precision of a run of `nevents` events (MadEvent sizes its refine from nevents), once. Small runs
    are then drawn from it (lhe_run_gridpack), each from the same grids, instead of each small run integrating
    afresh: complete runs of 150 to 1000 events that integrate on their own differ from complete runs of many
    (catalog_coverage, docs/results.tex par:v4_short_runs). Returns the gridpack's .tar.gz path."""
    lhe_run_madevent(process, ev, nevents, iseed, log_path, mg_cores=mg_cores, run_name=run_name,
                     patches={"gridpack": "True", **(patches or {})})
    tars = sorted(glob.glob(f"{ev}/{run_name}_gridpack.tar.gz") + glob.glob(f"{ev}/*_gridpack.tar.gz"))
    if not tars:
        raise RuntimeError(f"{process}: no gridpack written under {ev}; log {log_path}")
    return tars[0]


def lhe_gridpack_header(lhe_path):
    """The LHE file's header (everything before the first <event>): the banner of the gridpack's integration, with its
    process, run and parameter cards (analysis/catalog_v4/short_run_bias.py physics_key)."""
    opener = gzip.open if str(lhe_path).endswith(".gz") else open
    out = []
    with opener(lhe_path, "rt") as f:
        for line in f:
            if line.lstrip().startswith("<event"):
                break
            out.append(line)
    return "".join(out)


GRIDPACK_XIMPROVE = ("max_iter", "min_iter", "max_event_in_iter", "min_event_in_iter", "gen_events_security")


def lhe_patch_ximprove(work, values):
    """In an unpacked gridpack under `work`, set class attributes of MadEvent's gen_ximprove_gridpack (the event
    generation of a gridpack run: per channel at most max_iter iterations, the points of each between min_event_in_iter
    and max_event_in_iter; madevent/bin/internal/gen_ximprove.py) to `values`, the integration grids untouched.
    Refused for a key outside GRIDPACK_XIMPROVE or not found in that class."""
    f = os.path.join(work, "madevent", "bin", "internal", "gen_ximprove.py")
    src = open(f).read()
    i = src.index("class gen_ximprove_gridpack(")
    j = src.find("\nclass ", i + 1)
    j = len(src) if j < 0 else j
    body = src[i:j]
    for k, v in values.items():
        if k not in GRIDPACK_XIMPROVE:
            raise ValueError(f"gridpack ximprove key {k!r} not in {GRIDPACK_XIMPROVE}")
        body, n = re.subn(rf"(?m)^(\s+{k}\s*=\s*)[^#\n]+", lambda m: m.group(1) + repr(v), body, count=1)
        if n != 1:
            raise ValueError(f"gridpack ximprove key {k!r} not found in gen_ximprove_gridpack of {f}")
    open(f, "w").write(src[:i] + body + src[j:])
    for c in glob.glob(os.path.join(os.path.dirname(f), "__pycache__", "gen_ximprove*.pyc")):
        os.remove(c)


def lhe_run_gridpack(gridpack, work, nevents, seed, log_path, timeout=None, ximprove=None):
    """`nevents` unweighted events drawn from a gridpack (lhe_make_gridpack) with random seed `seed`, in a fresh
    unpacked copy under `work` (one copy per run, so runs of one gridpack go in parallel). ximprove: the copy's
    gridpack event-generation limits changed first (lhe_patch_ximprove). Returns the LHE path."""
    os.makedirs(work, exist_ok=True)
    with tarfile.open(gridpack) as t:
        t.extractall(work)
    if ximprove:
        lhe_patch_ximprove(work, ximprove)
    with open(log_path, "a") as log:
        r = subprocess.run(["bash", "./run.sh", str(int(nevents)), str(int(seed))], cwd=work, stdout=log,
                           stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS="1"), timeout=timeout)
    lhe = os.path.join(work, "events.lhe.gz")
    if r.returncode != 0 or not os.path.exists(lhe):
        raise RuntimeError(f"gridpack run.sh {nevents} {seed} exited {r.returncode}, no {lhe}; log {log_path}")
    return lhe


def _lhe_rows(parts, pdg_ids, n_in=2):
    """One event's (status, pdg, p4) list -> momenta in `pdg_ids` row order (identical particles in
    LHE order). Incoming are status -1, outgoing status +1; intermediate resonances are dropped."""
    rows = []
    for want, st in ((pdg_ids[:n_in], -1), (pdg_ids[n_in:], 1)):
        pool = [(p, m) for s, p, m in parts if s == st]
        if len(pool) != len(want):
            raise ValueError(f"LHE event has {[p for p, _ in pool]} with status {st}, expected {list(want)}")
        for w in want:
            j = next((i for i, (p, _) in enumerate(pool) if p == w), None)
            if j is None:
                raise ValueError(f"LHE event has no pdg {w} among {[p for p, _ in pool]} (status {st})")
            rows.append(pool.pop(j)[1])
    return rows


def read_lhe_pp(lhe_path, pdg_ids):
    """Stream an LHE file -> (momenta (N, n, 4) as (E, px, py, pz) in `pdg_ids` row order,
    alpha_s per event (AQCDUP), scale per event (SCALUP))."""
    pdg_ids = [int(p) for p in pdg_ids]
    moms, aqcd, scal = [], [], []
    opener = gzip.open if str(lhe_path).endswith(".gz") else open
    with opener(lhe_path, "rt") as f:
        parts = head = None
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("<event"):
                parts, head = [], None
                continue
            if parts is None:
                continue
            if s.startswith("</event"):
                moms.append(_lhe_rows(parts, pdg_ids))
                aqcd.append(float(head[5]))
                scal.append(float(head[3]))
                parts = None
                continue
            if s[0] in "<#":
                continue                # <mgrwt>, <scales>, ... inside the event block
            c = s.split()
            if head is None:
                head = c                # NUP IDPRUP XWGTUP SCALUP AQEDUP AQCDUP
                continue
            st = int(c[1])
            if st in (-1, 1):
                parts.append((st, int(c[0]), (float(c[9]), float(c[6]), float(c[7]), float(c[8]))))
    return (np.asarray(moms, dtype=np.float64).reshape(len(moms), len(pdg_ids), 4),
            np.asarray(aqcd, dtype=np.float64), np.asarray(scal, dtype=np.float64))


def label_lhe_pp(process, momenta, alphas, sa=None):
    """|M|^2 of `process` at each event's momenta (rows in pdg_ids order) and alpha_s, through the
    standalone_cpp pipe driver."""
    cfg = PROCESSES[process]
    sa = sa or lhe_standalone_dir(process)
    backend, sub, drv, _ = detect_compiled_backend(sa)
    if backend != "cpp" or drv is None:
        raise RuntimeError(f"{process}: no compiled C++ driver under {sa} (lhe_prepare builds it)")
    perm = row_to_slot_perm(cfg["pdg_ids"], cfg["mg5_generate"])
    with CppDriverPipe(drv, sa) as pipe:
        amps = pipe.compute([(m, None) for m in momenta], perm=perm, alphas=alphas)
    return np.asarray(amps, dtype=np.float64)


def _mg5_version():
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(MG5_BIN)), "VERSION")) as f:
            return next((l.split("=", 1)[1].strip() for l in f if l.startswith("version")), None)
    except OSError:
        return None


def _banner_xsec(banner):
    try:
        m = re.search(r"Integrated weight \(pb\)\s*:\s*([-+0-9.eE]+)", open(banner).read())
        return float(m.group(1)) if m else None
    except (OSError, TypeError):
        return None


def lhe_parent_physics_check(process):
    """An lhe_pp entry made as the pp version of a catalog entry (`parent`, recipes/gen_catalog_v3_pp.py) must label its
    events with the parent's physics: the physics block of its standalone (model, card values, coupling-order clause,
    order vector, alpha_s rule, the b's particle-table row; tools/pool_physics.py) against the parent's. A difference
    raises, unless the entry gives `physics_mismatch_reason` (or its lhe block does). Returns the check."""
    from tools import pool_physics as pp
    cfg = PROCESSES[process]
    parent = cfg.get("parent")
    if not parent or parent not in PROCESSES:
        return None
    blk = pp.physics_block(process, cfg=cfg)
    pcfg = PROCESSES[parent]
    ref = [{"name": parent, "pdg": [int(p) for p in pcfg["pdg_ids"]], "block": pp.physics_block(parent, cfg=pcfg)}]
    bad, skipped, _ = pp.compare(blk, cfg["pdg_ids"], ref, pp.SAME_FIELDS)
    chk = {"probe": process, "parent": parent, "status": "mismatch" if bad else "ok", "mismatches": bad,
           "not_compared": skipped}
    why = cfg.get("physics_mismatch_reason") or (cfg.get("lhe") or {}).get("physics_mismatch_reason")
    if bad and not pp.apply_allowance(chk, why):
        raise RuntimeError(f"{process}: its standalone's physics differs from its parent {parent}'s "
                           f"({pp.summary(chk)}); fix the cards or model, or set physics_mismatch_reason on the entry")
    print(f"[LHE] {process}: physics vs parent {parent}: {chk['status']}"
          + (f" (not compared: {', '.join(skipped)})" if skipped else ""), flush=True)
    return chk


def build_lhe_pp(process, items, work_root=None, mg_cores=1, write_sidecars=True, keep_dir=None):
    """Generate the pools `items` = [(recipe, out_path), ...] of ONE lhe_pp process from one
    MadEvent run: MG5 output (+ standalone backend if missing), generate_events, stream-parse the
    LHE, seeded shuffle, slice each recipe's block, label it, write the .npy (and, if
    `write_sidecars`, its .recipe.json + manifest entry; a parallel caller writes them itself,
    serially, from the returned recipes). The banner is kept next to each pool directory, the
    MadEvent directory is deleted (AMP_LHE_KEEP=1 keeps it). Returns [(out_path, full_recipe)]."""
    cfg = PROCESSES[process]
    if cfg["lhe"].get("parts"):
        raise RuntimeError(f"{process}: lhe.parts is set, its pools are built from one-run part jobs "
                           f"(tools/lhe_parts.py plan/work), not in series")
    recs = [r for r, _ in items]
    if not recs or any(r["process"] != process or r.get("mode") != "lhe_pp" for r in recs):
        raise ValueError(f"build_lhe_pp({process}): items must be lhe_pp recipes of this process")
    runs = {(r["seed"], r["lhe"]["iseed"], r["lhe"]["n_requested"]) for r in recs}
    if len(runs) != 1:
        raise ValueError(f"build_lhe_pp({process}): recipes from different MadEvent runs {runs}")
    seed, iseed, n_req = runs.pop()
    keep_dir = os.environ.get("AMP_LHE_KEEP") == "1" if keep_dir is None else keep_dir
    work_root = work_root or lhe_work_dir()
    os.makedirs(work_root, exist_ok=True)
    log_path = os.path.join(work_root, f"{process}.mg5.log")
    open(log_path, "w").close()
    secs, t0 = {}, time.time()
    ev, sa = lhe_prepare(process, work_root, log_path)
    lhe_parent_physics_check(process)            # before MadEvent: a pool labelled with other physics is refused cheaply
    secs["output"] = time.time() - t0; t0 = time.time()
    # a process whose MadEvent run returns a fixed few thousand events however many are asked (its unweighting stalls)
    # runs n_runs times with seeds iseed + 7919 k (lhe.n_runs, in its pool identity), the unweighted events concatenated
    runs = lhe_runs(process, iseed, n_req)
    n_runs = len(runs)
    moms, aqcds = [], []
    for k, (iseed_k, n_k, patches_k, name_k) in enumerate(runs):
        lhe, b = lhe_run_madevent(process, ev, n_k, iseed_k, log_path, mg_cores, run_name=name_k, patches=patches_k)
        banner = banner if k else b
        m_k, a_k, _ = read_lhe_pp(lhe, cfg["pdg_ids"])
        moms.append(m_k); aqcds.append(a_k)
    secs["madevent"] = time.time() - t0; t0 = time.time()
    out = _lhe_finish(process, recs, items, moms, aqcds, runs, banner, secs, sa, seed, iseed, n_req, mg_cores,
                      write_sidecars)
    if not keep_dir:
        shutil.rmtree(ev, ignore_errors=True)
    return out


def _lhe_check_recipes(process, items):
    """(seed, iseed, n_req) of the one MadEvent layout the recipes `items` read."""
    recs = [r for r, _ in items]
    if not recs or any(r["process"] != process or r.get("mode") != "lhe_pp" for r in recs):
        raise ValueError(f"lhe build of {process}: items must be lhe_pp recipes of this process")
    runs = {(r["seed"], r["lhe"]["iseed"], r["lhe"]["n_requested"]) for r in recs}
    if len(runs) != 1:
        raise ValueError(f"lhe build of {process}: recipes from different MadEvent runs {runs}")
    return runs.pop()


def _lhe_finish(process, recs, items, moms, aqcds, runs, banner, secs, sa, seed, iseed, n_req, mg_cores,
                write_sidecars):
    """The events of the runs (moms, aqcds: per run, in run order) -> the pools of `items`: concatenated in run
    order, seeded shuffle, each recipe's block labelled and written. Shared by build_lhe_pp (runs in sequence in one
    job) and lhe_assemble (runs as separate part jobs), so both give the same bytes for the same runs."""
    cfg = PROCESSES[process]
    n_runs = len(runs)
    t0 = time.time()
    need = max(r["lhe"]["block"][1] for r in recs)
    per_run = [len(m_k) for m_k in moms]            # as returned, before the per-energy cut
    moms, aqcds = lhe_equal_shares(process, moms, aqcds, runs, need)
    mom, aqcd = np.concatenate(moms), np.concatenate(aqcds)
    secs["parse"] = time.time() - t0; t0 = time.time()
    n_lhe = len(mom)
    if n_lhe < need:
        raise RuntimeError(f"{process}: MadEvent returned {n_lhe} events, the blocks need {need} "
                           f"(asked {n_req} in {n_runs} run(s)); raise lhe.margin, or lhe.n_runs if a larger "
                           f"request returns no more events")
    order = np.random.default_rng(np.random.SeedSequence([seed, iseed])).permutation(n_lhe)
    mom, aqcd = mom[order], aqcd[order]
    xsec, version = _banner_xsec(banner), _mg5_version()     # the first run's banner (each run's own events counted below)
    npart = int(cfg["nfinal"]) + 2
    pdg_row = np.asarray(cfg["pdg_ids"], dtype=np.float64)
    out = []
    for rec, path in items:
        lo, hi = rec["lhe"]["block"]
        m, a = mom[lo:hi], aqcd[lo:hi]
        a_up = a
        rule = cfg["lhe"].get("label_alphas", "aqcdup")
        if rule == "running_sqrt_shat":
            q = m[:, 0] + m[:, 1]
            a = compute_alphas(np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 1e-300, None)), alphas_mz=0.118)
        elif rule != "aqcdup":
            raise ValueError(f"{process}: unknown lhe.label_alphas {rule!r}")
        amps = label_lhe_pp(process, m, a, sa)
        bad = ~np.isfinite(amps) | (amps <= 0)
        if bad.any():
            raise RuntimeError(f"{process} {rec['role']}: {int(bad.sum())} labels non-finite or <= 0")
        n = hi - lo
        data = np.concatenate([m.reshape(n, npart * 4), np.tile(pdg_row, (n, 1)), amps[:, None]], axis=1)
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp.npy"
        np.save(tmp, data)
        os.replace(tmp, path)
        # each event's alpha_s (the one it was labelled at), row-aligned with the pool: the coupling-order factor of
        # the target (data.target_physics_orders, D93) divides it out; MadEvent's own AQCDUP kept beside it when the
        # label used another
        np.save(path[:-len(".npy")] + ".alphas.npy", np.asarray(a, dtype=np.float64))
        if rule != "aqcdup":
            np.save(path[:-len(".npy")] + ".aqcdup.npy", np.asarray(a_up, dtype=np.float64))
        bcopy = None
        if banner:
            bcopy = os.path.join(d, f"{process}_lhe_iseed{iseed}.banner.txt")
            shutil.copy2(banner, bcopy)
        full = dict(rec)
        full["effective_seed"] = seed
        full["backend"] = "cpp"
        full["lhe_run"] = {"n_lhe_events": n_lhe, "xsec_pb": xsec, "mg5_version": version,
                           "iseeds": [r[0] for r in runs],
                           "events_per_run": per_run,
                           "banner": bcopy, "mg_cores": int(mg_cores),
                           "alphas_range": [float(a.min()), float(a.max())],
                           "amp_range": [float(amps.min()), float(amps.max())],
                           "seconds": {k: round(v, 1) for k, v in secs.items()}}
        if write_sidecars:
            write_recipe(path, full)
        out.append((path, full))
        print(f"[LHE] {process} {rec['role']}: {n} events [{lo}:{hi}] of {n_lhe} -> {path}  "
              f"|M|^2 in [{amps.min():.3e}, {amps.max():.3e}]", flush=True)
    secs["label+write"] = time.time() - t0
    print(f"[LHE] {process}: done (iseed {iseed}, xsec {xsec} pb, MG5 {version}); seconds "
          + ", ".join(f"{k} {v:.1f}" for k, v in secs.items()), flush=True)
    return out


# One-run part jobs (catalog_coverage step 7: a pool of several MadEvent runs, built in parallel). A part job runs ONE
# of an lhe entry's runs (lhe_runs: the same iseed, events and patches as the k-th run of build_lhe_pp) in a fresh
# MadEvent directory and keeps its events, unlabelled and in LHE order, as a part file named by the run's whole
# identity. Once every run's part is there, lhe_assemble concatenates them in run order and finishes as build_lhe_pp
# does (_lhe_finish: seeded shuffle, blocks, labels). Only entries with lhe.parts are built this way, and that flag is
# in their pool identity ("build": "parts"): a fresh MadEvent directory per run is not shown to give the events the
# serial build's reused directory gives, so the two never share a recipe id.

def lhe_parts_dir():
    d = os.environ.get("AMP_LHE_PARTS_DIR")
    if d:
        return d
    s = os.environ.get("SCRATCH")
    return os.path.join(s, "lhe_parts") if s else os.path.join(OUTPUT_DIR, ".lhe_parts")


def _lhe_events_id(rec):
    """The MadEvent events' identity of an lhe_pp recipe: everything but which role and block of them it reads."""
    ev = {k: v for k, v in rec.items() if k not in ("role", "n_events")}
    ev["lhe"] = {k: v for k, v in rec["lhe"].items() if k != "block"}
    return recipe_id(ev)


def lhe_part_paths(process, items):
    """[(part path, run)] of the runs of the pools `items`, in run order. A part is named by the events' identity
    (cards' sha256, generate lines, model, iseed, layout: _lhe_events_id) and its run, so a part made under other cards
    or another layout is never taken for this one."""
    seed, iseed, n_req = _lhe_check_recipes(process, items)
    ids = {_lhe_events_id(r) for r, _ in items}
    if len(ids) != 1:
        raise ValueError(f"{process}: recipes of different MadEvent events {ids}")
    eid = ids.pop()
    runs = lhe_runs(process, iseed, n_req)
    if len({r[0] for r in runs}) != len(runs):
        raise ValueError(f"{process}: two of its {len(runs)} runs share a MadEvent iseed")
    d = os.path.join(lhe_parts_dir(), process)
    return [(os.path.join(d, f"{process}_{eid}_{r[3]}_seed{r[0]}_ev{r[1]}.npz"), r) for r in runs]


def lhe_gridpacks(process, iseed, n_req):
    """[(grid iseed, grid events, run-card patches, name)] of an lhe.gridpack entry's gridpacks: one per energy of an
    e+e- entry (the patches of that energy's runs), one for pp. A gridpack's iseed continues the run seeds' sequence
    past its runs (seed_k with k = runs + j; 7919 and 30000 coprime, so no run and no other gridpack shares it)."""
    l = PROCESSES[process]["lhe"]
    if not l.get("gridpack"):
        return []
    runs = lhe_runs(process, iseed, n_req)
    pats = list({json.dumps(r[2], sort_keys=True): r[2] for r in runs}.values())    # in run order, one per energy
    base = int(l.get("grid_seed_runs") or len(runs))       # pinned: the gridpacks of the layout the probe measured
    return [((iseed + 7919 * (base + j) - 1) % 30000 + 1, int(l["gridpack"]), dict(p), f"grid{j:02d}")
            for j, p in enumerate(pats)]


def lhe_gridpack_of_run(run, grids):
    """Index in `grids` (lhe_gridpacks) of the gridpack run `run` (of lhe_runs) is drawn from: the one of its patches."""
    return [g[2] for g in grids].index(run[2])


def lhe_gridpack_paths(process, items):
    """[(gridpack .tar.gz path, gridpack)] of the pools `items`, named like their parts (the events' identity, which
    holds lhe.gridpack, and the gridpack's name, iseed and events), so a gridpack of another layout is never used."""
    seed, iseed, n_req = _lhe_check_recipes(process, items)
    grids = lhe_gridpacks(process, iseed, n_req)
    if not grids:
        return []
    # lhe.gridpack_eid: the gridpacks are those built under that identity (a layout that differs only in its run sizes,
    # tests/test_lhe_gridpack.py checks it), so the runs are drawn from the very gridpacks the run-size probe measured
    eid = PROCESSES[process]["lhe"].get("gridpack_eid") or _lhe_events_id(items[0][0])
    d = os.path.join(lhe_parts_dir(), process)
    return [(os.path.join(d, f"{process}_{eid}_{g[3]}_seed{g[0]}_ev{g[1]}_gridpack.tar.gz"), g) for g in grids]


def lhe_build_gridpack(process, items, j, mg_cores=1, work_root=None, keep_dir=False):
    """Make the j-th gridpack of the pools `items` (lhe_make_gridpack at its iseed, events and patches, in a fresh
    MadEvent directory) and copy it to its path (atomic). Returns the path; an existing gridpack is kept. Its work
    directory is deleted afterwards unless keep_dir (the queue worker, whose tidy deletes it with a manifest)."""
    if not PROCESSES[process]["lhe"].get("parts"):
        raise RuntimeError(f"{process}: no lhe.parts, its pools are built in series (build_lhe_pp)")
    path, (giseed, gev, patches, gname) = lhe_gridpack_paths(process, items)[j]
    if os.path.exists(path):
        return path
    work_root = os.path.join(work_root or lhe_work_dir(), "parts", f"{process}__{gname}")
    os.makedirs(work_root, exist_ok=True)
    log_path = os.path.join(work_root, f"{process}.mg5.log")
    open(log_path, "w").close()
    t0 = time.time()
    ev, _ = lhe_prepare(process, work_root, log_path, madevent=True, standalone=False)
    tar = lhe_make_gridpack(process, ev, gev, giseed, log_path, mg_cores=mg_cores, patches=patches)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-len(".tar.gz")] + f".tmp{os.getpid()}.tar.gz"
    shutil.copy2(tar, tmp)
    os.replace(tmp, path)
    if not keep_dir:
        shutil.rmtree(work_root, ignore_errors=True)
    print(f"[LHE GRIDPACK] {process} {gname}: {gev} events' precision, iseed {giseed}, {os.path.getsize(path)} bytes "
          f"in {time.time() - t0:.0f}s -> {path}", flush=True)
    return path


def lhe_build_part(process, items, k, mg_cores=1, work_root=None, keep_dir=False):
    """Run the k-th MadEvent run of the pools `items` and write its part file (atomic). Returns its path; an existing
    part is kept (a rerun does nothing). Its run directory is deleted once the part holds every event asked, unless
    keep_dir (the queue worker, whose tidy_run deletes it with a manifest)."""
    cfg = PROCESSES[process]
    if not cfg["lhe"].get("parts"):
        raise RuntimeError(f"{process}: no lhe.parts, its pools are built in series (build_lhe_pp)")
    path, (iseed_k, n_k, patches_k, name_k) = lhe_part_paths(process, items)[k]
    if os.path.exists(path):
        return path
    work_root = os.path.join(work_root or lhe_work_dir(), "parts", f"{process}__{name_k}")
    os.makedirs(work_root, exist_ok=True)
    log_path = os.path.join(work_root, f"{process}.mg5.log")
    open(log_path, "w").close()
    t0 = time.time()
    if cfg["lhe"].get("gridpack"):
        # lhe.gridpack: the run is drawn from its energy's gridpack (lhe_build_gridpack, made first) with the run's
        # seed and events; its banner is the LHE header, the gridpack integration's cards
        grids = lhe_gridpack_paths(process, items)
        tar = grids[lhe_gridpack_of_run((iseed_k, n_k, patches_k, name_k), [g for _, g in grids])][0]
        if not os.path.exists(tar):
            raise RuntimeError(f"{process} {name_k}: its gridpack {tar} is not made yet")
        t_out = 0.0
        lhe = lhe_run_gridpack(tar, os.path.join(work_root, "gridpack_run"), n_k, iseed_k, log_path)
        t_me = time.time() - t0
        m_k, a_k, _ = read_lhe_pp(lhe, cfg["pdg_ids"])
        banner_txt = lhe_gridpack_header(lhe)
    else:
        ev, _ = lhe_prepare(process, work_root, log_path, madevent=True, standalone=False)
        t_out = time.time() - t0; t0 = time.time()
        lhe, banner = lhe_run_madevent(process, ev, n_k, iseed_k, log_path, mg_cores, run_name=name_k,
                                       patches=patches_k)
        t_me = time.time() - t0
        m_k, a_k, _ = read_lhe_pp(lhe, cfg["pdg_ids"])
        banner_txt = open(banner).read() if banner else ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-len(".npz")] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, mom=m_k, aqcd=a_k, seconds=np.array([t_out, t_me]), mg_cores=np.array(int(mg_cores)),
             second_refine=np.array(lhe_second_refine() if lhe_second_refine() is not None else np.nan),
             banner=np.array(banner_txt))
    os.replace(tmp, path)
    if len(m_k) >= n_k and not keep_dir:     # a short run keeps its directory (its logs)
        shutil.rmtree(work_root, ignore_errors=True)
    print(f"[LHE PART] {process} {name_k}: {len(m_k)} events (asked {n_k}, iseed {iseed_k}) in {t_me:.0f}s "
          f"-> {path}", flush=True)
    return path


def lhe_assemble(process, items, mg_cores=1, write_sidecars=True, work_root=None):
    """The pools `items` from their part files (every run's, lhe_build_part): the standalone built if missing, the
    parent physics check, then _lhe_finish on the runs' events in run order. A missing part raises."""
    parts = lhe_part_paths(process, items)
    missing = [p for p, _ in parts if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"{process}: {len(missing)} of {len(parts)} part files missing, e.g. {missing[0]}")
    seed, iseed, n_req = _lhe_check_recipes(process, items)
    work_root = work_root or lhe_work_dir()
    os.makedirs(work_root, exist_ok=True)
    log_path = os.path.join(work_root, f"{process}.assemble.mg5.log")
    open(log_path, "w").close()
    t0 = time.time()
    _, sa = lhe_prepare(process, work_root, log_path, madevent=False, standalone=True)
    lhe_parent_physics_check(process)
    secs = {"output": time.time() - t0, "madevent": 0.0}
    if not PROCESSES[process]["lhe"].get("parts"):
        raise RuntimeError(f"{process}: no lhe.parts, its pools are built in series (build_lhe_pp)")
    complete = bool(PROCESSES[process]["lhe"].get("complete_runs"))
    moms, aqcds, banner, used, dropped = [], [], None, [], []
    for k, (p, run) in enumerate(parts):
        z = np.load(p)
        m_k = np.asarray(z["mom"])
        secs["output"] += float(z["seconds"][0]); secs["madevent"] += float(z["seconds"][1])
        if complete and len(m_k) < int(run[1]):            # a run short of its ask: not used (lhe.complete_runs)
            dropped.append({"run": run[3], "iseed": int(run[0]), "asked": int(run[1]), "got": int(len(m_k))})
            continue
        moms.append(m_k); aqcds.append(np.asarray(z["aqcd"])); used.append(run)
        mg_cores = max(int(mg_cores), int(z["mg_cores"]))     # recorded: the most cores a part ran on
        if banner is None and str(z["banner"]):
            banner = p[:-len(".npz")] + ".banner.txt"
            with open(banner, "w") as f:
                f.write(str(z["banner"]))
    if complete:
        need = max(r["lhe"]["block"][1] for r, _ in items)
        if sum(len(m) for m in moms) < need:              # checked before anything is set aside
            raise RuntimeError(f"{process}: its {len(moms)} complete runs hold {sum(len(m) for m in moms)} events, the "
                               f"blocks need {need} ({len(dropped)} runs stopped short); raise lhe.n_runs")
        try:                                               # an e+e- pool: each energy's equal share, also before
            lhe_equal_shares(process, moms, aqcds, used, need)
        except RuntimeError as e:
            raise RuntimeError(f"{e} (complete runs only, {len(dropped)} stopped short: raise lhe.n_runs)") from None
        lhe_set_aside(process, items)
    # secs["madevent"]: summed over the part jobs (each run's wall time, not this job's)
    out = _lhe_finish(process, [r for r, _ in items], items, moms, aqcds, used, banner, secs, sa,
                      seed, iseed, n_req, mg_cores, False)
    for path, full in out:
        if complete:
            full["lhe_run"]["runs_dropped_short"] = dropped
        if write_sidecars:
            write_recipe(path, full)
    return out


def lhe_set_aside(process, items):
    """Pools of `items` on disk under another identity (e.g. assembled from runs that stopped short) moved, with their
    sidecars and banner copy, to <pool dir>/set_aside/<process>_<time>/ before a pool of complete runs takes their
    path. Nothing is deleted; a pool with the identity asked is left in place. Returns the paths moved."""
    import datagen
    moved = []
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for rec, path in items:
        if not os.path.exists(path) or datagen._is_cached(path, recipe_id(rec)):
            continue
        dest = os.path.join(os.path.dirname(path), "set_aside", f"{process}_{stamp}")
        os.makedirs(dest, exist_ok=True)
        stem = path[:-len(".npy")]
        side = {}
        try:
            side = json.load(open(path + ".recipe.json"))
        except (OSError, ValueError):
            pass
        b = (side.get("lhe_run") or {}).get("banner")
        for f in [path, path + ".recipe.json", stem + ".alphas.npy", stem + ".aqcdup.npy"] + ([b] if b else []):
            if f and os.path.exists(f):
                shutil.move(f, os.path.join(dest, os.path.basename(f)))
                moved.append(f)
    if moved:
        print(f"[LHE] {process}: {len(moved)} files under another identity set aside", flush=True)
    return moved


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
        generate_mg5_process(args.process, config, events=True)

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
