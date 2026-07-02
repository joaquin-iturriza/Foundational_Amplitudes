# Internal-mass off-shellness feature + big-run physics-scan — progress

Status: the **propagator off-shellness** feature (`data.offshell_per_event`) is
implemented, validated, and incorporated as the production internal-mass conditioning
for the big-run pretraining. All numbers below are `val_loss_no_reg` (MSE on the
per-dataset-standardized log-amplitude), reported in scientific notation.

## 1. The design principle

Feature value = **(new information) × (learnable interaction)**. Three physics levers
tested against this:

- **coupling** α_s(√s): new per-event info via running; earns its keep ∝ (α range) ×
  (running steepness). Folded into the big run (wide α ∈ [5.0e-2, 2.5e-1], √s extended
  down to 25 GeV so the running shape survives per-dataset standardization).
- **external mass** (final-state t/τ/H): **dropped** — a final-state mass is a
  per-dataset offset on |M|² that per-dataset standardization removes; the model
  already reads it from the momenta (`mass_from_momenta`). No learnable per-event signal.
- **internal mass** (s-channel/internal propagator): the mass is **hidden** from the
  external momenta, so the propagator OFF-SHELLNESS `s_prop − M²` is genuinely new
  per-event info. A bare per-dataset mass scalar cannot use it (the √s×M interaction is
  the wall); feeding `s_prop − M²` directly to the main transformer solves it.

Key negative result: routing the off-shellness through the **diagram encoder** (Tier-B
virtuality) dilutes it — the encoder is a good conduit for topology, not for a precise
per-event kinematic scalar. It must be fed DIRECTLY into the `order_labels` channel.

## 2. The feature (`data.offshell_per_event`)

`experiment.py` + `diagram_graphs.py`. For each `data.internal_mass_pdgs` entry, use the
diagram's propagator masks (`build_process_virtuality`) to compute per-event
`s_prop = (Σ signed·p_slot)²` for the propagator(s) carrying that PDG, subtract the
(scanned) `M²`, nearest-pole aggregate when a PDG appears in several channels, signed-log
+ unit-std, and overwrite the internal-mass slot. Matches propagator PDGs on `|pdg|`
(self-conjugate γ/Z come out sign-arbitrary). Fed direct to the main transformer, NOT the
diagram encoder. Independent of `model.use_diagrams` (only needs the sidecar for masks).

## 3. The big-run recipe (`recipes/gen_scan_bigrun.py` → `scan_bigrun.yaml`)

448 datasets: 260 LO (20 anchors + 240 α_s scans) + 60 s-channel Z-mass + 32 exotic-mass
+ 96 NLO. All LO share one √s range `[25,1000]` (clamped up by threshold).

Internal-mass levers (all masses are free param_card inputs → hidden, scannable):
- **s-channel Z (pdg 23)** in `ee→ff̄`, M_Z ∈ ~[80,114] GeV × 12, resonance-dense √s
  windows. Floor ≥0.88× (~80 GeV, > M_W): below M_Z≈77 the on-shell relation
  `M_W²=M_Z²/2(1+√(1−4πα/√2 G_F M_Z²))` has no real root → NaN couplings.
- **top (pdg 6)** in `ee→W⁺W⁻bb̄` (`t→Wb` resonance in M(Wb)), MT ∈ ~[138,207] × 10.
- **Higgs (pdg 25)** in `ee→μ⁺μ⁻τ⁺τ⁻` (`ee→ZH, Z→μμ, H→ττ`; M(ττ) resonance), MH ∈
  ~[100,150] × 10. (`ee→WWbb` H→WW was rejected: external on-shell W → M(WW) ≥ 2M_W =
  161, so a physical M_H can never be reconstructed there.)
- **Z-in-4-lepton (pdg 23)** in `ee→4μ` (μμ-pair resonance), M_Z ∈ ~[80,114] × 12 —
  multi-Z, exercises the nearest-pole aggregation.

## 4. Pipeline bugs found and fixed along the way

- **RAMBO has no cuts** — variable-energy generation samples flat phase space and
  bypasses MadGraph's run-card cuts entirely, so every dataset carried the full
  soft/collinear/forward/low-mass IR tail (log|amp| span up to ~31). Added fiducial cuts
  (pt>10, |cosθ|<0.9, ΔR>0.4, m(pair)>10 GeV) on massless visible finals via
  oversample-and-reject in `sample_nbody_phase_space`/`sample_2to2_phase_space`; threaded
  through the NLO path too. Cut params enter `recipe_id`. σ dropped e.g. bhabha 2.33→1.72.
- **Stale param_card on standalone reuse** — standalones are keyed by process name and
  reused across recipe versions; the param_card was only patched on first compile, so a
  mass-scan whose masses changed silently generated at the OLD mass (NaN when the stale
  M_Z fell below the on-shell root). Fixed: `repatch_standalone_param_cards()` on every
  `ensure_backend` reuse for own-backend (mass/EW) scans.
- **Amplitude transform resolved globally** — one NLO-virtual dataset with negative
  amplitudes forced `signedlog` GLOBALLY. `signedlog(x)=sign(x)·log1p(|x|)` only
  compresses |x|≫1, but every |M|² is ≪1 (down to 1e-13), so it left the targets in
  LINEAR scale → per-dataset standardization couldn't reach unit variance (target std
  8–30, one up to 1.2e4) → most datasets did not train (combined val stuck at 1.7,
  ee_aa above predict-the-mean). Fixed: resolve the trafo PER-DATASET (positive → log,
  signed NLO virt → signedlog). 416 log / 32 signedlog; global prepd std 282 → 1.06.
- **Concurrent numpy import segfaults on Lustre** — the NLO/MadLoop path spawns a fresh
  interpreter per chunk; several importing numpy's C-extension from Lustre at once
  segfault (`PyCapsule_Import datetime`). Run NLO prebuild at `--workers 1`. LO is
  immune (imports numpy once in-process).
- **Login-node fork exhaustion** — running LO (24w) + NLO (8w) + MadLoop children at once
  hit `BlockingIOError` on `_fork_exec`. Run one generation job at a time, ≤~12 workers.

## 5. Validation — focused off-vs-offshell A/B

44 lever datasets (ee_mumu Z-scan, ee_mumumumu Z-4ℓ, ee_wwbb top, ee_mumutautau Higgs),
FULL train data, 30k iters. off = no features; offshell = internal_mass_pdgs=[23,6,25].

| lever            | off (median)   | offshell (median) |
|------------------|----------------|-------------------|
| Z (2→2 s-channel)| 1e-1 .. 1.6    | 1e-6 .. 2e-4      |
| top (2→4)        | 4.2e-1         | 5.4e-2            |
| Z-in-4ℓ (2→4)    | 1.7e-1         | 5.9e-2            |
| Higgs (2→4)      | 1.0e-1         | 7.7e-2            |

The s-channel Z is textbook: `off` traces the regress-to-mean U (worst at the wings
where M_Z is farthest from typical), `offshell` flattens it to ~1e-6..1e-4. Not a leak
or a degenerate metric: the `off` arm gets 1.6 on the SAME val set (it lacks only the M_Z
info); amplitudes are deterministic (no noise floor) and ee→μμ is a smooth low-D
(s, cosθ) target, so with the feature it is genuinely easy. The 2→4 levers land at
sensible ~5e-2 (more amplitude structure), which is the tell that the small Z number is
physics, not a bug.

The joint 448-dataset smoke plateaued the levers at ~2e-1 — that was **dilution +
data-starvation** (`train_subsample=2000` cap on 5k–10k-event datasets), NOT a 2→4
limit: with full data and no dilution the levers reach the values above.

## 6. Open items

- **Higgs lever is weak** (H→ττ, Z→ττ swamps the small τ-Yukawa). Keep as a scalar-
  propagator example, or build the strong version H→bb̄ with a massive b (4-flavor scheme).
- **NLO signedlog residual**: the 32 negative-amplitude NLO datasets sit at target std
  ~1.5 (vs 1.0), since signedlog still under-compresses sub-1 magnitudes near sign flips.
- **Full 448-dataset joint A/B** (foundation-model setting): needs full data
  (`train_subsample=null`, now set) and a higher fidelity than 5000 steps — re-estimate
  GPU cost before launching.

## Production setting
`data.mass_from_momenta=true data.coupling_scalars=true data.internal_mass_scalars=true
data.offshell_per_event=true data.internal_mass_pdgs=[23,6,25]` on the recipe path with a
diagram sidecar per process; per-dataset preprocessing; fiducial cuts on
(`AMP_FIDUCIAL_CUTS=on`); cut-tagged cache `$SCRATCH/{datasets,amp_cache}_scanbig_cut`.
