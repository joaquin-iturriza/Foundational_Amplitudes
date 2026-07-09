# Behaviour near amplitude divergences & extrapolation — new research direction

Opened 2026-07-09, following collaborator feedback.

## 1. The direction

So far the project has been judged mostly on **aggregate accuracy**: can one
Lorentz-equivariant foundation model *learn* many processes jointly, and *transfer*
(fine-tune) to new processes/orders with good pooled val loss. Collaborators are
interested in a sharper, physics-driven question:

> How does the model behave on the **divergence structure** of the amplitudes —
> the regions where |M|² grows large or varies fastest (soft/collinear limits,
> s-channel resonances, production thresholds, Sudakov/IR logs at NLO)?
> Does accuracy **degrade** there, and can the model **extrapolate** into those
> regions (including sparsely-sampled tails and kinematics outside training
> support)?

Why it matters: in real use the peaked / near-divergent regions are exactly the
ones that dominate cross sections and are hardest for surrogates. A foundation
model for amplitudes is only useful if it is *reliable where the physics lives*,
not just on average. Aggregate `val_loss_no_reg` can look excellent while a thin,
high-|M|² tail is mispredicted — and that tail can carry the phenomenology.

This reframes the evaluation from a single pooled number to a **resolved, phase-
space-local** picture: error as a function of where you are in phase space and of
how large / how rare the amplitude is.

## 2. Concrete questions to answer

1. **Degradation map.** Bin the model error (in log|M|²) over the physical phase
   space and localize where it is worst. Is the worst error co-located with the
   divergences (low √s / resonance / forward-backward peaks / threshold)?
2. **Error vs amplitude magnitude.** Does |Δ log|M|²| rise with |M|² percentile?
   (i.e. is the model systematically worse on the largest, most physical events?)
3. **Interpolation vs extrapolation.** Separate "sparsely sampled but in-support"
   from "outside training support". A clean test: fine-tune on a *restricted*
   kinematic window (e.g. |cosθ*| < 0.8, or √s below some cut) and measure error
   in the held-out window beyond the cut — true extrapolation, not just rare bins.
4. **Order/threshold specifics.** Near-threshold e+e-→ttbar (√s → 2m_t), the
   Z-pole region for e+e-→uu (√s ≈ 91 GeV), and eventually the genuine IR
   structure of real-emission NLO (uug, uugg) where soft/collinear singularities
   are explicit.

## 3. First deliverable — true vs predicted across the 2→2 phase space

For the **best fine-tuned** models of the two NLO virtual processes, visualize the
true and predicted log-amplitude across the phase space and localize the error.

- Processes / datasets: `ee_uu_nlo_virt_e4` and `ee_ttbar_nlo_virt_e4`
  (both 2→2: e-, e+, f, f̄; s-channel γ*/Z).
- Best fine-tunes (lowest `val_loss` = best non-reg val loss, over all
  `sweeps/*/results/*.json`; both fine-tuned from `pretrain_full_nh8/trial_0271`,
  D=100k, 40k steps, `reset_output_head:false`):
  - eeuu: `runs/finetune_scaling_virt_002_eeuunlovirte4_t40000/trial_0154`
    (val 4.12e-8, test 9.98e-8)
  - eett: `runs/finetune_scaling_virt_002_eettbarnlovirte4_t40000/trial_0154`
    (val 2.65e-9, test 1.25e-7)
- Phase-space coordinates: `√s` (Lorentz invariant of the initial pair) and the
  c.o.m. scattering angle `cosθ*` between the incoming e⁻ and the outgoing
  fermion. Both are invariant under the input Lorentz augmentation, so they are
  recovered directly from the (boosted, unit-scaled) momenta the model sees.
- Plots (per process): 2D maps of ⟨log|M|²⟩ truth vs model and the error map
  ⟨|Δ log|M|²|⟩ over (√s, cosθ*), plus 1D projections onto √s and onto cosθ*
  (truth vs model, with the mean |error| on a twin axis), plus a predicted-vs-true
  hexbin. A sparse-statistics contour flags the extrapolation-prone bins.

### Tooling
- `analysis/divergences/extract_preds.py` — reloads a fine-tune run's exact config
  + preprocessing + μP model, loads the fine-tuned checkpoint, forward pass (GPU;
  xformers block-diagonal attention is CUDA-only, so this runs via `sbatch`, not on
  CPU — a CPU forward is ~1.5 s/event, infeasible), and saves per-event
  `(√s, cosθ*, true/pred log-amp, split)`.
- `analysis/divergences/make_plots.py` — builds the figures (png + pdf) from the npz.
- `analysis/divergences/extract.sh` — sbatch driver: extract both processes + plot.
- Figures: `analysis/divergences/figs/phase_space_{eeuu,eett}_nlo.{png,pdf}`.

### First results (2026-07-09)

Reproduction is faithful: the extracted per-split MSE(prepd) matches each run's own
end-of-training log to 4–5 sig figs (e.g. eett val 2.660e-9 vs logged 2.6599e-9;
eeuu val 4.117e-8 vs 4.117e-8). **Gotcha fixed:** these runs (2026-06-17) predate
the name-based `amp_orders` resolver (commit accef58, 2026-06-26). The stale 8-entry
`amp_orders` in the config fed `order_labels=[0,0]` at training time, but today's
resolver derives `[1,0]` from "nlo_virt" — feeding that wrong constant degraded
*every* prediction ~20–20000× (even memorized train events). `extract_preds.py`
pins the training-time value. Always sanity-check extracted MSE against the run's
log before trusting an error map.

Findings (both models, over 100k events across √s∈[91,1000] (uu) / [350,1000] (tt) GeV):

- The model reproduces log|M|² across the **whole** 2→2 phase space:
  RMS Δlog|M|² ≈ 1.7e-4 (eeuu), 9.4e-5 (eett); MAE ≈ 5e-5 / 2e-5.
- **No degradation at the divergences.** The near-divergent structure — the
  s-channel 1/s growth toward low √s, the ttbar threshold (√s→2m_t), and the
  forward/backward angular peaks — is tracked accurately; the projected error
  curves stay flat *through* the sharp rises, i.e. error is not correlated with
  |M|² magnitude here.
- The largest (still ~1e-4) errors sit at the **phase-space boundaries**
  (cosθ*→±1, √s extremes) and the sparsely-sampled corners — the extrapolation
  edges, not the physical peaks.

Caveat: these are 2→2 *virtual* NLO amplitudes, whose "divergences" are mild
(finite 1/s growth, threshold, angular peaks) — there is no explicit soft/collinear
IR singularity. The genuinely divergent case is **real-emission** NLO (uug, uugg),
which is the natural next target (see §4).

### Jointly-pretrained processes (NOT fine-tuned) — 2026-07-09

Extended the same phase-space view to processes the model learned *jointly* in a
pretraining mixture (no per-process fine-tune), from `pretrain_full_nh8/trial_0271`
(8-process joint pretrain: ee→WWZ, WW, ttbar, uug, uugg, γγ, γγγ, uu). Chose three
well-learned **2→2** processes with *distinct* divergence structures:

| process | pretrain test MSE | divergence structure |
|---|---|---|
| ee→γγ (`ee_aa`) | 9.0e-10 | t/u-channel **collinear** peaks at cosθ*→±1 (~1/sin²θ), √s-independent |
| ee→W⁺W⁻ (`ee_WW`) | 2.5e-8 | t-channel ν exchange → **forward** peak, grows with √s |
| ee→uu (`ee_uu`) | 6.1e-9 | s-channel, ~(1+cos²θ), no angular divergence (control) |

Findings: the joint model reproduces all three across the whole (√s,cosθ*) plane at
RMS Δlog|M|² ≈ 3.5e-4 (γγ) / 4.9e-4 (WW) / 4.5e-4 (uu). The **collinear divergence**
of ee→γγ (log|M|² rising to ≈10 at cosθ*→±1) and the **forward peak** of ee→WW are
tracked faithfully *through* the singular region; error rises only mildly toward the
cosθ*→±1 edges and sparse corners — same story as the fine-tuned NLO cases, now for
jointly-learned tree amplitudes. So the "no degradation at the divergence, small
error only at the phase-space boundary" behaviour is a property of the pretrained
foundation model, not just of fine-tuning.

Tooling: `analysis/divergences/extract_pretrain.py` (+ `extract_pretrain.sh`).
It loads ALL pretrain datasets together (needed to reproduce the global amp stats
and dataset-0 momentum norm faithfully), forwards the chosen 2→2 processes, and gets
exact physical kinematics by inverting the deterministic `default_rng(42)` event
shuffle back to the raw rows. It is scope-aware: detects GLOBAL vs PER-DATASET amp
preprocessing (`len(prepd_mean)>1`) and un-preprocesses each process with its own
stats — important for recipe-path pretrains (per-dataset), though `pretrain_full_nh8`
is files-path (global). A per-event `preprocess(raw_amp, stats)==stored true_prepd`
check (max|Δ|=0 here) guards the alignment/stats; the per-process MSE matches the
pretrain log (e.g. ee→uu 5.9e-9 vs logged 6.1e-9).
Figures: `analysis/divergences/figs/phase_space[_3d]_pretrain_{aa,ww,uu}.{png,pdf}`.

**Collinear-resolved view** (now **row 3 of every `make_plots.py` figure**, so it ships
in the same per-process PDF as the maps/projections/hexbin).
The mean-per-bin maps/projections *hide* the divergence: it lives in a razor-thin
cos→±1 sliver (for ee→γγ only 0.003% of events exceed log|M|²=6, all at |cosθ*|>0.9999,
peak 9.86), so a linear-cosθ* bin averages log|M|² down to ~+0.35 in its outer bin.
Re-plotting vs the signed collinear coordinate
  η = sign(cosθ*)·log₁₀[1/(1−|cosθ*|)]   (η=0 central, η→±6 collinear)
stretches the beam directions and turns the divergence into a visible linear ramp
(log|M|² ~ −ln(1−|cosθ*|)). Shows MEAN and MAX per η-bin. Result: ee→γγ is a symmetric
V ramping to ≈+8 at both edges; ee→WW an asymmetric forward-only ramp; ee→uu flat
(no collinear divergence — control, mild forward tilt = γ/Z A_FB). Model tracks truth
up the ramps; the error panel shows accuracy degrades only in the most collinear,
sparsest bins (ee→WW: into the forward peak; ~1e-3 elsewhere).

### Real IR channels: ee→uug (2→3) and ee→uugg (2→4) — 2026-07-09

The genuine test: gluon-emission channels with real SOFT (E_g→0) and COLLINEAR
(g∥quark) singularities, from the same 8-process joint pretrain (zero-shot).
IR observables, all Lorentz invariants from the raw momenta (`extract_ir.py`):
x_g = 2(p_g·Q)/s (gluon energy fraction; x_gmin = softest), y_ij = (p_i+p_j)²/s over
gluon-involving colored pairs, **y_min** = master IR-resolution variable (→0 in both
the soft and collinear limits). Plots (`make_ir.py`): 2D maps over
(log₁₀ y_min, log₁₀ x_gmin), the IR ramp log|M|² vs log₁₀ y_min (mean & max), the
soft-gluon ramp vs log₁₀ x_gmin, a pred-vs-true hexbin, and — for ee→uug — the Dalitz
plane (x_q, x_q̄). Faithful (align max|Δ|=0; MSE matches pretrain log: uug 1.5e-6 vs
4.4e-6 all-splits-vs-test, uugg 2.1e-4 vs 2.3e-4).

- **ee→uug** (pretrain MSE 4.4e-6): the model reproduces the soft (~1/x_g²) and
  soft+collinear (~1/y_min) ramps and the **full bremsstrahlung Dalitz** (collinear
  walls at x_q→1 / x_q̄→1 meeting at the soft (1,1) corner) — truth≈model across
  ~30 orders of magnitude in |M|², RMS Δlog|M|²=0.007. No IR degradation.
- **ee→uugg** (pretrain MSE 2.3e-4 — the HARDEST process in the set): the model still
  tracks the MEAN IR ramps faithfully into the deep IR, but this is the **first clear
  case of divergence-region degradation** — per-event error grows monotonically into
  the IR: ⟨|Δlog|M|²|⟩ ≈ 0.03 at y_min~0.1 → **0.27 at y_min~1e-4**, and the pred-vs-
  true band visibly widens (RMS 0.085). The error maps localize it to the deep
  soft/collinear corner.

**Bottom line for collaborators:** across 2→2 (incl. the ee→γγ collinear divergence)
and ee→uug, accuracy does NOT degrade at the divergences — only at phase-space
boundaries / sparse bins. Degradation *into* the singular region appears only for the
hardest, highest-multiplicity process (ee→uugg) in its deepest soft/collinear limits.
Figures: `analysis/divergences/figs/ir_pretrain_{uug,uugg}.{png,pdf}` (+ `_uug_dalitz`).

## 4. Next steps

- Add an **error-vs-|M|²-percentile** curve (question 2) to quantify tail behaviour.
- Run the **restricted-window fine-tune** experiment (question 3) for a clean
  extrapolation number.
- ee→uugg deep-IR degradation: does a short **fine-tune** (or more pretrain weight on
  uugg) recover the deep soft/collinear limit? Natural follow-up to the finding above.
