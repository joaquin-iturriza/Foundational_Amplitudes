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

## 4. Next steps

- Add an **error-vs-|M|²-percentile** curve (question 2) to quantify tail behaviour.
- Run the **restricted-window fine-tune** experiment (question 3) for a clean
  extrapolation number, on the same two processes.
- Extend to real-emission NLO (uug/uugg) to probe genuine IR divergences.
