# Empirical HPO search-range rules (harvested from all Bayesian sweeps)

Aggregated **771** DyHPO sweeps on disk (`sweeps/**/dyhpo_state.pkl`). Kept the
**422** that are *converged* in your sense: the effective learning-rate knob
(`training.lr` for pretrain/solo/scaling sweeps, `fine_tune.lr_scale` for
finetune sweeps) does **not** sit at the edge of the explored range
(reusing `analyze_lr_boundary.analyze_sweep`'s tested-range edge test), and the
sweep has ≥6 observations.

Regenerate: `python sweep/aggregate_hpo_optima.py` → `hpo_optima.{json,csv}`,
then `analyze_optima.py` / `hp_importance.py` / `make_fig.py` on the json.

Split of converged sweeps: **338** `training.lr` sweeps, **84**
`fine_tune.lr_scale` sweeps. Comparison metric is each sweep's own best
val_loss (DyHPO objective); optima are the HP config at that minimum.

---

## Headline result #1: lr transfers across width (μP), but NOT across iterations

**`training.lr` optimum is flat in width — that part of μP holds** — but it is
**strongly non-monotonic in training length** (an inverted-U). Best-lr geomean by
axis:

| axis | range scanned | best-lr behaviour |
|------|---------------|-------------------|
| **width** `num_heads` | 2 → 32 (16×) | 6.0e-3, 3.6e-3, 3.4e-3, 3.0e-3, 2.9e-3 — **flat** (μP transfer ✓) |
| **iterations** `t_steps` | 4 → 6×10⁵ | **INVERTED-U** at fixed D — see below (a single slope is ≈0 and *misleading*) |
| **data size** `n_train` | 700 → 70000 | **weak positive**: `lr* ∝ n_train^{+0.15..0.3}` at fixed t (~2× over 100× data, saturating) |
| **batch / #proc** | 256↔16384, solo↔8-proc | **cannot isolate** — entangled with t_steps and regime |

Disentangled on the clean `scaling_p` 2D grid (`nh × n_train × t_steps`, width
marginalized since lr is width-flat) — see `lr_2d_disentangled.png`:
* **down each column (fix t, vary data): weak.** Optimal lr rises mildly with
  dataset size, slope +0.15–0.3 in log-log, saturating. *This is the dataset-size
  dependence — it's real but small, which is why it never jumped out of the
  marginal plots.*
* **along each row (fix data, vary t): the inverted-U, at every single D.**

> ⚠️ Correction to an earlier version of this note: I first reported lr as
> "invariant to iterations." That was wrong — it came from fitting one straight
> line in log-log, whose slope (+0.09/dec, r=0.19) averages the rise and fall of a
> hump to nearly zero. Binning best-lr by `t_steps` and plotting the geomean (as
> was already done for width/batch) makes the trend obvious.

### The iterations law (inverted-U, peak t\* ≈ 3000 steps)

Splitting at the peak gives two clean power laws, consistent across regimes
(numbers = slope `dlog10 lr / dlog10 t`, Pearson r):

| branch | joint 8-proc (bs 256) | solo (bs 16384) | pooled |
|--------|-----------------------|-----------------|--------|
| ascending  (t ≲ 3000) | **+0.49** (r 0.68) | +0.51 (r 0.71) | +0.43 (r 0.63) |
| descending (t ≳ 3000) | **−0.58** (r 0.68) | −0.21 (r 0.22) | −0.57 (r 0.62) |

i.e. `lr*(t) ≈ lr_peak · min[(t/t*)^{+0.5}, (t/t*)^{−0.55}]`, with **t\* ≈ 3×10³**,
**lr_peak ≈ 8×10⁻³** (joint) to ~1.5×10⁻² (solo). Geomean lr by t_steps (well-
populated bins): t=4→7.8e-4, t=100→1.1e-3, t≈1–3k→**6–8e-3 (peak)**, t=10k→2.5e-3,
t=31k→1.6e-3, t=100k→7e-4.

**The peak is at an absolute ~3×10³ steps, independent of dataset size**
(argmax-t = 3162 for *every* n_train from 700 to 70000, a 100× range). The
convergence knee (where `best_val_loss` floors, panel C) instead shifts *later*
with D — so the peak is **not** "the convergence point"; it's pinned to an
absolute step scale. That points to an **optimizer/schedule timescale**, not a
data effect: Adam's 2nd-moment EMA has timescale `1/(1−β₂) ≈ 10³` steps, warmup is
`warmup_frac·t` (and optimal warmup collapses toward 0 at the shortest horizons,
0.064 @ t=10), and the weight EMA adds another ~10²–10³-step scale.

### Why this matches (not contradicts) intuition

* **Descending branch (t ≳ 3k), the intuitive one:** past the optimizer's
  steady-state/convergence scale, extra steps are spent polishing an
  already-fit model, and optimal peak-lr falls as ≈ `t^{−0.5..−0.6}` — exactly
  "more iterations ⇒ lower lr." This is where standard intuition lives.
* **Rising branch (t ≲ 3k), the counterintuitive one:** with fewer steps than
  Adam's variance-EMA and warmup timescales you're **not in steady state** — a
  large peak lr can't be exploited (you'd end mid-warmup or overshoot with no
  time to anneal), so the best lr is *lower* and grows as you add steps. We just
  rarely sweep lr this deep in the undertrained regime, so we never built
  intuition for it. These are also mostly the cheap low-fidelity DyHPO probes,
  not real runs.

### What this means for a real (scaled-up) run — corrects an earlier claim

A production run grows **both** data and steps. The two structured effects then
**partly cancel**: the negative t-slope past the peak vs the positive D-slope. So
along a realistic compute-scaling ray optimal lr stays **near the peak, ~5e-3 to
1e-2** (at the base μP width) and is fairly *stable* — which is why the marginal
looked flat. My earlier "10⁵ steps ⇒ lr 1e-3" was wrong: that came from
*over-training small subsampled datasets* (fixed tiny D, many steps → deep into
the descending branch). On full data at 10⁵ steps you're still under-converged →
stay near the peak lr (~1e-2), **don't** lower it just because t is large. Only
lower lr when you knowingly train a fixed dataset well past its loss floor.

μP still earns its keep: **lr transfers across width at matched t_steps**, so
re-sweeping lr per width remains pure waste.

`fine_tune.lr_scale` optimum ≈ **1.0** (median 1.26, p5–p95 = 0.13…7.7): the best
finetune lr is ≈ the pretrain lr. Declared `[5e-3, 50]` (4 decades) is ~2 decades
too wide; `layer_decay` optimum ≈ 0.88 (p5–p95 0.74…1.0).

## Which knobs actually matter (mean |Spearman(HP, val_loss)| within a sweep)

`training.lr` sweeps: **lr 0.29** > sampler_alpha_ema 0.27 > reg_lambda 0.23 >
warmup_frac 0.22 > ema_decay 0.20 ≈ eta_min 0.20 ≈ min_alpha_frac 0.20.
`fine_tune.lr_scale` sweeps: **lr_scale 0.48** > layer_decay 0.29 ≈ warmup 0.29 >
reg_lambda 0.27 ≈ eta_min 0.26.

So **lr is the dominant knob** (much more so for finetune). The rest carry
*moderate* signal — narrow them, don't blindly fix them. The exotic sampler
variants (`sampler_sig_k`, `sampler_deficit_*`, …) show |ρ|≈0.08–0.11 (n=1 sweep
each) → noise, drop them.

## Where optima land vs the declared range (⇒ how much to cut)

| HP | declared | optima p5–p95 | recommendation |
|----|----------|---------------|----------------|
| `training.lr` | 3.2e-5…3e-1 (4 dec) | 3.5e-4…2.1e-2 (1.8 dec) | **center on lr\*(t_steps)** (below), sweep ±½ dec |
| `regularization_lambda` | 1e-11…1e-2 (9 dec) | 2.5e-11…9.8e-7 | **cap high at 1e-6**; use [1e-10, 1e-6] |
| `cosanneal_eta_min` | 1e-11…1e-6 | spans full range, weakest | **fix at ~1e-8** (or [1e-10,1e-7]) |
| `cosanneal_warmup_frac` | 0…0.2 | 0.007…0.2 (median 0.15) | **[0.05, 0.2]** |
| `ema_decay` | 0.9…1.0 | 0.90…1.0 (median 0.96) | keep [0.9, 0.999] |
| `sampler_alpha_ema` | 0.3…0.95 | spans full | keep (2nd most important) |
| `sampler_min_alpha_frac` | 0.05…0.5 | spans full | keep, or fix ~0.3 |
| `fine_tune.lr_scale` | 5e-3…50 (4 dec) | 0.13…7.7 (1.8 dec) | **[0.1, 10]** (2 dec), center 1 |
| `fine_tune.layer_decay` | 0.65…1.0 | 0.74…1.0 | **[0.75, 1.0]** |

## Recommended default search space (pretrain / solo `training.lr` sweeps)

Center the lr window on the horizon **relative to the ~3k-step optimizer scale**,
and sweep only ±½ decade around it:

```
lr_center(t) = 1.0e-2 * min( (t/3000)**0.5, (t/3000)**-0.55 )   # peak 1e-2 @ ~3k steps
low  = lr_center / 3 ;  high = lr_center * 3
```
BUT the decay side (`t>3k`) only applies when the dataset is **fixed and being
over-trained**. For a **real scaled run** (data grows with steps ⇒ under-converged)
just **sweep the peak band, `[3e-3, 3e-2]`**, at every horizon — the t and D
effects cancel and the optimum stays near the peak. Rule of thumb: is
`best_val_loss` still dropping at your budget? → under-converged → use the peak
band. Has it floored? → over-converged → apply the `t^{-0.55}` decay.
Either way the range is ~1 decade vs the current 4.

```yaml
search_space:
- {name: training.lr,                     type: float_log,     low: LR_LOW,  high: LR_HIGH}  # from lr_center(t_steps) ±3×; was 3e-5..3e-1
- {name: training.cosanneal_warmup_frac,  type: float_uniform, low: 0.05,    high: 0.2}
- {name: training.ema_decay,              type: float_uniform, low: 0.9,     high: 0.999}
- {name: training.regularization_lambda,  type: float_log,     low: 1.0e-10, high: 1.0e-6}   # was ..1e-2
# fix (do not sweep): training.cosanneal_eta_min = 1e-8
# pretrain only, keep: sampler_alpha_ema [0.3,0.95], sampler_min_alpha_frac [0.05,0.5]
```

Finetune sweeps: `fine_tune.lr_scale` float_log **[0.1, 10]**,
`fine_tune.layer_decay` **[0.75, 1.0]**, same reg/warmup/eta_min rules.

## Compute savings

Current sweeps: median **300 candidates / ~15 observations** over a 4-decade lr
range + 5–7 loosely-bounded knobs.

1. **Never re-sweep lr across width** — μP transfers it. Tune HPs once at the
   smallest width, reuse for all larger widths. On a width-scaling grid
   (nh ∈ {2,4,8,16,32}) that's up to a **5× cut in sweeps**.
2. **Center lr on lr\*(t_steps)** (the inverted-U law) and sweep ±½ decade
   instead of a flat 4-decade prior; fix eta_min, cap reg_lambda, drop the exotic
   sampler knobs. Together this shrinks the search volume ~10–100×, so the *same*
   trial budget explores it far more densely — or **drop trials/candidates roughly
   in half** (e.g. 300→150 candidates, ~15→~8 obs) at equal resolution.
3. **Seed new cells from the law, not a flat prior.** lr\*(t) is smooth in
   t_steps and transfers across width, so a good lr at one horizon predicts its
   neighbours — start DyHPO's candidates around lr_center(t) rather than sampling
   the whole range. (Do NOT assume batch/data-size invariance — those axes are
   entangled with t_steps here and were never cleanly isolated.)

Caveat: derived `n_train` / `eff_bs` are exact for `subsample`+`train_test_val`
sweeps (majority); source/`train_subsample` sweeps (pretrain25, encab, preproc)
are estimated (`size_reliable=false` in the json). This mostly affects the
data-size / batch axes — which are already entangled with t_steps and not used
for a strong claim. The width-transfer and iterations-hump results stand on the
clean joint grid (fixed batch & #processes, width and t_steps varied on a grid).
