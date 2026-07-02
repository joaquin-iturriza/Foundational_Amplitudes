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

## Headline result: the optimum barely moves — μP + Adam deliver

**`training.lr` optimum ≈ 3×10⁻³, and it is essentially invariant to everything
we scanned.** Best-lr geomean by axis:

| axis | range scanned | best-lr behaviour |
|------|---------------|-------------------|
| **width** `num_heads` | 2 → 32 (16×) | 6.0e-3, 3.6e-3, 3.4e-3, 3.0e-3, 2.9e-3 — **flat** (slope ≈ −0.004/head, r=−0.07) |
| **batch size** | 256 → 16384 (64×) | all ~2.5–4.8e-3 — **flat** (slope +0.08/decade) |
| **data size** `n_train` | 7 distinct | slope +0.06/decade, r=+0.09 — **flat** |
| **iterations** `t_steps` | 4 → 6×10⁵ | slope +0.09/decade, r=+0.19 — weak, near-flat |
| **# processes** | solo=5.1e-3, 8-proc=2.7e-3 | within 2× |

μP is doing exactly its job: **lr transfers across width**, so re-sweeping lr per
width is pure waste. Adam + μP also make lr near-insensitive to batch size and
data size in this regime. **90% of all lr optima fall inside a single decade**
(`p5–p95 = 3.5e-4 … 2.1e-2`), vs the declared 4-decade range `[3.2e-5, 3e-1]`.

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
| `training.lr` | 3.2e-5…3e-1 (4 dec) | 3.5e-4…2.1e-2 (1.8 dec) | **[1e-3, 1e-2]** (1 dec), center 3e-3 |
| `regularization_lambda` | 1e-11…1e-2 (9 dec) | 2.5e-11…9.8e-7 | **cap high at 1e-6**; use [1e-10, 1e-6] |
| `cosanneal_eta_min` | 1e-11…1e-6 | spans full range, weakest | **fix at ~1e-8** (or [1e-10,1e-7]) |
| `cosanneal_warmup_frac` | 0…0.2 | 0.007…0.2 (median 0.15) | **[0.05, 0.2]** |
| `ema_decay` | 0.9…1.0 | 0.90…1.0 (median 0.96) | keep [0.9, 0.999] |
| `sampler_alpha_ema` | 0.3…0.95 | spans full | keep (2nd most important) |
| `sampler_min_alpha_frac` | 0.05…0.5 | spans full | keep, or fix ~0.3 |
| `fine_tune.lr_scale` | 5e-3…50 (4 dec) | 0.13…7.7 (1.8 dec) | **[0.1, 10]** (2 dec), center 1 |
| `fine_tune.layer_decay` | 0.65…1.0 | 0.74…1.0 | **[0.75, 1.0]** |

## Recommended default search space (pretrain / solo `training.lr` sweeps)

```yaml
search_space:
- {name: training.lr,                     type: float_log,     low: 1.0e-3,  high: 1.0e-2}   # was 3e-5..3e-1
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
2. **lr range 4→1 decade + fix eta_min + cap reg_lambda + drop exotic sampler
   knobs** shrinks the search volume by ~10–100×, so the *same trial budget*
   explores it far more densely — or **drop trials/candidates roughly in half**
   (e.g. 300→150 candidates, ~15→~8 obs) for equal resolution.
3. lr is also **batch- and data-size-robust**, so a good lr found at one
   (D, t_steps) cell transfers to neighbours — you can seed new cells at 3e-3
   instead of re-searching from a flat prior.

Caveat: derived `n_train` / `eff_bs` are exact for `subsample`+`train_test_val`
sweeps (majority); source/`train_subsample` sweeps (pretrain25, encab, preproc)
are estimated (`size_reliable=false` in the json). The invariance conclusions
hold regardless since lr is flat along *every* axis.
