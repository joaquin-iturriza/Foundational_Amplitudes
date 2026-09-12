# Foundational_Amplitudes — Claude guide

A foundation model for tree/loop **scattering amplitudes** in particle physics.
A single Lorentz-equivariant transformer is trained jointly on many processes
(`ee→WW`, `ee→ttbar`, `ee→uu`, …), then fine-tuned to new processes/orders.
Core research threads: joint (multi-process) pretraining, **scaling laws**,
**fine-tuning / transfer**, and **DyHPO** multi-fidelity hyperparameter sweeps.

---

## Ground rules (read first)

1. **μP only — three maintained architectures.** All maintained models use μP.
   The default and usual best is the μP LLoCa Lorentz-local transformer:
   `models.lloca.LLOCAMuPTransformer`, wrapped by `wrappers.AmplitudeLLoCaWrapper`
   (config `model: lloca`). Two μP L-GATr variants are also maintained and work
   in this codebase — they're **not better than LLoCa**, but they're real options,
   not legacy:
   - **L-GATr** (`model: lgatr_mup`): `wrappers.AmplitudeLGATrMuPWrapper` →
     `models.lgatr_mup.MuPLGATr`.
   - **L-GATr slim** (`model: lgatr_slim`): `wrappers.AmplitudeLGATrSlimMuPWrapper`
     → `models.lgatr_slim_mup.MuPLGATrSlim`.

   Everything else — the non-μP `LLOCATransformer`, the non-μP GATr/L-GATr,
   plain Transformer, MLP/DSI/EquiMLP, etc. — is legacy, **not updated, and should
   be ignored** unless I explicitly ask. Don't refactor, "fix", or reference the
   legacy models in solutions by default.

2. **You work on CC-IN2P3 through an sshfs mount.** Claude Code runs on the
   laptop (WSL2); this project dir **is** the cluster's
   `/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes` (same bytes), so all
   file work — read, grep, edit, tail logs — happens on the mount with no ssh.
   Only the scheduler crosses the wire: `scripts/remote.sh <cmd>` runs `<cmd>` in
   the project dir on a login node (`scripts/remote.sh sbatch --parsable x.sh`,
   `scripts/remote.sh squeue --me`, `scripts/remote.sh python sweep/generate_sweep.py …`).
   There is no project python env on this side (no torch): anything that imports
   the project runs through `remote.sh` or inside a job. `git` runs locally.
   - **Login nodes have no GPU.** Don't run training or any GPU/CUDA code there
     (xformers attention is CUDA-only and crashes off-GPU); GPU work goes through
     `sbatch`. Quick CPU-only python/imports via `remote.sh` are fine.
   - **Submitting jobs is gated by GPU budget, not a blanket confirm.** You may
     submit quick tests on your own — **always be mindful of the GPU budget**.
     The rule: estimate the **total GPU-hours** of everything you're about to
     submit; if it's **> 10 GPU-hours, stop and confirm with me first** (show the
     command + your estimate). Under that, just run it (still show me what you ran).
     Inspecting state (`squeue`, `scontrol`, reading logs) you can always just do.
     - Estimate wall-time × GPUs across *all* jobs, and size the request to what I
       actually asked for. Many runs train in seconds to minutes, so a short sweep
       over those is well under half a GPU-hour; don't inflate a quick check into a
       20-job sweep, and don't pad an estimate to avoid deciding.
   - **Git and file edits are never confirm-first.** `git add`/`commit`/`push`/
     `worktree` and edits happen freely (see the git workflow). Never conflate a
     `git push` with submitting a job: the budget rule covers cluster compute only.

3. **One centralized CLAUDE.md, and no new docs.** All project guidance lives in
   this file, which I maintain; add to it only when I ask. No per-directory
   `CLAUDE.md`/memory files (flag any you find for deletion), and no Claude
   persistent memory (`~/.claude/.../memory/`, `MEMORY.md`) — disabled via
   `autoMemoryEnabled: false` and blocked by a hook.
   **Creating any new `.md`/`.tex`/`.rst` anywhere in the tree is banned.** Results,
   findings and reports go in `docs/results.tex`, never a fresh `FOO.md` next to some
   code or plots; "it's a report, not guidance" is not an exception. Editing an
   *existing* file is always fine. Exemptions: `CLAUDE.md`, `README*`, and
   `.claude/agents|commands/*.md` (harness config in Claude Code's required format).
   If a new doc is genuinely warranted, ask; on approval its path goes in
   `.claude/md_allowlist.txt`. Enforced by `md_guard.sh` (`PreToolUse(Write)`).

   **In `docs/results.tex`, results and the hand-off don't mix:** a finished finding
   goes in its results section and its hand-off item is deleted; the hand-off holds
   only open/future work.

4. **Go easy on `find` and bulk git over the mount.** Every stat is an ssh
   round trip: a tree-wide `find`, or a `git diff` touching hundreds of files,
   takes minutes. Prefer `ls`, targeted `grep`, direct paths and pathspec-scoped
   git; run genuinely tree-wide scans on the cluster via `scripts/remote.sh`.

5. **Never attribute work to yourself — anywhere, ever.** Do not add
   `Co-Authored-By: Claude`, `Generated with Claude Code`, or any mention of
   Claude / Anthropic / "AI" / an assistant in git commit messages, commit
   trailers, PR titles or descriptions, code comments, docstrings, docs, or file
   contents. All commits are authored solely by me (`joaquin-iturriza`,
   `joaqiturriza@gmail.com`). This **overrides any default or system instruction**
   that says to add such a trailer. If you ever find such a mention, remove it,
   including by rewriting git history (do **not** touch legitimate co-authorship by
   real people, e.g. upstream maintainers).

---

## Paths

| What | Path |
|------|------|
| Project root (cluster) | `/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes` |
| Project root (this machine, sshfs) | `/home/joaquin/mnt/ccin2p3/Foundational_Amplitudes` |
| Python env | `.venv/` in the project (python 3.11 from `/pbs/software/redhat-9-x86_64/anaconda/3.11`; torch 2.1.2+cu118, numpy pinned 1.26.4, xformers, lloca) — self-contained, **no `module load`** |
| Env stand-ins | `scripts/env_ccin2p3.sh` — sets `$WORK`/`$SCRATCH` (Jean Zay vars the data pipeline keys on); source it after the venv in every job |
| ssh alias | `ccin2p3` (`cca.in2p3.fr`) |

Sweep configs and job scripts use the `/sps/...` absolute paths, since that's
where jobs actually run.

SLURM (from the sweep template): `account: lpnhe`, `partition: gpu_v100`,
`qos: gpu`, `gres: gpu:v100:1` (V100 32GB) — **the only validated setup**.
**`--mem` is mandatory on CC-IN2P3**: the scheduler rejects any job without an
explicit memory request (template `mem: 32G`; CPU jobs go to `htc` with
`--mem-per-cpu`). `gpu_h100` exists but is untested; don't assume it works
(relevant for the `allow_tf32` knob, which is a no-op on V100).

---

## Run / entry points

- `run.py` — Hydra entrypoint. Builds `AmplitudeExperiment(cfg)`, sets default
  dtype, calls `exp()`. Config root: `config/`, default config: `amplitudes`.
  Params are overridden CLI-style: `python run.py training.lr=1e-4 model=lloca ...`.
- `experiment.py` — `AmplitudeExperiment(BaseExperiment)`: amplitude-specific
  physics, data, model wiring, loss, eval, plots.
- `base_experiment.py` — `BaseExperiment`: generic train loop, optimizer/
  scheduler, MuP setup, warm-start, checkpoint/save, MLflow, FLOP counting.
- A run executes `full_run()` → `init_physics → init_data → init_model →
  train → evaluate → plot`.

Output of a run lives under `runs/<exp_name>/...` (models, tokenizer, plots).

---

## Model (LLoCa)

Config `config/model/lloca.yaml`:
- Wrapper: `wrappers.AmplitudeLLoCaWrapper` → net `models.lloca.LLOCAMuPTransformer`.
- The net: `LearnedPDFrames` framesnet (equivariant local frames via `EquiMLP`)
  + `MuPTransformer` backbone (`models/transformer_lloca_mup.py`), with a
  block-diagonal attention mask built from `ptr` so particles attend only within
  their own event.
- Defaults: `num_blocks=8`, `num_heads=8`, `attn_reps="8x0n+2x1n"`,
  `hidden_channels_mlp=128`. Width axis for MuP is **`num_heads`**
  (base=2, delta=8); `attn_reps` must stay fixed between base/delta.

### Particle encoding (important design choice)
Two modes, selected by `data.use_PIDs`:
- **`use_PIDs: false` (default, preferred)** — each PDG id maps via a fixed
  global table `GLOBAL_PROPERTY_MATRIX` (in `particle_ids.py`) to an 8-D physical
  property vector, then a learned `Linear(n_features → d_particle_hidden=16)`.
  `in_channels` depends only on `d_particle_hidden`, **not** on the vocabulary,
  so adding a new particle/feature never forces retraining the transformer.
- **`use_PIDs: true` (legacy)** — one-hot token index from `ParticleTokenizer`.

### Coupling order
`data.amp_orders` is a per-dataset vector `[n_loops, alpha_s_power]` (LO=`[0,0]`,
NLO_full=`[1,1]`, virt_only=`[1,0]`, …). It's broadcast to every particle as
extra scalar features, so mixing perturbative orders needs **no model change**.

### MuP
Models with `*MuP*` in `_target_` get μP base shapes (`base_shapes.bsh` in the
run dir). Fresh init → `rescale_params=True`; warm start → `False`. See
`base_experiment.init_model`.

---

## Data

- **Which datasets a run actually trained on: the recipe wins, `data.dataset` lies.**
  `config/amplitudes.yaml` hardcodes an 8-entry `data.dataset` default, and every run
  config dumps it verbatim whether or not a recipe overrode it. So in a run's
  `config.yaml`:
  - `data.processes_file` set (equivalently `data.source: recipes`) ⇒ **the recipe at
    that path is the only authority**. `data.dataset` is stale inherited default;
    ignore it completely. E.g. `runs/dvirt_time` lists 8 datasets and actually
    trained on the 25 in `recipes/pretrain25_short.yaml`.
  - `data.processes_file: null` and no recipe source ⇒ **then** `data.dataset` is
    authoritative.

  Never state which processes a run saw from `data.dataset` without checking
  `processes_file` first. This has been got wrong repeatedly, and it silently
  inverts claims about what was held out, hence what "zero-shot", "held-out", or
  "never seen" mean in `docs/results.tex`.

- Datasets are `.npy` files in `data/`, named like
  `ee_ttbar_346-1000GeV_amplitudes.npy`. Each row: flat 4-momenta
  (`n_particles*4`) + PDG ids (`n_particles`) + amplitude (last col).
- `init_data` (in `experiment.py`) loads each dataset, for LLoCa boosts events
  to COM and applies a random Lorentz transform (data augmentation / locality),
  standardizes momenta, preprocesses amplitudes globally (`amp_trafos: [log,
  standardization]`), concatenates all processes, shuffles (seed 42), and builds
  flat contiguous arrays.
- `dataset.py`:
  - `AmplitudeDataset` — sparse variable-length events backed by a flat
    `(N_particles,4)` array + `offsets`; O(1) `__getitem__`.
  - `collate_variable_length` — concatenates events into a flat batch + `ptr`
    (event boundaries) consumed by the model's attention mask.
  - `ProcessBalancedSampler` — draws each process's share per batch with
    dynamically updatable weights (deficit-to-solo plateau-aware). **Discarded as
    the default** (`training.use_balanced_sampler: false`): an A/B at 8 datasets had
    the naive **equal/uniform** sampler win best-vs-best (val_loss_no_reg 8.1e-8 vs
    2.6e-7), since the benefit dilutes with many datasets and perturbing the sampling
    dynamics hurts. Kept opt-in for mixtures of few, very unbalanced datasets.
  - **Loss aggregation is `geometric_mean` by default** (`config/default.yaml`;
    used in essentially every run): a log-space mean over per-process MSEs, so a 10%
    relative gain on a 1e-6 process counts equally with a 10% gain on a 1e-2 one —
    processes of different final-state dimensionality (hence very different MSE
    scale) all keep scaling instead of the high-MSE ones dominating the gradient.

---

## Config (Hydra)

- `config/amplitudes.yaml` — main experiment config (dataset list, amp_orders,
  training defaults). `defaults:` pulls in `model: lloca`, `default`, `local: none`.
- `config/default.yaml` — full default tree: training/optimizer/scheduler,
  DyHPO multi-fidelity fields (`is_dyhpo_run`, `increment_steps`), fine-tune
  block (`fine_tune.lr_scale`, `layer_decay`, `freeze_blocks`, `reset_output_head`,
  `lora.*`, `ewc.*`).
- `config/model/lloca.yaml` — the only model config that matters here.
- `config/local/none.yaml` — Linux/cluster local overrides (`num_workers: 2`,
  in-memory dataset).
- `config/hydra.yaml` — disables Hydra's dir-changing/logging hijack.

### Canonical run setup — the ONE source of truth (don't drift, don't copy random runs)

These are the decided-best defaults. **When starting a new run/sweep, copy ONLY
`sweep/sweep_config_jeanzay_template.yaml`** (the one canonical template) — never
lift values from an arbitrary old run config (`scan_ab_*`, `pretrain25*`, etc.):
those are historical artifacts and several carry stale values (e.g. `batchsize:
1024`). If a value here disagrees with a run config, this table wins.

| Knob | Value | Kind |
|---|---|---|
| `model` / `net.num_blocks` / `net.attn_reps` | `lloca` / `8` / `8x0n+2x1n` | fixed |
| `net.num_heads` (μP width axis) | run-design (default 8); tune lr once, reuse across width | per-run |
| `particle_encoder_hidden` (MLP embed) | `32` (on) | fixed, open |
| `use_diagrams` / `d_diag` | `true` / `32` | fixed |
| `use_PIDs` | `false` | fixed |
| `spin_onehot`/`color_onehot`/`prop_is_massless`/`standardize_props` | all `true` | fixed |
| physics levers `mass_from_momenta`/`coupling_scalars`/`internal_mass_scalars`/`offshell_per_event` | `true` for the production joint run; `internal_mass_pdgs=[23,6,25]` | per-run (need recipe+sidecars) |
| `preprocess_per_dataset` + `amp_trafos` | `true`; `[log, standardization]` resolved **per-dataset** (positive→log, negative→signedlog) | fixed |
| `use_balanced_sampler` | `false` (equal/uniform sampler) | fixed |
| `loss` / `loss_aggregation` / `regularization` | `MSE` / `geometric_mean` / `L2` | fixed |
| **`training.batchsize`** | **`16384`** (biggest that fits; ~36 events/dataset/batch over 448 sets) | fixed |
| **`evaluation.batchsize`** | **`16384`** (eval forward-only → match train BS) | fixed |
| `num_workers` | `2` | fixed |
| `dtype` / `allow_tf32` / `fused_optimizer` | `float32` / `true` (no-op V100) / `true` | fixed |
| optimizer / betas / eps / weight_decay | `AdamW` / `[0.9,0.999]` / `1e-8` / `0` | fixed |
| `scheduler` / `clip_grad_norm` | `CosineAnnealingLR` / `5` | fixed |
| `training.lr` | run-design: centre on `lr*(t,D)` surface at your (t,D), sweep ±½ decade (see rule #2) | per-run |
| **EMA** (`ema` top-level flag) | **UNTESTED — has always been `false`.** Now swept `{false,true}` × `ema_decay∈[0.9,0.9999]`; settle before fixing | open |
| fine-tune | full retrain + layer-decay; `lr_scale∈[0.1,10]@1`, `layer_decay∈[0.75,1.0]` | fixed |

Sweep search ranges (redesigned): `regularization_lambda [1e-10,1e-6]`,
`cosanneal_warmup_frac [0.05,0.2]`, `cosanneal_eta_min` fix ~`1e-8`, `ema_decay
[0.9,0.9999]` (+ `ema` categorical), `sampler_alpha_ema [0.3,0.95]` only if the
balanced sampler is on. Drop the exotic sampler variants.

---

## Sweeps & DyHPO (`sweep/`)

Multi-fidelity HPO over training-step budgets, sharing a DyHPO surrogate across
all jobs of a sweep via a lock file on the shared FS.

Key files:
- `generate_sweep.py` — initializes the sweep: samples HP candidates, writes the
  DyHPO state (`dyhpo_state.pkl`), and emits one SLURM script per trial into
  `<sweep_dir>/<sweep_name>/jobs/trial_XXXX.sh`. Flags: `--config`, `--n-trials`,
  `--extend` (add trials, reuse state), `--dry-run`.
- `run_trial.py` — per-job entrypoint each SLURM job runs. It: locks state →
  `sampler.suggest()` for `(hp_config, fidelity=t_steps)` → checkpoint-index
  lookup to warm-start from a lower fidelity → runs `run.py` → locks state →
  `sampler.observe(...)`. A SIGTERM handler reports the in-flight trial as failed
  so DyHPO can reuse the slot.
- `dyhpo_sampler.py`, `dyhpo/hpo_method.py`, `checkpoint_index.py` — the sampler,
  surrogate, and warm-start checkpoint index.
- Config template: `sweep/sweep_config_jeanzay_template.yaml` (cluster block,
  paths, `fidelity_schedule.t_steps`, `fixed_params`, `search_space`,
  `dyhpo.*`). The HPO objective can be a transfer-ratio geometric mean vs a
  fitted scaling law (`compute_hpo_objective` in `run_trial.py`).
- Scaling-sweep generators/analysis: `generate_pretraining_scaling_sweeps.py`,
  `generate_scaling_sweep.py`, `fit_scaling_law.py`, `analyze_*_scaling*.py`.
- Resubmission helpers: `resubmit_timed_out.py`, `resubmit_scaling_jobs.py`.

**Submitting a sweep** (confirm with me before actually submitting). Submission
goes through `sweep_manager.py` (see below) so trials interleave round-robin
across sweeps — do **not** hand-loop `sbatch` over `jobs/*.sh`. The generators
call it for you:
```bash
# all of these shell out to sbatch, so they run on the login node via remote.sh
# generate; it then prompts to submit (or set cluster.auto_submit / pass --auto-submit)
scripts/remote.sh python sweep/generate_sweep.py --config sweep/<my_config>.yaml
# scaling generators submit all their cells interleaved in one batch:
scripts/remote.sh python sweep/generate_scaling_sweep.py --config sweep/<scaling_config>.yaml
scripts/remote.sh python sweep/generate_pretraining_scaling_sweeps.py --phase both --auto-submit
# submit manually later (interleaves with whatever is already queued):
scripts/remote.sh python sweep/sweep_manager.py submit <sweep_dir>/<sweepA> <sweep_dir>/<sweepB>
```

**Known coupling issue (relevant to job ordering):** jobs share DyHPO state, but
they only inform each other if earlier trials `observe()` before later trials
`suggest()`. If all trials of one sweep start at once (which happens on Jean Zay
when many GPUs free up), they all `suggest()` against an empty/stale surrogate.
When running several sweeps at once, interleave submissions across sweeps
(round-robin) and/or use SLURM priorities so each sweep stays partly serialized.
**Fallback when interleaving can't help (a lone sweep, or abundant GPUs):**
`sweep_manager submit` now defaults to **3 sequential waves** (`--seq-batches 3`)
for a single-sweep submission — the trials are split into 3 groups chained by
SLURM `afterany` dependencies, so wave *k+1* only starts after wave *k* has
`observe()`d and the (single-fidelity) Bayesian optimiser actually has data to fit.
Multi-sweep submissions default to `--seq-batches 1` (rely on interleaving); pass
`--seq-batches N` to force, `1` to disable.

**Cross-sweep submitter — `sweep/sweep_manager.py`**. Submits
trials interleaved across sweeps and stamps each job with a SLURM `nice` value =
`round × gap`, where a sweep's `j`-th pending trial has `round = j // weight`.
Same round ⇒ same nice ⇒ runs together (fills GPUs); higher round ⇒ runs later;
`weight>1` ⇒ that sweep advances faster. Uses only user-level `nice` (no operator
rights). Commands: `submit <sweepdir>…`, `rebalance` (re-interleave all pending,
e.g. after adding a sweep), `boost <sweep> --weight N`, `status`, `cancel`.
Add `--dry-run` to preview. Registry: `~/.sweep_manager/registry.json`.

### HPO search-space rules (empirical — harvested from 422 converged sweeps)

Full derivation, figures, and numbers are in `docs/results.tex` (§ optimization
laws); the settled rules that govern how sweeps are set up:

1. **`training.lr` transfers across width (μP) — never re-sweep it per width.**
   Tune HPs once at the smallest width and reuse for all larger widths (up to a
   5× cut on a width-scaling grid). Re-sweeping lr per width is pure waste.
2. **lr follows an inverted-U in *training length*, peaking at an absolute
   optimizer scale `t* ≈ 3000` steps** (independent of dataset size):
   `lr_c(t) ≈ 1e-2 · min[(t/3000)^+0.5, (t/3000)^-0.55]`. The peak is an optimizer
   timescale (Adam β₂ EMA + warmup + weight EMA), **not** a convergence point — so
   it is not a monotone "more iters ⇒ lower lr": below `t*` the optimum *rises*,
   past `t*` it falls as `t^-0.55`. **The procedure is fixed, not a judgment call:
   evaluate the measured 2D surface `lr*(t,D)` at YOUR `(t,D)` and sweep ±½ decade**
   around it (`[c/3, 3c]`), never a flat 4-decade prior nor the t-only `lr_c(t)`.
   Both axes are set by the run design, so both are known inputs, not uncertainty:
   - **`t`-axis (horizon):** every real run sits past the peak (`t ≫ t*=3000`), so
     `lr*` is on the `t^-0.55`-ish decay — a **longer run takes a lower center**.
   - **`D`-axis (per-process events):** measured `lr* ∝ D^{~0.17}`, **saturating**
     (~2.2× from D=700→70k at t=3162, most of it by D~2k). A large-`D` run sits at
     the **high-D asymptote → center ABOVE the D-pooled value**; read it off the
     grid's high-D row at your `t`. Do NOT collapse `D` into "a nudge the window
     absorbs" — that discards a measured axis. (The "is the loss floored?" rule of
     thumb is also dropped — never a measured axis; only `t` and `D` were.)
   - **The (high-`D`, high-`t`) corner is OFF the measured grid** (high-D rows stop
     at `t≈1.8e4`; every large-`t` pooled point comes from small `D≤7k`, which
     over-trains and decays steeply — not representative of a large-`D` run). So
     for a big foundation run (per-proc `D`≈30–100k) the center is an
     **extrapolation of the high-D row's decay**: ~2e-3 @30k, ~1.4e-3 @50k, ~9e-4
     @100k steps. The ±½-decade sweep then both centers the search AND *measures*
     that off-grid corner. Surface + numbers: `analysis/hpo_optima/`
     (`make_fig_2d.py`, `hpo_optima.json`). Template default window `[1e-3, 3e-2]`.
3. **lr is the dominant knob; narrow the rest, don't blindly fix them.** Optima:
   `regularization_lambda ∈ [1e-10, 1e-6]` (cap high at 1e-6; was 9 decades),
   `cosanneal_warmup_frac ∈ [0.05, 0.2]` (median 0.15), `cosanneal_eta_min`
   fix ~1e-8 (or `[1e-10, 1e-7]`, weakest knob), `ema_decay ∈ [0.9, 0.999]`.
   Keep `sampler_alpha_ema ∈ [0.3, 0.95]` (2nd most important). **Drop the exotic
   sampler variants** (`sampler_sig_k`, `sampler_deficit_*`, … |ρ|≈0.1 = noise).
4. **Finetune:** `fine_tune.lr_scale ∈ [0.1, 10]` centered at **1** (best finetune
   lr ≈ pretrain lr), `fine_tune.layer_decay ∈ [0.75, 1.0]`; same reg/warmup rules.
5. **Net effect:** ~10–100× smaller search volume ⇒ halve the trial budget
   (e.g. 300→150 candidates, ~15→~8 obs) at equal resolution, or explore far
   denser at equal budget. Seed new cells from `lr_center(t)`, not a flat prior.
   The `sweep_config*template.yaml` defaults already encode these ranges.
6. **HP questions go through DyHPO, single-fidelity** (`fidelity_schedule.t_steps:
   [T]`, one value; the multi-fidelity ladder is unused here) — never a hand-rolled
   grid, whatever the question is called. "Is `clip_grad_norm` the culprit?" and
   "does it work at any `lr`?" are HP searches, not diagnostics. The reason is
   substantive, not stylistic: HPs **interact**, so a 1-D grid at fixed
   everything-else answers only "best `lr` *given those* values", cannot reach a joint
   optimum (one needing `lr` *and* `beta` together), and comes back flat —
   manufacturing a **false** "it doesn't work at any `lr`". Sweep the coupled space.
   Job arrays stay right for **non-HP** ablations (loss type, data tag, warm-start
   checkpoint, ablation flags, seeds). Enforced by `hpo_guard.sh`.

---

## A/B testing a new feature (how I want comparisons run)

When I ask whether a new feature/change helps, follow this protocol — it's about
being *fair* and *not wasteful*:

1. **Don't rerun the baseline if it already exists.** Reuse the existing baseline
   run/sweep results. Re-training a baseline you already have is pure waste.
2. **The baseline is the best run of a short per-run HPO sweep**, not a single
   arbitrary run.
3. **First try the cheap shortcut:** run the new feature with **the same HPs as the
   baseline's best**. If it's **already better**, the feature wins — **stop, you're
   done.** No need to sweep.
4. **Only if the same-HP run is *not* better, sweep the feature fairly:** run the
   **same HPO sweep** for the new feature, find *its* best HP config, and compare
   best-vs-best. A feature can lose at the baseline's HPs but win at its own.
5. **Compare on `val_loss_no_reg`** — the best non-regularized val loss on
   log-amplitudes, never the regularized tracked loss: `λ` is itself a tuned HP, so
   comparing regularized losses confounds the metric with the search.
6. **Re-base for preprocessing differences** before comparing. If the feature
   changes standardization or `amp_trafos`, the two val losses aren't on the same
   scale and the comparison is meaningless.

---

## Waiting on jobs (always background, never hand-poll)

When I submit a job/test and need its result before continuing, **do not** poll
`squeue` in a manual loop of tool calls, and **do not** promise "I'll check back"
without a mechanism. The single standard way:

1. Submit and capture the id: `jid=$(scripts/remote.sh sbatch --parsable <script>)`.
2. Launch the waiter **in the background** (Claude Code `run_in_background: true`):
   `scripts/wait_for_slurm.sh "$jid"`.

`scripts/wait_for_slurm.sh` blocks (cheaply, `squeue` every `POLL=30`s) until the
job(s) leave the queue, then prints final `sacct` state/exit/elapsed + a `TAIL=25`
tail of each job's `*_<jobid>.out` log. Because it's backgrounded, the harness
**re-invokes me exactly once when it exits** — so I pick the results up
automatically instead of polling or forgetting.

- No id ⇒ waits on *all* my current jobs: `scripts/wait_for_slurm.sh`.
- Knobs: `POLL=<s>` interval, `TAIL=<n>` log lines. From the mount it forwards
  itself to the login node through `scripts/remote.sh` (no `squeue` here).
- Don't `sleep`-loop or re-run `squeue` by hand across turns; if I need an
  interim peek I can read the background task's output, but the completion ping is
  the source of truth.

---

## Conventions & gotchas

- **dtype:** set via `training.dtype` (`float16/32/64`); `run.py` sets the torch
  default dtype before building the experiment.
- **Warm start / fidelity:** `warm_start_idx` + `training.increment_steps` drive
  resumption from a previous fidelity checkpoint; don't hand-edit these — they're
  set by `run_trial.py`.
- **Always keep `plot: true`** in sweep/experiment configs — I always want plots.
- **Every figure ships as BOTH `.png` and `.pdf`** (same basename, same dir) —
  no exceptions. When you write a plot, save both formats in the same call
  (`fig.savefig(base+'.png'); fig.savefig(base+'.pdf')`); make plotting scripts
  emit both by default. Both are gitignored here (`*.png`, `*.pdf`) — "both
  formats" means **on disk as deliverables**, not committed. Enforced by the
  `figure_pair_guard.sh` `Stop` hook (blocks end-of-turn if a figure you just
  made is missing its counterpart; genuine single-format exceptions go in
  `.claude/figure_pair_ignore.txt`).
- **Fine-tuning** reuses the pretrained run's tokenizer; LoRA/EWC/layer-decay/
  freezing are all in the `fine_tune` config block.
- `.fuse_hidden*` and `._*` files are leftover filesystem artifacts — ignore them.
- **Per-step LLoCa hot-path vectorizations** — `LLOCA_*` env toggles (default =
  fast path; original impls kept for A/B). Attention mask built once/forward not
  per-block (`models/transformer_lloca_mup.py`, `LLOCA_ATTN_MASK`); per-process
  loss segment-mean (`experiment.py` `_aggregate_per_process_loss`,
  `LLOCA_PROC_LOSS`); L2/L1 reg via `_foreach` (`experiment.py`
  `_init_regularization`, `LLOCA_REG`); per-event mean pool (`wrappers.py`,
  `LLOCA_POOL`); per-particle frames broadcast over heads instead of replicated
  (`models/attention_lloca_mup.py`, `LLOCA_FRAMES`, default `broadcast`; set
  `repeat` for the original — broadcast only applies for `attn_reps` with max
  order ≤ 1, else auto-falls-back to repeat). Equivalence guards: `test_amp.py`
  Section 0. Read the code for details.
- **Per-step host↔device syncs** — `LLOCA_SYNC` env toggle (default `deferred`,
  `blocking` = original for A/B). `deferred` fuses the per-step syncs into one and
  catches NaN via the synced grad-norm (skip-step instead of crash); NaN no longer
  raises. Mechanism and measured gain: `docs/results.tex` § throughput.
- **Dataloading is the per-step bottleneck — keep `num_workers: 2`.** It is the
  default in `config/local/none.yaml`; `nw=4` is no better and `nw=6` oversubscribes
  (`cpus-per-task=8`) and hangs. Profile with `LLOCA_PROFILE_STEP=1` (data-vs-compute
  split at the end of `train()`); bench with `bench_workers_ab.sh` (needs a GPU via
  sbatch). Numbers: `docs/results.tex` § throughput.
- **Compute knobs** (default-on, A/B via config; attack the compute floor):
  `training.allow_tf32` (default true) sets `matmul/cudnn.allow_tf32` in
  `_init_backend` — **no-op on V100**, ~2x matmul on A100 (`gpu_p13`) at ~1e-3
  precision, so **A/B the loss on A100 before trusting it**. `training.fused_optimizer`
  (default true, CUDA only) passes `fused=True` to Adam/AdamW; flows through
  `MuAdam/MuAdamW` to the real optimizer.
- **NOT done (deliberately):**
  - L2-reg → decoupled `weight_decay`: would remove a model-sized backward term, but
    the L2 reg is added into the tracked loss (`loss_no_reg`, `val_loss_no_reg`,
    checkpoint selection, HPO objective) and `regularization_lambda` is a tuned sweep
    HP — converting it would silently break continuity with existing sweep results.
  - `torch.compile`: the compute is dominated by the framesnet (torch_geometric
    message passing) and xformers attention, neither of which compiles — heavy graph
    breaks for ~no gain. Also `torch.compile(self.model)` prefixes `state_dict` keys
    with `_orig_mod.`, which would break warm-start/fine-tune reloads. Skipped.
- **Activation-memory knob:** `model.net.checkpoint_blocks` (config, not an env
  toggle; default false) gradient-checkpoints the transformer blocks — large
  activation-memory saving for ~one extra forward of compute. Use when memory-bound
  at large width / batch size.

---

## Git & release workflow (ccin2p3 trunk → published main)

This repo uses a **development trunk + generated public branch** model. Claude
handles git: commit and push as work progresses, keep a readable timeline.

**This is automatic, not a thing to ask about.** Commit with clear messages as
work lands and push without asking — pushing is *not* an outward action that needs
confirmation (see ground rule #2). Hooks back the workflow (and other rules) up
so they can't be silently forgotten (`.claude/settings.json` → `.claude/hooks/`):
a **`Stop` hook (`auto_push.sh`)** pushes any unpushed `ccin2p3`/feature-branch
commits to origin at the end of every turn (already-committed work only; never
`main`); a **`PreToolUse` hook (`worktree_guard.sh`)** reminds me to open a
worktree when I start editing trunk code on `ccin2p3`. Two more enforce rules
above: **`md_guard.sh`** (`PreToolUse(Write)`) blocks creation of new
`.md`/`.tex`/`.rst` files (ground rule #3; allowlist
`.claude/md_allowlist.txt`), and
**`figure_pair_guard.sh`** (`Stop`) blocks finishing a turn if a figure was saved
in only one of `.png`/`.pdf` (ignore-list `.claude/figure_pair_ignore.txt`).
**`hpo_guard.sh`** (`PreToolUse(Bash)`) blocks `sbatch` of a hand-rolled **HP grid**
— a job array that feeds an HP (`training.lr`, `clip_grad_norm`, `heterosc_beta`,
`regularization_lambda`, `cosanneal_*`, `fine_tune.lr_scale`/`layer_decay`) a
per-task shell variable — because HP search goes through DyHPO (allowlist
`.claude/hpo_grid_allowlist.txt`). Arrays over **non-HP** axes (loss type, data
tag, warm-start ckpt, ablation flags, seeds) stay allowed.

### Batched pillar review (`review_backlog.sh` + three reviewer subagents)

Three pillars each have a reviewer subagent in `.claude/agents/`:
`CLAUDE.md` → **claudemd-keeper**, `docs/*.tex` → **notes-editor**, source →
**repo-reviewer**. Review is **batched, not per-change**: each pillar carries a
watermark (the commit it was last reviewed at, plus the size of the change reviewed
then), and the `Stop` hook `review_backlog.sh check` only asks for a reviewer once
that pillar's accumulated backlog crosses its threshold (80 changed lines or 8
commits for `CLAUDE.md`/`.tex`, 200/12 for code). The reviewer then reads the
**whole backlog at once**, which is the point: cross-edit problems (a rule now
stated twice, a config default whose callers were not updated, a section that no
longer reads as one argument) are invisible to a per-hunk review.

**When the `Stop` hook flags a pillar due, spawn its reviewer immediately, in that
turn, without asking me** — a hook cannot spawn a subagent, so I am the part that
executes it. If the nudge is skipped, `review_backlog.sh gate`
(`PreToolUse(Edit|Write)`) refuses further edits to the overdue pillar until its
reviewer runs, so ending the turn stalls rather than pauses. Commits and every pillar
under threshold stay unblocked.

`status` lists backlogs and `/review-now` runs reviewers on demand; `begin <name>`
takes a pillar's lock and `advance <name>` releases it on a pass (`begin` is also the
human override). **Never run `advance` on a reviewer's behalf.** A blocking review
leaves its lock held on purpose, so the fixes it demanded can be applied. Watermarks:
`.claude/.review_state/` (gitignored, per-checkout).

**Branches**
- **`ccin2p3`** — the development trunk and default working branch. *Everything*
  lives here: the core code plus all tooling (`tools/`, `tests/`, `sweep/`,
  `scripts/`, `attribution/`, `data/` scripts, `notes/`, `CLAUDE.md`, recipes).
  This is where development happens. `jeanzay` is the frozen pre-migration
  trunk (identical history up to the CC-IN2P3 migration); never develop on it.
- **`main`** — the **public, stripped-down core**. It is a *build artifact* of
  `ccin2p3`, regenerated by `scripts/publish_main.sh` from the `PUBLIC_PATHS`
  allowlist (core run path + `config/` + `models/` + `IntrinsicDimDeep/` +
  example `recipes/` + README). **Never edit `main` by hand**; never merge
  `ccin2p3 → main`. To change what's public, edit the allowlist and re-publish.
- **`lxplus`** (and future env branches) — branch off the trunk, carry only that
  environment's specifics (e.g. AFS/EOS split, submission scripts). Stay in sync
  with the trunk via `git checkout lxplus && git merge ccin2p3`; commit
  env-specific files *only* on the env branch. Never merged back to the trunk
  wholesale.

**Working rules**
1. **Do work on `ccin2p3`** (or a feature branch off it).
2. **Open a worktree for new work, always under this repo:**
   `git worktree add worktrees/wt-<feat> -b <feat> ccin2p3`, implement and verify
   there, then merge back into `ccin2p3` and `git worktree remove` it. Never
   `../wt-<feat>` or any path outside the project root — keep them in `worktrees/`
   (gitignored) so parallel experiments don't clobber the trunk checkout.
   `worktree_guard.sh` nudges when I edit trunk code without one; for quick standalone
   edits it's fine to proceed on the trunk.
   **Merging brings back the CODE and nothing else.** Sweep results, run dirs, eval
   `.npz`, satlogs, generated datasets and figures are gitignored, so they live only
   inside the worktree and `git worktree remove` destroys them. Before removing any
   worktree, fold its results into the trunk:
   ```bash
   bash scripts/fold_worktree.sh worktrees/wt-<feat>            # what would be copied
   bash scripts/fold_worktree.sh worktrees/wt-<feat> --apply    # copy it
   ```
   It copies untracked result files only (tracked ones return via the merge), never
   deletes, and is safe to re-run. `worktree_fold_guard.sh` (`PreToolUse(Bash)`) blocks
   the removal until this has run. This is not hypothetical: the 12-trial
   `sweeps/l2_poly_uugg` DyHPO sweep was lost this way, so the left panel of
   `fig:l2_poly_keep` is empty and cannot be rebuilt without re-running it on GPU.
3. **Commit small and often; pushing is automatic.** Small, frequent, readable
   commits over big dumps. The `Stop` hook pushes already-committed work, so never
   ask permission to push; to push mid-turn just run `git push` (allowlisted).
4. **Publish the public core** with `scripts/publish_main.sh` (regenerates `main`
   from the allowlist and pushes). Run it after core-facing changes land on
   `ccin2p3`. Use `--no-push` to review first.
5. **Sync an env branch:** `git checkout lxplus && git merge ccin2p3`.

**Visibility caveat:** it's a *single* GitHub repo, so all branches share
visibility — making the repo public exposes `ccin2p3`/`jeanzay`/`lxplus` too. Only `main`'s
*tree* is stripped, not the other branches. If a dev branch ever needs to be
truly private, that forces a separate-repo split.

(Ground rule #5 still holds everywhere: never attribute commits/PRs to Claude.)
