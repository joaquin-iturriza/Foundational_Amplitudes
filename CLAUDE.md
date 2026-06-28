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

2. **You are running directly on Jean Zay.** You are on the cluster itself (no
   mount in between), so you **can** execute things: `sbatch`, `squeue`,
   `scontrol`, `srun`, `conda activate`, `module`, `git`, `python`, and the
   read-heavy aggregations (scanning result JSONs, grepping the tree) that used
   to be too slow — the data, GPUs and env are all here.
   - **But you're typically on a login node — no GPU.** Don't run training or any
     GPU/CUDA code directly (xformers attention is CUDA-only and crashes on login
     nodes); GPU work goes through `sbatch`. Quick CPU-only python/imports are fine.
   - **Submitting jobs is gated by GPU budget, not a blanket confirm.** You may
     submit quick tests on your own — **always be mindful of the GPU budget**.
     The rule: estimate the **total GPU-hours** of everything you're about to
     submit; if it's **> 10 GPU-hours, stop and confirm with me first** (show the
     command + your estimate). Under that, just run it (still show me what you ran).
     Inspecting state (`squeue`, `scontrol`, reading logs) you can always just do.
     - **Estimate total wall-time × GPUs across *all* jobs, and understand "small."**
       "A quick A/B test" does **not** mean "a 20-job HPO sweep at 30 min each"
       (= 10 GPU-h). Many of our runs train in **seconds to a few minutes**, so a
       short per-run sweep over those can total **well under half a GPU-hour** —
       *that* is small. Size the request to what I actually asked for; don't inflate
       a quick check into a full sweep.
   - **Git is NOT in the confirm-first set.** `git add`/`commit`/`push` (and
     `git worktree`) happen freely and automatically — see the git workflow
     section. Never ask permission to commit or push, and never conflate a `git
     push` with submitting a job. The confirm-first rule is about cluster compute
     (`sbatch`/sweeps), not git.
   - File edits: do them directly.

3. **One centralized CLAUDE.md.** Keep all project guidance in this file.
   Do not create per-directory `CLAUDE.md`/memory files; if you find others,
   flag them for deletion. This also covers the Claude persistent-memory system
   (`~/.claude/.../memory/`, `MEMORY.md`) — it's disabled in settings
   (`autoMemoryEnabled: false`) and a hook blocks writes to it. Everything goes
   in this file, which I maintain; only add here when I explicitly ask.

4. **Go easy on `find` over large trees.** This is Lustre, not a slow network
   mount anymore, so `find` is allowed — but it can still be slow on huge
   directories (metadata-heavy). Prefer `ls`, targeted `grep`, and direct paths
   when you already know roughly where to look.

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

| What | Path (on Jean Zay) |
|------|--------------------|
| Project root | `/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes` |
| Conda env | `/lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational` |

Sweep configs use these `/lustre/...` absolute paths, since that's where jobs
actually run.

SLURM account/partitions (from the sweep template): `account: itg@v100`,
`partition: gpu_p2` (V100 32GB) — **the only validated setup; all runs use V100**.
A100 (`gpu_p13` + `itg@a100`) is an *untested aspiration* in the configs: submitting
it fails with "Invalid job type for the account" — the account isn't entitled to it.
Don't assume A100 works (relevant for the `allow_tf32` knob, which is a no-op on V100).

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
    dynamically updatable weights (used for loss-balancing across processes).

---

## Config (Hydra)

- `config/amplitudes.yaml` — main experiment config (dataset list, amp_orders,
  training defaults). `defaults:` pulls in `model: lloca`, `default`, `local: none`.
- `config/default.yaml` — full default tree: training/optimizer/scheduler,
  DyHPO multi-fidelity fields (`is_dyhpo_run`, `increment_steps`), fine-tune
  block (`fine_tune.lr_scale`, `layer_decay`, `freeze_blocks`, `reset_output_head`,
  `lora.*`, `ewc.*`).
- `config/model/lloca.yaml` — the only model config that matters here.
- `config/local/none.yaml` — Linux/cluster local overrides (`num_workers: 0`,
  in-memory dataset).
- `config/hydra.yaml` — disables Hydra's dir-changing/logging hijack.

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
# generate; it then prompts to submit (or set cluster.auto_submit / pass --auto-submit)
python sweep/generate_sweep.py --config sweep/<my_config>.yaml
# scaling generators submit all their cells interleaved in one batch:
python sweep/generate_scaling_sweep.py --config sweep/<scaling_config>.yaml
python sweep/generate_pretraining_scaling_sweeps.py --phase both --auto-submit
# submit manually later (interleaves with whatever is already queued):
python sweep/sweep_manager.py submit <sweep_dir>/<sweepA> <sweep_dir>/<sweepB>
```

**Known coupling issue (relevant to job ordering):** jobs share DyHPO state, but
they only inform each other if earlier trials `observe()` before later trials
`suggest()`. If all trials of one sweep start at once (which happens on Jean Zay
when many GPUs free up), they all `suggest()` against an empty/stale surrogate.
When running several sweeps at once, interleave submissions across sweeps
(round-robin) and/or use SLURM priorities so each sweep stays partly serialized.

**Cross-sweep submitter — `sweep/sweep_manager.py`**. Submits
trials interleaved across sweeps and stamps each job with a SLURM `nice` value =
`round × gap`, where a sweep's `j`-th pending trial has `round = j // weight`.
Same round ⇒ same nice ⇒ runs together (fills GPUs); higher round ⇒ runs later;
`weight>1` ⇒ that sweep advances faster. Uses only user-level `nice` (no operator
rights). Commands: `submit <sweepdir>…`, `rebalance` (re-interleave all pending,
e.g. after adding a sweep), `boost <sweep> --weight N`, `status`, `cancel`.
Add `--dry-run` to preview. Registry: `~/.sweep_manager/registry.json`.

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
5. **The comparison metric is the best NON-REGULARIZED val loss on the log-amplitude
   values** — this matters a lot. Compare `val_loss_no_reg` (no L2/L1 reg term),
   on log-amplitudes, **not** the regularized tracked loss.
6. **Account for any preprocessing differences** between the two sides before
   comparing — if the feature changes standardization/amp_trafos/etc., the val
   losses aren't on the same scale and a raw comparison is meaningless. Make sure
   you're comparing like with like.

---

## Waiting on jobs (always background, never hand-poll)

When I submit a job/test and need its result before continuing, **do not** poll
`squeue` in a manual loop of tool calls, and **do not** promise "I'll check back"
without a mechanism. The single standard way:

1. Submit and capture the id: `jid=$(sbatch --parsable <script>)`.
2. Launch the waiter **in the background** (Claude Code `run_in_background: true`):
   `scripts/wait_for_slurm.sh "$jid"`.

`scripts/wait_for_slurm.sh` blocks (cheaply, `squeue` every `POLL=30`s) until the
job(s) leave the queue, then prints final `sacct` state/exit/elapsed + a `TAIL=25`
tail of each job's `*_<jobid>.out` log. Because it's backgrounded, the harness
**re-invokes me exactly once when it exits** — so I pick the results up
automatically instead of polling or forgetting.

- No id ⇒ waits on *all* my current jobs: `scripts/wait_for_slurm.sh`.
- Knobs: `POLL=<s>` interval, `TAIL=<n>` log lines. Works on the login node
  (`squeue`/`sacct` only — no GPU).
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
  `blocking` = original for A/B). `deferred` collapses the ~4 syncs/step into one
  fused `torch.stack([loss, grad_norm(, loss_no_reg)]).tolist()` in
  `base_experiment._step`, drops the pre-backward `loss.item()` and the per-step
  `isfinite` assert (NaN now caught via the synced grad-norm → skip-step instead of
  crash), and computes the xformers block-diagonal `seq_lens` on the **CPU** ptr in
  `experiment._batch_loss_lloca` (threaded as `seq_lens=` through wrappers→net→
  `build_block_diagonal_bias`) so the mask never `.tolist()`s the GPU ptr. Worth
  only ~2% (the step was data-bound, not GPU-bound — see next bullet). `loss_no_reg`
  is now a detached tensor, materialized at the consumer.
- **Dataloading is the real per-step bottleneck — use `num_workers≥2`.** Profiling
  (`LLOCA_PROFILE_STEP=1`, prints a data-vs-compute split at the end of `train()`)
  showed dataloading was **69% (260ms) of a 380ms step** at `num_workers=0`: 8192
  serial `__getitem__` + collate `cat` on the main thread with the GPU idle.
  `num_workers=2` + `pin_memory` (auto for nw>0) + `non_blocking=True` H2D
  (`_batch_loss_lloca`) overlaps it under compute → **0.356→0.128 s/iter (2.78x)**,
  at the ~0.119s compute floor. **`config/local/none.yaml` now defaults
  `num_workers: 2`** (was 0). `nw=4` no better; `nw=6` oversubscribes (`cpus-per-
  task=8`) and hangs — stick with 2. Bench: `bench_workers_ab.sh` (sbatch, needs a
  GPU — xformers attention is CUDA-only, crashes on login nodes).
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

## Git & release workflow (jeanzay trunk → published main)

This repo uses a **development trunk + generated public branch** model. Claude
handles git: commit and push as work progresses, keep a readable timeline.

**This is automatic, not a thing to ask about.** Commit with clear messages as
work lands and push without asking — pushing is *not* an outward action that needs
confirmation (see ground rule #2). Two hooks back this up so it can't be silently
forgotten (`.claude/settings.json` → `.claude/hooks/`): a **`Stop` hook
(`auto_push.sh`)** pushes any unpushed `jeanzay`/feature-branch commits to origin
at the end of every turn (already-committed work only; never `main`); a
**`PreToolUse` hook (`worktree_guard.sh`)** reminds me to open a worktree when I
start editing trunk code on `jeanzay`.

**Branches**
- **`jeanzay`** — the development trunk and default working branch. *Everything*
  lives here: the core code plus all tooling (`tools/`, `tests/`, `sweep/`,
  `scripts/`, `attribution/`, `data/` scripts, `notes/`, `CLAUDE.md`, recipes).
  This is where development happens.
- **`main`** — the **public, stripped-down core**. It is a *build artifact* of
  `jeanzay`, regenerated by `scripts/publish_main.sh` from the `PUBLIC_PATHS`
  allowlist (core run path + `config/` + `models/` + `IntrinsicDimDeep/` +
  example `recipes/` + README). **Never edit `main` by hand**; never merge
  `jeanzay → main`. To change what's public, edit the allowlist and re-publish.
- **`lxplus`** (and future env branches) — branch off `jeanzay`, carry only that
  environment's specifics (e.g. AFS/EOS split, submission scripts). Stay in sync
  with the trunk via `git checkout lxplus && git merge jeanzay`; commit
  env-specific files *only* on the env branch. Never merged back to `jeanzay`
  wholesale.

**Working rules**
1. **Do work on `jeanzay`** (or a feature branch off it).
2. **Open a worktree for new work, by default:**
   `git worktree add ../wt-<feat> -b <feat> jeanzay`, implement + verify there,
   then merge back into `jeanzay` and `git worktree remove` it. Lets parallel
   experiments coexist without clobbering the trunk checkout. **If a worktree
   ends up unused** (no commits beyond `jeanzay`, no diff), just delete it —
   it cost nothing. The `worktree_guard.sh` hook nudges me when I edit trunk code
   without one; for genuinely quick/standalone edits it's fine to proceed on the
   trunk.
3. **Commit small and often; pushing is automatic.** Commit with a clear message
   after every significant change (small, frequent, readable commits over big
   dumps). The `Stop` hook pushes for me — I do **not** ask permission to push and
   do **not** need to remember `git push`; if I want it pushed mid-turn I just run
   `git push` (allowlisted, no prompt).
4. **Publish the public core** with `scripts/publish_main.sh` (regenerates `main`
   from the allowlist and pushes). Run it after core-facing changes land on
   `jeanzay`. Use `--no-push` to review first.
5. **Sync an env branch:** `git checkout lxplus && git merge jeanzay`.

**Visibility caveat:** it's a *single* GitHub repo, so all branches share
visibility — making the repo public exposes `jeanzay`/`lxplus` too. Only `main`'s
*tree* is stripped, not the other branches. If a dev branch ever needs to be
truly private, that forces a separate-repo split.

(Ground rule #5 still holds everywhere: never attribute commits/PRs to Claude.)
