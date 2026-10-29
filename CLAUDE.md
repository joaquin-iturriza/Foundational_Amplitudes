# Foundational_Amplitudes — Claude guide

A foundation model for tree/loop **scattering amplitudes** in particle physics.
A single Lorentz-equivariant transformer is trained jointly on many processes
(`ee→WW`, `ee→ttbar`, `ee→uu`, …), then fine-tuned to new processes/orders.
Core research threads: joint (multi-process) pretraining, **scaling laws**,
**fine-tuning / transfer**, and **DyHPO** multi-fidelity hyperparameter sweeps.

---

## Ground rules (read first)

0. **Start every message to me with my first name, "Joaquin", no matter what.** It is my check on
   context degradation: a message that does not open with it tells me the session is drifting.

0b. **Never change the scope of what I asked for on your own.** "Run X" means all of X: every
   probe, parent, arm, D, seed and horizon of the design it belongs to. Running a subset ("only
   the best rung", "for now", "the leader at the time"), fewer trials or seeds than the sibling
   cells, or a different horizon or pool is MY decision: ask before generating the configs, never
   after. If capacity or time forces a cut, propose it with what it costs the comparison; that entry waits
   for my answer (0g), the rest continues.
   Any difference between what was asked and what was submitted goes in the same message that
   reports the submission, in its first lines, never only in a commit message or the notes'
   hand-off. A figure or number that mixes cells run at different compute says so wherever it is
   shown. The ponytail plugin's "smallest change" governs code, never a study's design: its arms, seeds,
   probes and horizons follow this rule.

0c. **A plan I was told is a commitment: its cores, jobs and wall time.** When a tool refuses it ("fits NOWHERE",
   a cap, a wrong default), the refusal is a gap to fix or route around (split the work into parallel jobs of what fits
   so the stated wall time holds), never a reason to shrink the job. If neither works, the smaller version and its new
   time go to me in the first lines, before it is submitted. A delay's cause is read from the run records, never
   assumed.

0d. **When I point out a mistake, no apology and no quick patch presented as the end of it.** Answer with: what in
   your process produced it, traced in the transcript, logs or code (a cause nothing was read for says "not checked",
   and is checked before the message ends); the change that stops it recurring (a rule here, a guard, a tool fix), made
   in the same turn; then the fix of the instance. A change to this file is shown to me as a diff in the same message;
   a prevention that needs a permission I hold (a hook registration, a setting) is proposed to me as a concrete diff.
   A claim about current state (a setting's level, a job's status, what an agent runs on) is read from its live
   source in the same turn, never inferred from a default, a config file or memory; unread, it says "not checked".
   A number I state says what it counts and how it was selected; it comes from a committed tool or a script shown
   with it, never a throwaway one-liner. A proxy (one quantity standing for another) is checked on a few cases
   before it is reported, and an estimate says it is one, with its assumption.
   A message saying work is running, or not submitted, reads it in that turn from `site runs` and, for sweep trials
   (submitted by `sweep_manager`, absent from `site runs`), from `site status <site>` or the item's check.

0e. **Plain names, no internal codes, in anything I read.** Messages to me and the notes name a thing by what it is
   ("the dimensionless target with the context vector", "my decision to delete catalog_v2", "the prep job's failure"),
   never by a code: no D-numbers, E-numbers, run ids, arm codes or recipe tags on their own. An id may follow the
   description in brackets when I need it to look something up. Plain names are for things (runs, arms, datasets,
   axes): a quantity is written as its formula, in math, never spelled out in words, and a figure carries short labels,
   its explanation going in the caption or the talk, never inside the plot. New arms, sweeps and run directories get descriptive
   names (`dimensionless_context`, not `Qc`).
   **A result reaches me self-contained**, never as a continuation I must decode: first the question it answers and
   why it was asked (which study, which decision it feeds), then how it was designed (what is compared against what,
   on which data, how many runs and seeds, what is held fixed), stated again in every message that carries a result
   of that study, then the answer with its numbers, then what it changes for what we do next. A figure goes with what
   its axes and points are. A figure is sent once per version; a redrawn one says in its caption that it replaces
   the earlier one and what changed, and an unchanged one is referred to, never sent again; a redraw that does not
   visibly change what the figure shows (a fix that barely moves its numbers) counts as unchanged. The main session sends
   it with SendUserFile and never opens it with Read (a Read image is shown to me too, so it arrives twice); the
   step agent that drew it looks at it and says what it shows. A step agent's report is rewritten to this shape before it reaches me. A request for my go
   names the rule behind it (the GPU budget, or which decision of mine sends the work back at its design).

0g. **Never stop the turn for my answer.** No interactive question prompt (AskUserQuestion, refused by
   `no_blocking_question.sh`). A question for me goes in the message; only the part of the entry that needs the answer waits
   (`plan ask --blocks`; the whole entry only with `--holds-all` and its reason), everything else continues. An interrupt or a rejected tool call that comes with a question
   is answered and the work continues; only my words "stop" or "wait" stop it. Every word written in a turn is to me:
   a hook whose check is wrong gets one line to me naming the check, what it flagged and why it is wrong (a silent
   end is refused by Claude Code, and the hook's text is hidden from me), never a reply addressed to the hook.

0h. **A question of mine is answered in text first, before any tool call of that turn.** Then the work my answer
   unblocks goes out, in that same turn, before write-ups, events or other entries. A batch of jobs goes out in
   parallel: `site sync <site> FA` once, then every launch with `NOSYNC=1` at once (`scripts/launch_cc_only.sh`), never
   one sync per job in series.

0i. **The plan is `plan/plan.yaml`; `plan next` decides what moves; step agents do the work.** One entry per piece of
   work: its question, status (`running`, `waiting_user`, `held`, `done`), my words that authorise or shape it, next
   step, evidence, dead ends and `done_when`, a shell check that passes only when the entry is finished. It changes only
   through `python3 tools/plan.py` (`plan_files_guard.py` refuses anything else); where hand-off prose in
   `docs/results.tex` disagrees with it, the plan wins.
   - **Routing.** On every wake (an event, my message, an agent's report) the main session runs `plan next --prompts`,
     which lists, by fixed rules, each entry that can move and the exact prompt of its step agent
     (`plan/step_prompt.txt`). Each listed entry gets a background agent (Agent tool, `run_in_background`) started
     from that prompt in that turn and recorded with `plan step <entry> --agent <id>`. The main session works no entry
     itself: it launches step agents, sends me their questions and findings, and talks to me. Delegated work goes to a
     fresh agent with a written prompt, never a fork (a fork carries the main session's whole context).
     Independent pieces of work are separate next steps of the entry (`plan set-next <entry> "<step>" --also "<step>"`,
     `add-next`), each listed and launched on its own (`plan step ... --next N`), never one step chained with "then".
   - **A step agent does one step and exits:** an analysis, a fix, a write-up, or a job (`site submit ... --item
     <entry>`, recorded with `plan run`). It records evidence, dead ends and the next step through the plan CLI (each
     step with the decision it feeds and its branches, `--feeds` and two `--if "<result>: <action>"`, and its write-up
     says which branch the result took) and never waits on a job: when the job ends, `plan next` lists the entry again
     and a fresh step agent takes it. No entry has a long-lived owner. A job needs a recorded design (`plan design`: the question, the measure and the check, on
     existing data, that it moves with the tested change, what is fixed, the matched control, seeds, the outcome for
     yes and for no) passed by the design-reviewer subagent, which the step agent runs itself before submitting;
     `design_guard.py` refuses the submission otherwise. A mechanical step (requeue or resubmit a job, read a log and
     ack, check a count, collect, sync and submit an already-designed job) runs at `effort: low`;
     a low-effort step that finds something off reports it and stops, and the next step takes it at the default.
     Analyses, write-ups, designs, implementation and the reviewers run at the default (my preference, 2026-10-26).
   - **Prior work first.** Before proposing or designing any run or analysis, search `docs/results.tex`, `plan show`
     and `site decisions` for prior work on the same question (its processes, arms, data and quantity), and name what
     was found with its result, or "none found" with what was searched. Proposing again what exists, or answering a
     decision without the work it named, is a bug; `plan design --prior` records the search and the design reviewer
     redoes it.
   - **Questions.** Before a question reaches me, the rules here and `site decisions --project FA` are searched for
     what settles it; only a call none of them settles is asked, in a message to me, and recorded with `plan ask
     <entry> --ask "<the question>" --searched "<what was read, and why none settles it>" --blocks "<what waits on it>"`. My answer goes in with
     `plan answer <entry> --kind K --quote "<my words>"`, which records the decision and changes the entry in one step.
     Inside a running entry I approved, a new step (a design, an analysis, a job under the GPU budget of 2) goes ahead without
     coming to me; only work a decision of mine sends back at its design stops there (`site decisions`: the "stop at
     the design" hold, the larger synthetic catalogs' "consult with me on the design").
   - **Done** is `plan done <entry> --writeup <commit>`: it passes only when `done_when` passes and the write-up is
     committed. Every number added to `docs/results.tex` carries its provenance (`% src: python3 <script> @<commit>`,
     `% claim: key=value`); `tools/verify_claims.py` reruns the script and refuses a commit or merge whose numbers it
     does not reproduce (`.githooks`, `git_verify_guard.py`).
   - **The Stop check** `.claude/hooks/plan_check.py` keeps a turn from ending while `plan next` lists a step not
     launched, a `waiting_user` entry's question is in no message sent to me, a running entry has no approval of mine
     in force or a held one no hold, numbers on trunk fail the claims check, or a site event is open.

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

2. **The working copy is the local checkout `~/work/FA`; clusters are sites.** The general rules and the `site` verbs
   are in `~/work/CLAUDE.md`; this file adds only what is FA's. The laptop has a CPU copy of the CC env, `.venv-cpu`
   (`bash scripts/cpu_env.sh`; no xformers): tests run there, only those a change touches
   (`.venv-cpu/bin/pytest --testmon --continue-on-collection-errors tests/`), and CPU dry-runs of the sweep generators;
   training and anything CUDA runs inside a job on a site.
   - **The sites.** Jean Zay: A100 through `itg@a100`/`gpu_p5` (`-C a100`, QOS `qos_gpu_a100-t3`); `itg@v100` is in
     over-consumption. A pending job moved by `scontrol` needs Account, Partition, Features **and** QOS, else it sits on
     `InvalidQOS`. lxplus: HTCondor, code on EOS, submission from AFS; its ssh master needs a 2FA code only I can enter.
   - **Login nodes have no GPU.** No training or CUDA code there (xformers attention is CUDA-only); GPU work goes
     through `site submit`.
   - **GPU budget.** Estimate the total GPU-hours of everything about to be submitted (wall time × GPUs over all jobs,
     sized to what I asked, never padded); above 30 GPU-h (10 on Jean Zay, my call of 2026-10-26 while it is unused),
     confirm with me first unless a decision of mine in force
     already approves that work at that cost. Under it, run it and show me what ran. Every submission names its plan
     entry's work item (`--item`, `~/work/CLAUDE.md` "Decisions and work items"), quick tests included. Reading state
     is always free.
   - **Git and file edits are never confirm-first.** The budget rule covers cluster compute only.

3. **One centralized CLAUDE.md, and no new docs.** All project guidance lives in
   this file, which I maintain: it changes when I ask, or under 0d, shown to me as a diff; claudemd-keeper reviews
   every change. No per-directory
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
   goes in its results section, its plan entry is closed (`plan done`) and its hand-off
   item is deleted; the hand-off holds only open/future work.

4. **Never attribute work to yourself, anywhere, ever.** Do not add
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
| Working copy (where you edit) | `~/work/FA`, branch `trunk` |
| Checkout on CC-IN2P3 | `/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes` — `.venv/` (python 3.11, torch 2.1.2+cu118, numpy pinned 1.26.4, xformers, lloca) |
| Checkout on Jean Zay | `/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes` — conda env `foundational` (same torch stack) |
| Checkout on lxplus | `/eos/user/j/joiturri/Foundational_Amplitudes` — `.venv/` (python 3.11) on EOS; sweeps and Condor submission under `/afs/cern.ch/user/j/joiturri/Foundational_Amplitudes` |
| Site facts | `sites/sites.yaml` — per site: `project_dir`, `sweep_dir`, `scratch`, `data_dir`, the env recipe, and the `cluster` block (scheduler, partition, account, qos, gpu flag, mem, CPU ceiling). The only file, with `sites/activate.sh`, that names a cluster |
| Env activation | `sites/activate.sh` — resolves the site (`CCORCH_SITE`, exported by `site submit`, else sniffed from its own path), activates that site's env, exports `PROJECT_DIR`, `WORK`, `SCRATCH`, `DATA_DIR`, `SUBMIT_DIR` |
| Tier-0 mirror | `~/.local/share/ccorch/artifacts/FA/<run>/` (`site fetch`) |

**Nothing hardcodes a cluster path.** Python asks `siteconf`: `siteconf.PROJECT_DIR`,
`siteconf.SWEEP_DIR`, `siteconf.DATA_DIR`; `siteconf.resolve(cfg)` fills the
site-owned half of a sweep config (paths, `cluster` directives, `${PROJECT_DIR}` /
`${SWEEP_DIR}` / `${DATA_DIR}` / `${SCRATCH}` expansion, re-rooting of a config
generated on another site); `siteconf.slurm_header(cluster, job_name, out, err)`
renders the whole `#SBATCH` block for the site it runs on, and
`siteconf.cpu_header(...)` the CPU-partition one. Data is found through the environment `sites/activate.sh` exports: recipe runs
read `$WORK/datasets` and the cache under `$SCRATCH`; `DATA_DIR` is the same
`datasets` dir on every site. The old `data_path: data/` default in the configs is
per-checkout and empty on the clusters -- use recipes. Every job script starts with

```bash
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
```

(SLURM runs a *copy* of the script out of its spool, so `$0` cannot locate the
checkout; `site submit` exports `CCORCH_SITE` and `CCORCH_PROJECT_DIR`.) A job
script keeps only what is the job's business: `--job-name`, `--cpus-per-task`,
`--time`, `--output`/`--error`, and `--gres=gpu:N` as a plain count. Partition,
account, qos, `--gpus` vs `--gres`, `--mem` and the CPU ceiling are the site's
business and are added at submit time from `sites/sites.yaml` and
`~/.config/ccorch/sites.toml`.

SLURM (`sites/sites.yaml`): `account: lpnhe`, `partition: gpu_v100,gpu_h100` (either, whichever
starts first), `qos: gpu`, one GPU through `--gpus=1`.
**`--mem` is mandatory on CC-IN2P3**: the scheduler rejects any job without an
explicit memory request (template `mem: 32G`; CPU jobs go to `htc` with
`--mem-per-cpu`). `gpu_h100` (3 nodes of 4 H100 NVL) runs FA; `allow_tf32` is a no-op on V100, active on H100.

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
  global table `GLOBAL_PROPERTY_MATRIX` (in `particle_ids.py`) to a 9-D physical
  property vector (charge, spin, log-mass, T3, B, L, colour charge, Casimir,
  generation), then a learned `Linear(n_features → d_particle_hidden=16)`. The
  table's masses are the **generator's** (u, d, s, c, e, μ massless; b 4.7, t 172.5),
  never PDG values, and the generation column is what tells d from s and u from c.
  `in_channels` depends only on `d_particle_hidden`, **not** on the vocabulary,
  so adding a new particle/feature never forces retraining the transformer.
- **`use_PIDs: true` (legacy)** — one-hot token index from `ParticleTokenizer`.

### Coupling order
`data.amp_orders` is a per-dataset vector `[L_QCD, L_EW, alpha_s_max, alpha_ew_max]`
of the stored target (`mg5_pipeline_final.order_vector`; LO `ee→uū`=`[0,0,0,2]`,
`+g`=`[0,0,1,2]`, mixed four-quark `[0,0,2,2]`, stripped QCD-virt `[1,0,0,2]`; legacy
2-vectors are widened). It's broadcast to every particle as
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
  `processes_file` first: it decides what "zero-shot", "held-out" and "never seen" mean.

- **Processes equal up to relabelling are one dataset.** `tools/equal_processes.py` (CPU, CC) finds
  the catalog trees whose |M|² is a constant multiple of another's up to a leg relabelling (18 classes,
  `analysis/process_equality/equal_processes.json`: massless flavour relabels, `ee_X = uubar_X` for
  colourless bosons, ...). A held-out process is held out with its whole class.
- **Every role of a process samples the same distribution.** A recipe entry's
  `sampling` policy (catalog_v2: the mixture) shapes train, val and test alike; the
  shaped pools carry `_smix` in their file names. Validation and test were once forced
  onto uniform √s while training was shaped, which scored the model on a measure it
  never trained on (a finite-sample fit's error follows the training density; a
  validation set estimates the trained risk only from the same measure). Never
  reintroduce a train/eval measure split; the uniform-√s window is a dataset
  convention, not a "physical measure".
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
    **Caveat:** the same `1/MSE` weighting starves a process stuck far above the bulk
    (a resonance needle); the arithmetic mean recovers it at a cost elsewhere, and the
    τ-floor `loss_aggregation_tau` is an HP for the wave's DyHPO, not a fix with a natural
    value (`docs/results.tex` `tab:agg_ab` and the catalog hand-off). `loss_aggregation`
    and `loss_aggregation_tau` shape the **gradient only**; the validation aggregate
    (`val_loss_no_reg`, checkpoint selection, HPO objective) is fixed by
    `training.val_aggregation` (geometric mean) and never follows a training-side knob.

### Data generation (catalog_v3): what the pools contain and how they are sampled

Files: `mg5_pipeline_final.py` (unprefixed line numbers), `recipes/gen_catalog_v3.py`, `recipes/select_catalog_v3.py`.
**open (unsourced)** = chosen by an agent with no decision of mine behind it; changing it needs my decision.

| Knob | Value | Source |
|---|---|---|
| fiducial cuts `FIDUCIAL_CUTS` (:2664) | pT > 10 GeV, abs(cos θ) < 0.9, ΔR > 0.4, pair mass > 10 GeV; partonic c.o.m. frame, by rejection with top-up (:2738-2790) | **open (unsourced)** (commit bfb2d7a) |
| which finals are cut (:2752) | only m_final == 0 and not a neutrino: b (4.7) and τ (1.777) never cut, nor W/Z/t/H | **open (unsourced)** |
| √ŝ window (`gen_catalog_v3.py:245`, `select_catalog_v3.py:162`) | [max(25 GeV, 1.05 × Σ m_final), 1000 GeV] | [threshold, 1 TeV] kept by my decision to leave the sampler as it was (2026-10-08, "We'll keep things as they are for now") [D17]; the 1.05 and the 25 GeV floor **open (unsourced)** (commit f9f5ad7) |
| heavy-quark pool window (`select_catalog_v3.py:179`) | [lo, max(1000, lo + 500)] | **open (unsourced)** |
| √ŝ draw (:2848, :2967, :3214) | uniform in the window (bulk, all roles) | **open (unsourced)**; a dataset convention (Data above) |
| mixture `f_flat` (:2999) | 0.35 of a pool flat in log abs(M)² | `docs/results.tex` `fig:eeuu_genmix_fraction_sweep` |
| mixture `oversample` (:3000) | 6 candidates per flat event | **open (unsourced)** |
| mixture `democratic` (:3001) | 0.5 of multi-leg candidates from the splitter, rest RAMBO | `docs/results.tex` `sec:div-sampling` |
| mixture `y_lo` (:3002) | 1e-6 (deepest invariant / s) | `docs/results.tex` `fig:deep_sampling_antenna` (at lowered cuts, not the catalog's) |
| adaptive √ŝ pass (:3003, :3124) | on, ≤ 200 bins, ≥ 50 candidates per bin | **open (unsourced)** |
| phase space shape (:3167, :3097) | 2→2 isotropic; the splitter is final-final only (no initial-state collinear corner) | **open (unsourced)** |
| one-loop pools (:3015) | never shaped (uniform √ŝ, RAMBO) whatever the recipe says | **open (unsourced)** |
| α_s (:74, :3318, :3400) | α_s(M_Z) = 0.118, one-loop running, n_f = 5, μ = √ŝ per event, trees and one-loop | one-loop: `tools/nlo_conventions.py:39-41` (collaborators' reference); trees: **open (unsourced)** |
| α_s points (`recipes/catalog_v3_gen_stage3.yaml`) | 0.09, 0.15 trained, 0.13 held out | my approval of the generation (2026-10-24, "go ahead with the generation") [D102] |
| coupling orders (`gen_catalog_v3.py:111`, :192) | none in the generate strings: MadGraph's minimal QED order, so no VBF/VBS topologies | MadGraph default |
| external W, Z, t, H | stable, on-shell, unpolarised | MadGraph default |
| masses and widths | MadGraph `sm` card (G_μ scheme); MT = ymt = 172.5 only when a top is external (:464, `gen_catalog_v3.py:240`), 173.0 when internal; WT 1.4915. A width is read only from a DECAY line in column 1 (fixed 7b00eb21); before that every patched width was ignored, so catalog_v3's six loop-induced HH/HHH pools (`loop_qcd_qed_sm`) have the model's Z 2.48883 and W 2.08465, not 2.441404 / 2.0476 (`analysis/catalog_v4/data/card_width_audit.json`) | card: MadGraph default; 172.5: `tools/nlo_conventions.py` (collaborators' reference); external-only patch **open (unsourced)**; catalog_v3's loop-induced Z/W widths: not regenerated, **open** |
| heavy-quark layer (`select_catalog_v3.py:135`) | the top's quantum numbers at m_Q ∈ {50, 300, 600} GeV, width scaled from 1.4915 at LO (0, 8.7956, 71.3498); the pre-fix pools had 1.4915 at every mass, being rebuilt at the scaled widths (`tools/width_regen.py`) | layer: my catalog design approval (2026-10-23, "the proposals for the catalog are good") [D98, D100]; rebuild: my call (2026-10-26, "For the others, if it's not too much time for regeneration, regenerate.") [D145]; width rule **open (unsourced)** |
| b beams (:2796) | on their 4.7 GeV shell | **open (unsourced)** |
| BSM points `NONSM_POINT` (:1649-1662) | Z′ MY1 500 GeV (gVl 0.3, gAl 0.1, gVq 0.25 model default, dark matter decoupled), WY1 18.64129; 2HDM H± 200, h2 250, h3 300 GeV, second-doublet Yukawas zero, h1 width 6.382e-3; WZ/WW/WH aligned to the `sm` card (`_SM_ALIGN`), WT 1.4915. The pre-fix pools (both catalogs) had the UFO defaults (WY1 10.0, 2HDM h1 1.0, WT 1.508336, WZ 2.4952, WW 2.085, Z′ model WH 0.00407); being rebuilt at these widths (`tools/width_regen.py`) | layer: "at least one BSM process that can be encoded with our current particle embedding" (2026-10-23) [D98]; rebuild: [D145]; the points **open (unsourced)** |
| pool storage | float64 momenta, IDs and amplitudes; never propose float32 again | my call after the float32 check (2026-10-24, "Okay, fine, keep it like that then"): float32 momenta shift the target's physics factors by up to 0.022, and single-process precision studies reuse the pools |
| pool sizes | trees 1e5 / 1e4 / 1e4 (train / val / test), one-loop 5e4 / 5e3 / 5e3 | my approval of the generation (2026-10-24, "go ahead with the generation") [D102] |
| pp pool sizes (`recipes/gen_catalog_v3_pp.py:88`) | 1e4 / 2e3 / 2e3 | **open (unsourced)** |
| pp pools' cards (`recipes/catalog_v3_pp/`) | Daniel's 13 TeV run card as is (nn23lo1, pT > 10 GeV, abs(η) < 5); catalog_v3's param card | the pp study (2026-10-25, "build it, run a few checks"; "that's acceptable yes") [D117, D119] |


#### catalog_v4 (my coverage decision, 2026-10-25, "yes to 1 to 3, yes to electroweak production and the smaller fixes, no to the decayed final states" [D130])

Files: `mg5_pipeline_final.py` (`V4_*`, :2664-2730), `recipes/gen_catalog_v4.py` (entries and recipes from catalog_v3's,
names `v4_*`), check `analysis/catalog_v4/validate_v4.py`. catalog_v3's pools, recipes and recipe ids are unchanged (the
v4 fields are absent from its entries). Rows not listed are catalog_v3's.

| Knob | Value | Source |
|---|---|---|
| acceptance `V4_CUTS` | hadronic beams: pT > 10 GeV, abs(η) < 5, ΔR > 0.4, no pair-mass cut; e+e-: abs(cos θ) < 0.99, the rest as v3; b and τ cut like the other legs (`cut_pdgs`) | [D130] |
| √ŝ window and draw (`v4_sqrts_lo`) | [the least √ŝ the masses and the pT cut allow, 6 TeV], log-uniform (`sqrts_measure`); the adaptive pass bins in log √ŝ | [D130]; the 10 GeV edge of the 22 trees with neither a cut leg nor a mass **open (unsourced)** |
| forward component (`rotate_forward`, `V4_SAMPLING`) | 0.15 of each pool, and a third of the flat part's candidates, turned so one final is uniform in log(1/(1 − abs(cos θ))) up to the cut edge; splitter, f_flat 0.35, same mixture in every role | design [D130]; 0.15 and 1/3 **open** (the coverage check's settings) |
| coupling orders | `QED<=b` on every tree, b = n − (external gluons) when a quark is external (0 all-gluon, n colourless): every tree diagram (VBF, VBS, EW-QCD interference); QCD unbounded; BSM strings as v3 | [D130] |
| top | MT = ymt = 172.5 in every entry, internal and one-loop tops too; WT 1.4915 (`V4_WT`) in every entry but one: the one-loop b b~ > h w+ w- (`v4_bbx_hwpwm_nlo`, `WIDTH_OVERRIDE` in `recipes/gen_catalog_v4.py`) has WT 0, exact there (t-channel top only; the fixed width broke its pole, 19% of points refused). The pre-fix cards asked 1.480327 on an ignored line, so the SM tree and `loop_sm` pools already have 1.4915; the BSM pools had 1.508336 and the six loop-induced HH/HHH pools (`loop_qcd_qed_sm`) 1.36728 with Z 2.48883, W 2.08465: those are being rebuilt at WT 1.4915 (the loop-induced ones with Z 2.441404, W 2.0476), the others migrated (`tools/width_regen.py`). The heavy-quark layer takes the LO width at its mass (`top_width_lo`: 0, 8.729709, 70.81529); its pre-fix pools had 1.4915, being rebuilt | 172.5 everywhere: [D130]; WT 1.4915: my call (2026-10-26, "do 1.4915 for all, let's not waste time regenerating everything now") [D145]; the zero-width exception: my call (2026-10-26, "if this one is blowing up, then it's clearly wrong") [D154]; the heavy-quark width rule (LO, massless b, V_tb = 1) **open (unsourced)** |
| b | massless in beams and finals: `sm-no_b_mass` (`loop_sm-no_b_mass` for MadLoop); `particle_ids.py` gives it a massless species row (±5000000, `b_mass_of_process`), catalog_v3's b stays 4.7 | [D130]; the particle table: commit 3abde40c |
| classes, roles, sizes | v4's own (`recipes/select_catalog_v4.py`: the equality step's exact classes joined with near pairs, the rule that reproduces catalog_v3's 1586): splits keep their v3 class's role; a merge with a held-out class is held out whole; changes listed in `analysis/catalog_v4/data/class_changes.json`; sizes as catalog_v3's; the energy axis is the 6 to 20 TeV probes | the class rule: "Processes equal up to relabelling are one dataset" above; energy probes: my approval (2026-10-25, "Yes, okay, that's fine, go ahead.") [D137] |

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
| `particle_encoder_hidden` (MLP embed) | `0` (off, linear embed; `docs/results.tex` `tab:bigrun_arms`) | fixed |
| `use_diagrams` / `d_diag` | `false` (`tab:bigrun_arms`; revisit after the catalog pretraining, catalog hand-off) / `32` | fixed |
| `use_PIDs` | `false` | fixed |
| `spin_onehot`/`color_onehot`/`generation_onehot`/`prop_is_massless`/`standardize_props` | all `true` | fixed |
| physics levers `coupling_scalars`/`internal_mass_scalars`/`offshell_per_event` | `true` for the production joint run (`internal_mass_scalars` carries the off-shellness columns); `internal_mass_pdgs=[23,6,25]`. `mass_from_momenta` stays `false` (redundant, `tab:levers`). **Off-shellness needs the diagram sidecars** `data/diagrams/<process>.diagrams.json` (gitignored; `tools/dump_diagrams.py --all`, CPU); the run log line `offshell_per_event: built propagator masks for N/P processes` must show N = P, else the flag is silently a no-op | per-run (need recipe+sidecars) |
| `preprocess_per_dataset` + `amp_trafos` | `true`; `[log, standardization]` resolved **per-dataset** (positive→log, negative→signedlog) | fixed |
| target-side levers `data.target_propagators` (+`target_propagator_tchannel`, `_max_final: 2`) / `training.sign_head` | adopted at the catalog working point (`docs/results.tex` catalog census, three seeds each; off by default, **switch on for the full-horizon catalog sweep**). **Not for single-process runs:** in the transfer study (`docs/results.tex` `sec:ladder`) the t-channel factor sets a loss floor and the Breit-Wigner factor loses in all seeded cells, so the study runs both off (`tp3_`; `tp2_` = t-channel off only); they change the target, so their `val_loss_no_reg` is not comparable to runs without them (`analysis/catalog_v2/signed_compare.py` re-bases the signed pools); frozen stats record `target_propagators` and a mismatch asserts | per-run |
| `use_balanced_sampler` | `false` (equal/uniform sampler) | fixed |
| `loss` / `loss_aggregation` / `regularization` | `MSE` / `geometric_mean` (default) / `L2` | `MSE`, `L2` fixed; the aggregation is **open at catalog scale**: the arithmetic mean wins there (`tab:agg_ab`, `tab:catv2_short_smix`, catalog hand-off) |
| **`training.batchsize`** | **`16384`** (biggest that fits; ~36 events/dataset/batch over 448 sets) | fixed |
| **`evaluation.batchsize`** | **`16384`** (eval forward-only → match train BS) | fixed |
| `num_workers` | `2` | fixed |
| `dtype` / `allow_tf32` / `fused_optimizer` | `float32` / `true` (no-op V100) / `true` | fixed |
| optimizer / betas / eps / weight_decay | `MuonAdamW` (`optim/muon.py`; Muon on μP hidden matrices, AdamW `[0.9,0.999]` / `1e-8` / `0` on the rest) for pretraining and fine-tuning, set in the template's `fixed_params` (`training.optimizer`), my call (2026-10-26, "seems obvious that we should switch to muon"; "why would finetunes stay on adamW?") [D155], `par:v3_muon`. HPO rules 2 and 4 (lr surface, fine-tune lr_scale and layer_decay) were measured with AdamW: a Muon sweep centres on its own search, not on them | optimizer fixed; `muon_weight_decay` swept |
| `scheduler` / `clip_grad_norm` | `CosineAnnealingLR` / `5` | fixed |
| `training.lr` | run-design: centre on `lr*(t,D)` surface at your (t,D), sweep ±½ decade (see rule #2) | per-run |
| **EMA** (`ema` top-level flag) | swept `{false,true}` × `ema_decay∈[0.9,0.999]`; the big-run DyHPO and the catalog quick sweeps chose it on for their best trial (`docs/results.tex`, sec:bigrun, catalog quick sweeps), the catalog wave's best had it off; not settled | open |
| fine-tune | full retrain + layer-decay; `lr_scale∈[0.1,10]@1`, `layer_decay∈[0.75,1.0]` | fixed |

Sweep search ranges: HPO rule 3 below.

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

**Submitting a sweep** (the GPU budget rule of 2 applies). Submission
goes through `sweep_manager.py` (see below) so trials interleave round-robin
across sweeps — do **not** hand-loop `sbatch` over `jobs/*.sh`. The generators
call it for you:
```bash
# the generators render the #SBATCH block for the site they run on
# (siteconf.slurm_header) and submit through sweep_manager.py, so they run on the
# login node of the site where the sweep lives, through
#   site run <site> FA -- python sweep/generate_sweep.py --config sweep/<my_config>.yaml
# (checkout, env activated; never an ssh by hand). On the local checkout they are
# dry-runs only (no scheduler, no torch).
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
**Fallback when interleaving can't help (a lone sweep, or abundant GPUs):**
`sweep_manager submit` now defaults to **3 sequential waves** (`--seq-batches 3`)
for a single-sweep submission — the trials are split into 3 groups chained by
SLURM `afterany` dependencies, so wave *k+1* only starts after wave *k* has
`observe()`d and the (single-fidelity) Bayesian optimiser actually has data to fit.
Multi-sweep submissions default to `--seq-batches 1` (rely on interleaving); pass
`--seq-batches N` to force, `1` to disable.
**Chained submission (opt-in, `sweep_manager submit --chain [--capacity N]`).** Holds guided trials until the
preceding trials have observed; use it when free GPUs would start a whole sweep blind. `--weight`/`boost` do not
apply to chained jobs. On HTCondor the chain is the DAG (`cluster.chain_width`). Details: the script's docstring.

**Cross-sweep submitter — `sweep/sweep_manager.py`**. Submits
trials interleaved across sweeps and stamps each job with a SLURM `nice` value =
`round × gap`, where a sweep's `j`-th pending trial has `round = j // weight`.
Same round ⇒ same nice ⇒ runs together (fills GPUs); higher round ⇒ runs later;
`weight>1` ⇒ that sweep advances faster. Uses only user-level `nice` (no operator
rights). Commands: `submit <sweepdir>…`, `rebalance` (re-interleave all pending,
e.g. after adding a sweep), `boost <sweep> --weight N`, `status`, `cancel`.
Add `--dry-run` to preview. Registry: `~/.sweep_manager/registry.json`.

**Rebalancing across sites — `sweep/rebalance.py PREFIX... [--apply] [--allow-jeanzay]`.** `site pick` places a
batch once; this moves trials that have not started from the latest-finishing site to the earliest (by backlog and
achieved rate), deletes nothing (the source is cancelled and marked `MOVED_TO`), and copies a fine-tune parent's
whole run directory if missing. Run by the supervisor over `tp3_`; without `--apply` it prints the plan. Placement
follows `site pick` and this tool, never a hand pick and never a question for me.

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
   - **`D`-axis (training events summed over the sweep's 8 processes):** measured `lr* ∝ D^{~0.17}`, **saturating**
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

### Reported values

**Every reported loss is read at the best checkpoint**: the validation step with the lowest aggregate
`val_loss_no_reg` (the checkpoint selection, `experiment._result_extra`; `docs/results.tex`
`def:metric`), each process at that same step. Never the last validation, never the regularized
loss, never a per-process minimum over steps. A post-training evaluation is the best checkpoint
only when `training.es_load_best_model` is true; say which it is. **A run that blows up is reported
separately** (listed, with where it blew up), never dropped silently and never by a loss
threshold; its best checkpoint is still its value wherever the run is reported.

### Scaling fits

**Every scaling fit uses the floor-aware law `L(C) = A·C^(−α) + L_∞`** (`docs/results.tex`
`eq:scaling`), for learning curves in steps as much as in compute: an exponent quoted from
the bare power law `L ∝ C^(−α)` is wrong whenever the curve plateaus, and most of ours do.
The bare law is a diagnostic only, far above the floor, and says so where it is used.

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

## Waiting on jobs (a mechanism, never hand-polling)

A job belongs to a plan entry: submitted with `--item <entry>` and recorded with `plan run <entry> <run>` (a sweep: `plan run <entry> --sweep <site>:<name>`, done by `plan next` itself when missed). Nobody
waits on it by hand and no step agent stays alive for it. `site supervise --project FA` (CC feeder, rebalancer,
`watchdog.py --units` as every item's check, overruns, run ends) and one `site events --wait --project FA`, both in
the background, are the whole watch: the run's end opens an event that wakes the main session, and `plan next` then
lists the entry for its next step agent, which reads the log (`site logs <run>`) before reporting anything and acks
the event with what the log shows. New sweep families go into `analysis/transfer/expect_tp3.json` so their item's
check counts them.

- Never `sleep` in the foreground or rerun `site poll` by hand across turns (`sleep_guard.sh`); a foreground Bash call
  is capped at 10 min.
- **Quick tests must return quickly**: pass `evaluation.train_subsample=2000` (the
  post-training train-split pass otherwise runs the whole pool; default 10000).

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
- A stray generated `py.py` at the root shadows the `py` module and breaks `pytest`
  in the venv; test modules carry a `__main__` runner (`python tests/<file>.py`).
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
  `_init_backend` — **no-op on V100**, faster matmul on A100 at ~1e-3 precision; A/B'd
  on A100 (2026-10-02, `docs/results.tex` sec:ladder hand-off): TF32 on and off agree within
  the run-to-run spread, so it stays on. `training.fused_optimizer`
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

## Git & release workflow (trunk → published main)

This repo uses a **development trunk + generated public branch** model. Claude
handles git: commit and push as work progresses, keep a readable timeline.

**This is automatic, not a thing to ask about.** Commit with clear messages as
work lands and push without asking — pushing is *not* an outward action that needs
confirmation (see ground rule #2). Hooks back the workflow (and other rules) up
so they can't be silently forgotten (`.claude/settings.json` → `.claude/hooks/`):
a **`Stop` hook (`auto_push.sh`)** pushes any unpushed `trunk`/feature-branch
commits to origin at the end of every turn (already-committed work only; never
`main`); a **`PreToolUse` hook (`worktree_guard.sh`)** reminds me to open a
worktree when I start editing trunk code on `trunk`. Two more enforce rules
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
that pillar's accumulated backlog crosses its threshold (any change for `CLAUDE.md`, 200 changed lines or 20
commits for `.tex`; for code 520 lines weighted by their area's and kind's measured rate of blocking errors, about two errors waiting, `tools/review_weight.py`). The reviewer then reads the
**whole backlog at once**, which is the point: cross-edit problems (a rule now
stated twice, a config default whose callers were not updated, a section that no
longer reads as one argument) are invisible to a per-hunk review.

**When the `Stop` hook flags a pillar due, spawn its reviewer immediately, in that
turn, without asking me** — a hook cannot spawn a subagent, so I am the part that
executes it. If the nudge is skipped, `review_backlog.sh gate`
(`PreToolUse(Edit|Write)`) refuses further edits to the overdue pillar until its
reviewer runs, so ending the turn stalls rather than pauses. Shell writes count too:
`bashpost` (`PostToolUse(Bash)`) blocks when a command changed an overdue pillar's files,
and `bashpre` (`PreToolUse(Bash)`) refuses a `git commit` while an overdue, unlocked
pillar has uncommitted changes. Pillars under threshold stay unblocked; a lock expires
after 180 min.

`status` lists backlogs and `/review-now` runs reviewers on demand; `begin <name>`
takes a pillar's lock and `advance <name>` releases it on a pass (`begin` is also the
human override). **Never run `advance` on a reviewer's behalf.** A blocking review
leaves its lock held on purpose, so the fixes it demanded can be applied; the re-check of those fixes continues the
same reviewer (SendMessage to it, naming the fix commits), never a fresh one, which would pay the start-up and re-read
the code again (enforced by `reviewer_resume.py`). Watermarks:
`.claude/.review_state/` (gitignored, per-checkout).

**Branches**
- **`trunk`** — the development trunk and the only working branch. *Everything*
  lives here: the core code plus all tooling (`tools/`, `tests/`, `sweep/`,
  `scripts/`, `attribution/`, `data/` scripts, `notes/`, `CLAUDE.md`, recipes,
  `sites/`). This is where development happens, and it is what every site checks
  out: `site sync <site> FA` pushes it and fast-forwards the site's copy. There
  are no per-site branches — a site's differences live in `sites/sites.yaml`, not
  in git.
- **`main`** — the **public, stripped-down core**. It is a *build artifact* of
  `trunk`, regenerated by `scripts/publish_main.sh` from the `PUBLIC_PATHS`
  allowlist (core run path + `config/` + `models/` + `IntrinsicDimDeep/` +
  example `recipes/` + README). **Never edit `main` by hand**; never merge
  `trunk → main`. To change what's public, edit the allowlist and re-publish.
- **Retired: `ccin2p3`, `jeanzay`.** The former per-cluster branches still exist
  on GitHub, have no commits `trunk` lacks, and are not to be checked out, merged
  or ported to.

**Working rules**
1. **Do work on `trunk`** (or a feature branch off it).
2. **Open a worktree for new work, always under this repo:**
   `git worktree add worktrees/wt-<feat> -b <feat> trunk`, implement and verify
   there, then merge back into `trunk` and `git worktree remove` it. Never
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
   the removal until this has run.
3. **Commit small and often; pushing is automatic.** Small, frequent, readable
   commits over big dumps. The `Stop` hook pushes already-committed work, so never
   ask permission to push; to push mid-turn just run `git push` (allowlisted).
4. **Publish the public core** with `scripts/publish_main.sh` (regenerates `main`
   from the allowlist and pushes). Run it after core-facing changes land on
   `trunk`. Use `--no-push` to review first.

**Visibility caveat:** it's a *single* GitHub repo, so all branches share
visibility — making the repo public exposes `trunk` and the retired branches too. Only `main`'s
*tree* is stripped, not the other branches. If a dev branch ever needs to be
truly private, that forces a separate-repo split.

(Ground rule #4 still holds everywhere: never attribute commits/PRs to Claude.)

## Sweeps across sites: `sweep/drive.py`

A sweep no longer has to live on one cluster. `generate_sweep` + `sweep_manager` keep the
DyHPO state in a file that every trial locks on one site's filesystem; `sweep/drive.py`
keeps that state on the laptop and sends stateless trials wherever `site pick` says:

    python sweep/drive.py --config sweep/sweep_config_X.yaml [--config ...] \
        [--n-trials N] [--parallel P] [--mem 8G] [--cpus 4] [--hours 0.5] \
        [--sites ccin2p3 lxplus] [--allow-jeanzay] [--poll 60]

- State and results: `sweeps_local/<sweep_name>/` (dyhpo_state.pkl, driver_state.json,
  results/*.json, summary.txt). Resumable: run the same command again.
- Placement per trial through `site pick` (fitting free GPUs, submit caps, expected waits);
  `--sites` restricts, Jean Zay only with `--allow-jeanzay`.
- Trials run `sweep/trial_job.sh` via `site submit` with `--detached --hp-idx --t-steps --hp k=v`
  (`run_trial.py --detached`): no shared state, result returned as one RESULT_JSON log line.
- Pools: before the first trial on a site the recipe's datasets are checked there and, if
  missing, built by `scripts/prebuild_recipes.sh` as a CPU job on that site (SLURM or HTCondor).
- Detached trials cold-start (no cross-fidelity warm start); fine for single-fidelity sweeps.
- Code must be committed: the driver syncs each site once at start and submits with --no-sync.
- The driver runs on the laptop in `.venv-driver` (CPU torch + scikit-learn); it builds it
  itself on first use via `sweep/driver_env.sh`.
Use the classic path only for a sweep that must stay on one site (multi-fidelity warm starts).
