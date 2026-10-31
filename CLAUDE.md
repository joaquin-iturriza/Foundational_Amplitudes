# Foundational_Amplitudes

<!-- orientation -->

A foundation model for tree and loop scattering amplitudes in particle physics. A single Lorentz-equivariant
transformer is trained jointly on many processes (ee→WW, ee→ttbar, ee→uu, …), then fine-tuned to new processes and
orders. Research threads: joint multi-process pretraining, scaling laws, fine-tuning and transfer, and DyHPO
hyperparameter sweeps; the training data is a MadGraph process catalog generated here (catalog_v4).

Entry points: `run.py` (Hydra, config root `config/`, default `amplitudes`) builds the experiment of `experiment.py`
on `base_experiment.py`; the model is μP LLoCa (`models/lloca.py`, `config/model/lloca.yaml`); sweeps live in
`sweep/`; generation is `mg5_pipeline_final.py` driven by `recipes/gen_catalog_v4.py`; work in progress is
`plan/plan.yaml`, read and written through `tools/plan.py`; results are `docs/results.tex`. The working copy is
`~/work/FA` on branch `trunk`; the laptop has a CPU environment `.venv-cpu` for tests and dry runs, and GPUs exist only
on the sites (`~/work/CLAUDE.md`).

<!-- /orientation -->

## Talking to me

- Start every message to me with "Joaquin".
- Answer a question of mine in text first, before any tool call of that turn; then send out, in that turn, the work the answer unblocks.
- Never stop a turn for my answer and never use AskUserQuestion: the question goes in the message, only the work that needs the answer waits, and only my words "stop" or "wait" stop work.
- Ask me only a call that no rule here and no decision in force (`site decisions --project FA`) settles; a choice with one correct answer (exact over approximate, a bug fix, a convention the field settles) is made, not asked.
- Name a thing by what it is, never by a code alone (decision, event or run ids, arm codes, recipe tags; an id may follow in brackets); new arms, sweeps and run directories get descriptive names, and a quantity is written as its formula.
- A result reaches me self-contained, in every message that carries it: the question and the decision it feeds; the design (what against what, on which data, runs and seeds, what is fixed); the answer with its numbers; what it changes next.
- An agent's findings and questions reach me rewritten once, in that shape, by the main session; routine events never reach me.
- An owner's hand-back that starts "ROUTINE:" gets no message to me.
- Send a figure once per version, with what its axes and points are; a redraw says what it replaces and what changed; the agent that drew it looks at it once per version, at low resolution, and says what it shows; the main session sends it without opening it.
- A request for my go names the rule behind it: the GPU budget, or the decision of mine that holds the work at its design.
- When I point out a mistake: no apology; trace what in the process produced it ("not checked" until read, and read before the message ends), make the change that stops it recurring in that turn (a rule, a check, a tool fix; one needing a setting I hold goes to me as a concrete diff), then fix the instance.
- A fix is reported by what it changed and the named cases it was tested on; failure modes not tested are listed as not fixed.
- A statement about current state (a setting, what an agent runs on) is read from its live source in that turn, or says "not checked".
- A statement to me about what waits on me is checked against my own words (`site decisions`, my messages), never against the plan's notes.

## What you may do

- Never change the scope I asked for: "run X" means every probe, parent, arm, D, seed and horizon of its design; a subset, fewer trials or seeds than the sibling cells, or another horizon or pool is my call, asked before the configs are made, with what the cut costs the comparison.
- The "smallest change" of the ponytail plugin governs code, never a study's arms, seeds, probes or horizons.
- A plan I was told is a commitment (its cores, jobs and wall time): a tool's refusal is a gap to fix or route around (parallel jobs of what fits), never a reason to shrink the job; when neither works, the smaller version and its new time come to me before submission.
- Any difference between what I asked and what was submitted opens the message that reports the submission; a figure or number mixing cells run at different compute says so wherever it is shown.
- GPU budget: estimate wall time × GPUs over everything about to be submitted, sized to what I asked; above 30 GPU-h confirm with me unless a decision in force approves that work at that cost [D153]; under it, run it and show me what ran.
- Reading state, git and file edits are never confirm-first.
- Work exists only as a plan entry, changed only through `tools/plan.py`; every submission names its entry's item (`--item`).
- Before proposing or designing a run or analysis, search `docs/results.tex`, the plan and `site decisions` for prior work on the same question and name what was found with its result, or "none found" with what was searched.
- A job runs only under a design the design reviewer passed: the question, a measure checked on existing data to move with the tested change, what is fixed, the matched control, seeds, the outcome for yes and for no.
- Inside an entry I approved, new designs, analyses, jobs within the budget and repairs of its defects go ahead without me; work a decision of mine holds at its design stops there.
- The main session talks to me and directs agents; it never does an entry's work itself: no code edits, analyses, tests or submissions.
- One agent owns each piece of work in progress and covers all of it: it is resumed on every event of that work, stopped while it waits on jobs, and closed near 300k tokens of context with a hand-off it writes on its entry.
- A brief to an agent starts from `plan/step_prompt.txt`; what is added may narrow the task, never override its scope or capacity rules.
- μP only: the maintained models are μP LLoCa (`model: lloca`, the default), μP L-GATr (`lgatr_mup`) and μP L-GATr slim (`lgatr_slim`); everything else is legacy, ignored unless I ask, never fixed or used in a solution.
- A hyperparameter question goes through a single-fidelity DyHPO sweep, never a hand-rolled grid, whatever it is called; job arrays are for non-HP axes (seeds, data tags, ablation flags).
- Lookup agents (Explore, claude-code-guide) run at low effort.

## Clusters and jobs

- State every batch's wall-time target before submission and record it on its plan entry.
- A batch goes out only after a small probe (yield, time and disk per unit), laid out to finish in one wave at the width `site pick` offers.
- A margin above 2× (events asked over events kept, a run's wall time over its target) is a finding to fix, not to absorb.
- A failed task goes to the agent that owns its work, which reads its cause in the log before anything is resubmitted.

## Code and git

- Edit in `~/work/FA` on `trunk`; new work goes in a worktree under `worktrees/`, a finished worktree is removed only with `bash scripts/remove_worktree.sh <worktree>` (folds its results, removes it, deletes its merged branch), never by hand.
- Commit small and often, only your own files, named (`git commit -m "..." -- <paths>`): other agents share this checkout; pushing needs no asking.
- Never edit, merge into or check out `main` (generated by `scripts/publish_main.sh`) or the retired `ccin2p3` and `jeanzay` branches.
- A merge into `trunk`, or the close of a piece of work, waits for its backlog's reviewer to pass; one reviewer per backlog, and its re-check continues that same reviewer.
- On the laptop run only the tests a change touches, in `.venv-cpu`; training and CUDA code run only in jobs on sites.
- A check that refuses wrongly is never answered as if it were addressed, and never left misfiring: fixed at once when the fix is obvious, otherwise disabled with `tools/hook_incident.py <hook> --why "<why>"`, which opens an incident; it reaches me, as one line (which check, what it flagged, why it is wrong), only for a call that is mine.
- All guidance lives in this file, `~/work/CLAUDE.md` and the sections they index, changed when I ask or after a mistake and shown to me as a diff; no other CLAUDE.md, no memory file, no Claude persistent memory.
- Never create a new `.md`, `.tex` or `.rst` file (exempt: `CLAUDE.md`, `README*`, `.claude/agents|commands/*.md`); a new doc needs my approval.

## Results and reporting

- Results go in `docs/results.tex`: a finished finding in its results section with its hand-off item deleted; the hand-off holds only open work.
- Every number reported comes from a committed tool at a named commit, never a throwaway one-liner, and says what it counts and how it was selected; a number in `docs/results.tex` carries `% src:` and `% claim:` lines that reproduce.
- A proxy is checked on a few cases before it is reported; an estimate says it is one, with its assumption.
- Every reported loss is read at the best checkpoint (lowest aggregate `val_loss_no_reg`, each process at that step); a run that blows up is listed apart with where it blew up, never dropped.
- Every scaling fit uses `L(C) = A·C^(−α) + L_∞`; the bare power law is a labelled diagnostic only.
- A feature is judged against the existing baseline (the best of a short HPO sweep, never retrained): first at the baseline's best HPs, done if better; else the same sweep for the feature, best against best, on `val_loss_no_reg`, re-based when preprocessing differs.
- Every figure is saved as both `.png` and `.pdf`, and no run is configured with plotting off.

## Reference

<!-- index -->

- working-on-the-plan: plan entries, owners, resume and close, hand-offs, questions, designs, done; pushed to the main session at start and on `tools/plan.py`, Agent, SendMessage.
- fa-generation: how a generation batch is probed, cut, laid out and watched; pushed on `lhe_*`, `gen_catalog_*`, `mg5_pipeline_final`.
- fa-catalog-reference: what catalog_v3 and catalog_v4 pools contain and how they are sampled; pushed with fa-generation.
- fa-canonical-setup: the decided run defaults and the one sweep template; pushed on sweep configs and `site submit`.
- fa-sweeps-and-hpo: DyHPO, `sweep_manager`, `drive.py`, rebalancing, the HPO search ranges; pushed on `sweep/**`.
- fa-data: which datasets a run saw, process classes, sampling roles, the data pipeline; pushed on `dataset.py`, recipes, run configs.
- fa-model: the model code map, particle encoding, coupling orders, μP; pushed on `models/**`, `wrappers.py`.
- fa-run-and-config: `run.py`, the experiment classes, the Hydra configs; pushed on `run.py`, `config/**`.
- fa-performance: per-step toggles, dataloading, compute knobs, what was not done; pushed on the hot-path files, `LLOCA_*`.
- fa-sites-and-paths: checkouts, `siteconf`, the job-script header, FA's site directives; pushed on `sites/**`, `site submit|run FA`.
- fa-results-notes: provenance lines, the claims check, results against hand-off; pushed on `docs/results.tex`.
- fa-figures: the figure pair and style; pushed on plotting scripts and `savefig`.
- fa-git-and-release: trunk, worktrees, the public `main`; pushed on `git worktree|merge|push`.
- fa-review: backlogs, reviewers, the merge and close requirement; pushed on `review_backlog.sh`, `.claude/agents/**`.
- fa-hooks: the check set, refusals against notes, the dispatcher, incidents; pushed on `.claude/hooks/**`, settings.
- fa-testing: the CPU environment and test runner; pushed on `pytest`, `tests/**`.

<!-- /index -->
