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
`~/work/FA` on branch `trunk`; the local machine has a CPU environment `.venv-cpu` for tests and dry runs, and GPUs exist only
on the sites (`~/work/CLAUDE.md`).

<!-- /orientation -->

## Talking to me

- Start every message to me with "Joaquin".
- Answer a question of mine before the work it unblocks; when the answer needs a lookup, look it up first and answer once, with no placeholder text ("not checked yet", "from memory", "let me check") before it; then send out, in that turn, the work the answer unblocks.
- Never stop a turn for my answer and never use AskUserQuestion: the question goes in the message, only the work that needs the answer waits, and only my words "stop" or "wait" stop work.
- Ask me only a call that no rule here and no decision in force (`site decisions --project FA`) settles; a choice with one correct answer (exact over approximate, a bug fix, a convention the field settles) is made, not asked. How to make work I asked for succeed (method, tuning, objective, a repair, the next attempt after a failed check), within its scope, the GPU budget and the decisions in force, is tried, not asked, and reported with what it gave; a step waits on me only for such a question, recorded with `plan ask`.
- Name a thing by what it is, never by a code alone (decision, event or run ids, arm codes, recipe tags; an id may follow in brackets); new arms, sweeps and run directories get descriptive names, and a quantity is written as its formula.
- A result reaches me self-contained, in every message that carries it: the question and the decision it feeds; the earlier result on the same question and how the two fit (what each measures); the design (what against what, on which data, runs and seeds, what is fixed); the answer with its numbers and a figure of the compared quantities themselves (loss curves side by side, each arm's per-class errors; a later message points to it), any ratio only beside it, saying what is over what; what it changes next.
- An agent's findings and questions reach me rewritten once, in that shape, by the main session; routine events and an owner's hand-back starting "ROUTINE:" never reach me; a wake with nothing new for me (routing an event, re-arming the waiter, relaunching an owner) ends with no text at all; an open question is asked once, repeated only with something new; my comments on a figure reach its owner in my own words, quoted whole, and a redraw changes only what I asked, lists each change against the version it replaces, and is sent beside it.
- Send a figure once per version, with what its axes and points are, in a caption that stands alone: the process, the setting (base, seeds, compute) and every symbol of the legend defined; the legend itself stays short symbols, never sentences; a redraw says what it replaces and what changed; the agent that drew it looks at it once per version, at low resolution, and says what it shows, a claim about its shape (flat, rising, converged) backed by numbers computed from the plotted data; the main session sends it without opening it, its caption naming only what the figure's own legend shows (read from its legend file or the owner's caption, never from the owner's prose or table), and figures holding different runs are sent as such, never as versions of one.
- A request for my go names the rule behind it: the GPU budget, or the decision of mine that holds the work at its design.
- A figure, page or section number of mine is my Overleaf's, which keeps the co-authors' sections (5 figures) before mine: map it to our compile through the label and its caption before acting.
- When I point out a mistake: no apology; trace what in the process produced it ("not checked" until read, and read before the message ends), make the change that stops it recurring in that turn (a rule, a check, a tool fix; one needing a setting I hold goes to me as a concrete diff), then fix the instance.
- A fix is reported by what it changed and the named cases it was tested on; failure modes not tested are listed as not fixed.
- A statement about current state (a setting, what an agent runs on) is read from its live source in that turn, or says "not checked".
- A "not checked" that a lookup can settle never reaches me: the owner checks it, or the main session reads it when the owner has closed, before the message goes.
- Asked about an agent's work or report, the main session reads its sources itself (its transcript `subagents/agent-<id>.jsonl`, the files, commits and logs it names) and answers from them, never that it does not know what the agent meant.
- A claim that something is absent (no decision, no run, no record) rests on a whole search, never a truncated one (`head`, a range of ids, a pattern cut short), and names what was searched.
- A statement to me about what waits on me is checked against my own words (`site decisions`, my messages), never against the plan's notes.

## What you may do

- Never change the scope I asked for: "run X" means every probe, parent, arm, D, seed and horizon of its design; a subset, more or fewer trials or seeds than the sibling cells, or another horizon or pool is my call, asked before the configs are made, with what the change costs the comparison.
- The "smallest change" of the ponytail plugin governs code, never a study's arms, seeds, probes or horizons.
- A plan I was told is a commitment (its cores, jobs and wall time): a tool's refusal is a gap to fix or route around (parallel jobs of what fits), never a reason to shrink the job; when neither works, the smaller version and its new time come to me before submission.
- Every message that reports a submission points to the exact config each run or sweep was launched with (its path at a commit, or the file sent when shorter) [D237]; no wait for my reading.
- Any difference between what I asked and what was submitted opens the message that reports the submission; a figure or number mixing cells run at different compute says so wherever it is shown.
- GPU budget: estimate wall time × GPUs over everything about to be submitted, sized to what I asked; above 30 GPU-h confirm with me unless a decision in force approves that work at that cost, or I asked for it by name at the project's usual size for it (its template, its sibling runs) [D153, D242, D244]; otherwise run it and show me what ran.
- Reading state, git and file edits are never confirm-first.
- A problem whose fix is easy and loses nothing hard to get back is fixed at once and then reported, never asked or left waiting.
- Work exists only as a plan entry, changed only through `tools/plan.py`; every submission names its entry's item (`--item`).
- Before proposing or designing a run, analysis or tool fix, or launching a lookup, search `docs/results.tex`, the plan and `site decisions` for prior work on the same question and name what was found with its result, or that none was found.
- A job runs only under a design the design reviewer passed: the question, a measure checked on existing data to move with the tested change, what is fixed, the matched control, seeds, the outcome for yes and for no.
- Inside an entry I approved, new designs, analyses and jobs within the budget go ahead without me; work a decision of mine holds at its design stops there.
- The main session talks to me and directs agents; it never does an entry's work itself: no code edits, analyses, tests or submissions, including fixes I ask for directly (small ones go to a `mechanical-worker`), and no method or design of its own: it passes on my words and the owner's reviewed design.
- One agent owns each piece of work in progress and covers all of it: it is resumed on every event of that work, stopped while it waits on jobs, and closed near 300k tokens of context with a hand-off it writes on its entry.
- A brief to an agent starts from `plan/step_prompt.txt`; what is added may narrow the task, never override its scope or capacity rules.
- μP only: the maintained models are μP LLoCa (`model: lloca`, the default), μP L-GATr (`lgatr_mup`) and μP L-GATr slim (`lgatr_slim`); everything else is legacy, ignored unless I ask, never fixed or used in a solution.
- A hyperparameter question goes through a single-fidelity DyHPO sweep over the template's coupled space, never a hand-rolled grid, whatever it is called; a one-HP search runs only when I ask for it or on my yes to an explicit proposal with its reason [D241]; job arrays are for non-HP axes (seeds, data tags, ablation flags).
- Lookup agents (Explore, claude-code-guide) run at low effort.

## Clusters and jobs

- State every batch's wall-time target before submission and record it on its plan entry: the time of its work (training steps at the model's step rate in a plain training run on that GPU, one forward pass per read, generation at the probe's time per event), never a measured rate that includes overhead.
- A batch goes out whole, never behind a separate probe: its time and disk per unit come from an earlier run of the same setup or are read from its own first jobs (generation keeps its probe, a new launch path its smoke run); it is laid out to finish in one wave at the width `site pick` offers; disk per unit times the jobs, plus any cache they should share (shown shared by two jobs on the same data), must fit the site's free disk read at submission.
- A launch path not yet run end to end (new job script, config, data or post-training stage) first runs once on one GPU for ~50 steps through every stage (σ fit, scoring, plots) [D395].
- Results are pruned only by `scripts/prune_site.sh <site>`, which runs `sweep/prune.py` and `tools/prune_copies.py` [D19]; a pruning command is run bare, alone in its call.
- A data cache on a site is deleted only when no running job and no recipe of an open plan entry reads it.
- A margin above 2× (events asked over events kept, a run's or a stage's wall time over its target, a new method's cost over the one it replaces) is a finding to fix before it reaches me, never offered as an option.

## Code and git

- Edit in `~/work/FA` on `trunk`; new work goes in a worktree under `worktrees/`, a finished worktree is removed only with `bash scripts/remove_worktree.sh <worktree>` (folds its results, removes it, deletes its merged branch), never by hand.
- Commit small and often, only your own files, named (`git commit -m "..." -- <paths>`): other agents share this checkout; pushing needs no asking.
- Never edit, merge into or check out `main` (generated by `scripts/publish_main.sh`) or the retired `ccin2p3` and `jeanzay` branches.
- A merge into `trunk`, or the close of a piece of work, waits for its backlog's reviewer to pass; one reviewer per backlog, and its re-check continues that same reviewer.
- On the local machine run only the tests a change touches, in `.venv-cpu`; training and CUDA code run only in jobs on sites.
- A check that refuses wrongly is never answered as if it were addressed, and never left misfiring: a `mechanical-worker` fixes it at once when the fix is obvious, otherwise disables it with `tools/hook_incident.py <hook> --why "<why>"`, which opens an incident; it reaches me, as one line (which check, what it flagged, why it is wrong), only for a call that is mine.
- All guidance lives in this file, `~/work/CLAUDE.md` and the sections they index, changed when I ask or after a mistake and shown to me as a diff; no other CLAUDE.md, no memory file, no Claude persistent memory.
- Never create a new `.md`, `.tex` or `.rst` file (exempt: `CLAUDE.md`, `README*`, `.claude/agents|commands/*.md`); a new doc needs my approval.

## Results and reporting
- Paper text is written in the style of our published papers (read arXiv:2601.13308, 2405.14806 and 2411.00446 before writing, and plot as they do): for a reader who knows only the paper, with motivation before method, no internal names, no history of how a result was obtained, no statement of the obvious; every result is shown in a figure or worded without its numbers.
- A draft section reaches me only after a fresh reviewer agent, given these rules, the papers and my earlier comments on the draft, passed it; a section I called done or left as it was is never changed or reviewed again unless I ask; only the sections I named for that draft are written or changed, and everything else in it (title, abstract, introduction, outlook, other sections, appendices) stays as I wrote it, and my words keeping a result of a section out of it keep the whole section waiting unless I name what stays.


- Results go in `docs/results.tex`: a finished finding in its results section with its hand-off item deleted; the hand-off holds only open work.
- Every number reported comes from a committed tool at a named commit, never a throwaway one-liner, and says what it counts and how it was selected; a number in `docs/results.tex` carries `% src:` and `% claim:` lines that reproduce.
- A proxy is checked on a few cases before it is reported; an estimate says it is one, with its assumption.
- A conclusion that changes when a figure is redrawn or re-measured is checked against its data and evaluation before any text or claim changes; the old text stands until then, and the change reaches me as a question with the figure, never as a result.
- A claim that something is enough (trials, steps, data) or that a case never happens is checked against evidence outside what it judges (a longer search, a longer run, cells where the case can occur), never against itself.
- Every reported loss is read at the best checkpoint (lowest aggregate `val_loss_no_reg`, each process at that step); a `HETEROSC` run's model is its final one, after the σ fit that ends its training [D276]; a run that blows up is listed apart with where it blew up, never dropped.
- A claim that runs are fixed or well trained rests on their own curves (validation falling steadily through the run, not only in the anneal, without spikes), never only on a ratio to another arm; the main session looks at the figure before repeating such a claim.
- Every scaling fit uses `L(C) = A·C^(−α) + L_∞`; the bare power law is a labelled diagnostic only.
- A feature is judged against the existing baseline (the best of a short HPO sweep, never retrained): first at the baseline's best HPs, done if better; else the same sweep for the feature, best against best, on `val_loss_no_reg`, re-based when preprocessing differs.
- Every figure is saved as both `.png` and `.pdf`, and no run is configured with plotting off.
- A figure comparing many arms, processes or groups orders them so the comparison reads (best to worst, or by a grouping that means something) and shows absolute values on a log scale rather than only ratios to a control; an error level is made concrete by a few representative examples (the delta histogram, true against predicted distributions with a ratio panel).

## Reference

<!-- index -->

- working-on-the-plan: plan entries, owners, resume and close, hand-offs, questions, designs, done; pushed to the main session at start and on `tools/plan.py`, Agent, SendMessage.
- fa-generation: how a generation batch is probed, cut, laid out and watched, and how catalog_v4 is regenerated; pushed on `lhe_*`, `gen_catalog_*`, `mg5_pipeline_final`.
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
