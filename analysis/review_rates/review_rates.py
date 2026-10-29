#!/usr/bin/env python3
"""How often the code reviewer (repo-reviewer) finds a blocking error, per changed line and per commit.

  python3 analysis/review_rates/review_rates.py [--fig]     -> analysis/review_rates/review_rates.json (+ figure)

Runs: every Agent launch with subagent_type == "repo-reviewer", in the main transcript of each session of this project
and in its subagents' transcripts (step agents launch reviewers too), mapped to the reviewer's transcript by the
"agentId: <id>" line of the launch's tool result (as tools/agent_costs.py does). Never selected by prompt text.

Reviews: the reviewer's watermark moves (the "watermark advanced to <sha>" lines in the reviewer transcripts) cut
trunk's first-parent history into intervals; each interval is one review episode and every run is assigned to the
interval its last report falls in (re-checks after fixes and concurrent reviewers of one backlog: same episode).
Exposure: lines added + deleted, commits and files in the code pathspec of .claude/hooks/review_backlog.sh, between
consecutive watermarks. The threshold that fired: the backlog from the episode's start to the trunk commit at its
first run's launch (committed work only; uncommitted files are not counted).

Findings: FINDINGS below, read by hand from each run's reports (the reviewer's own blocking / high / medium items; nits,
minor, low-hygiene, advisory, cross-pillar relays and notes-only items excluded). A finding re-reported by a later run of
the same episode is listed once. Tags: b = defect in code, s = scope or design choice with no decision of the user's,
pre = defect in code older than the backlog (kept out of the rates), x = withdrawn by the reviewer (kept out). A finding
whose file the backlog did not change (searched in its review's diff and the next one's; for the two reviews of worktree
code in WORKTREE, also the two after) is retagged pre,
or outside when the file is not in the code pathspec at all (its lines are never counted): both are kept out.
Intervals: 68% from resampling reviews (bootstrap), because findings cluster within a review; the Poisson interval
per line is kept in the JSON for comparison and is about 2.5 times narrower.
Kind of change: the file's status in the diff where it was found by the search above:
new file, edit or deletion."""
import fnmatch, glob, json, os, re, subprocess, sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJ = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")
OUT = os.path.join(REPO, "analysis/review_rates/review_rates.json")
START = "cab4940"                       # watermark the first review started from (its prompt: "since cab4940")
PATHS = ["*.py", "*.sh", "config", "recipes", "models", "sweep", "tools", "tests", "scripts", "IntrinsicDimDeep",
         ":(exclude)*.json", ":(exclude)*.log", ":(exclude)*.out", ":(exclude)*.err", ":(exclude)runs/*",
         ":(exclude)sweeps/*"]       # review_backlog.sh PILLARS, repo-reviewer row
LINES_T, COMMITS_T = 600, 30
CUTOFF = "2026-10-28T00:00:00Z"        # runs launched and watermark moves made after this are left out (reproducible numbers)
SKIP = {"ac57e4c2d"}   # a review of the unmerged branch diag-llm4lhc, launched by the main session before the merge
                       # (2026-10-08); the same lines are reviewed again after the merge, so counting both would count them twice
WORKTREE = {"ab3551ca1", "a48d20fe6"}   # their reports review worktrees/wt-decomp and .claude/worktrees/agent-a6859ebf5e...
NO_CONFIGS = [":(exclude,glob)sweep/**/*.yaml", ":(exclude,glob)recipes/**/*.yaml"]


def F(*a):
    return list(a)


FINDINGS = {  # reviewer agent id (first 9 chars) -> [(file or glob, tag)]
 "a15940777": F(("sweep/generate_sweep.py", "b"), ("sweep/generate_sweep.py", "b")),
 "acd62e7b4": F(("analysis/catalog_v2/steps_tuned.py", "b"), ("analysis/catalog_v2/steps_tuned.py", "b")),
 "afd051fb3": F(("analysis/catalog_v2/steps_tuned.py", "b"), ("analysis/catalog_v2/steps_tuned.py", "b"),
                ("analysis/catalog_v2/solo_b1k.py", "b"), ("sweep/generate_pretraining_scaling_sweeps.py", "b"),
                ("config/model/lloca.yaml", "b"), ("scripts/job_catalog_short_ab.sh", "b"),
                ("analysis/catalog_v2/unweight_excess.py", "b")),
 "acc8d0fe3": F(("sweep/compare_finetune_methods.py", "b"), ("sweep/fit_scaling_law.py", "b"),
                ("scripts/rebuild_figures.sh", "b"), ("analysis/hpo_optima/make_fig.py", "b")),
 "a7de931dc": F(("analysis/catalog_v2/solo_old_new.py", "b")),
 "a92bf6cc8": F(("sweep/trial_job.sh", "b"), ("sweep/sweep_config_drivetest_ee_aa.yaml", "b")),
 "af5a6c16d": F(("experiment.py", "b"), ("experiment.py", "b"), ("sweep/gen_transfer_pilot.py", "b")),
 "a9e37c971": F(("scripts/job_transfer_calib.sh", "b")),
 "ac43132d8": F(("scripts/job_transfer_calib.sh", "b"), ("sweep/sweep_manager.py", "b"),
                ("analysis/transfer/collect_calib.py", "b"), ("analysis/transfer/collect_calib.py", "b")),
 "a93a02b7a": F(("analysis/transfer/seeded_ab.py", "b"), ("analysis/transfer/seeded_ab.py", "b"),
                ("analysis/transfer/residual_map.py", "b"), ("analysis/transfer/transfer_ch1.py", "b")),
 "a4a1b8d46": F(("tools/steer_pool.py", "b"), ("sweep/gen_transfer_pilot.py", "b"),
                ("sweep/sweep_config_tps_scr_ee_WW_d*.yaml", "b"), ("tools/steer_pool.py", "b"), ("tools/steer_pool.py", "b")),
 "a0354809a": F(("sweep/gen_transfer_pilot.py", "b")),
 "a38741f7b": F(("sweep/gen_transfer_pilot.py", "b"), ("tools/rebuild_run.py", "b"), ("experiment.py", "pre")),
 "a97fcee5b": F(("experiment.py", "b"), ("analysis/transfer/transfer_all.py", "b"), ("sweep/gen_transfer_pilot.py", "b"),
                ("experiment.py", "b"), ("analysis/transfer/transfer_all.py", "b")),
 "a910e5ae3": F(("analysis/transfer/collect_ladder.py", "b")),
 "a9334aa16": F(("sweep/sweep_config_tp3_uufte_*.yaml", "b"), ("scripts/archive_runs.sh", "b")),
 "aafd86097": F(("sweep/rebalance.py", "b"), ("sweep/rebalance.py", "b"), ("sweep/rebalance.py", "b"),
                ("analysis/transfer/rung_focus.py", "b")),
 "ae3470054": F(("sweep/sweep_config_tp3_scr32k_*.yaml", "b"), ("sweep/refill.py", "b"), ("sweep/rebalance.py", "b"),
                ("sweep/refill.py", "b")),
 "af2efedee": F(("analysis/transfer/pretrain_hp.py", "b")),
 "a98c71584": F(("analysis/transfer/finetune_hp.py", "b"), ("analysis/transfer/finetune_hp.py", "b")),
 "a322eb9f9": F(("analysis/transfer/cells.py", "s"), ("analysis/transfer/cells.py", "b"), ("analysis/transfer/star_arms.py", "b"),
                ("analysis/transfer/horizon32k.py", "b"), ("analysis/transfer/rung_grid.py", "b")),
 "a4cf7ea3f": F(("sweep/run_trial.py", "b"), ("sweep/gen_transfer_pilot.py", "b")),
 "a76f41a4f": F(("tools/synthetic_amplitudes.py", "b"), ("tools/synthetic_amplitudes.py", "b"), ("mg5_pipeline_final.py", "b")),
 "a5d63d760": F(("tools/zero_shot_ft.py", "b")),
 "a7cbd14ab": F(("scripts/job_config_smoke.sh", "b")),
 "a6ae99aba": F(("sweep/sweep_config_tp3_accrr*.yaml", "x"), ("sweep/sweep_config_tp3_accrr*.yaml", "b")),
 "a9d814495": F(("sweep/prune.py", "b"), ("sweep/prune.py", "b"), ("sweep/prune.py", "b"), ("sweep/prune.py", "b"),
                ("sweep/sweep_config_tp3_wadd_*.yaml", "b"), ("sweep/sweep_config_tp3_wadd_*.yaml", "b")),
 "a53367123": F(("sweep/gen_highmult.py", "b"), ("analysis/transfer/expect_tp3.json", "b")),
 "aede3c40f": F(("experiment.py", "b"), ("sweep/sweep_config_tp3_waddm_*.yaml", "b"), ("base_experiment.py", "b")),
 "a4bf90155": F(("sweep/sweep_config_tp3_waddm_frz*.yaml", "b")),
 "a67b99cf8": F(("analysis/transfer/waddm_tradeoff.py", "b")),
 "a7248e31f": F(("sweep/sweep_config_tp3_waddr_*.yaml", "b"), ("analysis/transfer/highmult.py", "b"), ("sweep/refill.py", "b")),
 "a43d545b2": F(("tools/synthetic_amplitudes.py", "b"), ("tools/synthetic_amplitudes.py", "b")),
 "aa272a42f": F(("tools/synthetic_amplitudes.py", "s")),
 "a3d2a9d15": F(("analysis/transfer/delta_plots.py", "s"), ("analysis/transfer/delta_plots.py", "b"), ("scripts/hand_place.sh", "b")),
 "a94cca42c": F(("recipes/gen_catalog_daniel.py", "b"), ("recipes/gen_catalog_daniel.py", "b"), ("experiment.py", "pre")),
 "a7bdf0beb": F(("models/lloca.py", "b"), ("models/lloca.py", "s"), ("tools/multiplicity_extrap.py", "b"),
                ("config/default.yaml", "s")),
 "a3945263e": F(("recipes/gen_catalog_daniel.py", "s"), ("recipes/gen_catalog_daniel.py", "b"), ("scripts/publish_main.sh", "b")),
 "a57daaf1e": F(("analysis/transfer/het_sweep.py", "b")),
 "ab3551ca1": F(("experiment.py", "b"), ("experiment.py", "b"), ("experiment.py", "b"), ("config/default.yaml", "b")),
 "ae9b06af2": F(("analysis/transfer/het_long.py", "b"), ("base_experiment.py", "b"), ("data_mmap.py", "b"), ("data_mmap.py", "b"),
                ("tools/synthetic_loops.py", "b"), ("mg5_pipeline_final.py", "b"), ("data_mmap.py", "b")),
 "abd3d065b": F(("sweep/sweep_config_tp_nonsm_*.yaml", "s"), ("tests/test_particle_encoding.py", "b")),
 "afa298df1": F(("analysis/transfer/het_ab.py", "b"), ("analysis/transfer/het_long.py", "b"),
                ("analysis/transfer/delta_plots.py", "b"), ("analysis/transfer/zero_shot_jcaus.py", "b")),
 "a9969ec01": F(("analysis/transfer/het_delta_summary.py", "b"), ("analysis/transfer/het_delta_summary.py", "b"),
                ("analysis/catalog_daniel/make_heldout_twins.py", "b"), ("sweep/gen_het23_sweeps.py", "b"),
                ("analysis/catalog_daniel/dancat_het_pre.py", "b"), ("scripts/launch_het_delta.sh", "b")),
 "acb38e238": F(("scripts/job_catalog_norm.sh", "b"), ("data_mmap.py", "b"), ("experiment.py", "b"),
                ("docs/results.tex", "b"), ("base_experiment.py", "b")),
 "a95715bd8": F(("experiment.py", "b"), ("experiment.py", "b"), ("recipes/gen_catalog_2to4.py", "s"), ("experiment.py", "s")),
 "a31eb18ec": F(("models/diagram_encoder_llm4lhc.py", "b"), ("models/diagram_encoder_llm4lhc.py", "b"),
                ("scripts/job_catalog_norm.sh", "b"), ("experiment.py", "b"), ("experiment.py", "b"), ("experiment.py", "b"),
                ("recipes/gen_catalog_2to4.py", "s")),
 "a6e2bf0a1": F(("sweep/gen_finale_het_sweeps.py", "b"), ("sweep/gen_finale_het_sweeps.py", "b"), ("scripts/job_het_*.sh", "b")),
 "a073c0f55": F(("tools/prune_copies.py", "b"), ("analysis/transfer/het_mumse22_summary.py", "b"), ("tools/prune_copies.py", "b")),
 "acfd7e951": F(("tools/item_checks.py", "b"), ("experiment.py", "b"), ("tools/write_own_alphas.py", "b"),
                ("recipes/catalog_*order*.yaml", "s"), ("scripts/job_loop_probes_prep.sh", "s")),
 "a42f13239": F(("recipes/catalog_loop_signed.yaml", "s"), ("recipes/catalog_loop_signed.yaml", "b"), ("tools/zero_shot_parent.py", "b")),
 "a3ef68c2d": F(("tools/item_checks.py", "s"), ("analysis/catalog_daniel/level_range.py", "b"), ("data_mmap.py", "b"),
                ("recipes/gen_catalog_v3.py", "b")),
 "ace557ba2": F(("recipes/select_catalog_v3.py", "b"), ("recipes/gen_catalog_v3.py", "s")),
 "acf8c4fac": F(("recipes/select_catalog_v3.py", "b"), ("mg5_pipeline_final.py", "b"), ("mg5_pipeline_final.py", "b"),
                ("mg5_pipeline_final.py", "b"), ("tests/test_beam_order.py", "b"), ("analysis/catalog_v3/pool_precision.py", "b"),
                (".claude/hooks/continuous_list_guard.py", "b")),
 "a885dac3f": F(("data_mmap.py", "b"), ("analysis/catalog_daniel/shape_learn.py", "b")),
 "a01601dd5": F(("recipes/gen_catalog_v3.py", "s"), ("recipes/gen_catalog_v3.py", "b"), ("recipes/catalog_v3_loops.yaml", "b")),
 "abf3fa721": F(("recipes/select_catalog_v3.py", "s"), ("tools/nlo_pole_check.py", "b")),
 "aac7c1218": F(("recipes/catalog_v3_gen_stage2.yaml", "s"), ("tools/nlo_virtual_pipeline.py", "b"),
                ("tools/nlo_virtual_pipeline.py", "b"), ("tools/nlo_virtual_pipeline.py", "b")),
 "a49b5bd73": F(("tools/nlo_pole_check.py", "pre")),
 "acb2a5366": F(("analysis/transfer/nogain_fig.py", "b"), ("analysis/transfer/expect_tp3.json", "b"),
                ("sweep/sweep_config_tp3_*ftelo*.yaml", "s")),
 "a65d9c470": F(("wrappers.py", "b"), ("tools/kshot_context.py", "b")),
 "ab9d9d6e0": F(("recipes/make_catalog_v3_recipes.py", "b"), ("recipes/make_catalog_v3_recipes.py", "b"),
                ("recipes/make_catalog_v3_recipes.py", "b"), ("recipes/make_catalog_v3_recipes.py", "s"),
                ("recipes/make_catalog_v3_recipes.py", "b"), ("scripts/job_catalog_v3_short.sh", "b"),
                ("docs/results.tex", "b"), ("analysis/catalog_daniel/kshot_context_cmp.py", "b")),
 "a710a3cdd": F(("recipes/make_catalog_v3_recipes.py", "s"), ("recipes/make_catalog_v3_recipes.py", "b"),
                ("mg5_pipeline_final.py", "b"), ("scripts/job_catalog_v3_short.sh", "b"), ("scripts/job_catalog_v3_short.sh", "b")),
 "a8c6e01f8": F(("recipes/gen_catalog_v3_pp.py", "s"), ("recipes/gen_catalog_v3_pp.py", "b"), ("mg5_pipeline_final.py", "b"),
                ("tools/factorised_2to4.py", "b"), ("analysis/catalog_v3/short_reads.py", "b")),
 "ab95446ad": F(("scripts/job_factorised_2to5_net_v3.sh", "s"), ("analysis/catalog_v3/kshot_context.py", "b")),
 "a3690edb5": F(*[("tools/plan.py", "b")] * 4, (".claude/hooks/plan_check.py", "b"), (".claude/hooks/plan_check.py", "b"),
                *[("tools/plan.py", "b")] * 3, ("plan/migrate.py", "b"), ("sweep/gen_lowd_*.py", "s")),
 "a50158cd1": F((".claude/hooks/claims_guard.py", "b"), ("tools/verify_claims.py", "b"), *[("tools/plan.py", "b")] * 3,
                ("sweep/gen_lowd_*.py", "s")),
 "a44fb18c2": F(("tools/plan.py", "b"), ("tools/plan.py", "b"), (".githooks/pre-commit", "b"), ("tools/verify_claims.py", "b"),
                ("tools/plan.py", "b"), ("sweep/gen_lowd_dyhpo.py", "b"), ("analysis/transfer/expect_tp3.json", "b"),
                ("tools/plan.py", "b"), ("tools/plan.py", "s")),
 "aa6fc3446": F(("tools/verify_claims.py", "b"), ("tools/plan.py", "b"), ("tools/plan.py", "b"),
                (".claude/hooks/git_verify_guard.py", "b"), ("tools/switch_over.sh", "b")),
 "ad1f0034c": F(("tools/plan.py", "b"), (".claude/hooks/plan_check.py", "b"), ("tools/plan.py", "b")),
 "ad795b8d4": F(("tools/plan.py", "b"), ("tools/verify_claims.py", "b")),
 "a1495e04b": F(("sweep/gen_transfer_pilot.py", "b")),
 "ae502ad7f": F(("analysis/catalog_v3/pp_compare.py", "b"), ("analysis/catalog_v3/prep_level.py", "b"),
                ("analysis/catalog_v3/prep_level.py", "b")),
 "a40c942e8": F(("analysis/catalog_v3/coverage_check.py", "b"), ("analysis/catalog_v3/coverage_check.py", "b"),
                ("experiment.py", "b"), ("experiment.py", "b"), (".claude/hooks/review_backlog.sh", "b"),
                ("tools/verify_claims.py", "b"), ("mg5_pipeline_final.py", "b"), ("recipes/gen_catalog_v4.py", "s")),
 "ae360bb92": F(("analysis/storage/free_space.py", "b")),
 "a1f6bef38": F(("tools/plan.py", "b"), ("tests/test_setup_scenarios.py", "b"), (".claude/hooks/review_backlog.sh", "b")),
 "a483146da": F(("tests/test_setup_scenarios.py", "b"), ("tools/plan.py", "b")),
 "a76c9cfa0": F(("tools/plan.py", "b"), ("prebuild_recipes.py", "b")),
 "a1fa13a24": F(("mg5_pipeline_final.py", "b"), ("analysis/catalog_v3/prep_sampling_pools.py", "b"),
                ("analysis/catalog_v3/prep_sampling_pools.py", "b"), ("analysis/catalog_v3/prep_signed*.py", "b")),
 "a46a7ebc4": F(("tools/mixed_order_virt_check.py", "b")),
 "a01f2ee05": F(("analysis/transfer/nogain_readout_za.py", "b"), ("tools/pole_guard_diag.py", "b")),
 "a96789955": F(("scripts/job_catalog_v3_short.sh", "b"), ("wrappers.py", "b"), ("wrappers.py", "b")),
 "ab79ab9cb": F(*[("analysis/transfer/ema_start_audit.py", "b")] * 6),
 "a0e3f283e": F(("analysis/catalog_v3/muon_compare.py", "b")),
 "a48d20fe6": F(("tests/test_per_process_accumulation.py", "b")),
 "a7b008ece": F(("analysis/catalog_v4/validate_v4.py", "b")),
 "a7067849c": F(("analysis/catalog_v3/pair_head_blowup.py", "b"), ("analysis/catalog_v3/pair_head_blowup.py", "b")),
 "ae234e93d": F(("experiment.py", "b")),
 "ac2e48370": F(("scripts/job_catalog_v3_short.sh", "s"), ("base_experiment.py", "b"),
                ("sweep/sweep_config_ccin2p3_sign_head_search.yaml", "b"), ("sweep/sweep_config_ccin2p3_sign_head_search.yaml", "b")),
 "a4ddbbde7": F(("experiment.py", "b"), ("experiment.py", "s"), ("experiment.py", "b"), ("analysis/catalog_v3/muon_hp_ft.py", "b")),
 "aac268be6": F(*[("analysis/catalog_v4/compare_v3.py", "b")] * 3),
 "a7c8e72a9": F(("analysis/catalog_v4/compare_pp_trained.py", "b"), ("analysis/catalog_v4/compare_pp_trained.py", "b")),
 "ad8653270": F(("tools/plan.py", "b")),
 "a31553fb5": F(*[("analysis/catalog_v4/density_check.py", "b")] * 3),
 "a1ed4d0d6": F((".claude/hooks/send_check.py", "b"), ("talks/progress_oct2026b/make_figs.py", "b")),
 "a3d7629c2": F((".claude/hooks/send_check.py", "b")),
}

TOOLING = re.compile(r"^(\.claude/|\.githooks/|plan/|tools/(plan|verify_claims|item_checks|switch_over|agent_costs)|tests/test_setup)")
GENERATION = re.compile(r"^(mg5_pipeline_final\.py|prebuild_recipes\.py|recipes/|tools/(nlo_|synthetic_|steer_pool|write_own_alphas|"
                        r"width_regen|pole_|multiplicity_extrap|dump_diagrams|equal_processes)|tests/test_beam_order)")
TRAINING = re.compile(r"^(experiment\.py|base_experiment\.py|models/|wrappers\.py|dataset\.py|data_mmap\.py|particle_ids\.py|"
                      r"run\.py|config/|optim/|tests/)")
JOBS = re.compile(r"^(scripts/|sweep/|sites/|siteconf\.py|tools/prune_copies)")
ANALYSIS = re.compile(r"^(analysis/|tools/|attribution/|talks/)")
CONFIGS = re.compile(r"^(sweep|recipes)/.*\.ya?ml$")   # sweep configs and recipes, written in bulk by generators
AREAS = ["tooling", "analysis", "training", "jobs", "generation", "configs", "other"]


def area(p):
    for name, rx in (("configs", CONFIGS), ("tooling", TOOLING), ("generation", GENERATION), ("training", TRAINING), ("jobs", JOBS),
                     ("analysis", ANALYSIS)):
        if rx.match(p):
            return name
    return "other"


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, check=True).stdout


def runs():
    """Every repo-reviewer run of every session: id, launch time, last report time, watermark moves (time, sha)."""
    out = []
    for main in sorted(glob.glob(PROJ + "/*.jsonl")):
        sid = os.path.basename(main)[:-6]
        launch, ids = {}, {}
        for f in [main] + sorted(glob.glob(os.path.join(PROJ, sid, "subagents", "agent-*.jsonl"))):
            for line in open(f):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                c = (r.get("message") or {}).get("content")
                for x in c if isinstance(c, list) else []:
                    if not isinstance(x, dict):
                        continue
                    if x.get("type") == "tool_use" and x.get("name") == "Agent":
                        launch[x["id"]] = ((x.get("input") or {}).get("subagent_type"), r.get("timestamp"))
                    elif x.get("type") == "tool_result" and x.get("tool_use_id") in launch:
                        m = re.search(r"agentId: (\w+)", json.dumps(x.get("content")))
                        if m:
                            ids.setdefault(m.group(1), launch[x["tool_use_id"]])
        for aid, (typ, ts) in ids.items():
            f = os.path.join(PROJ, sid, "subagents", "agent-%s.jsonl" % aid)
            if typ != "repo-reviewer" or not os.path.exists(f) or ts >= CUTOFF:
                continue
            adv, end, verdict = [], ts, None
            for line in open(f):
                r = json.loads(line)
                c = (r.get("message") or {}).get("content")
                for x in c if isinstance(c, list) else []:
                    if not isinstance(x, dict):
                        continue
                    if x.get("type") == "tool_result":
                        s = json.dumps(x.get("content"))
                        adv += [(r["timestamp"], m.group(1)) for m in re.finditer(r"watermark advanced to ([0-9a-f]{7,40})", s)]
                    rep = (x.get("input") or {}).get("message") if x.get("name") == "SubagentHandback" else \
                        (x.get("text") if x.get("type") == "text" and r.get("type") == "assistant" else None)
                    if rep and len(rep) > 150:
                        end = r["timestamp"]
                        if verdict is None:
                            verdict = "fix first" if re.search(r"(?i)fix first", rep[:600]) else "pass"
            out.append(dict(session=sid, aid=aid[:9], start=ts, end=end, advances=adv, verdict=verdict))
    return sorted(out, key=lambda r: r["start"])


def poisson68(n):
    from scipy.stats import chi2
    return (chi2.ppf(0.16, 2 * n) / 2 if n else 0.0), chi2.ppf(0.84, 2 * n + 2) / 2


def boot68(ns, ls, n_boot=4000, seed=0):
    """68% interval of sum(ns)/sum(ls) from resampling reviews (findings cluster within a review, so a Poisson
    interval per line is too narrow: the per-review dispersion of the counts is several times Poisson's)."""
    import numpy as np
    ns, ls = np.asarray(ns, float), np.asarray(ls, float)
    idx = np.random.default_rng(seed).integers(0, len(ns), size=(n_boot, len(ns)))
    tot = ls[idx].sum(1)
    r = ns[idx].sum(1)[tot > 0] / tot[tot > 0]
    return [float(np.percentile(r, 16)), float(np.percentile(r, 84))]


def rate(ns, ls, cs=None):
    """ns, ls, cs: per-review findings, lines and commits."""
    n, lines = sum(ns), sum(ls)
    d = dict(n=n, lines=lines, per_kline=1e3 * n / lines if lines else None)
    if lines:
        d["per_kline_68"] = [1e3 * x for x in boot68(ns, ls)]
        d["per_kline_poisson68"] = [1e3 * x / lines for x in poisson68(n)]
    if cs:
        d.update(commits=sum(cs), per_commit=n / sum(cs), per_commit_68=boot68(ns, cs))
    return d


def main(argv):
    R = runs()
    fp = git("rev-list", "--first-parent", "--reverse", "trunk").split()
    pos = {s: i for i, s in enumerate(fp)}

    def on_trunk(sha):                    # first first-parent commit that contains sha
        full = git("rev-parse", sha).strip()
        if full in pos:
            return pos[full]
        lo, hi = 0, len(fp) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            ok = subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", full, fp[mid]]).returncode == 0
            lo, hi = (lo, mid) if ok else (mid + 1, hi)
        return lo

    def head_at(ts):
        return pos[git("rev-list", "-1", "--first-parent", "--before=" + ts, "trunk").strip()]

    advs = sorted((t, on_trunk(s), r["aid"]) for r in R for t, s in r["advances"])
    cuts, last = [], on_trunk(START)
    advs = [x for x in advs if x[0] < CUTOFF]
    for t, i, aid in advs:                # a watermark move that does not move trunk forward starts no new interval
        if i > last:
            if cuts and cuts[-1][2] == aid:   # one run moving the watermark several times: one episode
                cuts.pop()
            cuts.append((t, i, aid))
            last = i
    bounds = [on_trunk(START)] + [i for _, i, _ in cuts]
    ep_of = {}
    for r in R:
        if r["aid"] in SKIP:
            continue
        ep_of[r["aid"]] = next((n for n, (t, _, a) in enumerate(cuts) if t >= r["end"] or (a == r["aid"] and t >= r["start"])), None)
    episodes = []
    for k, (t, i, aid) in enumerate(cuts):
        a, b = fp[bounds[k]], fp[i]
        members = [r for r in R if ep_of.get(r["aid"]) == k]
        files = {}
        for line in git("diff", "--numstat", "--no-renames", a, b, "--", *PATHS).splitlines():
            ad, de, p = line.split("\t")
            files[p] = (0 if ad == "-" else int(ad)) + (0 if de == "-" else int(de))
        status = {l.split("\t")[1]: l.split("\t")[0] for l in
                  git("diff", "--name-status", "--no-renames", a, b, "--", *PATHS).splitlines()}
        first = members[0] if members else None
        h0 = max(head_at(first["start"]), bounds[k]) if first else i
        tl = sum(int(x) for l in git("diff", "--numstat", "--no-renames", a, fp[h0], "--", *PATHS).splitlines()
                 for x in l.split("\t")[:2] if x != "-")
        tc = int(git("rev-list", "--count", "%s..%s" % (a, fp[h0]), "--", *PATHS))
        ch = int(git("rev-list", "--count", "%s..%s" % (a, b), "--", *PATHS, *NO_CONFIGS))
        fired = "both" if tl >= LINES_T and tc >= COMMITS_T else "lines" if tl >= LINES_T else \
            "commits" if tc >= COMMITS_T else "neither"
        episodes.append(dict(start=a[:10], end=b[:10], runs=[m["aid"] for m in members],
                             first_verdict=first["verdict"] if first else None, lines=sum(files.values()),
                             commits=int(git("rev-list", "--count", "%s..%s" % (a, b), "--", *PATHS)), n_files=len(files),
                             commits_hand=ch, trigger_lines=tl, trigger_commits=tc, fired=fired, _files=files, _status=status, findings=[]))
    tracked = git("ls-files", "--", *PATHS).split()
    for aid, fl in FINDINGS.items():
        k = ep_of.get(aid)
        assert k is not None, aid
        for path, tag in fl:
            ep, kind = episodes[k], "none"
            # the review's own diff, then the next one's (the fix committed after the review); later ones only for the
            # reviews whose reports name worktree paths (code merged to trunk later)
            for j in (k, k + 1) + ((k + 2, k + 3) if aid in WORKTREE else ()):
                st = episodes[j]["_status"] if 0 <= j < len(episodes) else {}
                hit = {v[0] for p, v in st.items() if fnmatch.fnmatch(p, path)}
                if hit:
                    kind = "new file" if "A" in hit else "edit" if "M" in hit else "deletion"
                    break
            if tag in ("b", "s") and kind == "none":
                # outside the code pathspec (its lines are never counted), or code the backlog did not change
                tag = "pre" if any(fnmatch.fnmatch(p, path) for p in tracked) else "outside"
            ep["findings"].append(dict(reviewer=aid, file=path, tag=tag, area=area(path), kind=kind))
    for ep in episodes:
        ep["n_blocking"] = sum(f["tag"] in ("b", "s") for f in ep["findings"])

    counted = [f for ep in episodes for f in ep["findings"] if f["tag"] in ("b", "s")]
    assert not any(f["kind"] == "none" for f in counted)
    L, C = sum(ep["lines"] for ep in episodes), sum(ep["commits"] for ep in episodes)
    CH = sum(ep["commits_hand"] for ep in episodes)
    al, kl = defaultdict(int), defaultdict(int)
    for ep in episodes:
        for p, n in ep["_files"].items():
            al[area(p)] += n
            if area(p) != "configs":
                kl[{"A": "new file", "D": "deletion"}.get(ep["_status"].get(p, "M")[0], "edit")] += n
        ep["lines_hand"] = sum(n for p, n in ep["_files"].items() if area(p) != "configs")
        ep["_al"], ep["_kl"] = defaultdict(int), defaultdict(int)
        for p, n in ep["_files"].items():
            ep["_al"][area(p)] += n
            if area(p) != "configs":
                ep["_kl"][{"A": "new file", "D": "deletion"}.get(ep["_status"].get(p, "M")[0], "edit")] += n

    def nf(pred):
        return [sum(1 for f in e["findings"] if f["tag"] in ("b", "s") and pred(f)) for e in episodes]
    hand = [f for f in counted if f["area"] != "configs"]
    LH = sum(al[a] for a in AREAS if a != "configs")
    res = dict(
        n_runs=len(R), n_runs_by_session={s: sum(r["session"] == s for r in R) for s in sorted({r["session"] for r in R})},
        n_episodes=len(episodes),
        fired={f: sum(e["fired"] == f for e in episodes) for f in ("lines", "commits", "both", "neither")},
        episodes_with_blocking=sum(e["n_blocking"] > 0 for e in episodes),
        episodes_with_code_defect=sum(any(f["tag"] == "b" for f in e["findings"]) for e in episodes),
        lines=L, commits=C, median_lines=sorted(e["lines"] for e in episodes)[len(episodes) // 2],
        commits_hand=CH,
        overall=rate(nf(lambda f: True), [e["lines"] for e in episodes], [e["commits"] for e in episodes]),
        overall_code_only=rate(nf(lambda f: f["tag"] == "b"), [e["lines"] for e in episodes], [e["commits"] for e in episodes]),
        hand_written=rate(nf(lambda f: f["area"] != "configs"), [e["lines_hand"] for e in episodes],
                          [e["commits_hand"] for e in episodes]),
        hand_written_code_only=rate(nf(lambda f: f["area"] != "configs" and f["tag"] == "b"),
                                    [e["lines_hand"] for e in episodes]),
        findings_per_episode=len(counted) / len(episodes),
        by_area={a: rate(nf(lambda f, a=a: f["area"] == a), [e["_al"][a] for e in episodes]) for a in AREAS},
        by_kind={k: rate(nf(lambda f, k=k: f["area"] != "configs" and f["kind"] == k), [e["_kl"][k] for e in episodes])
                 for k in ("new file", "edit", "deletion")},
        excluded={t: sum(f["tag"] == t for e in episodes for f in e["findings"]) for t in ("pre", "outside", "x")},
    )
    r1, r2 = res["overall"]["per_kline"] / 1e3, res["hand_written"]["per_kline"] / 1e3
    res["expected_per_review"] = {"hook_lines": {str(n): round(r1 * n, 2) for n in (600, 1500)},
                                  "hand_written_lines": {str(n): round(r2 * n, 2) for n in (600, 1500)}}
    res["median_lines_hand"] = sorted(e["lines_hand"] for e in episodes)[len(episodes) // 2]
    big = [e for e in episodes if e["fired"] != "neither"]
    res["fired_rows"] = dict(n=len(big), with_blocking=sum(e["n_blocking"] > 0 for e in big),
                             rate=rate([e["n_blocking"] for e in big], [e["lines"] for e in big]))
    res["episodes"] = [{k: v for k, v in e.items() if not k.startswith("_")} for e in episodes]
    json.dump(res, open(OUT, "w"), indent=1)
    a, k = res["by_area"], res["by_kind"]
    S = dict(n_runs=res["n_runs"], n_episodes=res["n_episodes"], fired_lines=res["fired"]["lines"] + res["fired"]["both"],
             fired_both=res["fired"]["both"], fired_commits_only=res["fired"]["commits"], fired_neither=res["fired"]["neither"],
             ep_blocking=res["episodes_with_blocking"], n_find=res["overall"]["n"], n_code=res["overall_code_only"]["n"],
             n_scope=res["overall"]["n"] - res["overall_code_only"]["n"], n_hand=res["hand_written"]["n"],
             n_configs=a["configs"]["n"], hand_klines=round(res["hand_written"]["lines"] / 1e3, 1),
             all_klines=round(L / 1e3), configs_klines=round(a["configs"]["lines"] / 1e3), commits=C,
             per_ep=round(res["findings_per_episode"], 2), median_hand=res["median_lines_hand"],
             rate=round(res["hand_written"]["per_kline"], 2), rate_lo=round(res["hand_written"]["per_kline_68"][0], 2),
             rate_hi=round(res["hand_written"]["per_kline_68"][1], 2),
             rate_code=round(res["hand_written_code_only"]["per_kline"], 2),
             per_commit=round(res["hand_written"]["per_commit"], 3), rate_hook=round(res["overall"]["per_kline"], 2),
             rate_configs=round(a["configs"]["per_kline"], 3),
             exp600=res["expected_per_review"]["hand_written_lines"]["600"],
             exp1500=res["expected_per_review"]["hand_written_lines"]["1500"],
             exp600_hook=res["expected_per_review"]["hook_lines"]["600"],
             exp1500_hook=res["expected_per_review"]["hook_lines"]["1500"], n_pre=res["excluded"]["pre"],
             n_outside=res["excluded"]["outside"],
             tooling_top_review=max(sum(f["tag"] in ("b", "s") and f["area"] == "tooling" for f in e["findings"]) for e in episodes), commits_hand=CH, rate_p_lo=round(res["hand_written"]["per_kline_poisson68"][0], 2),
             rate_p_hi=round(res["hand_written"]["per_kline_poisson68"][1], 2),
             **{"%s_%s" % (ar, q): round(v, 2) for ar in ("tooling", "analysis", "training", "jobs", "generation")
                for q, v in (("n", a[ar]["n"]), ("klines", a[ar]["lines"] / 1e3), ("rate", a[ar]["per_kline"]),
                             ("lo", a[ar]["per_kline_68"][0]), ("hi", a[ar]["per_kline_68"][1]))},
             **{"%s_%s" % (kk, q): round(v, 2) for kk, name in (("new", "new file"), ("edit", "edit"))
                for q, v in (("n", k[name]["n"]), ("klines", k[name]["lines"] / 1e3), ("rate", k[name]["per_kline"]),
                             ("lo", k[name]["per_kline_68"][0]), ("hi", k[name]["per_kline_68"][1]))})
    res["summary"] = S
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(S))
    if "--fig" in argv:
        figure(res)


def figure(res):
    sys.path.insert(0, REPO)
    import plot_style as ps
    fig, ax = ps.figure()
    items = [(a, v) for a, v in res["by_area"].items() if a not in ("other", "configs") and v["lines"]]
    for j, (a, v) in enumerate(items):
        lo, hi = v["per_kline_68"]
        ax.errorbar([j], [v["per_kline"]], yerr=[[v["per_kline"] - lo], [hi - v["per_kline"]]], fmt="o", color=ps.C.blue,
                    label="area" if j == 0 else None)
    ax.axhline(res["hand_written"]["per_kline"], color=ps.C.vermillion, ls="--", label="all hand-written code")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([{"training": "train", "generation": "gen"}.get(a, a) for a, _ in items])
    ax.set_ylabel(r"findings / $10^3$ lines")
    ps.legend(ax, "upper left")
    ps.save(fig, os.path.join(REPO, "analysis/review_rates/figs/review_rates_by_area"))


if __name__ == "__main__":
    main(sys.argv)
