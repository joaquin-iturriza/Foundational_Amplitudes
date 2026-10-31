#!/usr/bin/env python3
"""The matched model trial (agent_redesign step 4, D203): the same small tool-fix tasks run by Sonnet 5.5 and by
Opus 5.5 at low effort, each task's pass check its own, every output read by hand.

    python3 tools/model_trial.py validate [TASK ...]      each task's check fails at the parent and passes at the fix
    python3 tools/model_trial.py setup TASK ARM           worktree worktrees/zz_trial_<task>_<arm> at the fix's parent
    python3 tools/model_trial.py brief TASK ARM           the task text given to the agent (identical across arms)
    python3 tools/model_trial.py check TASK ARM           the task's pass check on the agent's committed work
    python3 tools/model_trial.py cost AGENT_ID [...]      $ of one agent's transcript at list price (usage_windows)
    python3 tools/model_trial.py report --out JSON        per arm: tasks passed, $ per task, $ per task with rework
    python3 tools/model_trial.py size [--out JSON]        the extension's sizing from the first ten pairs
    python3 tools/model_trial.py altcheck                 the hand-written alternative fixes (other internals than the
                                                          original's) pass their checks: the check accepts another fix

A task is a past fix of this repo (commit C) with its own test: the agent gets a worktree at C^ (the defect present,
C's test not in it) and a brief that states the defect as it was seen, never the fix. Pass check: C's test files
copied over the agent's work and run in .venv-cpu (every test of those files passes), then restored. Rework: a task
failed on the first attempt is resumed once with the failing check's output; its $ is the sum over both attempts,
and it counts as passed with rework when the second check passes. Records: analysis/agent_redesign/model_trial.json.
"""
import argparse, glob, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv-cpu", "bin", "python")
REC = os.path.join(ROOT, "analysis", "agent_redesign", "model_trial.json")
ARMS = ("sonnet", "opuslow", "sonnetlow")      # sonnetlow: Sonnet 5.5 at effort low (extension, D212)
# the ten tasks whose check fails at the parent and passes at the fix (validate, 2026-10-31); the other candidates
# below failed that test (passed at the parent: emaload, runperm, sizeprobe, ppsidecar, memmask; failed at the fix on
# the laptop: planwait, statusjson; check over 15 min: ggh) and are kept only as the record of what was tried
TRIAL = ["defshare", "ftorders", "poolphys", "skipclaim", "stallshort", "zsrole", "ordsrc", "hybridcp", "loopenc",
         "poolrow"]

# id: (fix commit, its test files, the defect as seen)
TASKS = {
    "runperm": ("cdcae2cd", ["tests/test_run_perm_calibration.py"],
                "short_run_bias.run_perm gives p-values that are too small when the two groups of units have unequal "
                "sizes: on iid events, 66 runs of 1000 events against 18 runs of 555, p came out under 0.05 in 19 "
                "percent of repeats instead of about 5 percent. Its permutation test is not calibrated."),
    "poolphys": ("bb70a04a", ["tests/test_pool_physics.py"],
                 "The pool physics guard (tools/pool_physics.py) refused pp13_dg_adg and pp13_dg_dhz on alpha_s: their "
                 "only training entries are the alpha_s scan points, whose alpha_s differs from the probe's by design. "
                 "An entry that scans a field (an alpha_s point, a heavy-quark mass) should not count as a reference "
                 "for that field."),
    "planwait": ("9dad7d3a", ["tests/test_plan_status.py"],
                 "tools/plan.py status shows an entry as STALLED when its step waits on something that is not a run "
                 "(a wait that names no run id), so the Stop gate refuses it; such an entry is waiting, not stalled."),
    "sizeprobe": ("b08f0c73", ["tests/test_lhe_parts.py"],
                  "recipes/gen_catalog_v4_default.py: a size_from_probe pool whose probe data lack one energy's "
                  "complete share is silently sized at 0.8 for that energy instead of being refused."),
    "skipclaim": ("c9b5c8f0", ["tests/test_lhe_parts.py"],
                  "tools/lhe_parts.py plan --skip-claimed Q: the new plan adopted none of the parts already made in "
                  "the queues it takes over (catalog_v4_w2's first plan linked zero existing parts to the new layout's "
                  "names), so finished runs would be generated again."),
    "stallshort": ("53c80201", ["tests/test_lhe_parts.py"],
                   "tools/lhe_parts.py: 11 of pp13v4_ud_dggu's 32 runs were failed as short, but that pool uses the "
                   "stall layout, whose runs each ask the whole pool and return a few thousand events by design "
                   "(lhe.n_runs is sized from that). Short parts should fail a task only where the run is meant to "
                   "deliver its events."),
    "defshare": ("91a639fe", ["tests/test_default_share.py"],
                 "With sampling.f_default = 0.5, building a chunked pool of [334, 333, 333] events overshoots by one "
                 "event: the shares (default, forward, bulk) do not sum to each chunk's count exactly."),
    "ftorders": ("6e0eece0", ["tests/test_orders_source.py"],
                 "tools/catalog_ft.py: a fine-tune's data.orders_from_diagrams does not follow its parent's "
                 "coupling-monomial source, so a parent trained with orders read one way is fine-tuned with them read "
                 "the other way; a parent whose data stats record only alpha_s orders should be refused."),
    "emaload": ("cff8039a", ["tests/test_ema_checkpoint.py"],
                "Fine-tuning from an EMA checkpoint fails on real runs (FA-ccin2p3-072f43d3: "
                "tp3_synfte_ee_bb_nlo_d8_lad, tp3_finfte_uubar_Zg_d8 trial_0031): the parent's state dict carries "
                "entries that are neither parameters nor buffers of the model, and the EMA shadow cannot be mapped "
                "past the buffers. An unmappable shadow should be refused with a clear error, a mappable one loaded."),
    "ggh": ("ce0e8873", ["tests/test_synthetic_versions.py"],
            "The synthetic amplitude generator's legal_diagram_count undercounts processes whose last leg is the H "
            "(e.g. u ubar -> g Z H): the effective ggH vertex is missed at the root of its recursion."),
    "zsrole": ("9702a254", ["tests/test_zero_shot_split.py"],
               "tools/zero_shot_parent.py fails on a probe that has no pool of some role (the EE4 probes, "
               "catalog_v4_ee_test.yaml: n_train 0, n_val 0); it should read that role from a role the probe has."),
    "ordsrc": ("77b43ae3", ["tests/test_orders_source.py"],
               "With the dimensionless target, the coupling monomial ignores data.orders_from_diagrams (it is always "
               "taken one way), and the frozen data stats do not say where it came from, so a run reloaded with the "
               "other setting silently uses a different target. With orders_from_diagrams false the table's target "
               "should be used (a mixed-order entry refused), and a reload under another source refused."),
    "hybridcp": ("6b387fbb", ["tests/test_hybrid_cp_frozen.py"],
                 "target_physics_hybrid: a fine-tune onto a process its parent already trained on refits that "
                 "process's c_p instead of reading the parent's, so the fine-tune's target differs from the parent's "
                 "for the same process even though the frozen data stats carry it."),
    "ppsidecar": ("a2bd74ae", ["tests/test_orders_source.py"],
                  "E3058: the smoke run refused pp13v4_udx_adxgu as a mixed-order entry with no diagram sidecar, but "
                  "its sidecar is present: the lhe_pp pools (pp13_, pp13v4_, ee4_, dc_, dsh_, dE100_) do not read "
                  "their coupling powers from their sidecars the way their trees, features and target do."),
    "memmask": ("9087df0f", ["tests/test_memmap_data.py"],
                "With data.memmap on, the propagator masks come out wrong or fail to build: they need the slot PDG "
                "ids, which are not loaded yet when the masks are built."),
    "loopenc": ("ebeb0954", ["tests/test_diagram_encoder_llm4lhc.py"],
                "The llm4lhc diagram encoder stops on PDG 82 from rung 9's one-loop processes (the GPU smoke run "
                "died there): MadGraph's one-loop ghosts and Goldstones (82, 9000001-9000004, 250, 251) have no "
                "label, so loop processes cannot be encoded."),
    "statusjson": ("9a0d29ac", ["tests/test_plan_status.py"],
                   "tools/plan.py status has no machine-readable output: the supervisor needs `plan status --json` "
                   "(per entry its state MOVING / WAITING / STALLED and the reason) to raise its STALLED check."),
    "poolrow": ("b5173fa7", ["tests/test_dipole_pool_rows.py"],
                "The dipole read cannot be tied back to the pool: a zero-shot read's events are in the split's order "
                "(the role's shuffle), and nothing maps a read position to its row in the pool file, so per-event "
                "quantities computed from the pool (the dipole variables) cannot be joined to the read's errors. "
                "Zero-shot reads should save each event's pool row."),
    # extension candidates (D212, 2026-10-31): defect text and interface written for those that pass validate
    "handledruns": ("e5cd7bb1", ['tests/test_plan_handled_runs.py'], ""),
    "modelguard": ("0eff43b9", ['tests/test_redesign_d.py'], ""),
    "replygrant": ("49fd91b4", ['tests/test_reply_check.py'], ""),
    "replyexc": ("77261ced", ['tests/test_reply_check.py'], ""),
    "replyneg": ("f49724fd", ['tests/test_reply_check.py'], ""),
    "gatecount": ("9cb31aa5", ['tests/test_hooks_real.py'], ""),
    "loopread": ("5380dd62", ['tests/test_hooks_real.py'], ""),
    "claimdead": ("4609381f", ['tests/test_lhe_parts.py'], ""),
    "repeatfp": ("2a16b36f", ['tests/test_hooks_real.py'], ""),
    "letthrough": ("382a662f", ['tests/test_hooks_real.py'], ""),
    "refusestand": ("c868f490", ['tests/test_hooks_real.py'], ""),
    "sharesreg": ("e887b556", ['tests/test_lhe_parts.py'], ""),
    "continuous": ("948c59d5", ['tests/test_setup_scenarios.py'], ""),
    "donecont": ("9d75f5e3", ['tests/test_setup_scenarios.py'], ""),
    "proposed": ("80367b7d", ['tests/test_setup_scenarios.py'], ""),
    "compactskip": ("e3698073", ['tests/test_setup_scenarios.py'], ""),
    "statssrc": ("5cd8c2b3", ['tests/test_orders_source.py'], ""),
    "alphapow": ("1d9d6a29", ['tests/test_order_features.py'], ""),
    "valtotal": ("577164a6", ['tests/test_val_total.py'], ""),
    "replylist": ("4fe75ce5", ['tests/test_setup_scenarios.py'], ""),
    "replyblock": ("85eeb4f0", ['tests/test_setup_scenarios.py'], ""),
    "quietstop": ("e830f4ca", ['tests/test_setup_scenarios.py'], ""),
    "emastart": ("ddfc0314", ['tests/test_ft_ema_start.py'], ""),
    "endedrun": ("39886d3a", ['tests/test_setup_scenarios.py'], ""),
    "eventblock": ("e7cddfce", ['tests/test_setup_scenarios.py'], ""),
    "inflight": ("ced2a62f", ['tests/test_setup_scenarios.py'], ""),
    "backlogdepth": ("240b0f14", ['tests/test_setup_scenarios.py'], ""),
    "recheck": ("41faddd6", ['tests/test_setup_scenarios.py'], ""),
    "holditem": ("22fbd7b9", ['tests/test_setup_scenarios.py'], ""),
    "claimtime": ("ee42732d", ['tests/test_setup_scenarios.py'], ""),
    "scanwindow": ("324510e2", ['tests/test_beam_order.py'], ""),
    "massconv": ("27c5ffc8", ['tests/test_beam_order.py'], ""),
    "lightprop": ("d406f965", ['tests/test_synthetic_amplitudes.py'], ""),
    "quotedcut": ("23e149a8", ['tests/test_redesign_d.py'], ""),
    "topsampling": ("4e844204", ['tests/test_pool_physics.py'], ""),
    "switchB": ("69c9b567", ['tests/test_hooks_real.py'], ""),
    "twoword": ("5bf79e0d", ['tests/test_hooks_real.py'], ""),
    "acceptyield": ("b2050260", ['tests/test_lhe_parts.py', 'tests/test_supervisor_c.py'], ""),
    "reviewlock": ("275845c9", ['tests/test_setup_scenarios.py'], ""),
    "asksearch": ("9ce1cb16", ['tests/test_setup_scenarios.py'], ""),
}

# source-level assertions of a check (they read the original fix's exact lines): turned into `pass` in the copied
# test; what they asked for (the wiring) is judged in the hand read instead (design review FIX, 2026-10-31)
STRIP = {
    "zsrole": ['assert "datagen.ensure_split_set = stand_in_split_set(" in zs'],
    "ordsrc": ['assert re.search(r"self\\._check_frozen_orders_source\\(stats\\)", body)'],
    "hybridcp": ['assert re.search(r\'_cp_frozen\\s*=', 'assert re.search(r"hybrid_cps\\(names,',
                 'assert \'"target_physics_hybrid_cp": getattr(self, "_phys_cp", None)\' in src'],
}

# what each task's check calls, written from the check (never from the fix): names, signatures, returns;
# given in the brief so a correct fix is not failed on a name
INTERFACE = {
    "defshare": "mg5_pipeline_final.default_share_rows(f_default, counts) gives each chunk's (start, stop) of the "
                "default rows, and build_mixture_dataset(...) with a chunk's default rows must return exactly that "
                "chunk's count; the chunks together read the default rows [0, round(f_default * N)) exactly once.",
    "ftorders": "tools/catalog_ft.py: parent_orders_flag(parent_dir) -> (orders_from_diagrams flag for the fine-tune, "
                "the parent's source, how it was read): the source from the parent's data_stats.json key "
                "target_physics_orders_source, else AmplitudeExperiment._infer_orders_source(parent_dir); source "
                "'table' -> False, 'diagrams_highest_alphas' -> True, any other (an alpha_s-only parent) raises "
                "ValueError. command(..., orders_from_diagrams=flag) puts ++data.orders_from_diagrams=true|false on "
                "the run.py command line.",
    "poolphys": "tools/pool_physics.check_probe(...) on such a probe returns status 'ok' and lists the skipped field "
                "(e.g. 'alphas_mz') in its result's not_compared.",
    "skipclaim": "tools/lhe_parts.cmd_plan(args) with args.skip_claimed = [queue dirs]: parts already made under a "
                 "taken-over queue's layout (mg5_pipeline_final.lhe_part_paths) are linked to the new layout's part "
                 "paths and not put in the new queue's todo/.",
    "stallshort": "the stall layout is a pool registered with lhe run_events None; through the queue (cmd_plan, "
                  "the worker, assembly) its short parts are done, not failed, and the pool assembles from their sum.",
    "zsrole": "tools/rebuild_run.py: recipe_role_counts(recipe_yaml) -> {process: {'train': n, 'val': n, 'test': n}}; "
              "stand_in_role(counts, process, role, scored_split) -> (role read, n events): its own role when it has "
              "events, else the scored split first, else another role it has, else its own (0); "
              "stand_in_split_set(real_ensure_split_set, counts, scored_split) -> a function with real's signature "
              "(specs, role, seed, dest_dir=None, require_cache=False) that reads each spec from its stand-in role. "
              "tools/zero_shot_parent.py installs it per recipe in place of datagen.ensure_split_set and records "
              "own_stats_role (the role the probe's own statistics came from) in its output.",
    "ordsrc": "experiment.AmplitudeExperiment: with data.orders_from_diagrams false the target's monomial is the "
              "table's (a table-mixed entry refused with an AssertionError saying 'coupling homogeneity is not exact'), "
              "with it true the diagrams' highest alpha_s power; _save_data_stats writes target_physics_orders_source "
              "('diagrams_highest_alphas' or 'table'); _check_frozen_orders_source(stats) raises AssertionError naming "
              "target_physics_orders_source on a mismatch while data.target_physics_orders is on (stats without the "
              "key count as 'table'; orders_from_diagrams unset counts as true), passes when it is off; init_data "
              "calls it on every reload of frozen stats.",
    "hybridcp": "physics_scales.hybrid_cp(P) -> float: the c_p of one process's momenta P (n, N, 4), the formula of "
                "experiment._physics_hybrid_cp as a module function (0.0 for a 2 -> 2), and "
                "physics_scales.hybrid_cps(names, events_of, frozen=None) -> (cps {name: c_p}, reused [names]): a name "
                "in `frozen` keeps that value and its events are not read, any other is fitted with hybrid_cp. "
                "experiment.py wires it: the frozen-stats path reads the stats key target_physics_hybrid_cp and passes "
                "it to the fit as frozen, and _save_data_stats writes the fitted c_p under that key.",
    "loopenc": "models/diagram_encoder_llm4lhc.py: a module-level _LOOP_LINES (the one-loop ghost and Goldstone PDG "
               "ids); the vocabulary is the 30 SM labels (0..29) then one label per _LOOP_LINES entry; "
               "_label(pdg, vocab) maps -250 to vocab[250] and keeps 82 and -82 apart; an unknown PDG (9000099) still "
               "raises KeyError in load_llm4lhc_diagram_set.",
    "poolrow": "analysis/catalog_v4/dipole_residual.pool_rows(z) for a saved read z (z['idx'] positions in the "
               "role-contiguous train|val|test array, z['split'] the role): the pool rows they came from, each role "
               "shuffled by one numpy default_rng(seed=42) drawing rng.permutation(n) for train, val, test in turn as "
               "experiment._init_data does; a saved z['pool_row'] wins.",
}


def git(*a, cwd=ROOT, check=True):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=check).stdout


def wt(task, arm):
    return os.path.join(ROOT, "worktrees", "zz_trial_%s_%s" % (task, arm))


def stripped(text, task):
    """The check's text with the task's source-level assertions turned into `pass` (same indentation)."""
    out = []
    for line in text.splitlines(True):
        if any(line.strip().startswith(x) for x in STRIP.get(task, [])):
            line = line[:len(line) - len(line.lstrip())] + "pass\n"
        out.append(line)
    return "".join(out)


def run_check(path, commit, tests, timeout=900, task=None):
    """(passed, output tail) of `commit`'s test files run over the tree at `path` (source-level assertions of `task`
    stripped); the tree is restored after."""
    saved = {}
    for t in tests:
        p = os.path.join(path, t)
        saved[t] = open(p).read() if os.path.exists(p) else None
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(stripped(git("show", "%s:%s" % (commit, t)), task))
    try:
        ok, out = True, ""
        for t in tests:        # a test file with its own __main__ runner is run by it (some set globals there)
            src = open(os.path.join(path, t)).read()
            cmd = [PY, t] if '__name__ == "__main__"' in src or "__name__ == '__main__'" in src else \
                [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", t]
            r = subprocess.run(cmd, cwd=path, capture_output=True, text=True, timeout=timeout,
                               env=dict(os.environ, CLAUDE_PROJECT_DIR=path, PYTHONDONTWRITEBYTECODE="1"))
            ok, out = ok and r.returncode == 0, out + (r.stdout + r.stderr)[-3000:]
    except subprocess.TimeoutExpired:
        ok, out = False, "timeout after %d s" % timeout
    finally:
        for t, s in saved.items():
            p = os.path.join(path, t)
            if s is None:
                os.unlink(p)
            else:
                open(p, "w").write(s)
    return ok, out


def validate(names, vt_name="zz_trial_validate"):
    vt = os.path.join(ROOT, "worktrees", vt_name)
    if not os.path.isdir(vt):
        git("worktree", "add", "--detach", vt, "HEAD")
    out = {}
    for name in names:
        c, tests, _ = TASKS[name]
        git("checkout", "-q", "--detach", c + "^", cwd=vt)
        before = run_check(vt, c, tests, task=name)
        git("checkout", "-q", "--detach", c, cwd=vt)
        after = run_check(vt, c, tests, task=name)
        out[name] = {"commit": c, "fails_at_parent": not before[0], "passes_at_fix": after[0],
                     "parent_tail": before[1][-600:], "fix_tail": after[1][-600:]}
        print("%-14s %s fails_at_parent=%s passes_at_fix=%s" % (name, c, not before[0], after[0]))
    return out


# hand-written alternative fixes, appended to the file at the fix's parent (analysis/agent_redesign/model_trial_alt/)
ALT = {"poolrow": "analysis/catalog_v4/dipole_residual.py", "hybridcp": "physics_scales.py"}


def altcheck():
    vt = os.path.join(ROOT, "worktrees", "zz_trial_validate")
    if not os.path.isdir(vt):
        git("worktree", "add", "--detach", vt, "HEAD")
    out = {}
    for task, target in ALT.items():
        c, tests, _ = TASKS[task]
        git("checkout", "-q", "--detach", c + "^", cwd=vt)
        with open(os.path.join(vt, target), "a") as f:
            f.write("\n\n" + open(os.path.join(ROOT, "analysis", "agent_redesign", "model_trial_alt", task + ".py")).read())
        try:
            ok, tail = run_check(vt, c, tests, task=task)
        finally:
            git("checkout", "-q", "--", ".", cwd=vt)
        out[task] = {"passed": ok, "tail": tail[-400:]}
        print(task, "alternative fix passes its check:", ok)
    return out


def setup(task, arm):
    c = TASKS[task][0]
    p = wt(task, arm)
    git("worktree", "add", "-b", "zz_trial_%s_%s" % (task, arm), p, c + "^")
    print(p)


def brief(task, arm):
    c, tests, defect = TASKS[task]
    return ("Task (model trial, one small fix). Work only in the git worktree %s (a checkout of this repository at an "
            "older commit); read and edit files there by absolute path, never in ~/work/FA itself, never another "
            "worktree. Do not read git history past the worktree's HEAD (no git log/show/diff of trunk or other "
            "branches): the task is to find the fix, not to look it up. No jobs, no site commands, no plan.py "
            "writes, no reviewers, no other agents.\n\n"
            "The defect, as it was seen:\n%s\n\nWhat the check that will judge the fix calls (keep these names and "
            "signatures exactly):\n%s\n\n"
            "Fix it in the code (the smallest correct change), add or update a test of it if the repository has a "
            "test file for that code, run the tests you touched from inside the worktree with %s <file> when the "
            "file has a __main__ runner, else with that python -m pytest -q <file>, then commit everything you changed on the worktree's branch: git -C %s commit -am "
            "\"<what you changed>\" (git add new files first). End with a short report: the cause, the change, the "
            "test run's result." % (wt(task, arm), defect, INTERFACE[task], PY, wt(task, arm)))


def check(task, arm):
    c, tests, _ = TASKS[task]
    p = wt(task, arm)
    dirty = git("status", "--porcelain", "--untracked-files=no", cwd=p).strip()
    ok, out = run_check(p, c, tests, task=task)
    return {"task": task, "arm": arm, "passed": ok, "uncommitted": bool(dirty),
            "diff_stat": git("diff", "--stat", c + "^", "HEAD", cwd=p)[-1500:], "tail": out[-1500:]}


def cost(agent):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import usage_windows as uw
    fs = glob.glob(os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA/*/subagents/agent-%s.jsonl" % agent))
    seen, usd, models = set(), 0.0, set()
    for f in fs:
        for line in open(f, errors="replace"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            for m, _ in uw._messages(d):
                if m["id"] in seen:
                    continue
                seen.add(m["id"])
                u = m["usage"]
                cc = u.get("cache_creation") or {}
                c = {"model": m.get("model"), "in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0),
                     "cr": u.get("cache_read_input_tokens", 0),
                     "cw5": cc.get("ephemeral_5m_input_tokens", 0), "cw1h": cc.get("ephemeral_1h_input_tokens", 0),
                     "cwu": 0 if cc else u.get("cache_creation_input_tokens", 0)}
                usd += uw.cost(c) or 0
                models.add(m.get("model"))
    return {"agent": agent, "files": len(fs), "calls": len(seen), "usd": round(usd, 4), "models": sorted(models)}


def wilson(k, n, z=1.96):
    """95% Wilson interval of k successes in n."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return [round(c - h, 3), round(c + h, 3)]


def boot(vals, stat, n=10000, seed=0):
    """95% percentile bootstrap interval of stat over tasks (vals: one item per task, resampled together)."""
    import random
    rng = random.Random(seed)
    xs = sorted(stat([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(n))
    return [round(xs[int(0.025 * n)], 3), round(xs[int(0.975 * n) - 1], 3)]


def report(out):
    """Per arm over every task it ran (first ten and extension pooled): pass rates first and with rework (Wilson
    95%), hand-read flaws, $ per task with rework (bootstrap 95% over tasks), and on the tasks both it and opuslow ran
    the ratio of $ totals and the median paired ratio to opuslow (bootstrap 95% over tasks)."""
    rec = json.load(open(REC)) if os.path.exists(REC) else {"runs": []}
    usd = {(r["task"], r["arm"]): sum(cost(a)["usd"] for a in r["agents"]) for r in rec["runs"]}
    per = {}
    for arm in ARMS:
        rs = [r for r in rec["runs"] if r["arm"] == arm]
        n = len(rs)
        if not n:
            continue
        u = [usd[(r["task"], arm)] for r in rs]
        f1, f2 = sum(r["first_pass"] for r in rs), sum(r["final_pass"] for r in rs)
        ok = sum(r.get("hand_read_ok", False) for r in rs)
        d = {"tasks": n, "passed_first": f1, "first_ci": wilson(f1, n), "passed_with_rework": f2,
             "rework_ci": wilson(f2, n), "hand_read_ok": ok, "hand_read_flaws": [r["task"] for r in rs
                                                                                 if not r.get("hand_read_ok")],
             "usd_total": round(sum(u), 3), "usd_per_task": round(sum(u) / n, 3),
             "usd_per_task_ci": boot(u, lambda v: sum(v) / len(v))}
        if arm != "opuslow":
            pairs = [(usd[(t, arm)], usd[(t, "opuslow")]) for t in sorted({r["task"] for r in rs})
                     if (t, "opuslow") in usd and usd[(t, "opuslow")] > 0]
            if pairs:
                med = lambda v: sorted(a / b for a, b in v)[len(v) // 2]
                tot = lambda v: sum(a for a, _ in v) / sum(b for _, b in v)
                d["vs_opuslow"] = {"pairs": len(pairs), "ratio_of_totals": round(tot(pairs), 3),
                                   "ratio_of_totals_ci": boot(pairs, tot), "median_paired_ratio": round(med(pairs), 3),
                                   "median_paired_ratio_ci": boot(pairs, med),
                                   "min": round(min(a / b for a, b in pairs), 3),
                                   "max": round(max(a / b for a, b in pairs), 3)}
                # first-pass discordant pairs: [this arm passed and opuslow failed, this arm failed and opuslow passed]
                d["vs_opuslow"]["discordant_first"] = [
                    sum(1 for r in rs if r["first_pass"] and not fp(rec, r["task"], "opuslow")),
                    sum(1 for r in rs if not r["first_pass"] and fp(rec, r["task"], "opuslow") is True)]
        per[arm] = d
    print(json.dumps(per, indent=1))
    if out:
        rec["summary"] = per
        json.dump(rec, open(out, "w"), indent=1)


def fp(rec, task, arm):
    for r in rec["runs"]:
        if r["task"] == task and r["arm"] == arm:
            return r["first_pass"]
    return None


def size(n_list=(10, 20, 30, 35, 40)):
    """Sizing of the extension from the first ten pairs: the 95% half-width of the ratio of $ totals and of the
    mean paired ratio at n pairs (normal approximation, the first ten's paired spread), and the smallest difference
    in first-pass failure rate a paired McNemar test detects at n (alpha 0.05 two-sided, power 0.8, failures
    independent across arms, reference failure rate 0.1 as in the first ten)."""
    import math, statistics as st
    rec = json.load(open(REC))
    usd = {(r["task"], r["arm"]): sum(cost(a)["usd"] for a in r["agents"]) for r in rec["runs"]}
    ts = sorted({t for t, a in usd if a == "sonnet" and (t, "opuslow") in usd})
    rat = [usd[(t, "sonnet")] / usd[(t, "opuslow")] for t in ts]
    sd = st.stdev(rat)
    out = {"first_pairs": len(ts), "paired_ratio_sd": round(sd, 4), "by_n": {}}
    for n in n_list:
        hw = 1.96 * sd / math.sqrt(n)
        dd = None
        for k in range(1, 90):
            d, p1 = k / 100, 0.1
            p2 = p1 + d
            pd = p1 * (1 - p2) + p2 * (1 - p1)
            need = (1.96 * math.sqrt(pd) + 0.8416 * math.sqrt(max(pd - d * d, 1e-9))) ** 2 / (d * d)
            if need <= n:
                dd = d
                break
        out["by_n"][n] = {"ratio_ci_halfwidth": round(hw, 3), "detectable_failure_rate_difference": dd,
                          "pass_rate_ci_at_0.9": wilson(round(0.9 * n), n)}
    # live check after a switch (agent_tasks.py, tool-fix cell): revert when failures among the next N Sonnet
    # tool-fix tasks reach K; chance of a revert when Sonnet fails at Opus's live rate (0.04: 24/25 stood) and at
    # that rate plus 0.16 (a 0.2 failure rate)
    def tail(k, n, p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
    out["live_revert_rule"] = {"N": 30, "K": 5, "p_revert_at_0.04": round(tail(5, 30, 0.04), 3),
                               "p_revert_at_0.20": round(tail(5, 30, 0.2), 3),
                               "p_revert_at_0.25": round(tail(5, 30, 0.25), 3)}
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["validate", "setup", "brief", "check", "cost", "report", "altcheck", "size"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--vt", default="zz_trial_validate", help="validate: the worktree name (parallel validates)")
    a = ap.parse_args()
    if a.cmd == "validate":
        res = validate(a.args or sorted(TASKS), a.vt)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
    elif a.cmd == "setup":
        setup(*a.args)
    elif a.cmd == "brief":
        print(brief(*a.args))
    elif a.cmd == "check":
        print(json.dumps(check(*a.args), indent=1))
    elif a.cmd == "cost":
        print(json.dumps([cost(x) for x in a.args], indent=1))
    elif a.cmd == "altcheck":
        res = altcheck()
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
    elif a.cmd == "size":
        res = size()
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
    else:
        report(a.out)


if __name__ == "__main__":
    main()
