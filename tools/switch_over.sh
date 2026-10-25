#!/bin/sh
# The switch-over of the setup redesign (D121), in one step, run by the main session in the main checkout on trunk
# once the reviewers pass:   sh tools/switch_over.sh <branch>
#   1. merge <branch> into trunk (--no-ff), after tools/verify_claims.py passes on its change to docs/results.tex
#   2. migrate the hand-off into plan/plan.yaml again (it moved since the branch was built), commit it
#   3. core.hooksPath = .githooks (tools/install_githooks.sh): the claims check on every commit and merge
#   4. register the hooks (tools/switch_settings.py --apply), delete the four retired hook files, commit
#   5. print the CLAUDE.md lines that still describe what was retired, for the CLAUDE.md pass
# Stops at the first failure and says which steps are done. Left to the user: ledger_guard.sh (global, every
# project) still runs `site guard --stop`, whose waiter and supervisor checks plan_check.py now covers for FA.
set -eu
branch=${1:?usage: sh tools/switch_over.sh <branch>}
top=$(git rev-parse --show-toplevel)
cd "$top"
[ "$(git rev-parse --git-dir)" = ".git" ] || { echo "switch_over: run it in the main checkout, not a worktree"; exit 1; }
[ "$(git rev-parse --abbrev-ref HEAD)" = "trunk" ] || { echo "switch_over: not on trunk"; exit 1; }
git diff --quiet && git diff --cached --quiet || { echo "switch_over: the tree has uncommitted changes"; exit 1; }
done_steps=""
trap 'echo "switch_over: stopped; done: ${done_steps:-nothing}"' EXIT
base=$(git merge-base HEAD "$branch")
git show "$branch:tools/verify_claims.py" > /tmp/switch_over_verify.$$.py
python3 /tmp/switch_over_verify.$$.py --root "$top" --rev "$base..$branch"
rm -f /tmp/switch_over_verify.$$.py
git merge --no-ff -m "setup redesign: the plan, its check and guards (D121)" "$branch"
done_steps="merge"
python3 tools/plan_migrate.py --out plan/plan.yaml
git add plan/plan.yaml
git commit -m "plan: migrated from the hand-off at the switch-over (D121)"
done_steps="$done_steps, migration"
sh tools/install_githooks.sh
done_steps="$done_steps, git hooks"
python3 tools/switch_settings.py --apply
git add .claude/settings.json
git rm -q .claude/hooks/plan_guard.py .claude/hooks/continuous_guard.py .claude/hooks/continuous_list_guard.py \
    .claude/hooks/slurm_waiter_guard.sh
git commit -m "hooks: plan_check, plan_files_guard, git_verify_guard registered; the four idle-work guards retired (D121)"
done_steps="$done_steps, hooks"
trap - EXIT
echo "switch_over: done ($done_steps). Still the user's: ledger_guard's idle checks for FA (global hook)."
echo "For the CLAUDE.md pass, the lines that name what was retired or replaced:"
grep -n -e plan_guard -e continuous_guard -e continuous_list -e slurm_waiter_guard -e continuous.json \
    -e '.slurm_monitor_jobs' -e 'background Bash waiter' CLAUDE.md || echo "  (none)"
