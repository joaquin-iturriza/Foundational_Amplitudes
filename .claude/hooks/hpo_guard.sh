#!/usr/bin/env bash
# PreToolUse(Bash) hook — block hand-rolled HYPERPARAMETER grids/scans submitted via sbatch.
#
# Why: CLAUDE.md says HP search goes through the DyHPO Bayesian optimiser
# (sweep/generate_sweep.py + sweep/sweep_manager.py), single-fidelity. The model (me)
# violated this REPEATEDLY — even after an explicit user instruction — by rationalizing
# "this isn't an HPO, it's a *diagnostic* to test a hypothesis" and then reaching for an
# sbatch --array over lr / clip_grad_norm / beta / lambda. Framing an HP question as a
# mechanism question does not stop it being an HP search. (Exactly the same rationalization
# shape as the one md_guard.sh exists to stop: "it's a report, not guidance".)
#
# It is not merely a style violation — it CORRUPTS CONCLUSIONS. A 1-D grid over lr at fixed
# beta/lambda/warmup answers "best lr GIVEN those fixed values", not "does this method work at
# its own HPs". If the optimum needs (lr AND beta) jointly, the grid cannot reach it and comes
# back flat, manufacturing a false "the method is broken at every lr" verdict. The HPs interact;
# the search must be joint.
#
# Scope: Bash commands that `sbatch` a script which BOTH
#   (a) is a SLURM job array  (#SBATCH --array), and
#   (b) feeds a known HYPERPARAMETER a per-task shell variable (e.g. training.lr=${LR}).
# That is precisely an HP scan. Arrays that vary NON-HP things stay allowed — loss type, data
# tag/path, warm-start checkpoint, run name, ablation flags, seeds — because those are genuine
# controlled ablations, not HP searches.
#
# Escape hatch (deliberate + auditable): if the user explicitly approves a one-off HP grid, add
# the script's repo-relative or absolute path to  .claude/hpo_grid_allowlist.txt  (one per line).
set -uo pipefail
REPO="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ALLOWLIST="$REPO/.claude/hpo_grid_allowlist.txt"

input=$(cat)
cmd=$(printf '%s' "$input" | python3 -c 'import sys,json
try:
    print(json.load(sys.stdin).get("tool_input",{}).get("command",""))
except Exception:
    print("")' 2>/dev/null)
[ -z "$cmd" ] && exit 0

# Only interested in sbatch submissions.
printf '%s' "$cmd" | grep -qE '(^|[^[:alnum:]_])sbatch([^[:alnum:]_]|$)' || exit 0

# Hyperparameters, per CLAUDE.md's search spaces. Deliberately EXCLUDES run-design/scaling axes
# (num_heads, num_blocks, iterations, dataset size) and ablation flags — arrays over those are fine.
HP_RE='(training\.(lr|clip_grad_norm|clip_grad_value|regularization_lambda|cosanneal_warmup_frac|cosanneal_eta_min|heterosc_beta|ema_decay|sampler_alpha_ema)|fine_tune\.(lr_scale|layer_decay))'

# Candidate scripts referenced by the command (any token that looks like a shell script path).
scripts=$(printf '%s' "$cmd" | tr ' \t\n' '\n\n\n' | grep -E '\.sh$' || true)

for s in $scripts; do
  path="$s"
  [ -f "$path" ] || path="$REPO/$s"
  [ -f "$path" ] || continue

  # allowlisted one-off?
  rel=${path#"$REPO"/}
  if [ -f "$ALLOWLIST" ]; then
    skip=0
    while IFS= read -r line; do
      line=$(printf '%s' "$line" | sed 's/#.*//; s/^[[:space:]]*//; s/[[:space:]]*$//')
      [ -z "$line" ] && continue
      if [ "$line" = "$path" ] || [ "$line" = "$rel" ]; then skip=1; break; fi
    done < "$ALLOWLIST"
    [ "$skip" -eq 1 ] && continue
  fi

  # (a) is it a job array?
  grep -qE '^#SBATCH[[:space:]]+--array' "$path" || continue
  # (b) does it feed an HP a per-task shell variable?
  hit=$(grep -nE "${HP_RE}=[\"']?\\\$" "$path" | head -3 || true)
  [ -z "$hit" ] && continue

  {
    echo "BLOCKED by hpo_guard: '$rel' is a hand-rolled HYPERPARAMETER GRID (sbatch --array varying an HP)."
    echo "Offending line(s):"
    printf '%s\n' "$hit" | sed 's/^/    /'
    echo ""
    echo "CLAUDE.md: HP search goes through the DyHPO Bayesian optimiser, SINGLE-FIDELITY — never a grid:"
    echo "    python sweep/generate_sweep.py --config sweep/<my_config>.yaml   # fidelity_schedule.t_steps: [T]  (one value)"
    echo "    python sweep/sweep_manager.py submit <sweep_dir>/<sweep_name>"
    echo ""
    echo "Reframing an HP question as 'just a diagnostic / just testing a mechanism' does NOT make it a"
    echo "controlled ablation — lr, clip_grad_norm, beta, lambda, warmup, lr_scale, layer_decay are HPs."
    echo "A 1-D grid at fixed other-HPs cannot find a joint optimum and will manufacture a FALSE"
    echo "'it doesn't work at any <HP>' conclusion. Sweep the coupled space instead."
    echo ""
    echo "Arrays are still fine for NON-HP ablations (loss type, data tag, warm-start ckpt, seed, flags)."
    echo "If the user EXPLICITLY approved this one-off grid: add '$rel' to .claude/hpo_grid_allowlist.txt, then retry."
  } >&2
  exit 2
done

exit 0
