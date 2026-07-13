#!/usr/bin/env bash
# Stop hook — refuse to end a turn with MY SLURM jobs in flight and no background waiter.
#
# Why: CLAUDE.md ("Waiting on jobs — always background, never hand-poll") says: submit, then
# launch `scripts/wait_for_slurm.sh <jids>` with run_in_background so the harness re-invokes
# me exactly once when the jobs finish. It explicitly forbids promising "I'll report when they
# land" WITHOUT a mechanism. The model (me) submitted two 16-trial sweeps and then did exactly
# that — no waiter, just a promise. That silently drops the result on the floor: the turn ends,
# nothing re-invokes me, and the user has to notice and prod.
#
# Rule enforced: if `squeue -u $USER` shows any of my jobs RUNNING/PENDING and no
# wait_for_slurm.sh process is alive, block the Stop and make me launch the waiter.
#
# Escape hatch: `touch .claude/.no_waiter_needed` to allow one Stop with jobs in flight
# (deliberate fire-and-forget, e.g. the user said they'll check themselves). Auto-cleared.
set -uo pipefail
REPO="/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
BYPASS="$REPO/.claude/.no_waiter_needed"

command -v squeue >/dev/null 2>&1 || exit 0

if [ -f "$BYPASS" ]; then
  rm -f "$BYPASS"
  exit 0
fi

USER_NAME="${USER:-$(whoami)}"
jobs=$(squeue -u "$USER_NAME" -h -o "%i %j %T" 2>/dev/null | grep -viE 'prebuild' || true)
[ -z "$jobs" ] && exit 0

# A waiter alive? (wait_for_slurm.sh backgrounded by the harness)
if pgrep -f "wait_for_slurm.sh" >/dev/null 2>&1; then
  exit 0
fi

n=$(printf '%s\n' "$jobs" | wc -l | tr -d ' ')
{
  echo "BLOCKED by slurm_waiter_guard: $n of your SLURM job(s) are still in the queue and NO"
  echo "background waiter is running. Ending the turn now means nothing will re-invoke you when"
  echo "they finish — the result gets dropped and the user has to chase it."
  echo ""
  printf '%s\n' "$jobs" | head -8 | sed 's/^/    /'
  [ "$n" -gt 8 ] && echo "    ... ($n total)"
  echo ""
  echo "CLAUDE.md (Waiting on jobs): submit, then launch the waiter IN THE BACKGROUND —"
  echo "    scripts/wait_for_slurm.sh <jid> [<jid> ...]        # run_in_background: true"
  echo "  (pass job ids as SEPARATE args, not one comma-joined string.)"
  echo "Do NOT promise 'I'll report when they land' without that mechanism, and do NOT hand-poll squeue."
  echo ""
  echo "If the jobs are genuinely fire-and-forget: touch .claude/.no_waiter_needed and stop again."
} >&2
exit 2
