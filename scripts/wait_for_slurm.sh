#!/usr/bin/env bash
# wait_for_slurm.sh — block until SLURM job(s) leave the queue, then print a summary.
#
# Purpose: the agent (Claude) should NOT poll squeue by hand in a loop of tool
# calls. Instead start this once in the background; it blocks (cheaply) until the
# jobs finish and prints final states + a tail of each job's log. When run via
# Claude Code's run_in_background, the agent is pinged exactly once on exit.
#
# Usage:
#   scripts/wait_for_slurm.sh <jobid> [<jobid> ...]   # wait for specific jobs
#   scripts/wait_for_slurm.sh                          # wait for ALL my current jobs
#
# Env knobs:
#   POLL=30   poll interval in seconds (default 30)
#   TAIL=25   lines of each job's log to print at the end (default 25)
#
# Typical agent flow:
#   jid=$(sbatch --parsable scripts/job_test.sh)
#   POLL=30 scripts/wait_for_slurm.sh "$jid"        # <- launched with run_in_background

set -uo pipefail
POLL="${POLL:-30}"
TAIL="${TAIL:-25}"
USER_NAME="${USER:-$(whoami)}"

stamp() { date +%H:%M:%S; }

# --- resolve the set of job IDs to wait on -------------------------------------
if [ "$#" -gt 0 ]; then
  ids=("$@")
else
  mapfile -t ids < <(squeue -h -u "$USER_NAME" -o "%i" 2>/dev/null)
fi

if [ "${#ids[@]}" -eq 0 ]; then
  echo "[$(stamp)] no jobs to wait for (queue empty / none given). nothing to do."
  exit 0
fi

idcsv=$(IFS=,; echo "${ids[*]}")
echo "[$(stamp)] waiting on ${#ids[@]} job(s): $idcsv   (poll=${POLL}s)"

# --- wait loop -----------------------------------------------------------------
# A job is "done" once it no longer appears in the user's squeue listing. We list
# the whole user queue and intersect, which avoids `squeue -j <gone-id>` errors.
while :; do
  qids=$(squeue -h -u "$USER_NAME" -o "%i" 2>/dev/null); rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(stamp)] squeue error (rc=$rc) — retrying in ${POLL}s"
    sleep "$POLL"; continue
  fi

  remaining=()
  states=()
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    jid=${line%% *}; st=${line#* }
    for want in "${ids[@]}"; do
      if [ "$jid" = "$want" ] || [[ "$jid" == "${want}_"* ]]; then
        remaining+=("$jid"); states+=("$jid:$st")
      fi
    done
  done < <(squeue -h -u "$USER_NAME" -o "%i %T" 2>/dev/null)

  [ "${#remaining[@]}" -eq 0 ] && break
  echo "[$(stamp)] still running (${#remaining[@]}): ${states[*]}"
  sleep "$POLL"
done

# --- summary -------------------------------------------------------------------
echo
echo "============================================================"
echo "[$(stamp)] all target jobs left the queue"
echo "============================================================"

while IFS='|' read -r jid jname state exitcode elapsed workdir; do
  [ -z "$jid" ] && continue
  echo
  echo "Job $jid ($jname): $state   exit=$exitcode   elapsed=$elapsed"
  log=$(ls -t "$workdir"/*"${jid%%_*}"*.out 2>/dev/null | head -1)
  if [ -n "$log" ]; then
    echo "log: $log  (last $TAIL lines)"
    echo "------------------------------------------------------------"
    tail -n "$TAIL" "$log"
  else
    echo "(no *${jid%%_*}*.out log found under ${workdir:-?})"
  fi
done < <(sacct -X -j "$idcsv" -P --noheader \
            -o JobID,JobName,State,ExitCode,Elapsed,WorkDir 2>/dev/null)

echo
echo "[$(stamp)] done."
