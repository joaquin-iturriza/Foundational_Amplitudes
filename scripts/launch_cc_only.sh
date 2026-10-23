#!/bin/bash
# Launch-queue run command for a one-job submission whose inputs live only on CC-IN2P3 (a run dir written there):
# plan with `site pick --only ccin2p3` (the job cannot run elsewhere; planning over every site deferred it for as long
# as another site looked faster, which with CC busy was indefinitely) and submit; defer (exit 75; the supervisor
# requeues and retries on its next pass) only when CC is not in the plan, i.e. down.
#     [MEM=40G] bash scripts/launch_cc_only.sh <item> <est_gpu_h> "<note>" <job.sh> [job args...]
# MEM: the host memory to plan and request (default: the site template's)
# FA_SEED: passed to the job (a rerun at another seed; job_catalog_norm.sh reads it)
cd "$(dirname "$0")/.." || exit 1
ITEM="$1"; EST="$2"; NOTE="$3"; JOB="$4"; shift 4
# a submission already made for this item, job and arguments is skipped (exit 0): a launch that loops over several
# jobs and defers on a later one is retried whole by the supervisor, which otherwise submitted the earlier ones twice
DONE="$HOME/.local/share/ccorch/launch/FA/$ITEM.cc_only.submitted"; KEY="${FA_SEED:+FA_SEED=$FA_SEED }$JOB $*"
# RESUBMIT=1 submits again on purpose (a run that failed and is to be redone)
[ -z "$RESUBMIT" ] && grep -qxF "$KEY" "$DONE" 2>/dev/null && { echo "already submitted: $KEY"; exit 0; }
p=$(timeout 250 site pick FA --only ccin2p3 --jobs 1 --mem "${MEM:-16G}" --cpus 4 --hours "$EST" 2>&1 | tail -1 | sed "s/^ *//")
echo "$p"
[ "$p" = "pick: ccin2p3" ] || exit 75
site submit ccin2p3 FA "$JOB" ${MEM:+--hdr=--mem=$MEM} ${FA_SEED:+--env FA_SEED=$FA_SEED} --item "$ITEM" --est-gpu-hours "$EST" --note "$NOTE" -- "$@" || exit $?
echo "$KEY" >> "$DONE"
