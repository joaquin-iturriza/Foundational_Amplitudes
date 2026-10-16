#!/bin/bash
# Launch-queue run command for a one-job submission whose inputs live only on CC-IN2P3 (a run dir written there):
# plan with `site pick`, submit to CC when the plan puts the job there, otherwise defer (exit 75; the supervisor
# requeues and retries on its next pass).
#     [MEM=40G] bash scripts/launch_cc_only.sh <item> <est_gpu_h> "<note>" <job.sh> [job args...]
# MEM: the host memory to plan and request (default: the site template's)
cd "$(dirname "$0")/.." || exit 1
ITEM="$1"; EST="$2"; NOTE="$3"; JOB="$4"; shift 4
p=$(timeout 250 site pick FA --jobs 1 --mem "${MEM:-16G}" --cpus 4 --hours "$EST" 2>&1 | tail -1 | sed "s/^ *//")
echo "$p"
[ "$p" = "pick: ccin2p3" ] || exit 75
exec site submit ccin2p3 FA "$JOB" ${MEM:+--hdr --mem=$MEM} --item "$ITEM" --est-gpu-hours "$EST" --note "$NOTE" -- "$@"
