#!/bin/bash
# Fast waiter for design runs (the user, 2026-10-22: "a faster waiter should be in place for design runs in general"):
# the supervisor sees a run end only on its 20-min pass, up to a third of an hour-long design run. Run in the background
# right after a design run's LAUNCHED event, with its run ids:
#     bash scripts/wait_runs.sh FA-ccin2p3-xxxx [FA-ccin2p3-yyyy ...]      (run_in_background)
# It polls the scheduler every POLL seconds (default 90) through `site poll`, prints each run's state change, and exits
# once every run is terminal, which wakes the session. The run ids go in .claude/.slurm_monitor_jobs for the duration,
# so slurm_waiter_guard.sh accepts the turn ending.
POLL="${POLL:-90}"
cd "$(dirname "$0")/.." || exit 1
MON=".claude/.slurm_monitor_jobs"
for r in "$@"; do grep -qxF "$r" "$MON" 2>/dev/null || echo "$r" >> "$MON"; done
declare -A last
while :; do
    left=0
    out=$(timeout 120 site poll "$@" 2>/dev/null)
    for r in "$@"; do
        st=$(echo "$out" | awk -v r="$r" '$1 == r {print $NF}' | tail -1)
        [ -n "$st" ] && [ "$st" != "${last[$r]}" ] && { echo "$(date +%H:%M:%S) $r $st"; last[$r]=$st; }
        case "$st" in COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|OUT_OF_MEM|NODE_FAIL|HELD|REMOVED) ;; *) left=1 ;; esac
    done
    [ $left = 0 ] && break
    sleep "$POLL"
done
for r in "$@"; do sed -i "\|^$r\$|d" "$MON" 2>/dev/null; done
echo "all terminal: $*"
