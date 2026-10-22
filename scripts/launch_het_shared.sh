#!/bin/bash
# Launch-queue run command (site launch FA het_shared, D82): the other 24 2->3/2->4 Delta cells with the het loss at
# the het sweeps' shared HPs, one job per line of analysis/transfer/het_shared_jobs.txt (est. GPU-h, then its cells).
# Plans the remaining jobs with `site pick`, submits each site's allotment, records what it submitted; exit 75 (stays queued, retried on the
# supervisor's next pass) while any job is left. Every cell and its parent is on CC-IN2P3 (checked 2026-10-22).
cd "$(dirname "$0")/.." || exit 1
JOBS="analysis/transfer/het_shared_jobs.txt"
DONE="$HOME/.local/share/ccorch/launch/FA/het_shared.submitted"
touch "$DONE"
mapfile -t LEFT < <(grep -v '^#' "$JOBS" | while read -r t cells; do grep -qxF "$cells" "$DONE" || echo "$t $cells"; done)
[ ${#LEFT[@]} -eq 0 ] && { echo "all $(grep -vc '^#' "$JOBS") jobs submitted"; exit 0; }
plan=$(timeout 300 site pick FA --only ccin2p3 --jobs ${#LEFT[@]} --mem 16G --cpus 4 --hours 1.5 2>&1)
echo "$plan" | tail -4
i=0; ok=0     # i: jobs tried, ok: jobs submitted (a failed one stays left and is retried on the next pass)
for site in ccin2p3; do          # the cells and parents are complete on CC only (lxplus holds 9 of the copies)
    n=$(echo "$plan" | awk -v s="$site" '$1==s && $3 ~ /^jobs?$/ {print $2}' | tail -1)
    for ((k = 0; k < ${n:-0} && i < ${#LEFT[@]}; k++, i++)); do
        read -r t cells <<< "${LEFT[$i]}"
        # shellcheck disable=SC2086
        site submit "$site" FA scripts/job_het_delta.sh --item het_shared --est-gpu-hours "$t" \
            --note "het at shared HPs, Delta cells (D82)" -- --shared $cells && { echo "$cells" >> "$DONE"; ok=$((ok + 1)); }
    done
done
left=$(( ${#LEFT[@]} - ok ))
echo "submitted $ok, left $left"
[ "$left" -eq 0 ] || exit 75
