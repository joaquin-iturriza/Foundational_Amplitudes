#!/bin/bash
# Launch-queue run command (site launch FA het_delta, D47 released by D71): the het retrains of the Delta-plot cells,
# one job per line of analysis/transfer/het_delta_jobs_<set>.txt (est. GPU-h, then its cells). Plans the remaining jobs with
# `site pick`, submits each site's allotment, records what it submitted; exit 75 (stays queued, retried on the
# supervisor's next pass) while any job is left. Every cell and its parent is on CC-IN2P3 and lxplus (site copy).
cd "$(dirname "$0")/.." || exit 1
JOBS="analysis/transfer/het_delta_jobs_${1:?cclx or jz}.txt"   # cclx: cells first trained on CC/lxplus; jz: on Jean Zay
DONE="$HOME/.local/share/ccorch/launch/FA/het_delta.submitted"
touch "$DONE"
mapfile -t LEFT < <(grep -v '^#' "$JOBS" | while read -r t cells; do grep -qxF "$cells" "$DONE" || echo "$t $cells"; done)
[ ${#LEFT[@]} -eq 0 ] && { echo "all $(grep -vc '^#' "$JOBS") jobs submitted"; exit 0; }
plan=$(timeout 300 site pick FA --jobs ${#LEFT[@]} --mem 16G --cpus 4 --hours 1.5 2>&1)
echo "$plan" | tail -4
i=0
for site in ccin2p3 lxplus; do
    n=$(echo "$plan" | awk -v s="$site" '$1==s && $3 ~ /^jobs?$/ {print $2}' | tail -1)
    for ((k = 0; k < ${n:-0} && i < ${#LEFT[@]}; k++, i++)); do
        read -r t cells <<< "${LEFT[$i]}"
        # shellcheck disable=SC2086
        site submit "$site" FA scripts/job_het_delta.sh --item het_delta --est-gpu-hours "$t" \
            --note "het retrain of Delta cells (D47)" -- $cells && echo "$cells" >> "$DONE"
    done
done
left=$(( ${#LEFT[@]} - i ))
echo "submitted $i, left $left"
[ "$left" -eq 0 ] || exit 75
