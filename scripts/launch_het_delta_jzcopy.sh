#!/bin/bash
# Launch-queue run command (site launch FA het_delta_jzcopy, D47/D71): bring the Delta cells first trained on Jean Zay
# (analysis/transfer/het_delta_jobs_jz.txt) and their fine-tune parents to CC-IN2P3 and lxplus, site to site, so their
# het retrains run where `site pick` places them. Jean Zay's outbound copies time out at times: exit 75 (stays queued,
# retried on the supervisor's next pass) until every copy has gone through; then writes the marker that
# het_delta_jz's --ready waits for.
cd "$(dirname "$0")/.." || exit 1
MARK="$HOME/.local/share/ccorch/launch/FA/het_delta_jz.copied"
[ -e "$MARK" ] && { echo "already copied"; exit 0; }
CELLS=$(grep -v '^#' analysis/transfer/het_delta_jobs_jz.txt | cut -d' ' -f2- | tr '\n' ' ')
for dst in ccin2p3 lxplus; do
    # shellcheck disable=SC2086
    out=$(timeout 3600 site copy FA jeanzay "$dst" $CELLS --item het_delta 2>&1); echo "$out" | tail -3
    echo "$out" | grep -q "copy failed" && exit 75
done
# the parents, read from the copied configs on CC (absolute Jean Zay paths, re-rooted by job_het_delta.sh)
PAR=$(timeout 600 site run ccin2p3 FA -- "for c in $CELLS; do grep -m1 'pretrained_path:' \$c/config.yaml; done" 2>&1 \
      | grep 'pretrained_path:' | sed 's#.*pretrained_path: *##; s#.*/Foundational_Amplitudes/##; s#/models/.*##' \
      | grep -v '^null$' | sort -u | tr '\n' ' ')
echo "parents: $PAR"
for dst in ccin2p3 lxplus; do
    # shellcheck disable=SC2086
    [ -n "$PAR" ] && { out=$(timeout 3600 site copy FA jeanzay "$dst" $PAR --item het_delta 2>&1); echo "$out" | tail -3
                       echo "$out" | grep -q "copy failed" && exit 75; }
done
touch "$MARK"
