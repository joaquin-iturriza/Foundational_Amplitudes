#!/bin/bash
# Launch-queue run command (site launch FA het_delta_jzcopy, D47/D71): bring the Delta cells first trained on Jean Zay
# (analysis/transfer/het_delta_jobs_jz.txt) and their fine-tune parents to CC-IN2P3 and lxplus, site to site, so their
# het retrains run where `site pick` places them. Jean Zay's outbound copies time out at times: exit 75 (stays queued,
# retried on the supervisor's next pass) until every copy has gone through. Each destination is done on its own and
# leaves its own marker (<MARK>.<site>); the first one done writes the marker het_delta_jz's --ready waits for, so an
# unreachable lxplus (expired ticket) does not hold back the retrains on CC.
cd "$(dirname "$0")/.." || exit 1
MARK="$HOME/.local/share/ccorch/launch/FA/het_delta_jz.copied"
[ -e "$MARK.ccin2p3" ] && [ -e "$MARK.lxplus" ] && { echo "already copied"; exit 0; }
CELLS=$(grep -v '^#' analysis/transfer/het_delta_jobs_jz.txt | cut -d' ' -f2- | tr '\n' ' ')
# the parents, read from the cells' configs on Jean Zay (absolute Jean Zay paths, re-rooted by job_het_delta.sh)
PAR=$(timeout 600 site run jeanzay FA -- "for c in $CELLS; do grep -m1 'pretrained_path:' \$c/config.yaml; done" 2>&1 \
      | grep 'pretrained_path:' | sed 's#.*pretrained_path: *##; s#.*/Foundational_Amplitudes/##; s#/models/.*##' \
      | grep -v '^null$' | sort -u | tr '\n' ' ')
echo "parents: $PAR"
pending=0
for dst in ccin2p3 lxplus; do
    [ -e "$MARK.$dst" ] && continue
    # shellcheck disable=SC2086
    out=$(timeout 3600 site copy FA jeanzay "$dst" $CELLS $PAR --item het_delta 2>&1); echo "$out" | tail -3
    if echo "$out" | grep -q "copy failed"; then pending=1; continue; fi
    touch "$MARK.$dst"; [ -e "$MARK" ] || touch "$MARK"
done
[ "$pending" -eq 0 ] || exit 75
