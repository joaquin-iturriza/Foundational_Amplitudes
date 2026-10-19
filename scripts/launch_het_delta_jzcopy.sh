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
# the parents, read from the cells' sweep configs in this checkout (a lookup on Jean Zay timed out); only the runs/...
# part is kept, so the site each config was written for does not matter
PAR=$(for c in $CELLS; do s=${c#runs/}; s=${s%%/*}; grep -m1 'fine_tune.pretrained_path' "sweep/sweep_config_$s.yaml"; done \
      | sed -E 's#.*(runs/[^ "]+)/models/.*#\1#' | sort -u | tr '\n' ' ')
echo "parents: $PAR"
pending=0
# one path per `site copy`: it stops at the first path that fails and re-sends the earlier ones on the next try, and
# Jean Zay drops a connection every few paths, so the batch never finished. Paths done go in <MARK>.<site>.done.
for dst in ccin2p3 lxplus; do
    [ -e "$MARK.$dst" ] && continue
    touch "$MARK.$dst.done"; left=0
    for p in $CELLS $PAR; do
        grep -qxF "$p" "$MARK.$dst.done" && continue
        out=$(timeout 1800 site copy FA jeanzay "$dst" "$p" --item het_delta 2>&1); rc=$?
        if [ "$rc" -ne 0 ] || echo "$out" | grep -q "copy failed"; then left=$((left+1)); continue; fi
        echo "$p" >> "$MARK.$dst.done"
    done
    echo "$dst: $(wc -l < "$MARK.$dst.done") paths copied, $left failed this pass"
    if [ "$left" -gt 0 ]; then pending=1; continue; fi
    touch "$MARK.$dst"; [ -e "$MARK" ] || touch "$MARK"
done
[ "$pending" -eq 0 ] || exit 75
