#!/bin/bash
# kit_benchmark: protocol A fine-tuned (A_ft) from the family-held-out parent, the 12 arms matched to the protocol A
# from-scratch runs (design passed on the plan entry). Run on the laptop by `site launch` once the parent's resume
# segment has ended with RUN_DONE; submits the whole batch at once (one wave), then records each run on the plan.
#     bash scripts/launch_kit_aft.sh
set -u
cd "$(dirname "$0")/.."
P=runs/kit/kit_pretrain_heldout
site pick FA --jobs 12 || exit 75
sub() {  # <time> <dataset> <n> [args...]
    local t="$1"; shift
    out=$(site submit ccin2p3 FA scripts/job_kit.sh --item kit_benchmark --allow-dirty --est-gpu-hours 1.6 \
        --note "A_ft from $P" --env KIT_PARENT=$P --hdr=--time="$t" -- "$1" "$2" 8000 A_ft "${@:3}" 2>&1)
    echo "$out" | tail -2
    run=$(echo "$out" | grep -oE "FA-ccin2p3-[0-9a-f]{8}" | tail -1)
    [ -n "$run" ] && python3 tools/plan.py run kit_benchmark "$run"
}
OFF="+data.offshell_zero=true"
for n in 900000 100000; do
    sub 01:40:00 kit_ee_tt__virt_finite $n tag=offon
    sub 01:40:00 kit_ee_tt__virt_finite $n tag=offoff $OFF
    sub 01:40:00 kit_ee_tt__born $n tag=offon
    sub 01:40:00 kit_ee_tt__born $n tag=offoff $OFF
    sub 01:55:00 kit_ee_uug__me2 $n
    sub 02:10:00 kit_ee_uugg__me2 $n
done
