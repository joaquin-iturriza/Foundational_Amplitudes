#!/bin/bash
# Launch-queue run command (site launch FA jaff_zero_shot, D65): the zero-shot of the jaff control, tp3_jaff_r9
# (tp3_jcaus_r9 minus the three causal_* keys), on the multiplicity probes, once its pretraining has completed.
# The parent lives on CC-IN2P3; if the plan places the job elsewhere, defer (exit 75) and let the supervisor retry.
cd "$(dirname "$0")/.." || exit 1
p=$(timeout 250 site pick FA --jobs 1 --mem 16G --cpus 3 --hours 0.5 2>&1 | tail -1 | sed "s/^ *//")
echo "$p"
[ "$p" = "pick: ccin2p3" ] || exit 75
exec site submit ccin2p3 FA scripts/job_zero_shot_parent.sh --item jaff --est-gpu-hours 0.3 \
    --note "zero-shot of the jaff control (D65)" -- runs/tp3_jaff_r9/trial_0073
