#!/bin/bash
#SBATCH --job-name=standalones
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=standalones_%j.out
#SBATCH --error=standalones_%j.out
#
# CPU only. The compiled MadGraph standalones under $WORK/mg5amcnlo (tools/pool_physics.py) [D208, D209]:
#   STANDALONES_MODE=audit (default): their count and size, the pools whose sidecar has no physics block (their
#     standalones are kept), the prune's dry run, and the probe-read physics check from the sidecars only
#     (--sidecar-only) on each "<train recipe>:<role>:<probe recipe>" of STANDALONES_CHECKS (words joined by '+')
#   STANDALONES_MODE=apply: the prune (manifest under $WORK/mg5amcnlo/.prune_manifests first), the install kept
#     site submit --item <entry> ccin2p3 FA scripts/job_standalones.sh --env STANDALONES_MODE=audit --env STANDALONES_CHECKS=...
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
rc=0
df -B1G "$WORK" | tail -1
echo "STANDALONES $(ls -d "$WORK"/mg5amcnlo/*_standalone 2>/dev/null | wc -l)"
if [ "${STANDALONES_MODE:-audit}" = apply ]; then
    python tools/pool_physics.py prune-standalones --apply || rc=1
else
    mkdir -p "$SCRATCH/standalones"
    python tools/pool_physics.py sidecars --out "$SCRATCH/standalones/sidecars_${SLURM_JOB_ID:-local}.json" || rc=1
    python tools/pool_physics.py prune-standalones || rc=1
    for c in ${STANDALONES_CHECKS//+/ }; do
        IFS=: read -r train role probe <<< "$c"
        echo "CHECK $train $role $probe"
        python tools/pool_physics.py check --sidecar-only --quiet --role "$role" --train "$train" "$probe" | tail -n 3 || rc=1
    done
fi
df -B1G "$WORK" | tail -1
echo "RUN_SUMMARY standalones mode=${STANDALONES_MODE:-audit} rc=$rc"
exit $rc
