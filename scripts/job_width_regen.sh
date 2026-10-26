#!/bin/bash
#SBATCH --job-name=cov_width_regen
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=cov_width_regen_%j.out
#SBATCH --error=cov_width_regen_%j.out
#
# CPU only (catalog_coverage; the user 2026-10-26: "do 1.4915 for all, let's not waste time regenerating everything
# now. For the others, if it's not too much time for regeneration, regenerate."). tools/width_regen.py:
#   1. the families whose patched widths MadGraph ignored (heavy-quark layer and Z'/2HDM points of both catalogs,
#      catalog_v4's six loop-induced HH/HHH entries) built in a staging area: standalones, val/test and train pools
#   2. the staged cards read back (no indented DECAY line, every patched width at its value); a bad card stops here
#   3. once no catalog_v4 regeneration part runs: catalog_v4's tree sidecars migrated to WT 1.4915 (what their cards
#      gave), then the old family pools, chunk dirs and standalones moved to the quarantine dir (manifests there, nothing
#      deleted) and the staged pools moved into place
#   4. the three held one-loop entries (zero-width certification, then their build: scripts/job_v4_certify_then_regen.sh)
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
W="$SCRATCH/zz_width_regen"; Q="$W/quarantine"
mkdir -p "$W/mg" "$W/frozen" "$W/train" "$W/out"
NW="${SLURM_CPUS_PER_TASK:-16}"
T0=$(date +%s)
df -k "$DATA_DIR" | tail -1 | awk '{print "WIDTH_FREE_KB_START", $4}'
python tools/width_regen.py spec "$W/spec.yaml" || exit 1
MG5_WORK_DIR="$W/mg" MG5_OUTPUT_DIR="$W/out" AMP_FROZEN_DIR="$W/frozen" AMP_TRAIN_CACHE_DIR="$W/train" \
    python prebuild_recipes.py "$W/spec.yaml" --workers "$NW" --skip-cert-failed "$W/cert_failed.txt"
RC=$?
echo "WIDTH_BUILD_RC $RC wall $(( $(date +%s) - T0 )) s"
[ $RC -eq 0 ] || { echo "WIDTH_FAILED build"; exit $RC; }
MG5_WORK_DIR="$W/mg" python tools/width_regen.py verify-cards "$W/spec.yaml" || { echo "WIDTH_FAILED cards"; exit 3; }
while squeue -h -u "$USER" -n cov_v4_regen,v4_cert_regen | grep -q .; do echo "waiting for a running regeneration part"; sleep 120; done
python tools/width_regen.py migrate --quarantine "$Q/migrate" --apply || { echo "WIDTH_FAILED migrate"; exit 4; }
python tools/width_regen.py swap "$W/spec.yaml" --stage-frozen "$W/frozen" --stage-train "$W/train" --quarantine "$Q" \
    --apply || { echo "WIDTH_FAILED swap"; exit 5; }
echo "WIDTH_SWAP_DONE wall $(( $(date +%s) - T0 )) s"
CERT_ARGS="--only v4_bg_bwpwm v4_bbx_tapvtwm v4_bbx_hwpwm" \
V4REGEN_ONLY="v4_bg_bwpwm_nlo,v4_bbx_tapvtwm_nlo,v4_bbx_hwpwm_nlo" \
    bash "$_CCORCH_ROOT/scripts/job_v4_certify_then_regen.sh"
RC=$?
df -k "$DATA_DIR" | tail -1 | awk '{print "WIDTH_FREE_KB_END", $4}'
echo "WIDTH_DONE rc=$RC wall $(( $(date +%s) - T0 )) s cores $NW"
exit $RC
