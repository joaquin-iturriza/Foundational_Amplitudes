#!/bin/bash
#SBATCH --job-name=recover_alphas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=recover_alphas_%j.out
#SBATCH --error=recover_alphas_%j.out
# CPU only. Each lhe_pp pool's per-event alpha_s (<pool>.alphas.npy) for pools built before build_lhe_pp saved it
# (D93, data.target_physics_orders): tools/recover_alphas.py reruns the pool's own MadEvent run and checks the momenta
# reproduce before writing. ALPHAS_SPEC: the recipe (default the 40 2->4 probes); ALPHAS_ONLY: names to restrict to.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
# MadEvent's work directories on the node's local disk, removed when the job exits: 48 of them at once on /sps/lpnhe
# (the group's shared 512 GB) filled it on 2026-10-23 and killed every job writing there
export AMP_LHE_WORK_DIR="${TMPDIR:-/tmp}/fa_recover_alphas_${SLURM_JOB_ID}"; mkdir -p "$AMP_LHE_WORK_DIR"
trap 'rm -rf "$AMP_LHE_WORK_DIR"' EXIT
df -h "$AMP_LHE_WORK_DIR" | tail -1
python tools/recover_alphas.py "${ALPHAS_SPEC:-recipes/catalog_2to4_heldout.yaml}" --workers "${SLURM_CPUS_PER_TASK:-48}" \
    ${ALPHAS_ONLY:+--only $ALPHAS_ONLY}
