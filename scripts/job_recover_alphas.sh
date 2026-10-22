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
export AMP_LHE_WORK_DIR="${AMP_LHE_WORK_DIR:-$SCRATCH/lhe_work}"
python tools/recover_alphas.py "${ALPHAS_SPEC:-recipes/catalog_2to4_heldout.yaml}" --workers "${SLURM_CPUS_PER_TASK:-48}" \
    ${ALPHAS_ONLY:+--only $ALPHAS_ONLY}
