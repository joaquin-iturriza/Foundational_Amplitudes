#!/bin/bash
#SBATCH --job-name=catalog_v3_p5check
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=catalog_v3_p5check_%j.out
#SBATCH --error=catalog_v3_p5check_%j.out
#
# CPU only (D102, stage 3): MadGraph's check of catalog_v3's 2->5 probe candidates (gen_catalog_v3.py --stage3), as the
# census checked the trees. Writes $SCRATCH/zz_catalog_v3_census/checked_p5.json (entries are written from the laptop).
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python recipes/gen_catalog_v3.py --check --cands recipes/catalog_v3_candidates_p5.json --workers "${SLURM_CPUS_PER_TASK:-16}" \
    --out "$SCRATCH/zz_catalog_v3_census/checked_p5.json"
