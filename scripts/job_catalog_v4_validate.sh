#!/bin/bash
#SBATCH --job-name=cov_v4_validate
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=cov_v4_validate_%j.out
#SBATCH --error=cov_v4_validate_%j.out
#
# CPU only (D130): catalog_v4's validation (analysis/catalog_v4/validate_v4.py: a sample of every class family built,
# generated and checked: cuts, log-uniform sqrt(s_hat), the forward component, the 172.5 top and massless b, the EW
# diagram counts; the one-loop layer's configuration), then the v4 census check (recipes/gen_catalog_v4.py --check: every
# catalog_v3 tree on its v4 string through MadGraph, existence and diagrams). Pools go to $SCRATCH/zz_catalog_v4_val,
# the census to $SCRATCH/zz_catalog_v4/checked.json; nothing is written to $DATA_DIR. The v4 backends it compiles land
# in the shared MadGraph work dir under v4_* names, where the regeneration reuses them.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
N="${SLURM_CPUS_PER_TASK:-16}"
python analysis/catalog_v4/validate_v4.py run --n "${V4VAL_N:-2000}" --workers "$N"
python recipes/gen_catalog_v4.py --check --workers "$N" --out "$SCRATCH/zz_catalog_v4/checked.json"
