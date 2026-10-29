#!/bin/bash
#SBATCH --job-name=v4_loop_born
#SBATCH --cpus-per-task=17
#SBATCH --time=03:00:00
#SBATCH --output=v4_loop_born_%j.out
#SBATCH --error=v4_loop_born_%j.out
#
# CPU only (plan v3_prep_explore; par:v4_loop_born): MadLoop's Born re-evaluated on 2000 train events of 17 of
# catalog_v4's one-loop pools (tools/nlo_virtual_pipeline.py --born-of, one process per pool: the MadLoop module is a
# per-process singleton), the stored target recomputed and checked; per pool the mean of log|c0 + shift| for the
# pool-level proxy's check (analysis/catalog_v4/loop_born_events.py). Writes only under $SCRATCH/zz_v4_loop_born_<job>.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SETUPTOOLS_USE_DISTUTILS=stdlib
cd "$PROJECT_DIR"
python analysis/catalog_v4/loop_born_events.py --job || exit 1
echo "V4_LOOP_BORN_DONE"
