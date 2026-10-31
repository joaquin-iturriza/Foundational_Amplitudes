#!/bin/bash
#SBATCH --job-name=v4_multext_check
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=v4_multext_check_%j.out
#SBATCH --error=v4_multext_check_%j.out
#
# CPU only (plan v3_prep_explore, D216): the multiplicity extension's new trees (recipes/catalog_v4_multext_processes.yaml)
# on their v4 strings, then the class check. 1) recipes/gen_catalog_v4.py --check on the new entries: existence under
# sm-no_b_mass at full coupling orders -> $SCRATCH/zz_v4_multext/checked_v4.json; stops unless every string exists.
# 2) tools/equal_processes.py on recipes/catalog_v4_multext_classcheck.yaml (the new trees and the held-out 2->4 trees of
# H1, H2, H3 in their groups): exact and near pairs -> $SCRATCH/zz_v4_multext/equality/. Compiles the backends it lacks.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
N="${SLURM_CPUS_PER_TASK:-16}"
OUT="$SCRATCH/zz_v4_multext"
mkdir -p "$OUT"
NAMES=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('recipes/catalog_v4_multext_processes.yaml'))['processes']))")
python recipes/gen_catalog_v4.py --check --workers "$N" --only $NAMES --out "$OUT/checked_v4.json" || exit 1
python -c "import json,sys; r=json.load(open(sys.argv[1])); n=sum(c['exists'] for c in r); print('V4_STRINGS_EXIST', n, len(r)); sys.exit(n < len(r))" "$OUT/checked_v4.json" || exit 1
python tools/equal_processes.py --recipes recipes/catalog_v4_multext_classcheck.yaml --workers "$N" \
    --compile-workers "$N" --out "$OUT/equality" || exit 1
echo "V4_MULTEXT_CHECK done"
