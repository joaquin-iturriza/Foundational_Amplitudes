#!/bin/bash
#SBATCH --job-name=catalog_v3_prep
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=catalog_v3_prep_%j.out
#SBATCH --error=catalog_v3_prep_%j.out
#
# CPU only (D115): what catalog_v3's pretrainings need beyond the pools, for every process of its recipes
# (recipes/catalog_v3_train.yaml, catalog_v3_heldout_*.yaml): each event's alpha_s (<pool>.alphas.npy,
# tools/write_own_alphas.py; data.target_physics_orders reads it) and the diagram sidecars (tools/dump_diagrams.py; the
# diagram encoder and the off-shellness need one per process, named by its base, the run refuses a missing one). Both
# skip what exists.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
# H7E / H7c (stage 4) are prepared with their generation: their pools do not exist before it
R=(recipes/catalog_v3_train.yaml $(ls recipes/catalog_v3_heldout_*.yaml | grep -v '_H7'))
for f in "${R[@]}"; do
    python tools/write_own_alphas.py "$f" || exit 1
done
NAMES=$(python -c "
import sys, yaml
print(' '.join(sorted({'--process ' + p.get('base', p['name']) for f in sys.argv[1:] for p in yaml.safe_load(open(f))['processes']})))" "${R[@]}")
python tools/dump_diagrams.py $NAMES --skip-existing --workers 32 | tee "$SCRATCH/zz_v3_prep_diagrams.txt"
grep -q "Done: [0-9]* ok, 0 failed" "$SCRATCH/zz_v3_prep_diagrams.txt" || { echo "V3_PREP: a sidecar failed"; exit 1; }
echo "V3_PREP_DONE"
