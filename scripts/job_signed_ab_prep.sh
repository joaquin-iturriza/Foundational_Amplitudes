#!/bin/bash
#SBATCH --job-name=signed_ab_prep
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=signed_ab_prep_%j.out
#SBATCH --error=signed_ab_prep_%j.out
#
# CPU only: the inputs the signed pools' A/B (job_catalog_norm.sh SgnR0 / SgnHead) needs on catalog_v3's 62 signed
# one-loop pools (recipes/catalog_signed_ab_{indist,heldout}.yaml) beyond the pools themselves: each event's alpha_s
# (<pool>.alphas.npy, tools/write_own_alphas.py; data.target_physics_orders reads it) and the diagram sidecars
# (tools/dump_diagrams.py; the run's diagram encoder and off-shellness need one per process). Both skip what exists.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
for S in indist heldout; do
    python tools/write_own_alphas.py "recipes/catalog_signed_ab_$S.yaml" || exit 1
done
NAMES=$(python -c "
import yaml
print(' '.join('--process ' + p['name'] for s in ('indist', 'heldout')
               for p in yaml.safe_load(open(f'recipes/catalog_signed_ab_{s}.yaml'))['processes']))")
python tools/dump_diagrams.py $NAMES --skip-existing --workers 8 | tee "$SCRATCH/zz_signed_ab_prep_diagrams.txt"
# every pool needs its sidecar: the run refuses a missing one (experiment._setup_diagram_set_llm4lhc)
grep -q "Done: [0-9]* ok, 0 failed" "$SCRATCH/zz_signed_ab_prep_diagrams.txt" || { echo "SIGNED_AB_PREP: a sidecar failed"; exit 1; }
echo "SIGNED_AB_PREP_DONE"
