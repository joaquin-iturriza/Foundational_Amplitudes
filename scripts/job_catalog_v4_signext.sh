#!/bin/bash
#SBATCH --job-name=catalog_v4_signext
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=catalog_v4_signext_%j.out
#SBATCH --error=catalog_v4_signext_%j.out
#
# CPU only (plan v3_prep_explore, the design of 2026-10-29): catalog_v4's one-loop sign extension, everything its two
# pretrainings (scripts/job_catalog_v3_short.sh v4_dimensionless_muon_more_signed[_seed2], chained afterok) need:
#   1. build, pole-certify and generate recipes/catalog_v4_gen_signext.yaml (123 one-loop pools, 1e4 / 1e3 / 1e3); an
#      entry whose build or certification fails is skipped and listed (prebuild_recipes.py --skip-cert-failed)
#   2. audit the pools (analysis/catalog_v3/pool_audit.py --stages v4_signext)
#   3. the arm's training recipe (recipes/make_catalog_v4_train_signext.py): catalog_v4_train plus the pools whose
#      train split holds both signs; refuses (exit 3) when more than a fifth failed certification
#   4. each event's alpha_s and the diagram sidecars of the whole recipe (as scripts/job_catalog_v3_prep.sh)
# Everything it writes outside the pools goes to $SCRATCH/zz_v4_signext (the recipe the arm reads), never the checkout.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
W="$SCRATCH/zz_v4_signext"; mkdir -p "$W"
N="${SLURM_CPUS_PER_TASK:-32}"
python prebuild_recipes.py recipes/catalog_v4_gen_signext.yaml --workers "$N" --auto-workers \
    --skip-cert-failed "$W/cert_failed.txt"
rc=$?; [ "$rc" = 0 ] || [ "$rc" = 4 ] || { echo "V4_SIGNEXT: generation failed (exit $rc)"; exit 1; }
python analysis/catalog_v3/pool_audit.py --stages v4_signext --workers "$N" --out "$W/pool_audit_signext.json" || exit 1
python recipes/make_catalog_v4_train_signext.py --audit "$W/pool_audit_signext.json" --cert-failed "$W/cert_failed.txt" \
    --out "$W/catalog_v4_train_signext.yaml" --summary "$W/signext_summary.json" || exit 1
python tools/write_own_alphas.py "$W/catalog_v4_train_signext.yaml" || exit 1
NAMES=$(python -c "
import sys, yaml
print(' '.join(sorted({'--process ' + p.get('base', p['name']) for p in yaml.safe_load(open(sys.argv[1]))['processes']})))" \
    "$W/catalog_v4_train_signext.yaml")
python tools/dump_diagrams.py $NAMES --skip-existing --workers "$N" | tee "$W/diagrams.txt"
grep -q "Done: [0-9]* ok, 0 failed" "$W/diagrams.txt" || { echo "V4_SIGNEXT: a sidecar failed"; exit 1; }
cat "$W/signext_summary.json"
echo "V4_SIGNEXT_DONE"
