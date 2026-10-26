#!/bin/bash
#SBATCH --job-name=v4_cert_regen
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=v4_cert_regen_%j.out
#SBATCH --error=v4_cert_regen_%j.out
#
# CPU only (catalog_coverage): the pole certification of named catalog_v4 one-loop entries (scripts/job_v4_certify_all.sh,
# CERT_ARGS='--only <virt names>'), then their build (scripts/job_catalog_v4_regen.sh with V4REGEN_ONLY=<entry names>,
# one shard), in one job. The build runs whatever the certification says: its own guard (prebuild_recipes.py, the
# per-entry c2_rel_tol) decides per event, and the certification log says whether the zero-width check passed.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
: "${CERT_ARGS:?CERT_ARGS not set}" "${V4REGEN_ONLY:?V4REGEN_ONLY not set}"
echo "=== certify: $CERT_ARGS"
bash "$_CCORCH_ROOT/scripts/job_v4_certify_all.sh"
echo "CERT_RC $?"
echo "=== regen: $V4REGEN_ONLY"
V4REGEN_N=1 V4REGEN_K=0 bash "$_CCORCH_ROOT/scripts/job_catalog_v4_regen.sh"
