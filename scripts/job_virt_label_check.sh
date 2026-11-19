#!/bin/bash
#SBATCH --job-name=virt_label_check
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=virt_label_check_%j.out
#SBATCH --error=virt_label_check_%j.out
#
# CPU only. The lhe label rule 'virt' checked on the held-out one-loop classes' own catalog pools (tools/virt_label_check.py):
# VL_N rows (200) of each class's test pool relabelled through the rule's path and compared with the stored labels, one
# class per core at a time (each builds or reuses its MadLoop standalone, datagen.ensure_virt_backend); classes VL_CLASSES
# (default: every class of recipes/catalog_v4_heldout_H5.yaml). Ends with one RUN_SUMMARY line.
#     site submit ccin2p3 FA scripts/job_virt_label_check.sh --item <entry>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
C=${VL_CLASSES:-$(python -c "import yaml; print(' '.join(p['name'] for p in yaml.safe_load(open('recipes/catalog_v4_heldout_H5.yaml'))['processes']))")}
echo "VIRT_LABEL_CHECK_START code=$(git rev-parse --short HEAD) classes=$(echo $C | wc -w) n=${VL_N:-200}"
# VL_FROM_RUN=<run dir> [VL_SAVE=<dir>]: the train and val pools that run loaded, all CPUs on one class at a time
if [ -n "${VL_FROM_RUN:-}" ]; then
  for c in $C; do python tools/virt_label_check.py --n "${VL_N:-200}" --workers "${SLURM_CPUS_PER_TASK:-1}" \
    --from-run "$VL_FROM_RUN" ${VL_SAVE:+--save "$VL_SAVE"} "$c"; done
else
echo $C | tr ' ' '\n' | xargs -P "${SLURM_CPUS_PER_TASK:-1}" -I{} python tools/virt_label_check.py --n "${VL_N:-200}" --workers 1 {}
fi
echo "RUN_SUMMARY virt_label_check rc=$? checked=$(echo $C | wc -w)"
