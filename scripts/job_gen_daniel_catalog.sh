#!/bin/bash
#SBATCH --job-name=gen_daniel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --output=gen_daniel_%j.out
#SBATCH --error=gen_daniel_%j.out
# CPU only (no GPU). Generates the catalog_daniel pools (arXiv:2606.23791 regenerated with their
# MadEvent cards, D51; 100k train / 10k val / 10k test per process, D54): prebuild_recipes.py runs
# one MadEvent run per lhe_pp process, 48 processes at once on one core each (AMP_LHE_CORES=k: 48/k
# at once on k cores), labels the events with the standalone_cpp |M|^2 and writes the pools where the
# recipes find them (val/test frozen in $WORK/datasets, train in $SCRATCH/amp_data_cache), then the
# diagram sidecars data/diagrams/dc_*.diagrams.json (offshell_per_event, the diagram encoders).
# Idempotent and resumable: built pools are cache hits, existing sidecars are skipped, a failed
# process is listed at the end (exit 1) and redone by resubmitting. The spec is DANIEL_SPEC
# (default recipes/catalog_daniel_all.yaml, all 1491 processes), e.g. --env DANIEL_SPEC=recipes/...
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
# one thread per process: thread pools sized to the node count against the per-user process limit
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
SPEC="${DANIEL_SPEC:-recipes/catalog_daniel_all.yaml}"
W="${SLURM_CPUS_PER_TASK:-${DANIEL_WORKERS:-48}}"
# parallel parts: shard DANIEL_SHARD_K of DANIEL_SHARD_N (tools/shard_recipes.py: backend groups kept together, cost-
# balanced), or DANIEL_ONLY=a,b for a few entries; MG5_WORK_DIR/MG5_BIN pick the MadGraph (lxplus: 3.7.0 = CC's)
if [ -n "${DANIEL_SHARD_N:-}${DANIEL_ONLY:-}" ]; then
    N="${DANIEL_SHARD_N:-1}"; K="${DANIEL_SHARD_K:-0}"
    OUT="$SCRATCH/zz_gen_daniel/shard${K}_of_${N}_$(echo "${DANIEL_ONLY:-}" | md5sum | cut -c1-8).yaml"
    mkdir -p "$(dirname "$OUT")"
    python tools/shard_recipes.py "$SPEC" --n "$N" --k "$K" --only "${DANIEL_ONLY:-}" --out "$OUT" || exit 1
    SPEC="$OUT"
fi
echo "MG5 $(sed -n 's/^version = //p' "${MG5_WORK_DIR:-$WORK/mg5amcnlo}/VERSION" 2>/dev/null)"
export AMP_LHE_WORK_DIR="${AMP_LHE_WORK_DIR:-$SCRATCH/lhe_work}"   # MadEvent dirs, deleted per process
echo "host $(hostname)  spec $SPEC  workers $W  lhe cores ${AMP_LHE_CORES:-1}  work $AMP_LHE_WORK_DIR"
python prebuild_recipes.py "$SPEC" --workers "$W" --lhe-cores "${AMP_LHE_CORES:-1}"
rc=$?
python tools/dump_diagrams.py --prefix "${DANIEL_PREFIX:-dc_}" --skip-existing --workers "$W"   # d4_: the 2->4 probes
rc2=$?
echo "prebuild exit $rc, sidecars exit $rc2"
exit $(( rc != 0 || rc2 != 0 ))
