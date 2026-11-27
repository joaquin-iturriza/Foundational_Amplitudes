#!/bin/bash
#SBATCH --job-name=perevent_level
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
# CPU only (llm_compare, D566): analysis/catalog_daniel/perevent_level.py over catalog_daniel's cached train pools and
# diagram sidecars on CC. Its own job, not `site run --job`, for the memory (~2.7M events x 43 features, the CPU block's
# 2G is too small).
#     site submit <site with the pools: lxplus> FA scripts/job_perevent_level.sh --item llm_compare [--env PEL_MODEL=mlp] [--env PEL_SEARCH=12]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
# PEL_ANCHORED=1 (D566 attempt C): --anchored on the memmap base caches PEL_MMAP (<key>/<depth> under
# $SCRATCH/amp_mmap_cache, space-separated), output perevent_anchored_C.json; PEL_ANCHOR=process (attempt D): the
# process-intrinsic anchor, output perevent_anchored_D.json; PEL_ANCHOR=map (attempt E): network / tree anchor maps, _E.json
# PEL_WRITE=1 (step 28): data.level_events' coefficient file from the network + kinematics anchor map,
# analysis/catalog_daniel/data/level_events_E_kin.json (read back with site run cat, committed on trunk); PEL_SEEDS=N
# (step 27): the network over seeds 0..N-1, files _s<N>, with the matched per-process file level_file_B_diag_s<N>.json
if [ -n "${PEL_WRITE:-}" ]; then
  OMP_NUM_THREADS=4 python analysis/catalog_daniel/perevent_level.py --anchored --anchor map --model mlp --write-level-events \
    --level-seeds "${PEL_SEEDS:-3}" \
    --mmap $(for k in $PEL_MMAP; do echo "$SCRATCH/amp_mmap_cache/$k"; done) && echo "RUN_DONE"
  exit $?
fi
if [ -n "${PEL_ANCHORED:-}" ]; then
  A=${PEL_ANCHOR:-global}; J=analysis/catalog_daniel/data/perevent_anchored_$(case $A in process) echo D;; map) echo E;; *) echo C;; esac).json
  OMP_NUM_THREADS=4 python analysis/catalog_daniel/perevent_level.py --anchored --anchor "$A" --model mlp \
    --mmap $(for k in $PEL_MMAP; do echo "$SCRATCH/amp_mmap_cache/$k"; done) && echo "PEL_JSON $(python -c \
    "import json; d=json.load(open('$J')); print(json.dumps(d['summary']))")" \
    && echo "RUN_DONE"
  exit $?
fi
# PEL_SEARCH=N (step 20): N DyHPO trials first, output perevent_level[_mlp]_search.json
M=${PEL_MODEL:-hgb}; S=${PEL_SEARCH:-0}
J=analysis/catalog_daniel/data/perevent_level$([ "$M" = mlp ] && echo _mlp)$([ "$S" -gt 0 ] && echo _search).json
OMP_NUM_THREADS=4 python analysis/catalog_daniel/perevent_level.py --model "$M" --search "$S" && cat "$J" \
  | python -c "import json,sys; print('PEL_JSON', json.dumps(json.load(sys.stdin)))" && echo "RUN_DONE"
