#!/bin/bash
#SBATCH --job-name=cov_v4_regen
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=cov_v4_regen_%j.out
#SBATCH --error=cov_v4_regen_%j.out
#
# CPU only (D130, D136, D137): one shard of catalog_v4's regeneration, the four generation stages
# (recipes/catalog_v4_gen_stage{1,2,3,4}.yaml: trees, one-loop and loop-induced, the 2->5 probes and alpha_s points,
# the energy hold-out on [6, 20] TeV) split by tools/shard_recipes.py into $V4REGEN_N shards (env V4REGEN_K, V4REGEN_N),
# every entry sharing a backend with another in the same shard. prebuild_recipes.py assembles each pool the moment its
# last chunk lands and refuses to start when the shard's pools do not fit the free space. Resumable: built pools are
# cached, a rerun builds only what is missing. Val/test go to $DATA_DIR, train to the cache under $SCRATCH.
# V4REGEN_SKIP (optional, comma-separated entry names) leaves those entries out of this shard after sharding (the
# other entries are unchanged); each is logged as SHARD_SKIP, to be built by a later rerun without the skip.
# V4REGEN_STAGES (optional, default "1 2 3 4") picks the generation stages sharded; "5" builds the 175 tree classes the
# electroweak orders split off (recipes/catalog_v4_gen_stage5.yaml), its shard specs named shard_s5_<k>_of_<n>.yaml.
# V4REGEN_ONLY (optional, comma-separated entry names) builds just those entries and their backend-group mates (run it
# with V4REGEN_N=1 V4REGEN_K=0), each logged as SHARD_ONLY.
# V4REGEN_RECIPES (optional, space-separated recipe files) replaces the stages: each recipe is sharded on its own and
# built in turn, so recipes whose entries share names (catalog_v4_default50/25_train.yaml: the same classes at other
# sampling) never meet in one spec, and shard k of each lands in the same job (no backend built by two jobs at once).
# V4REGEN_DEFAULT_SHARE=1 keeps only the entries with a default-sampled share (sampling.f_default > 0) and their
# backend groups (the mixture arms' pools; the recipe's other pools are catalog_v4's, already built).
# A one-loop entry whose backend build or pole certification fails is skipped with its backend group (the entries
# sharing its backend), each logged as CERT_FAILED <name>: <reason>, and the rest of the shard is built; the job then
# exits nonzero listing them (V4REGEN_CERT_SKIPPED). They stay in the catalog, built by a rerun once fixed.
# When the pole guard refuses a chunk mid-generation, the group's pools and chunk dirs already on disk (any role, any
# group-mate) are moved with their sidecars to $SCRATCH/zz_catalog_v4_regen/quarantine/ and listed in the CERT file
# (#quarantined lines), so the rerun after the fix rebuilds the whole group under one one-loop definition.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
K="${V4REGEN_K:?V4REGEN_K (shard index) not set}"
N="${V4REGEN_N:?V4REGEN_N (number of shards) not set}"
W="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$SCRATCH/zz_catalog_v4_regen"
STAGES="${V4REGEN_STAGES:-1 2 3 4}"
echo "V4REGEN_STAGES $STAGES"
echo "V4REGEN_SKIP ${V4REGEN_SKIP:-none} V4REGEN_ONLY ${V4REGEN_ONLY:-all}"
echo "V4REGEN_RECIPES ${V4REGEN_RECIPES:-stages} V4REGEN_DEFAULT_SHARE ${V4REGEN_DEFAULT_SHARE:-0}"
DS=(); [ "${V4REGEN_DEFAULT_SHARE:-0}" = 1 ] && DS=(--only-default-share)
df -k "$DATA_DIR" | tail -1 | awk '{print "V4REGEN_FREE_KB_START", $4}'
T0=$(date +%s)
RC=0
build() {   # build(<tag> <recipe>...): shard k of the recipes, then prebuild it
    local TAG="$1"; shift
    local SPEC="$SCRATCH/zz_catalog_v4_regen/shard_${TAG}${K}_of_${N}.yaml"
    [ -n "${V4REGEN_ONLY:-}" ] && SPEC="$SCRATCH/zz_catalog_v4_regen/only_${TAG}$(echo "$V4REGEN_ONLY" | md5sum | cut -c1-8).yaml"
    python tools/shard_recipes.py "$@" --n "$N" --k "$K" --skip "${V4REGEN_SKIP:-}" \
        --only "${V4REGEN_ONLY:-}" "${DS[@]}" --out "$SPEC" 2>&1 || return 1
    echo "V4REGEN shard $K of $N ($*): $(grep -c '^- name:' "$SPEC") entries"
    local T1=$(date +%s)
    local CERTF="${SPEC%.yaml}_cert_failed.txt"
    rm -f "$CERTF"
    python prebuild_recipes.py "$SPEC" --workers "$W" --skip-cert-failed "$CERTF"
    local R=$?
    echo "V4REGEN_PART_WALL_S $(( $(date +%s) - T1 )) $*"
    if [ -s "$CERTF" ]; then
        echo "V4REGEN_CERT_SKIPPED $(grep -vc "^#" "$CERTF") entries (#quarantined lines: pools of theirs moved aside):"
        sed 's/^/  /' "$CERTF"
    fi
    return $R
}
if [ -n "${V4REGEN_RECIPES:-}" ]; then
    for r in $V4REGEN_RECIPES; do
        b=$(basename "$r" .yaml); build "${b}_${V4REGEN_DEFAULT_SHARE:+ds_}" "$r" || RC=$?
    done
else
    RECIPES=""; for s in $STAGES; do RECIPES="$RECIPES recipes/catalog_v4_gen_stage${s}.yaml"; done
    TAG=""; [ "$STAGES" = "1 2 3 4" ] || TAG="s$(echo $STAGES | tr -d ' ')_"
    build "$TAG" $RECIPES || RC=$?
fi
df -k "$DATA_DIR" | tail -1 | awk '{print "V4REGEN_FREE_KB_END", $4}'
echo "V4REGEN_WALL_S $(( $(date +%s) - T0 )) cores $W"
[ $RC -eq 0 ] || { echo "V4REGEN_FAILED rc=$RC"; exit $RC; }
echo "V4REGEN_DONE shard $K of $N"
