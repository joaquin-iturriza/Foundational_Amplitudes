#!/usr/bin/env bash
# port_cluster.sh — rewrite the cluster-specific lines of the tree for one cluster.
#
# The trunk (ccin2p3) and the Jean Zay branch (jeanzay) carry the same code; they
# differ ONLY by the rule table below: project/scratch paths, the python env lines
# in job scripts and sweep `setup_commands`, and the SLURM directives
# (partition/account/qos/gres/mem). Directives that have no Jean Zay counterpart
# (`--qos`, `--mem`, `--mem-per-cpu`, and the `qos:`/`gres:`/`mem:` keys of a
# sweep's cluster block, in shell, YAML or a Python dict) are not deleted on the
# Jean Zay side: they are kept as `#PORT <original line>` comment lines, which
# sbatch, YAML and Python all ignore, and the reverse port uncomments them. The
# round trip is therefore lossless (a 64G request comes back as 64G) and both
# directions are idempotent. Syncing the two branches is
#
#   git merge ccin2p3            # on jeanzay: take every change from the trunk
#   scripts/port_cluster.sh --to jeanzay
#   git commit -am "jeanzay: cluster overlay re-applied"
#
# and the reverse direction ports a file written on Jean Zay back to the trunk
# (a file born on Jean Zay carries no #PORT lines, so add the CC-IN2P3 `--mem`
# and `--qos` by hand once; from then on they travel with the file).
# Prose (CLAUDE.md, docs/, README) is left alone: it names both clusters on
# purpose and is maintained by hand.
#
# Usage:
#   scripts/port_cluster.sh --to jeanzay|ccin2p3 [--check] [path ...]
#     no paths      every tracked file that carries a reference to the other cluster
#     --check       print the files that would change, touch nothing
#     --stdin       filter stdin to stdout (no git needed; used by the self-test)
set -euo pipefail

TO=""; CHECK=0; STDIN=0; FILES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --to) TO="$2"; shift 2 ;;
    --check) CHECK=1; shift ;;
    --stdin) STDIN=1; shift ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) FILES+=("$1"); shift ;;
  esac
done
case "$TO" in jeanzay|ccin2p3) ;; *) echo "port_cluster.sh: --to jeanzay|ccin2p3 is required" >&2; exit 2 ;; esac

# --- the rule table -----------------------------------------------------------
CC_ROOT='/sps/lpnhe/jiturrizaramirez01'
CC_PROJ="$CC_ROOT/Foundational_Amplitudes"
CC_SCRATCH="$CC_ROOT/tmp"
JZ_ROOT='/lustre/fswork/projects/rech/itg/ulm49ia'
JZ_PROJ="$JZ_ROOT/Foundational_Amplitudes"
JZ_SCRATCH='/lustre/fsn1/projects/rech/itg/ulm49ia'
JZ_CONDA="$JZ_ROOT/conda/envs/foundational"
JZ_MODULE='module load anaconda-py3/2023.09 \&\& source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh'
SP='[[:space:]]'

# sed -E programs. Order matters: the specific env lines go before the generic path rule.
to_jeanzay() {
  cat <<EOF
s#source $CC_PROJ/\.venv/bin/activate#$JZ_MODULE#g
s#source $CC_PROJ/scripts/env_ccin2p3\.sh#conda activate $JZ_CONDA#g
s#$CC_PROJ/\.venv/bin/#$JZ_CONDA/bin/#g
s#\\\$PROJ/\.venv/bin/python#\$WORK/conda/envs/foundational/bin/python#g
s#$CC_SCRATCH#$JZ_SCRATCH#g
s#$CC_ROOT#$JZ_ROOT#g
s#^($SP*)\#SBATCH --partition=gpu_v100$SP*\$#\1\#SBATCH --partition=gpu_p2#
s#^($SP*)\#SBATCH --account=lpnhe$SP*\$#\1\#SBATCH --account=itg@v100#
s#^($SP*)\#SBATCH --gres=gpu:v100:1$SP*\$#\1\#SBATCH --gres=gpu:1#
s#^($SP*)\#SBATCH --partition=htc$SP*\$#\1\#SBATCH --partition=prepost#
s#^($SP*)(\#SBATCH --(qos|mem|mem-per-cpu)=[^ ]+)$SP*\$#\1\#PORT \2#
s#^($SP*)partition: gpu_v100 +\# CC-IN2P3 V100 32GB nodes \(gpu_h100 exists too; untested here\)$SP*\$#\1partition: gpu_p2          \# V100 32GB (15k hours); try gpu_p13 + itg@a100 for A100 (5k hours)#
s#^($SP*)partition: gpu_v100($SP*(\#.*)?)\$#\1partition: gpu_p2\2#
s#^($SP*)account: lpnhe$SP*\$#\1account: itg@v100#
s#^($SP*)((qos|gres|mem): [^ ]+($SP*\#.*)?)\$#\1\#PORT \2#
s#"account", "lpnhe"#"account", "itg@v100"#g
s#"account": "lpnhe"#"account": "itg@v100"#g
s#^(.*)"partition": "gpu_v100", "qos": "gpu", "gres": "gpu:v100:1", "mem": "32G"(.*)\$#\1"partition": "gpu_p2"\2  \# PORT: qos/gres/mem#
s#"partition": "gpu_v100"#"partition": "gpu_p2"#g
s#^($SP*)("(qos|gres|mem)": "[^"]+",)$SP*\$#\1\#PORT \2#
s#^($SP*)partition = "gpu_v100"$SP*\#.*\$#\1partition = "gpu_p2l" if use_32gb else "gpu_p2"#
s#^($SP*)PARTITION$SP*= "gpu_v100".*\$#\1PARTITION    = "gpu_p2"       \# V100 32GB; nh=4 ~6.8GB, nh=8 ~12GB at BS=16384 -> fits#
s#^($SP*)gpu$SP*= "gpu_v100".*\$#\1gpu     = "gpu_p2l" if needs_32gb(nh, bs_new) else "gpu_p2"#
s#cluster\.get\('partition', 'gpu_v100'\)#cluster.get('partition', 'gpu_p2')#g
s#cluster\.get\('account', 'lpnhe'\)#cluster.get('account', 'itg@v100')#g
s#^\# htc partition \(CPU; no GPU hours\)\. Self-skips cached\.\$#\# prepost partition (CPU billed at weight 0; no GPU hours). Self-skips cached.#
EOF
}

to_ccin2p3() {
  cat <<EOF
s#^($SP*)\#PORT (.*)\$#\1\2#
s#$JZ_MODULE#source $CC_PROJ/.venv/bin/activate#g
s#module purge; module load anaconda-py3/2023.09#source $CC_PROJ/.venv/bin/activate#g
s#module load anaconda-py3/2023.09( 2>/dev/null [|][|] true)?#source $CC_PROJ/.venv/bin/activate#g
\#^$SP*(- )?"?source (/gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda[.]sh|"[$][(]conda info --base[)]/etc/profile[.]d/conda[.]sh")"?,?$SP*\$#d
s#conda activate $JZ_CONDA#source $CC_PROJ/scripts/env_ccin2p3.sh#g
s#$JZ_CONDA/bin/#$CC_PROJ/.venv/bin/#g
s#\\\$WORK/conda/envs/foundational/bin/python#\$PROJ/.venv/bin/python#g
s#$JZ_SCRATCH#$CC_SCRATCH#g
s#$JZ_ROOT#$CC_ROOT#g
s#^($SP*)\#SBATCH --partition=gpu_p2$SP*\$#\1\#SBATCH --partition=gpu_v100#
s#^($SP*)\#SBATCH --account=itg@v100$SP*\$#\1\#SBATCH --account=lpnhe#
s#^($SP*)\#SBATCH --gres=gpu:1$SP*\$#\1\#SBATCH --gres=gpu:v100:1#
s#^($SP*)\#SBATCH --partition=prepost$SP*\$#\1\#SBATCH --partition=htc#
s#^($SP*)partition: gpu_p2 +\# V100 32GB \(15k hours\); try gpu_p13 \+ itg@a100 for A100 \(5k hours\)$SP*\$#\1partition: gpu_v100        \# CC-IN2P3 V100 32GB nodes (gpu_h100 exists too; untested here)#
s#^($SP*)partition: gpu_p2($SP*(\#.*)?)\$#\1partition: gpu_v100\2#
s#^($SP*)account: itg@v100$SP*\$#\1account: lpnhe#
s#"account", "itg@v100"#"account", "lpnhe"#g
s#"account": "itg@v100"#"account": "lpnhe"#g
s#^(.*)"partition": "gpu_p2"(.*)  \# PORT: qos/gres/mem\$#\1"partition": "gpu_v100", "qos": "gpu", "gres": "gpu:v100:1", "mem": "32G"\2#
s#"partition": "gpu_p2"#"partition": "gpu_v100"#g
s#^($SP*)partition = "gpu_p2l" if use_32gb else "gpu_p2"$SP*\$#\1partition = "gpu_v100"   \# every CC-IN2P3 V100 is the 32 GB part; use_32gb is moot here#
s#^($SP*)PARTITION$SP*= "gpu_p2".*\$#\1PARTITION    = "gpu_v100"     \# every CC-IN2P3 V100 is the 32 GB part#
s#^($SP*)gpu$SP*= "gpu_p2l" if needs_32gb\(nh, bs_new\) else "gpu_p2"$SP*\$#\1gpu     = "gpu_v100"   \# every CC-IN2P3 V100 is the 32 GB part#
s#cluster\.get\('partition', 'gpu_p2'\)#cluster.get('partition', 'gpu_v100')#g
s#cluster\.get\('account', 'itg@v100'\)#cluster.get('account', 'lpnhe')#g
s#^\# prepost partition \(CPU billed at weight 0; no GPU hours\)\. Self-skips cached\.\$#\# htc partition (CPU; no GPU hours). Self-skips cached.#
EOF
}

if [ "$TO" = jeanzay ]; then
  PROG=$(to_jeanzay)
  MARK=("$CC_ROOT" "gpu_v100" "account=lpnhe" "account: lpnhe" "'lpnhe'" '"lpnhe"' "partition=htc" "#SBATCH --qos" "#SBATCH --mem" "^ *qos: " "^ *gres: " "^ *mem: [0-9]" '^ *"\(qos\|gres\|mem\)": ')
else
  PROG=$(to_ccin2p3)
  MARK=("$JZ_ROOT" "gpu_p2" "itg@v100" "anaconda-py3" "partition=prepost" "#PORT " "# PORT: ")
fi

if [ "$STDIN" = 1 ]; then sed -E "$PROG"; exit 0; fi

# --- file selection ------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ ${#FILES[@]} -eq 0 ]; then
  args=(); for m in "${MARK[@]}"; do args+=(-e "$m"); done
  mapfile -t FILES < <(git grep -l -I "${args[@]}" -- . \
    ':!CLAUDE.md' ':!README*' ':!docs/*' ':!scripts/port_cluster.sh' ':!*.tex' ':!*.md' ':!sites/activate.sh' ':!*.json' ':!sites/sites.yaml' ':!siteconf.py')
fi
[ ${#FILES[@]} -eq 0 ] && { echo "port_cluster.sh: nothing to port"; exit 0; }

n=0
for f in "${FILES[@]}"; do
  [ -f "$f" ] || continue
  if [ "$CHECK" = 1 ]; then
    if ! sed -E "$PROG" "$f" | cmp -s - "$f"; then echo "$f"; n=$((n+1)); fi
  else
    tmp=$(mktemp); sed -E "$PROG" "$f" > "$tmp"
    if ! cmp -s "$tmp" "$f"; then cat "$tmp" > "$f"; n=$((n+1)); fi
    rm -f "$tmp"
  fi
done
echo "port_cluster.sh --to $TO: $n file(s) $([ "$CHECK" = 1 ] && echo 'would change' || echo 'rewritten')"
