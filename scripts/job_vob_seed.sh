#!/bin/bash
#SBATCH --job-name=vob_seed
#SBATCH --cpus-per-task=4
#SBATCH --time=01:27:00
#SBATCH --output=vob_seed_%j.out
#SBATCH --error=vob_seed_%j.out
#SBATCH --gres=gpu:1
# Wall-time target: 8000 steps at 0.372 s/step (batch 16384, 50k events, V100; vob_tt_V wave 1) = 50 min, plus the
# HETEROSC sigma stage and the evaluations: 0.9 h.
#
# Seed runs of the current-defaults V/B test (plan virtual_over_born, D366): one fixed-HP run (sweep/run_fixed_hp.py,
# the same command a sweep trial builds, sigma stage included) of a V sweep's best config, with the model seed and
# the target (V, or V/B: data.target_over_born) set in a copy of the sweep config; data.seed stays 42.
#     site submit ccin2p3 FA scripts/job_vob_seed.sh -- <sweep config> <name> <seed> <true|false> key=value ...
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
CFG="$1"; NAME="$2"; SEED="$3"; VB="$4"; shift 4
TMP="$SCRATCH/zz_vob_seed_${NAME}.yaml"
python - "$CFG" "$TMP" "$NAME" "$SEED" "$VB" <<'EOF' || exit 1
import sys, yaml
src, dst, name, seed, vb = sys.argv[1:]
c = yaml.safe_load(open(src))
c["sweep_name"] = name
c["fixed_params"]["seed"] = int(seed)
c["fixed_params"]["data.target_over_born"] = vb
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
EOF
python sweep/run_fixed_hp.py --config "$TMP" --name "$NAME" --steps 8000 "$@" && echo "VOB_SEED_DONE $NAME"
