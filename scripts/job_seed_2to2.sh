#!/bin/bash
#SBATCH --job-name=seed_2to2
#SBATCH --cpus-per-task=4
#SBATCH --time=01:40:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Seed spread of the 2->2 solo references (solo16k): one process, one seed, each of the seven step counts at that
# cell's best trial's HPs (analysis/catalog_v2/solo16k_besthp_2to2.json: lr, warm-up), everything else the cell's
# config. The sweeps ran seed 42; these add seeds 1, 2, 3. Non-HP axis (seed), same HPs: no search.
#     site submit <site|auto> FA scripts/job_seed_2to2.sh -- <process> <seed>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
P="$1"; SEED="$2"
for T in 63 125 250 500 1000 2000 4000; do
  HP=$(python -c "import json; h=json.load(open('analysis/catalog_v2/solo16k_besthp_2to2.json'))['$P|$T']['hp']; print(' '.join(f'{k}={v}' for k, v in h.items()))")
  python sweep/run_fixed_hp.py --config sweep/sweep_config_solo16k_t${T}_${P}.yaml --name solo16kseed_t${T}_${P}_s${SEED} --steps $T $HP seed=$SEED
done
