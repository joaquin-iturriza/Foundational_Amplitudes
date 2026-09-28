#!/bin/bash
#SBATCH --job-name=fixedhp_old
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# One process of the old from-scratch set (tab:scaling) on its old pool with the catalog solo setup, at that
# set's best HPs per step count (analysis/catalog_v2/solo_full_old_besthp.json): the A/B cheap shortcut, no sweep.
# Runs the five old step counts in sequence.   site submit <site|auto> FA scripts/job_fixed_hp_old.sh -- <process>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
P="$1"
for T in 4 126 400 1265 4000; do
  HP=$(python -c "import json; h=json.load(open('analysis/catalog_v2/solo_full_old_besthp.json'))['$P|$T']['hp']; print(' '.join(f'{k}={v}' for k, v in h.items() if k != 'training.ema_decay'))")
  python sweep/run_fixed_hp.py --config sweep/sweep_config_solo16kflat_t500_${P}.yaml --name solo16kflatold_t${T}_${P} --steps $T $HP
done
