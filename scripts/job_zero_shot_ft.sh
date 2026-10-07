#!/bin/bash
#SBATCH --job-name=zero_shot_ft
#SBATCH --cpus-per-task=3
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=zero_shot_ft_%j.out
#SBATCH --error=zero_shot_ft_%j.out
#
# Zero-shot loss of every pretraining on every transfer probe (tools/zero_shot_ft.py): one fine-tune run dir per
# (parent, probe) at D = 10^4 found on this site (the 8k grid's tp3_<parent>fte cells, star arms and 32k sweeps excluded).

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
# with `-- --parent <pretraining run dir>`: that pretraining on the twelve probes, through one rung 9 cell per probe
# (its config fixes the probe and its pool)
PAT="runs/tp3_uu64fte_*_d8 runs/tp3_r[1-9]fte_*_d8 runs_from_lxplus/tp3_r[1-9]fte_*_d8"
if [ "${1:-}" = "--parent" ]; then
  # any D of a probe scores on the same validation split: the largest D whose run dir is on this site
  PAT=$(for p in ee_ddbar ee_nnbar ee_ttbar ee_WW ee_dd_nlo ee_bb_nlo ee_Za ud_ud uubar_gg uubar_Zg uubar_Zgg uubar_Zggg; do
          for k in 8 7 6 5 4 3 2; do
            t=$(ls -d runs/tp3_r9fte_${p}_d$k/trial_* 2>/dev/null | while read t; do [ -f $t/config.yaml ] && [ -f $t/data_stats.json ] && { echo $t; break; }; done)
            [ -n "$t" ] && { echo runs/tp3_r9fte_${p}_d$k; break; }
          done; done)
fi
dirs=$(for s in $PAT; do
         [ -d "$s" ] || continue
         for t in "$s"/trial_*; do [ -f "$t/config.yaml" ] && [ -f "$t/data_stats.json" ] && { echo "$t"; break; }; done
       done)
echo "ZERO_SHOT_DIRS $(echo $dirs | wc -w)"
python tools/zero_shot_ft.py "$@" $dirs
