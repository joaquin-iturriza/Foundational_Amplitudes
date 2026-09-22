#!/bin/bash
#SBATCH --job-name=evalho
#SBATCH --account=lpnhe
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=analysis/divergences/evalho_%j.out
#SBATCH --error=analysis/divergences/evalho_%j.out
set -e
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/worktrees/wt-l2-uugg
export PYTHONDONTWRITEBYTECODE=1
# args: space-separated "run_dir:label" pairs via $RUNS
for spec in $RUNS; do
  rd="${spec%%:*}"; lb="${spec##*:}"
  python analysis/divergences/eval_heldout_uugg.py --run_dir "$rd" --label "$lb" || echo "EVAL FAILED: $rd"
done
echo "DONE evalho"
