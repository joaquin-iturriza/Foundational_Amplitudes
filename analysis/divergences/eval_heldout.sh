#!/bin/bash
#SBATCH --job-name=evalho
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=analysis/divergences/evalho_%j.out
#SBATCH --error=analysis/divergences/evalho_%j.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-l2-uugg
export PYTHONDONTWRITEBYTECODE=1
# args: space-separated "run_dir:label" pairs via $RUNS
for spec in $RUNS; do
  rd="${spec%%:*}"; lb="${spec##*:}"
  python analysis/divergences/eval_heldout_uugg.py --run_dir "$rd" --label "$lb" || echo "EVAL FAILED: $rd"
done
echo "DONE evalho"
