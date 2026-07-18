#!/bin/bash
#SBATCH --job-name=ev_l2
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=analysis/divergences/ev_l2_%j.out
#SBATCH --error=analysis/divergences/ev_l2_%j.out
# Eval every arm x round mu ckpt on the common RAMBO test -> logflat A/B + per-round migration.
# Usage: sbatch eval_l2_online.sh <tag> <rounds>
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-l2-gen
cd "$WT"
TAG=${1:-run}
ROUNDS=${2:-3}
TAGS=""
for arm in static l2 oracle; do
  for ((r=0; r<ROUNDS; r++)); do TAGS="$TAGS,${TAG}_${arm}_mu_r${r}"; done
done
TAGS=${TAGS#,}
python analysis/divergences/eval_eeuu_resonance.py \
  --runs_root "$WT/runs/eeuu_l2" --run_prefix "" --tags "$TAGS" \
  --test /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/data_test_eeuu/ee_uu_91-1000GeV_amplitudes.npy \
  --out_prefix "l2_online_${TAG}_" --summary "l2_online_summary.json"
echo "DONE eval_l2_online $TAG"
