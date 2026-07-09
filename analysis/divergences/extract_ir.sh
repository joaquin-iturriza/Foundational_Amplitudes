#!/bin/bash
#SBATCH --job-name=div_ir
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=analysis/divergences/ir_%j.out
#SBATCH --error=analysis/divergences/ir_%j.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

# 2->3 / 2->4 gluon channels from the 8-process joint pretrain (NOT fine-tuned):
#   ee->uug  : soft + single-collinear IR   (pretrain test MSE 4.4e-6, well learned)
#   ee->uugg : double soft/collinear IR      (pretrain test MSE 2.3e-4, the hardest set)
python analysis/divergences/extract_ir.py \
  --run_dir runs/pretrain_full_nh8/trial_0271 --tag pretrain8 \
  --subsample 2000000 --max_per_proc 300000 \
  --processes ee_uug_91-1000GeV_amplitudes,ee_uugg_91-1000GeV_amplitudes

echo "=== plots ==="
python analysis/divergences/make_ir.py \
  --npz analysis/divergences/preds_ir_pretrain8_ee_uug_91-1000GeV_amplitudes.npz \
  --label 'pretrained $e^+e^-\to u\bar u g$ (joint, zero-shot)' \
  --out_base analysis/divergences/figs/ir_pretrain_uug
python analysis/divergences/make_ir.py \
  --npz analysis/divergences/preds_ir_pretrain8_ee_uugg_91-1000GeV_amplitudes.npz \
  --label 'pretrained $e^+e^-\to u\bar u g g$ (joint, zero-shot)' \
  --out_base analysis/divergences/figs/ir_pretrain_uugg
echo "DONE"
