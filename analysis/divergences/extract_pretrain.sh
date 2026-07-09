#!/bin/bash
#SBATCH --job-name=div_pretrain
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=analysis/divergences/pretrain_%j.out
#SBATCH --error=analysis/divergences/pretrain_%j.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

# Well-learned 2->2 processes from the 8-process joint pretrain (NOT fine-tuned),
# with distinct divergence structures:
#   ee->gamma gamma : t/u-channel COLLINEAR peaks (cos->+-1)   [MSE 9e-10]
#   ee->W+ W-       : t-channel nu exchange, forward peak      [MSE 2.5e-8]
#   ee->u ubar      : s-channel, no angular divergence         [MSE 6e-9]
# subsample 2000000 (> file size) => full data => exact pretrain stats/scaling.
python analysis/divergences/extract_pretrain.py \
  --run_dir runs/pretrain_full_nh8/trial_0271 --tag pretrain8 \
  --subsample 2000000 --max_per_proc 300000 \
  --processes ee_aa_10-1000GeV_amplitudes,ee_WW_162-1000GeV_amplitudes,ee_uu_91-1000GeV_amplitudes

echo "=== plots ==="
declare -A LBL=(
  [ee_aa_10-1000GeV_amplitudes]='pretrained $e^+e^-\to\gamma\gamma$ (joint, zero-shot)'
  [ee_WW_162-1000GeV_amplitudes]='pretrained $e^+e^-\to W^+W^-$ (joint, zero-shot)'
  [ee_uu_91-1000GeV_amplitudes]='pretrained $e^+e^-\to u\bar u$ (joint, zero-shot)'
)
declare -A SHORT=(
  [ee_aa_10-1000GeV_amplitudes]=aa
  [ee_WW_162-1000GeV_amplitudes]=ww
  [ee_uu_91-1000GeV_amplitudes]=uu
)
for p in ee_aa_10-1000GeV_amplitudes ee_WW_162-1000GeV_amplitudes ee_uu_91-1000GeV_amplitudes; do
  npz=analysis/divergences/preds_pretrain8_${p}.npz
  s=${SHORT[$p]}
  python analysis/divergences/make_plots.py --npz "$npz" --label "${LBL[$p]}" \
    --out_base analysis/divergences/figs/phase_space_pretrain_${s} --split all
  python analysis/divergences/make_3d.py --npz "$npz" --label "${LBL[$p]}" \
    --out_base analysis/divergences/figs/phase_space_3d_pretrain_${s}
done
echo "DONE"
