#!/bin/bash
#SBATCH --job-name=amp_msstep_lg
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/compare_models/msstep_lgatr_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/compare_models/msstep_lgatr_%j.err

# Re-measure full L-GATr real training ms/step. fused_optimizer=false avoids the
# fused-Adam dtype crash on fresh MuP init (the opt step is sub-ms of lgatr's ~480ms
# step, so this does not affect the timing). Full output is captured (no grep filter)
# so any error is visible. iterations=300 is plenty for a steady-state slope.

cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational

rm -rf compare_models/_msstep_lgatr2
python run.py exp_name=compare_models/_msstep_lgatr2 \
  data.source=recipes \
  data.processes_file=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/recipes/pretrain25.yaml \
  data.preprocess_per_dataset=true data.require_cache=true \
  data.train_subsample=40000 data.eval_subsample=4000 \
  data.data_path=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/data/ \
  training.batchsize=1024 training.iterations=300 \
  training.validate_every_n_steps=100000 training.validate_frac=0.0 \
  training.save_intermediate=false training.get_ID=false \
  training.fused_optimizer=false plot=false \
  model=lgatr_mup model.net.num_blocks=8 model.net.num_heads=2 \
  model.net.hidden_mv_channels=22 model.net.hidden_s_channels=22
echo "LGATR DONE"
