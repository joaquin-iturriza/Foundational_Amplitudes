#!/bin/bash
#SBATCH --job-name=amp_msstep
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/compare_models/msstep_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/compare_models/msstep_%j.err

# Measure REAL training ms/step (overlapped dataloader+compute) for each model, by
# running the actual training loop for a few hundred steps. Validation/plot/save are
# off; a tiny train pool keeps init_data fast (per-step cost is O(batch), independent
# of pool size). Headline number = steady-state slope of the iteration-timing logs
# (plain loop, so num_workers=2 overlap is preserved). LLOCA_PROFILE_STEP is left OFF
# for the headline (its per-segment CUDA syncs serialize data vs compute and overstate
# the step) — a separate short pass at the end captures the data-vs-compute split.

cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational

ITERS=500
COMMON=(
  data.source=recipes
  data.processes_file=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/recipes/pretrain25.yaml
  data.preprocess_per_dataset=true
  data.require_cache=true
  data.train_subsample=40000
  data.eval_subsample=4000
  data.data_path=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/data/
  training.batchsize=1024
  training.iterations=$ITERS
  training.validate_every_n_steps=100000
  training.validate_frac=0.0
  training.save_intermediate=false
  training.get_ID=false
  plot=false
)

declare -A MODEL
MODEL[lloca]="model=lloca model.net.num_blocks=8 model.net.num_heads=8"
MODEL[lgatr]="model=lgatr_mup model.net.num_blocks=8 model.net.num_heads=2 model.net.hidden_mv_channels=22 model.net.hidden_s_channels=22"
MODEL[slim]="model=lgatr_slim model.net.num_blocks=8 model.net.num_heads=2 model.net.hidden_v_channels=52 model.net.hidden_s_channels=104"

for m in lloca lgatr slim; do
  echo "############ measuring $m ############"
  rm -rf compare_models/_msstep_$m
  python run.py exp_name=compare_models/_msstep_$m ${COMMON[@]} ${MODEL[$m]} 2>&1 \
    | grep -E "Finished iteration|Starting to train" \
    | sed "s/^/[$m] /"
done
echo "ALL DONE"
