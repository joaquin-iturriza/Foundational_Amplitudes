#!/bin/bash
#SBATCH --job-name=pool_ab
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=pool_ab_%j.out
#SBATCH --error=pool_ab_%j.err
#
# A/B benchmark for the vectorised event pooling in wrappers.py.
# Runs the SAME training (best trial of scaling_p1ext_nh4_D1e5_t316, hp_0266) twice
# on the same GPU: LLOCA_POOL=loop (old Python loop) vs vectorised (default).
# Compare the "avg N.NNNNs/iter" line and the val_loss in each result.json.

module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

DATA=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/data/

# Identical Hydra overrides for both runs (bash array => no quote/space splitting)
COMMON=(
  local=none
  model=lloca
  data.data_path="${DATA}"
  "data.dataset=[ee_wwz_255-1000GeV_amplitudes,ee_WW_162-1000GeV_amplitudes,ee_ttbar_346-1000GeV_amplitudes,ee_uug_91-1000GeV_amplitudes,ee_uugg_91-1000GeV_amplitudes,ee_aa_10-1000GeV_amplitudes,ee_aaa_10-1000GeV_amplitudes,ee_uu_91-1000GeV_amplitudes]"
  "data.amp_orders=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]"
  data.subsample=12500
  "data.train_test_val=[0.7,0.2,0.1]"
  model.net.num_blocks=8
  model.net.num_heads=4
  training.batchsize=8192
  evaluation.batchsize=8192
  training.get_ID=false
  training.loss_aggregation=geometric_mean
  training.regularization=L2
  training.save_intermediate=false
  training.scheduler=CosineAnnealingLR
  training.validate_frac=0.01
  plot=true
  seed=42
  training.iterations=316
  training.lr=0.0025372844166668
  training.regularization_lambda=4.696959391043144e-09
  training.cosanneal_warmup_frac=0.16410914901643991
  training.cosanneal_eta_min=2.6170438627072204e-09
  training.ema_decay=0.9551155311148614
  training.sampler_alpha_ema=0.49192759795114394
  training.sampler_min_alpha_frac=0.48420716463588176
)

echo "############################################################"
echo "## BASELINE: Python-loop pooling (LLOCA_POOL=loop)"
echo "############################################################"
LLOCA_POOL=loop python run.py "${COMMON[@]}" \
  exp_name=pool_ab_loop \
  run_dir="$PWD/runs/pool_ab_loop" \
  training.result_path="$PWD/runs/pool_ab_loop/result.json"

echo "############################################################"
echo "## VECTORISED pooling (LLOCA_POOL=vectorized, the new default)"
echo "############################################################"
LLOCA_POOL=vectorized python run.py "${COMMON[@]}" \
  exp_name=pool_ab_vec \
  run_dir="$PWD/runs/pool_ab_vec" \
  training.result_path="$PWD/runs/pool_ab_vec/result.json"

echo "############################################################"
echo "## SUMMARY"
echo "############################################################"
echo "loop  result: $(cat "$PWD/runs/pool_ab_loop/result.json" 2>/dev/null)"
echo "vec   result: $(cat "$PWD/runs/pool_ab_vec/result.json" 2>/dev/null)"
echo "Compare the 'avg ...s/iter' lines above (grep for 'avg')."
