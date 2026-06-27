#!/bin/bash
#SBATCH --job-name=allopts_ab
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=allopts_ab_%j.out
#SBATCH --error=allopts_ab_%j.err
#
# Cumulative A/B for ALL LLoCa hot-path vectorizations.
# Same training as bench_pooling_ab.sh (best trial of scaling_p1ext_nh4_D1e5_t316,
# hp_0266, seed 42), run twice on the same GPU:
#   BASELINE  = all 4 toggles on their OLD path
#   OPTIMIZED = all 4 toggles on their fast default
# Compare the "avg N.NNNNs/iter" line and traintime_hours in each result.json.
#
# Toggles: LLOCA_POOL (loop|vectorized), LLOCA_ATTN_MASK (per_block|per_forward),
#          LLOCA_REG (loop|foreach), LLOCA_PROC_LOSS (loop|vectorized).

module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

DATA=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/data/

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
echo "## BASELINE: all hot-path toggles on the OLD path"
echo "############################################################"
LLOCA_POOL=loop LLOCA_ATTN_MASK=per_block LLOCA_REG=loop LLOCA_PROC_LOSS=loop \
  python run.py "${COMMON[@]}" \
    exp_name=allopts_baseline \
    run_dir="$PWD/runs/allopts_baseline" \
    training.result_path="$PWD/runs/allopts_baseline/result.json"

echo "############################################################"
echo "## OPTIMIZED: all hot-path toggles on the fast default"
echo "############################################################"
LLOCA_POOL=vectorized LLOCA_ATTN_MASK=per_forward LLOCA_REG=foreach LLOCA_PROC_LOSS=vectorized \
  python run.py "${COMMON[@]}" \
    exp_name=allopts_fast \
    run_dir="$PWD/runs/allopts_fast" \
    training.result_path="$PWD/runs/allopts_fast/result.json"

echo "############################################################"
echo "## SUMMARY"
echo "############################################################"
echo "baseline traintime_hours: $(python3 -c "import json;print(json.load(open('$PWD/runs/allopts_baseline/result.json'))['traintime_hours'])" 2>/dev/null)"
echo "fast     traintime_hours: $(python3 -c "import json;print(json.load(open('$PWD/runs/allopts_fast/result.json'))['traintime_hours'])" 2>/dev/null)"
echo "Per-step (training only): grep 'avg' runs/allopts_baseline/out_*.log runs/allopts_fast/out_*.log"
