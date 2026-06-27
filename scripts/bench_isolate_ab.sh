#!/bin/bash
#SBATCH --job-name=iso_ab
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=iso_ab_%j.out
#SBATCH --error=iso_ab_%j.err
#
# Isolate the marginal speedup of each LLoCa hot-path optimization.
# Same training as the earlier benchmarks (hp_0266, seed 42). Each run turns ON
# exactly ONE optimization and leaves the other three on their OLD (baseline) path,
# so each result is directly comparable to the two you already have:
#     baseline (all old)  = runs/allopts_baseline/result.json (~0.134 h)
#     pooling only        = runs/pool_ab_vec/result.json
#
# Toggles: LLOCA_POOL(loop|vectorized) ATTN_MASK(per_block|per_forward)
#          REG(loop|foreach) PROC_LOSS(loop|vectorized)

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

run_one () {   # $1=tag  $2..=env assignments
  local tag="$1"; shift
  echo "############################################################"
  echo "## ISOLATED: $tag   ($*)"
  echo "############################################################"
  env "$@" python run.py "${COMMON[@]}" \
    exp_name="iso_${tag}" \
    run_dir="$PWD/runs/iso_${tag}" \
    training.result_path="$PWD/runs/iso_${tag}/result.json"
}

# Each: the named optimization FAST, the other three on OLD/baseline.
run_one attnmask  LLOCA_ATTN_MASK=per_forward LLOCA_POOL=loop        LLOCA_REG=loop    LLOCA_PROC_LOSS=loop
run_one reg       LLOCA_REG=foreach           LLOCA_POOL=loop        LLOCA_ATTN_MASK=per_block LLOCA_PROC_LOSS=loop
run_one procloss  LLOCA_PROC_LOSS=vectorized  LLOCA_POOL=loop        LLOCA_ATTN_MASK=per_block LLOCA_REG=loop

echo "############################################################"
echo "## SUMMARY (traintime_hours; compare to baseline ~0.134 h, pooling-only)"
echo "############################################################"
for r in allopts_baseline pool_ab_vec iso_attnmask iso_reg iso_procloss; do
  printf "%-18s " "$r"
  python3 -c "import json;print(json.load(open('$PWD/runs/$r/result.json'))['traintime_hours'])" 2>/dev/null \
    || echo "(no result.json)"
done
echo "Per-step (train only): grep 'avg' runs/iso_*/out_*.log"
