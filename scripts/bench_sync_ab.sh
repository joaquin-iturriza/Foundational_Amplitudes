#!/bin/bash
#SBATCH --job-name=sync_ab
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=sync_ab_%j.out
#SBATCH --error=sync_ab_%j.err
#
# A/B the per-step host/device sync cleanup (optimizations #1-#5).
#   LLOCA_SYNC=blocking  -> original: 4 syncs/step + per-forward ptr.tolist()
#   LLOCA_SYNC=deferred  -> new (default): 1 fused sync/step + CPU-side seq_lens
# Everything else is left on the FAST path so this isolates the sync change.
# Same training config as bench_isolate_ab.sh (hp_0266, seed 42), so traintime
# is comparable to the other isolated benchmarks.

module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

DATA=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/data/

# ---- microbenchmark first: equivalence + per-call timing on synthetic/real tensors ----
echo "################ verify_speedups.py (incl. #4 sync) ################"
python verify_speedups.py

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
  echo "## SYNC A/B: $tag   ($*)"
  echo "############################################################"
  env "$@" python run.py "${COMMON[@]}" \
    exp_name="sync_${tag}" \
    run_dir="$PWD/runs/sync_${tag}" \
    training.result_path="$PWD/runs/sync_${tag}/result.json"
}

run_one blocking  LLOCA_SYNC=blocking
run_one deferred  LLOCA_SYNC=deferred

echo "############################################################"
echo "## SUMMARY (traintime_hours; lower = faster)"
echo "############################################################"
for r in sync_blocking sync_deferred; do
  printf "%-16s " "$r"
  python3 -c "import json;print(json.load(open('$PWD/runs/$r/result.json'))['traintime_hours'])" 2>/dev/null \
    || echo "(no result.json)"
done
echo "Per-step avg (train only): grep 'avg' sync_ab_*.err"
