#!/bin/bash
#SBATCH --job-name=pretrain_iter_time
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=pretrain_iter_time_%j.out
#SBATCH --error=pretrain_iter_time_%j.err
#
# Real-conditions iteration-time benchmark for the full-dataset pretraining sweep.
#   - FULL datasets, NO subsampling (data.subsample=null -> all 8 x 12500 = 100k events).
#   - BS = 2**14 = 16384, num_workers=2 (the established fast-path setup).
#   - Measures avg s/iter for nh=4 and nh=8 so we can size t_steps for a ~10h budget.
# This supersedes the stale STEP_TIME_MS table (pre speed-optimisation) used by
# generate_pretraining_scaling_sweeps.py.
#
# Submit with:  sbatch bench_pretrain_iter_time.sh
# Both cells fit on gpu_p2 (V100 32GB): nh=4 ~6.8GB, nh=8 ~12GB peak at BS=16384.

module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

DATA=$PWD/data/
ITERS=300
BS=16384

COMMON=(
  local=none
  model=lloca
  data.data_path="${DATA}"
  "data.dataset=[ee_wwz_255-1000GeV_amplitudes,ee_WW_162-1000GeV_amplitudes,ee_ttbar_346-1000GeV_amplitudes,ee_uug_91-1000GeV_amplitudes,ee_uugg_91-1000GeV_amplitudes,ee_aa_10-1000GeV_amplitudes,ee_aaa_10-1000GeV_amplitudes,ee_uu_91-1000GeV_amplitudes]"
  "data.amp_orders=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]"
  data.subsample=null
  "data.train_test_val=[0.7,0.2,0.1]"
  model.net.num_blocks=8
  training.batchsize=${BS}
  evaluation.batchsize=8192
  training.num_workers=2
  training.iterations=${ITERS}
  training.validate_frac=0.5
  training.save_intermediate=false
  training.get_ID=false
  plot=false
  evaluate=false
  seed=42
)

run_cell () {   # $1=num_heads
  local nh="$1"
  local RUNDIR="$PWD/runs/itertime_nh${nh}"
  local LOG="$PWD/itertime_nh${nh}.log"
  rm -rf "$RUNDIR"
  echo "############################################################"
  echo "## nh=${nh}  BS=${BS}  iters=${ITERS}  (full datasets)"
  echo "############################################################"
  python run.py "${COMMON[@]}" \
    model.net.num_heads="${nh}" \
    exp_name="itertime_nh${nh}" run_dir="$RUNDIR" \
    > "$LOG" 2>&1
  if grep -q "Finished training" "$LOG"; then
    grep "Finished training" "$LOG" | tail -1
  else
    echo ">>> nh=${nh} did NOT finish — last 25 lines of $LOG:"; tail -25 "$LOG"
  fi
}

run_cell 4
run_cell 8

echo "############################################################"
echo "## SUMMARY — avg s/iter and t_steps for a ~10h training budget"
echo "############################################################"
python - <<'PY'
import re, pathlib
BUDGET_H = 10.0          # target training wall-clock per run
for nh in (4, 8):
    log = pathlib.Path(f"itertime_nh{nh}.log")
    if not log.exists():
        print(f"nh={nh}: no log"); continue
    m = None
    for line in log.read_text().splitlines():
        mm = re.search(r"avg ([0-9.]+)s/iter", line)
        if mm:
            m = float(mm.group(1))
    if m is None:
        print(f"nh={nh}: no 'avg s/iter' found"); continue
    t_steps = int(BUDGET_H * 3600 / m)
    print(f"nh={nh}:  {m:.4f} s/iter  ->  t_steps(~{BUDGET_H:.0f}h) = {t_steps}  "
          f"(at {m:.4f}s/iter that is {t_steps*m/3600:.2f}h of pure training)")
PY
