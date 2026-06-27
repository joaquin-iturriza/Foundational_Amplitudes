#!/bin/bash
#SBATCH --job-name=workers_ab
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=workers_ab_%j.out
#SBATCH --error=workers_ab_%j.err
#
# A/B the DataLoader num_workers (lever #6). SUBMIT WITH:  sbatch bench_workers_ab.sh
# (needs a GPU; xformers attention is CUDA-only, so it will crash on a login node.)

module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

DATA=$PWD/data/
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
  training.iterations=200
  training.validate_frac=0.5
  training.save_intermediate=false
  training.get_ID=false
  plot=false
  seed=42
)

for NW in 0 2 4 6; do
  RUNDIR="$PWD/runs/workers_nw${NW}"
  LOG="workers_nw${NW}.log"
  rm -rf "$RUNDIR"                 # avoid "Experiment already exists" abort
  echo "############################################################"
  echo "## num_workers=$NW   (full log -> $LOG)"
  echo "############################################################"
  python run.py "${COMMON[@]}" \
    training.num_workers="$NW" \
    exp_name="workers_nw${NW}" \
    run_dir="$RUNDIR" \
    > "$LOG" 2>&1
  if grep -q "Finished training" "$LOG"; then
    grep "Finished training" "$LOG" | tail -1
  else
    echo ">>> run did NOT finish — last 20 lines of $LOG:"
    tail -20 "$LOG"
  fi
done

echo "############################################################"
echo "## DONE. Lower avg s/iter = better (nw=0 ~0.32s is the baseline)."
echo "############################################################"
