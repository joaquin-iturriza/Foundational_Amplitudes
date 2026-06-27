#!/bin/bash
#SBATCH --job-name=compute_ab
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=compute_ab_%j.out
#SBATCH --error=compute_ab_%j.err
#
# A/B the compute knobs: #9 fused optimizer and #8 TF32.
#   - This runs on V100 (gpu_p2). TF32 is a NO-OP on V100 (Volta has no TF32), so the
#     tf32/both cells just confirm it does no harm; they will ~equal baseline/fused.
#     => this job measures the FUSED-OPTIMIZER win. TF32 can only be measured on A100.
# Submit with:  sbatch bench_compute_ab.sh
# Cells: baseline (both off) / +fused / +tf32 / +both. num_workers=2 so the step is
# compute-bound (otherwise dataloading hides the compute deltas).

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
  training.num_workers=2
  training.iterations=300
  training.validate_frac=0.5
  training.save_intermediate=false
  training.get_ID=false
  plot=false
  seed=42
)

run_cell () {   # $1=tag  $2..=extra overrides
  local tag="$1"; shift
  local RUNDIR="$PWD/runs/compute_${tag}"
  local LOG="compute_${tag}.log"
  rm -rf "$RUNDIR"
  echo "############################################################"
  echo "## $tag   ($*)"
  echo "############################################################"
  python run.py "${COMMON[@]}" "$@" \
    exp_name="compute_${tag}" run_dir="$RUNDIR" \
    > "$LOG" 2>&1
  if grep -q "Finished training" "$LOG"; then
    grep "Finished training" "$LOG" | tail -1
  else
    echo ">>> did NOT finish — last 20 lines of $LOG:"; tail -20 "$LOG"
  fi
}

run_cell baseline  training.fused_optimizer=false training.allow_tf32=false
run_cell fused     training.fused_optimizer=true  training.allow_tf32=false
run_cell tf32      training.fused_optimizer=false training.allow_tf32=true
run_cell both      training.fused_optimizer=true  training.allow_tf32=true

echo "############################################################"
echo "## DONE. Compare avg s/iter; also eyeball final val loss in each"
echo "##       compute_*.log to confirm TF32 didn't move it materially."
echo "############################################################"
