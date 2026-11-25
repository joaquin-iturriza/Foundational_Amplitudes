#!/bin/bash
#SBATCH --job-name=cbs_upd_zng
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=cbs_upd_zng_%j.out
#SBATCH --error=cbs_upd_zng_%j.out
# Wall-time target: 1953 steps x 0.036-0.074 s/step (A100/H100, sweep_config_tp_zng_d1e4_adam_b2048_zgggg.yaml) plus
# the noise reads (32 chunk gradients every 10 steps, 62 batch forwards every 50) and start-up: 0.3 h.
#
# critical_batch D560: the D507 Adam batch-2048 chosen trial (hp1152) on q q~ -> Z+4g at 10^4 events, rerun at its HPs
# (sweep/run_fixed_hp.py) with the gradient noise measurement and the update-direction terms on: 32 chunks of 64 events
# (b = 64 ... 1024), records in <run>/noise_scale.json. smoke: 50 steps, every read every 10 steps (D395).
#     site submit <site> FA scripts/job_cbs_update_zng.sh --item critical_batch -- <full|smoke>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
MODE="${1:?full or smoke}"
if [ "$MODE" = smoke ]; then STEPS=50; EVERY=10; NAME=cbs_upd_zng_adam_b2048_smoke; else STEPS=1953; EVERY=50; NAME=cbs_upd_zng_adam_b2048; fi
python sweep/run_fixed_hp.py --config sweep/sweep_config_tp_zng_d1e4_adam_b2048_zgggg.yaml --name "$NAME" --steps "$STEPS" \
    training.lr=0.004179596 training.regularization_lambda=6.567441e-07 training.cosanneal_warmup_frac=0.1320452 \
    training.cosanneal_eta_min=4.333021e-10 training.ema_decay=0.9976223 ema=false seed=42 \
    find_BS=true BS_finding.every=10 BS_finding.chunks=32 BS_finding.update=true BS_finding.update_every=$EVERY \
    && ls "$PROJECT_DIR/runs/$NAME/noise_scale.json" && echo "CBS_UPD_ZNG_DONE $NAME"
