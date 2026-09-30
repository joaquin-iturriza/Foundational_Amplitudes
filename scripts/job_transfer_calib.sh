#!/bin/bash
#SBATCH --job-name=tp_calib
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Transfer pilot, horizon calibration: one scratch run of a probe at one D = 10^(k/2) and a fixed
# HP point, at a given horizon. The HPs are passed explicitly (training.lr is required), so the run is
# reproducible from its command line; defaults for the rest: warm-up 0.1, lambda 1e-8, eta_min 1e-8,
# EMA off. The first calibration (analysis/transfer/calib_ee_ddbar.json, which records each run's HPs)
# used the centre of the cell's lr window as the generator had it then.
#     site submit <site|auto> FA scripts/job_transfer_calib.sh -- <probe> <k> <steps> training.lr=<lr> [key=value ...]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
P="$1"; K="$2"; T="$3"; shift 3
case " $* " in *" training.lr="*) ;; *) echo "training.lr=<lr> is required" >&2; exit 2;; esac
TAG="_lr$(printf '%s\n' "$@" | sed -n 's/^training.lr=//p' | head -1)"
# data.* overrides (an A/B arm, e.g. data.target_propagator_tchannel=false) go into the name too
TAG="$TAG$(printf '%s\n' "$@" | sed -n 's/^data\.\([^=]*\)=\(.*\)$/_\1-\2/p' | tr -d '\n')"
CFG=sweep/sweep_config_tp2_scr_${P}_d${K}.yaml   # t-channel factor off (tp2_)
python sweep/run_fixed_hp.py --config "$CFG" --name tp2_calib_${P}_d${K}_t${T}${TAG} --steps "$T" \
  training.cosanneal_warmup_frac=0.1 training.regularization_lambda=1e-8 \
  training.cosanneal_eta_min=1e-8 "$@"
