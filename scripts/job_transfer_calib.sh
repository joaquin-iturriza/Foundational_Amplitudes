#!/bin/bash
#SBATCH --job-name=tp_calib
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Transfer pilot, horizon calibration: one scratch run of a probe at one D = 10^(k/2), at fixed
# middle-of-the-search HPs (lr = the centre of the cell's batch-scaled window, warm-up 0.1,
# lambda 1e-8, eta_min 1e-8, EMA off) and a long horizon. The best checkpoint's step says how
# many steps that D needs; it sets the horizons of the pilot's HPO. No search: one run per D.
#     site submit <site|auto> FA scripts/job_transfer_calib.sh -- <probe> <k> <steps>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
P="$1"; K="$2"; T="$3"
CFG=sweep/sweep_config_tp_scr_${P}_d${K}.yaml
LR=$(python -c "import yaml; s={x['name']: x for x in yaml.safe_load(open('$CFG'))['search_space']}['training.lr']; print(f\"{(s['low']*s['high'])**0.5:.3g}\")")
python sweep/run_fixed_hp.py --config "$CFG" --name tp_calib_${P}_d${K}_t${T} --steps "$T" \
  training.lr=$LR training.cosanneal_warmup_frac=0.1 training.regularization_lambda=1e-8 \
  training.cosanneal_eta_min=1e-8
