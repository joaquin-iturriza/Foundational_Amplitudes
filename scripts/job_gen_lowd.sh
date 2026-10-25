#!/bin/bash
#SBATCH --job-name=gen_lowd
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# The small-D study's DyHPO round (D120, D123): generate the sweeps sweep/gen_lowd_dyhpo.py has not generated yet
# (each about 30 s of DyHPO set-up, too long for one login-node call). No submission.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
python3 sweep/gen_lowd_dyhpo.py --generate
for m in lpft l2sp freeze; do echo "GEN $m $(ls -d sweeps/tp3_lowd_${m}_* | wc -l) sweeps $(ls sweeps/tp3_lowd_${m}_*/jobs/*.sh | wc -l) trial scripts"; done
