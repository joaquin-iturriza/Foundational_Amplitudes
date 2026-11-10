#!/bin/bash
#SBATCH --job-name=ev_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=analysis/divergences/ev_all_%j.out
#SBATCH --error=analysis/divergences/ev_all_%j.out
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd $PROJECT_DIR
# subprocess-isolated re-eval of EVERY arm (leak fix). raw=f0, flatlogm=resampled(fails),
# mix025/050/075=generated mixtures, genflat=f1, l1{base,sigma,oracle}=sigma-reweight.
python analysis/divergences/eval_eeuu_resonance.py \
  --runs_root runs/eeuu_flatlogm --run_prefix ft_ \
  --tags raw,flatlogm,mix025,mix050,mix075,genflat,l1base,l1sigma,l1oracle \
  --out_prefix eeuu_reson_clean_ --summary eeuu_reson_clean_summary.json
echo "DONE eval_all"
