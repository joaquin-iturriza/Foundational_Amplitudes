#!/bin/bash
#SBATCH --job-name=ho_eval_addback
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=analysis/divergences/ho_eval_addback_%j.out
#SBATCH --error=analysis/divergences/ho_eval_addback_%j.out
#SBATCH --gres=gpu:1
#
# Regenerate the BASE ee->uug add-back eval (heldout_eval_ft_f<tag>.npz) for every add-back
# fraction. This is the companion of eval_heldout_studies.sh (uugg / soft / collinear), which
# was recorded while the base uug arm never was -- so when heldout_eval_ft_f100.npz was later
# overwritten by an unrelated eval (keys shrank to pred/true/sigma/y_min, losing `cut` and
# `x_gmin`) there was no recorded command to rebuild it, and both addback_curve and
# heldout_resid_f100 became unreproducible.
#
# All five tags are re-run together, not just f100, so every npz and the summary json share one
# provenance instead of mixing a July f000 with a fresh f100.
set -e
module purge
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"

python analysis/divergences/eval_heldout.py \
  --tags 000,005,015,050,100 \
  --heldout analysis/divergences/uug_heldtest.npz \
  --run_prefix ft_f --out_prefix heldout_eval_ft_f \
  --summary heldout_eval_summary.json --ckpt model_run0_best.pt

echo "DONE ho_eval_addback"
