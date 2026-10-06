#!/bin/bash
#SBATCH --job-name=seed_batch
#SBATCH --cpus-per-task=3
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=seed_batch_%j.out
#SBATCH --error=seed_batch_%j.out
#
# Seed repeats of fine-tune cells (docs/results.tex sec:ladder-open, plan step 5): runs the lines of one chunk file
# written by sweep/seed_list.py, one after another, each a fixed-HP run (sweep/run_fixed_hp.py) of a cell's best
# trial at another seed. A run whose result file exists is skipped, so a chunk can be resubmitted after a time-out.
#     site submit <site> FA scripts/job_seed_batch.sh -- analysis/transfer/seed_chunks/<chunk>.txt
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
CHUNK="$1"
W=$(python -c "import siteconf;print(siteconf.SWEEP_DIR)")
while read -r CFG NAME STEPS HP; do
  [ -z "$CFG" ] && continue
  if [ -f "$W/$NAME/results/fixed.json" ]; then echo "SEED_SKIP $NAME (done)"; continue; fi
  echo "SEED_RUN $NAME"
  python sweep/run_fixed_hp.py --config "$CFG" --name "$NAME" --steps "$STEPS" $HP evaluation.train_subsample=2000 \
    && echo "SEED_DONE $NAME" || echo "SEED_FAILED $NAME"
done < "$CHUNK"
