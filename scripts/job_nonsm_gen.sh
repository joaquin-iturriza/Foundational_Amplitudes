#!/bin/bash
#SBATCH --job-name=nonsm_gen
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=nonsm_gen_%j.out
#SBATCH --error=nonsm_gen_%j.out
#
# CPU only (D49, item nonsm): the non-SM pools of recipes/nonsm_all.yaml (mg5_pipeline_final.NONSM), their diagram
# sidecars (data.offshell_per_event needs them), and their validation (tools/nonsm_validate.py: the SM limit of each
# model against the SM standalone, analytic |M|^2 for ee -> mu mu with the Z' and ee -> H+ H-).
#   site submit ccin2p3 FA scripts/job_nonsm_gen.sh --item nonsm

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
set -e
PROCS="zp_ee_mumu zp_ee_uu zp_ee_Y1a thdm_ee_HpHm thdm_ee_h2h3 smeft_ee_mumu smeft_ee_ttbar"
MG5=$(python -c 'import mg5_pipeline_final as mg; print(mg.MG5_BIN)')
# each UFO model loaded once, serially, before the parallel backend builds (the first import of a model or of a
# restriction writes its cache into the model directory)
for m in DMsimp_s_spin1 2HDM SMEFTatNLO-LO; do printf 'import model %s\nquit\n' "$m" | "$MG5" >/dev/null 2>&1; done
python tools/dump_diagrams.py $(for p in $PROCS; do echo --process $p; done)
python prebuild_recipes.py recipes/nonsm_all.yaml --workers "${SLURM_CPUS_PER_TASK:-16}"
python tools/nonsm_validate.py
