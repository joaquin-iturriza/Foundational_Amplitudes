# Resolve the project root and this site's python env at RUNTIME.
#
# Source this from any job script instead of hardcoding a cluster path:
#
#     _CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
#     source "$_CCORCH_ROOT/sites/activate.sh"
#
# and the same script runs unchanged at every site. PROJECT_DIR comes from this
# file's own location, so it is right wherever the checkout happens to live;
# CCORCH_SITE is exported by `site submit` and sniffed from the path otherwise.
# Only the env activation genuinely differs per cluster -- that is the single
# thing enumerated below, and it is the only place a cluster is named.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export PROJECT_DIR

if [ -z "${CCORCH_SITE:-}" ]; then
  case "$PROJECT_DIR" in
    /sps/*)        CCORCH_SITE=ccin2p3 ;;
    /lustre/*)     CCORCH_SITE=jeanzay ;;
    /eos/*|/afs/*) CCORCH_SITE=lxplus ;;
    *)             CCORCH_SITE=local ;;
  esac
fi
export CCORCH_SITE

case "$CCORCH_SITE" in
  ccin2p3)
    source "$PROJECT_DIR/.venv/bin/activate"
    export WORK="${WORK:-/sps/lpnhe/jiturrizaramirez01}"
    export SCRATCH="${SCRATCH:-/sps/lpnhe/jiturrizaramirez01/tmp}"
    export DATA_DIR="${DATA_DIR:-/sps/lpnhe/jiturrizaramirez01/datasets}"
    export SUBMIT_DIR="${SUBMIT_DIR:-/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes}"
    ;;
  jeanzay)
    module load anaconda-py3/2023.09 2>/dev/null || true
    source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh
    conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
    export WORK="${WORK:-/lustre/fswork/projects/rech/itg/ulm49ia}"
    export SCRATCH="${SCRATCH:-/lustre/fsn1/projects/rech/itg/ulm49ia}"
    export DATA_DIR="${DATA_DIR:-/lustre/fswork/projects/rech/itg/ulm49ia/datasets}"
    export SUBMIT_DIR="${SUBMIT_DIR:-/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes}"
    ;;
  lxplus)
    source "$PROJECT_DIR/.venv/bin/activate"
    export WORK="${WORK:-/eos/user/j/joiturri}"
    export SCRATCH="${SCRATCH:-/eos/user/j/joiturri/tmp}"
    export DATA_DIR="${DATA_DIR:-/eos/user/j/joiturri/jitu/lorentz-gatr/data/data}"
    export SUBMIT_DIR="${SUBMIT_DIR:-/afs/cern.ch/user/j/joiturri/Foundational_Amplitudes}"
    ;;
  *)
    [ -d "$PROJECT_DIR/.venv" ] && source "$PROJECT_DIR/.venv/bin/activate"
    ;;
esac
