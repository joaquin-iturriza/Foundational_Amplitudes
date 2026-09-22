# Compatibility shim.
#
# The per-site environment now lives in sites/activate.sh, which is the single
# file in this repo allowed to name a cluster. This shim stays because 95 sweep
# configs and compare_models/make_scan_ab_sweeps.py still source this path by
# its absolute name; it will go when those move to sites/activate.sh too.
_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
# shellcheck source=../sites/activate.sh
source "$_here/sites/activate.sh"
