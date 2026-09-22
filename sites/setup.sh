#!/bin/bash
# Foundational_Amplitudes: MadGraph (datagen builds process standalones on demand), gfortran/g++, dataset dirs.
# Contract (run by `site env`, checked by `site pick`): with sites/activate.sh sourced,
#   bash sites/setup.sh            prepare this site for the project (idempotent)
#   bash sites/setup.sh --verify   fast, read-only: exit 0 iff runnable, one status line
set -u

MG5_VER="${MG5_VER:-3.7.0}"                    # CC-IN2P3 runs 3.7.0; Jean Zay 3.6.7
ROOT="${MG5_WORK_DIR:-$WORK/mg5amcnlo}"        # mg5_pipeline_final.py's default
BIN="${MG5_BIN:-$ROOT/bin/mg5_aMC}"

verify() {
  local miss=()
  command -v gfortran >/dev/null 2>&1 || miss+=("gfortran")
  command -v g++ >/dev/null 2>&1 || miss+=("g++")
  [ -x "$BIN" ] || miss+=("MadGraph at $ROOT")
  [ -d "$WORK/datasets" ] || miss+=("$WORK/datasets")
  if [ ${#miss[@]} -gt 0 ]; then echo "missing: ${miss[*]}"; return 1; fi
  echo "ok: MG5 $(sed -n 's/^version = //p' "$ROOT/VERSION" 2>/dev/null | head -1), gfortran, $WORK/datasets ($(ls "$WORK/datasets" | wc -l) files)"
}
[ "${1:-}" = "--verify" ] && { verify; exit $?; }

mkdir -p "$WORK/datasets" "${SCRATCH:-$WORK}/amp_data_cache"
if [ ! -x "$BIN" ]; then
  echo "installing MadGraph $MG5_VER into $ROOT"
  tmp=$(mktemp -d)
  ok=0
  for url in "https://launchpad.net/mg5amcnlo/3.0/${MG5_VER%.*}.x/+download/MG5_aMC_v${MG5_VER}.tar.gz" \
             "https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.6.7.tar.gz"; do
    echo "  $url"
    if curl -fsSL -m 900 -o "$tmp/mg5.tgz" "$url"; then ok=1; break; fi
  done
  [ "$ok" = 1 ] || { echo "download failed"; rm -rf "$tmp"; exit 1; }
  mkdir -p "$(dirname "$ROOT")"
  tar xzf "$tmp/mg5.tgz" -C "$tmp" && mv "$tmp"/MG5_aMC_v* "$ROOT"
  rm -rf "$tmp"
fi
# non-interactive smoke: MG5 starts and quits (writes its configuration on first run)
echo quit | timeout 600 "$BIN" >/dev/null 2>&1 || { echo "mg5_aMC does not start"; exit 1; }
verify
