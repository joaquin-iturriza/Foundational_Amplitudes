#!/bin/bash
#SBATCH --job-name=zng_fetch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
# zng_uq_benchmark setup, CPU: the Zenodo 16793011 archives (q qbar -> Z + n g, lloca-experiments-uq) into
# $DATA_DIR/zng_uq, only the files the UQ repo's split reads (<proc>_0,1,2.npy) extracted, then tools/zng_uq.py convert
# (uq_<proc>.h5, their split), the d d~ > Z + n g diagram sidecars and the |M|^2 cross-check of our MadGraph on 100 of
# their test events per multiplicity. Archives are deleted once their files are out (re-downloadable; md5 kept).
#     site submit ccin2p3 FA scripts/job_zng_fetch.sh --item zng_uq_benchmark
set -euo pipefail
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
T="$DATA_DIR/zng_uq"; mkdir -p "$T/archives" "$T/raw"
URL=https://zenodo.org/api/records/16793011/files
declare -A MD5=([zg.tar.xz]=6cc9476a0fb15f7f06d82ac1bf8a3acc [zgg.tar.xz]=4dd4eee71975d6bb8d83346fc2786650
    [zggg.tar.xz]=eb6143155dd1e869677a72a25d52c9ec [zgggg_part1.tar.xz]=11489a031d710ad99518b769b07babf1
    [zgggg_part2.tar.xz]=88d31cdefd5d5802e84e47dd8b6701d4)
have() { for i in 0 1 2; do [ -n "$(find "$T/raw" -name "${1}_${i}.npy" | head -1)" ] || return 1; done; }
fetch() {   # <archive> <proc>: download, verify, extract <proc>_0,1,2.npy, delete the archive
    local a="$T/archives/$1"
    curl -fL --retry 5 --retry-delay 30 -C - -o "$a" "$URL/$1/content"
    echo "${MD5[$1]}  $a" | md5sum -c - | tee -a "$T/MD5SUMS"
    tar -xJf "$a" -C "$T/raw" --wildcards "*${2}_[012].npy" || true    # a part may hold none of them
    rm -f "$a"
}
for p in zg zgg zggg; do have $p || fetch $p.tar.xz $p; done
have zgggg || fetch zgggg_part1.tar.xz zgggg
have zgggg || fetch zgggg_part2.tar.xz zgggg
for p in zg zgg zggg zgggg; do have $p || { echo "missing ${p}_0,1,2.npy"; exit 3; }; done
python tools/zng_uq.py convert "$T/raw"
python tools/dump_diagrams.py --skip-existing --process ddbar_Zg --process ddbar_Zgg --process ddbar_Zggg --process ddbar_Zgggg
python tools/zng_uq.py crosscheck --n 100
du -sh "$T"
echo "RUN_DONE zng_fetch"
