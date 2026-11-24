#!/bin/bash
#SBATCH --job-name=disk_inventory
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=disk_inventory_%j.out
#SBATCH --error=disk_inventory_%j.out
#
# CPU only, read-only: the disk inventory of the user's area (cc_disk, D542). CC's login nodes end a session after
# ~1-2 min of du (2026-11-24), so the long walks run here. Writes to $SCRATCH/zz_cc_disk/ (cheap, ours):
#   scratch_top.tsv   du -s (MiB) of every top-level entry of $SCRATCH; scratch_top_apparent.tsv the same, apparent size
#   top.tsv, top_apparent.tsv   du -s (MiB) of every entry of /sps/lpnhe, $WORK and the checkout, allocated and apparent
#                     (Isilon's allocated blocks summed to 1.28x the df "used" on 2026-11-24: both are kept)
#   inv_runs.jsonl    analysis/storage/inv_runs.py over the run roots (model files against the rest, training time)
#   census.tsv        analysis/storage/cache_census.py over $SCRATCH/amp_data_cache, every recipe counted in use
#   files_*.tsv       allocated 512-byte blocks, bytes and name of the loose files of datasets/ and the checkout root
#   lhe_queue.tsv     du -d 2 (MiB) of $SCRATCH/lhe_queue
# Deletes nothing.

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
O="$SCRATCH/zz_cc_disk"; mkdir -p "$O"
find "$SCRATCH" -mindepth 1 -maxdepth 1 -print0 | du -s --block-size=1M --files0-from=- > "$O/scratch_top.tsv" &
find "$SCRATCH" -mindepth 1 -maxdepth 1 -print0 | du -s --apparent-size --block-size=1M --files0-from=- > "$O/scratch_top_apparent.tsv" &
# one du per group of disjoint paths: given nested paths, du drops the inner one (counted once already)
for A in "" --apparent-size; do
  ( for G in "$(dirname "$WORK")" "$WORK" "$PROJECT_DIR"; do
      find "$G" -mindepth 1 -maxdepth 1 ! -path "$WORK" ! -path "$PROJECT_DIR" -print0 |
        du -s $A --block-size=1M --files0-from=- 2>/dev/null
    done ) > "$O/top${A:+_apparent}.tsv" &
done
df -B1M /sps/lpnhe > "$O/df.txt"
python analysis/storage/inv_runs.py runs runs_from_lxplus runs_from_jeanzay > "$O/inv_runs.jsonl" &
python analysis/storage/cache_census.py --in-use '*' --days 14 --workers 2 > "$O/census.tsv" &
find "$WORK/datasets" -maxdepth 1 -type f -printf "%b\t%s\t%f\n" > "$O/files_datasets.tsv"
find . -maxdepth 1 -type f -printf "%b\t%s\t%f\n" > "$O/files_checkout.tsv"
du -d 2 --block-size=1M "$SCRATCH/lhe_queue" > "$O/lhe_queue.tsv"
wait
wc -l "$O"/*
echo DISK_INVENTORY_DONE
