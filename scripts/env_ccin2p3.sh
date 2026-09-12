#!/usr/bin/env bash
# env_ccin2p3.sh — CC-IN2P3 stand-ins for the Jean Zay environment variables the data
# pipeline keys on. Source it AFTER activating the venv, in every job script.
#
# datagen.py / mg5_pipeline_final.py locate the MadGraph install and the dataset pools
# through $WORK and $SCRATCH (Jean Zay conventions); CC-IN2P3 sets neither, so without
# this they fall back to the laptop layout (/home/joaquin/...) and fail on a node.
#   $WORK/mg5amcnlo      MadGraph install (MG5_BIN / MG5_WORK_DIR default from it)
#   $WORK/datasets       frozen val/test pools + recipe sidecars (AMP_FROZEN_DIR)
#   $SCRATCH/amp_data_cache   purgeable, sweep-shared train cache (AMP_TRAIN_CACHE_DIR)
# Both must be on the shared /sps filesystem: a sweep's trials share the cache.
# Explicit MG5_BIN / MG5_WORK_DIR / MG5_OUTPUT_DIR / AMP_* exports still win.
export WORK="${WORK:-/sps/lpnhe/jiturrizaramirez01}"
export SCRATCH="${SCRATCH:-/sps/lpnhe/jiturrizaramirez01/tmp}"
