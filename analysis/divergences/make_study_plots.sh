#!/bin/bash
# CPU-only: build all figures for the uugg add-back and soft/collinear-cut studies from
# the eval npz files (run AFTER eval_heldout_studies.sh completes). Login-node safe.
set -e
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes
PY=/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/python
D=analysis/divergences

echo "===== uugg add-back curve ====="
$PY $D/plot_addback.py --tags 000,005,015,050,100 \
  --npz_prefix heldout_eval_uugg_f --out_base $D/figs/addback_curve_uugg \
  --summary_out $D/heldout_eval_uugg_summary.json \
  --process '$e^+e^-\to u\bar u gg$'
echo "===== uugg IR residual maps (f=0 vs f=1) ====="
$PY $D/make_ir.py --npz $D/heldout_eval_uugg_f000.npz --label 'ee->uugg hold-out f=0 (pure extrapolation)' --out_base $D/figs/heldout_uugg_resid_f000
$PY $D/make_ir.py --npz $D/heldout_eval_uugg_f100.npz --label 'ee->uugg hold-out f=1 (in-support)'      --out_base $D/figs/heldout_uugg_resid_f100

echo "===== soft-cut and collinear-cut add-back curves ====="
$PY $D/plot_addback.py --tags 000,005,015,100 \
  --npz_prefix heldout_eval_soft_f --out_base $D/figs/addback_curve_soft \
  --summary_out $D/heldout_eval_soft_summary.json \
  --process '$e^+e^-\to u\bar u g$'
$PY $D/plot_addback.py --tags 000,005,015,100 \
  --npz_prefix heldout_eval_coll_f --out_base $D/figs/addback_curve_coll \
  --summary_out $D/heldout_eval_coll_summary.json \
  --process '$e^+e^-\to u\bar u g$'
echo "===== soft-vs-collinear comparison (headline) ====="
$PY $D/plot_soft_vs_coll.py --tags 000,005,015,100

echo "ALL_STUDY_PLOTS_DONE"
