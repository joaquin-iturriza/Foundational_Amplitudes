#!/bin/bash
# Every catalog_v2 figure in docs/results.tex, from the runs it was measured on. Run in the
# checkout on the site that holds runs/ and sweeps/ (CC-IN2P3):
#     site run ccin2p3 FA -- bash analysis/catalog_v2/make_figures.sh
# Arm labels say what the arm does; they are the legend titles and entries of the figures.
set -e
cd "$(dirname "$0")/../.."
P="python analysis/catalog_v2"

$P/check_pools.py --plot-only >/dev/null

$P/loss_vs_range.py runs/catalog_short_smix_geo runs/catalog_short_smix_mean \
  "--labels=geometric mean,arithmetic mean"

$P/excess_ratio.py --ref=4=9.9e-4_5=4.4e-3_6=5.4e-3 \
  "geometric mean=runs/catalog_short_offsh_geo" \
  "arithmetic mean=runs/catalog_short_offsh_mean" \
  "reference-weighted=runs/_unweighted/catalog_short_offsh_excess"

$P/arms_compare.py --out=arms_compare_q \
  "geometric mean=runs/catalog_q_geo/trial_0058" \
  "arithmetic mean=runs/catalog_q_mean/trial_0266" \
  "reference-weighted=runs/catalog_q_excess/trial_0078"

$P/residual_vs_target.py runs/q_mean_best_preds
$P/residual_vs_target.py runs/t1000_tprop_s1 --out=residual_vs_target_tprop \
  --show=ee_mumu,ee_uu,ee_uu_nlo,ee_dd_nlo,uubar_ZZ,ee_ZZ,ee_uug,ee_uugg

$P/seed_arms.py --out=seed_arms_tprop \
  'target $\log|\mathcal{M}|^2$=runs/t1000_slq1e-2_s*' \
  'massive propagators divided out=runs/t1000_tprop_s*'

$P/seed_arms.py --out=seed_arms_tch22 \
  'massive propagators divided out=runs/t1000_tprop_s[123]' \
  '+ $t$-channel factor, every process=runs/t1000_tprop_tch2_s*' \
  '+ $t$-channel factor, $2\to2$ only (2 seeds)=runs/t1000_tprop_tch22_s[13]'

$P/seed_arms.py --out=seed_arms_slq \
  'signed log, $s$ at the $10^{-2}$ quantile=runs/t1000_slq1e-2_s*' \
  '$s$ at the $10^{-3}$ quantile=runs/t1000_slq1e-3_s*' \
  '$s$ at the $10^{-4}$ quantile=runs/t1000_slq1e-4_s*'

$P/joint_vs_solo.py
