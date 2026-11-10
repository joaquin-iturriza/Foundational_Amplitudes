"""The one-loop candidates of pretrain_data_saving step 6: the spread of ln|V/B| on each train pool against the share
of its events with V < 0, from analysis/data_saving/data/vb_spread.json (vb_spread.py). A target is kept when the
spread is >= 0.1 and no V is negative (V/B then trains as a plain log target, read as the MSE of ln|V|).

    python analysis/data_saving/plot_vb_spread.py
"""
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

LABELS = {
    "ee_uu_nlo_25-1000GeV": r"$e^+e^-\to u\bar u$",
    "ee_ttbar_nlo_362-1000GeV": r"$e^+e^-\to t\bar t$",
    "v4_epem_bbx_nlo_20-6000GeV": r"$e^+e^-\to b\bar b$",
    "v4_bbx_zz_nlo_183-6000GeV": r"$b\bar b\to ZZ$",
    "v4_uux_epemz_nlo_112-6000GeV": r"$u\bar u\to e^+e^-Z$",
    "v4_bg_ab_nlo_20-6000GeV": r"$bg\to\gamma b$",
    "v4_epem_ddxz_nlo_112-6000GeV": r"$e^+e^-\to d\bar dZ$",
    "v4_bbx_gg_nlo_20-6000GeV": r"$b\bar b\to gg$",
    "v4_dg_uwm_nlo_92-6000GeV": r"$dg\to uW^-$",
    "v4_bbx_avevex_nlo_20-6000GeV": r"$b\bar b\to\gamma\nu\bar\nu$",
    "v4_db_bdz_nlo_112-6000GeV": r"$db\to bdZ$",
    "v4_uc_cuz_nlo_112-6000GeV": r"$uc\to cuZ$",
}
FLOOR = 1e-4   # a pool with no V < 0 is drawn at this share (log axis)

rows = json.load(open(os.path.join(REPO, "analysis/data_saving/data/vb_spread.json")))
fig, ax = ps.figure()
markers = ["o", "s", "^", "D", "v", "P"]
for i, (stem, lab) in enumerate(LABELS.items()):
    r = rows.get(stem)
    if not r:
        continue
    ax.plot([max(r["neg_frac"], FLOOR)], [max(r["std"], 1e-3)], linestyle="none",
            marker=markers[i % len(markers)], color=ps.CYCLE[i % len(ps.CYCLE)],
            mfc=ps.CYCLE[i % len(ps.CYCLE)] if r.get("kept") else "none", label=lab)
ax.axhline(0.1, color=ps.C.grey, linestyle="--", label=r"$\sigma=0.1$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"share of events with $V<0$")
ax.set_ylabel(r"$\sigma(\ln|V/B|)$")
ps.shared_legend(fig, ax, ncol=4)
ps.save(fig, os.path.join(REPO, "analysis/data_saving/figs/vb_spread"))
for stem in LABELS:
    r = rows.get(stem)
    if r:
        print(f"{stem:32s} std {r['std']:.3g} neg {r['neg_frac']:.3f} kept {r.get('kept')}")
