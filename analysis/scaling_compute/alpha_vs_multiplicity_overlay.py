#!/usr/bin/env python3
"""Compute-scaling exponent alpha_C vs final-state multiplicity, overlaying the
from-scratch (solo) fits and the full-finetune fits on one axis, with the
theoretical lower bound alpha = 4/DOF (DOF = 3*n_fs - 4) drawn in grey.

Exponents are read from the two scaling_law_params.json files so the figure
tracks the actual fits; the ratio (virt/Born) targets are excluded as nonsense.
Usage: python alpha_vs_multiplicity_overlay.py [out_basename]
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
SOLO = os.path.join(ROOT, "sweeps/scaling_solo_full/scaling_law_params.json")
FT   = os.path.join(ROOT, "sweeps/finetune_scaling_virt_002/scaling_law_params.json")
out  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "analysis/scaling_compute/alpha_vs_multiplicity_overlay")

# final-state multiplicity per process key (2->n_fs). Ratio targets excluded.
NFS = {
    "ee_aa_10-1000GeV": 2, "ee_ttbar_346-1000GeV": 2, "ee_WW_162-1000GeV": 2,
    "ee_uu_91-1000GeV": 2, "ee_ttbar_nlo_virt_e4": 2, "ee_uu_nlo_virt_e4": 2,
    "ee_uug_91-1000GeV": 3, "ee_aaa_10-1000GeV": 3, "ee_wwz_255-1000GeV": 3,
    "ee_uugg_91-1000GeV": 4,
}
FT_NFS = {"ee_uu_nlo_virt_e4": 2, "ee_ttbar_nlo_virt_e4": 2,
          "ee_uu_nlo_virt": 2, "ee_ttbar_nlo_virt": 2}
def is_ratio(k): return "ratio" in k
def norm(k):  # JSON keys carry an "_amplitudes" suffix; NFS maps use the bare name
    return k[:-len("_amplitudes")] if k.endswith("_amplitudes") else k

def load(path, allow):
    try:
        d = json.load(open(path))
    except Exception as e:
        print("WARN could not read", path, e); return {}
    out = {}
    for k, v in d.items():
        nk = norm(k)
        if is_ratio(nk) or nk not in allow:
            continue
        a = v.get("alpha") if isinstance(v, dict) else None
        if a is not None:
            out[nk] = (allow[nk], float(a))
    return out

solo = load(SOLO, NFS)
ft   = load(FT, FT_NFS)

fig, ax = plt.subplots(figsize=(7.2, 5.0))
# theoretical lower bound alpha = 4/DOF, DOF = 3 n_fs - 4
ns = np.array([2, 3, 4]); dof = 3 * ns - 4; bound = 4.0 / dof
ax.plot(ns, bound, color="0.5", ls="--", lw=2, zorder=1,
        label=r"theoretical bound $\alpha=4/\mathrm{DOF}$")
ax.fill_between(ns, 0, bound, color="0.5", alpha=0.12, zorder=0)

def scatter(d, color, marker, lab):
    if not d: return
    xs = [n for (n, a) in d.values()]; ys = [a for (n, a) in d.values()]
    # small horizontal jitter so overlapping n_fs=2 points separate
    xj = np.array(xs, float) + np.linspace(-0.06, 0.06, len(xs))
    ax.scatter(xj, ys, s=70, color=color, marker=marker, zorder=3,
               edgecolor="k", linewidth=0.4, label=lab)

scatter(solo, "#0343DE", "o", "from scratch (solo)")
scatter(ft,   "#C1121F", "s", "finetuned (full, layer-decay)")

ax.set_xticks([2, 3, 4])
ax.set_xticklabels([r"$2{\to}2$", r"$2{\to}3$", r"$2{\to}4$"])
ax.set_xlabel("final-state multiplicity")
ax.set_ylabel(r"compute-scaling exponent $\alpha_C$")
ax.set_title(r"$\alpha_C$ vs multiplicity: from-scratch and finetuned vs the $4/\mathrm{DOF}$ bound")
ax.set_ylim(0, 2.6); ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(out + ".png", dpi=140); fig.savefig(out + ".pdf")
print("saved", out + ".png", out + ".pdf", "| solo:", len(solo), "ft:", len(ft))
