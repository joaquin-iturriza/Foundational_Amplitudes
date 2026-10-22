"""P1 per-event information and P2 extrapolation of the momentum-only surrogates (norm_surrogate.py).
Per-event (the surrogate as an input to the large model): within each process, the correlation r of the surrogate
with the true log|M|^2 on the same events and r^2, the fraction of the within-process variance a linear fit on it
explains; median over the 43 new, the 96 relabels, the 1352 training processes, and per multiplicity.
P2 in multiplicity on u ubar -> Z + n g and u dbar -> W + n g, n = 1..5 (data/surrogate_p2.json): (a) per family, the
level map y_mean = a + b * surrogate mean and the log-std map fitted on n <= 3, scored on n = 4, 5 (level error
(pred - true)^2 / var, |log std error|); (b) the catalog's maps (fitted on its 1352 training processes,
norm_surrogate_score.py's one-intercept form) applied to every n, which reaches N = 6..8 particles the catalog never has.
P2 in loop order: tree pools against their one-loop pools (ee -> u ubar, ee -> d dbar, ee -> b bbar, u ubar -> d dbar,
u ubar -> ZZ, and the EW one-loop ee -> d dbar): the one-loop pool's level and std against the tree pool's, the shift
the Catani-style surrogate predicts (its mean minus prop's mean: the log of the finite Catani factor), and the per-event
correlation of prop and catani with the one-loop target on the one-loop pool's own momenta. The tree pools are
mixture-sampled (smix) and the one-loop ones uniform, so their raw levels and stds also differ by sampling.
    python analysis/catalog_daniel/norm_surrogate_p2.py
"""
import json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["dim", "dimid", "eik", "pt", "prop", "propeik"]
S = {k: v for k, v in json.load(open(os.path.join(HERE, "data", "surrogate.json"))).items() if "error" not in v}
F = json.load(open(os.path.join(HERE, "data", "surrogate_p2.json")))
rel = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
held = set(json.load(open(os.path.join(HERE, "data", "norm_moments2.json")))["heldout"])
tr = sorted(n for n in S if n not in held); ho = sorted(n for n in S if n in held)
new = [n for n in ho if n not in rel]; rl = [n for n in ho if n in rel]
out = {"per_event": {}, "family": {}, "loop": {}}
print("per-event: median r (r^2) of the surrogate with log|M|^2 within a process")
for k in KEYS:
    row = {}
    for lab, names in (("train", tr), ("new", new), ("rel", rl), ("2to2", [n for n in S if S[n]["N"] == 4]),
                       ("2to3", [n for n in S if S[n]["N"] == 5])):
        r = np.array([S[n][k][2] for n in names])
        row[lab] = dict(r=float(np.median(r)), r2=float(np.median(r ** 2)), r2_p10=float(np.quantile(r ** 2, .1)))
    out["per_event"][k] = row
    print(f"  {k:8s} " + " | ".join(f"{lab} {v['r']:.2f} ({v['r2']:.2f})" for lab, v in row.items()))
fam = {"Z": [f"uubar_Z{'g' * i}" for i in range(1, 6)], "W": [f"udbar_W{'g' * i}" for i in range(1, 6)]}
print("per-event r (r^2) on the families, n = 1..5:")
for f, names in fam.items():
    for k in KEYS:
        r = [F[n][k][2] for n in names]
        out["per_event"][f"{f}|{k}"] = r
        print(f"  {f} {k:8s} " + " ".join(f"{x:.2f} ({x * x:.2f})" for x in r))


def lin(x, y):
    A = np.stack([np.ones(len(x)), x], 1); return np.linalg.lstsq(A, y, rcond=None)[0]


cat = {}
for k in KEYS:
    cat[k] = (lin(np.array([S[n][k][0] for n in tr]), np.array([S[n]["y_mean"] for n in tr])),
              lin(np.log([max(S[n][k][1], 1e-6) for n in tr]), np.log([S[n]["y_std"] for n in tr])))
print("multiplicity: (a) family maps fitted on n <= 3, scored at n = 4, 5; (b) the catalog's maps at every n")
for f, names in fam.items():
    for k in KEYS:
        fit, test = names[:3], names[3:]
        xm = np.array([F[n][k][0] for n in fit]); xs = np.log([max(F[n][k][1], 1e-6) for n in fit])
        cm = lin(xm, np.array([F[n]["y_mean"] for n in fit])); cs = lin(xs, np.log([F[n]["y_std"] for n in fit]))
        ea = [((cm[0] + cm[1] * F[n][k][0] - F[n]["y_mean"]) ** 2 / F[n]["y_std"] ** 2,
               abs(cs[0] + cs[1] * np.log(max(F[n][k][1], 1e-6)) - np.log(F[n]["y_std"]))) for n in test]
        (am, bm), (as_, bs) = cat[k]
        eb = [((am + bm * F[n][k][0] - F[n]["y_mean"]) ** 2 / F[n]["y_std"] ** 2,
               abs(as_ + bs * np.log(max(F[n][k][1], 1e-6)) - np.log(F[n]["y_std"]))) for n in names]
        out["family"][f"{f}|{k}"] = dict(a_n45=ea, b_n12345=eb)
        print(f"  {f} {k:8s} (a) n=4,5 level " + ", ".join(f"{e[0]:.3g}" for e in ea) + "; |dlog std| " +
              ", ".join(f"{e[1]:.2f}" for e in ea) + " | (b) level " + ", ".join(f"{e[0]:.3g}" for e in eb) +
              "; |dlog std| " + ", ".join(f"{e[1]:.2f}" for e in eb))
print("loop order: tree pool -> one-loop pool")
for t, l in (("ee_uu", "ee_uu_nlo"), ("ee_ddbar", "ee_dd_nlo"), ("ee_bbbar", "ee_bb_nlo"),
             ("uubar_ddbar", "uubar_ddbar_nlo"), ("uubar_ZZ", "uubar_ZZ_nlo"), ("ee_ddbar", "ee_dd_ew_nlo")):
    T, L = F[t], F[l]
    shift_true = L["y_mean"] - T["y_mean"]
    shift_cat = L["catani"][0] - L["prop"][0]
    rec = dict(tree_mean=T["y_mean"], tree_std=T["y_std"], loop_mean=L["y_mean"], loop_std=L["y_std"],
               loop_sign_frac=L["sign_frac"], shift_true=shift_true, shift_catani=shift_cat,
               r_prop=L["prop"][2], r_catani=L["catani"][2], r_prop_tree=T["prop"][2])
    out["loop"][l] = rec
    print(f"  {l:16s} level {T['y_mean']:.2f} -> {L['y_mean']:.2f} (shift {shift_true:+.2f}; Catani's log factor "
          f"{shift_cat:+.2f}), std {T['y_std']:.2f} -> {L['y_std']:.2f}, positive {L['sign_frac']:.2f}; per-event r on "
          f"the one-loop pool: prop {L['prop'][2]:.2f}, catani {L['catani'][2]:.2f} (prop on the tree pool {T['prop'][2]:.2f})")
json.dump(out, open(os.path.join(HERE, "data", "surrogate_p2_score.json"), "w"), indent=1)
