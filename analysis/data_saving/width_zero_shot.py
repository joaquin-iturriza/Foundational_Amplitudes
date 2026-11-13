"""Zero-shot error against model width, from committed reads only (no new training): catalog_v4 runs at mu-P widths
(attention heads) 8, 16, 32, 4264 steps, seeds 42 and 43, het loss, default share 0.25, each read as its final model.
(a) absolute MSE of log|M|^2 on the held-out 2->4 (6-particle, H2) classes, catalog-sampled and default-sampled, per seed
    (analysis/catalog_v4/data/abs_multiplicity.json, abs_multiplicity.py @a07b0fed; results.tex par:v4_mult_width_length).
(b) R = median over classes of the class's geometric mean of MSE(width w) / MSE(width 8), seeds pooled, with its 95%
    class bootstrap widened by the width-8 seed pair, on the test splits (PP4, EE4: trained classes, unseen events) and on
    every held-out class (default-sampled) (data/width_x1.json, mixture_ab.py width @20448b14; par:v4_width).
    python analysis/data_saving/width_zero_shot.py   -> figs/width_zero_shot
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

DATA = os.path.join(ROOT, "analysis", "catalog_v4", "data")
A = json.load(open(os.path.join(DATA, "abs_multiplicity.json")))["absolute"]
W = json.load(open(os.path.join(DATA, "width_x1.json")))["arms"]
WIDTHS = [8, 16, 32]

fig, (axa, axb) = ps.figure(ncols=2)
for read, col in (("catalog", ps.C.blue), ("default", ps.C.vermillion)):
    for s, mk in ((42, "o-"), (43, "s--")):
        axa.plot(WIDTHS, [A[f"w{w}_s{s}"][read]["mse"] for w in WIDTHS], mk, color=col,
                 label=f"{read}-sampled, seed {s}")
axa.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$, held-out $2\to4$")
ps.legend(axa, "lower left")

for (src, sub), lab, col in ((("PP4", "all"), "test, pp", ps.C.blue), (("EE4", "all"), r"test, $e^+e^-$", ps.C.sky),
                             (("held_default", "pooled"), "held out", ps.C.vermillion)):
    r = [1.0] + [W[f"w{w}"][src][sub]["R"] for w in WIDTHS[1:]]
    lo = [0.0] + [r[i + 1] - W[f"w{w}"][src][sub]["widened"][0] for i, w in enumerate(WIDTHS[1:])]
    hi = [0.0] + [W[f"w{w}"][src][sub]["widened"][1] - r[i + 1] for i, w in enumerate(WIDTHS[1:])]
    axb.errorbar(WIDTHS, r, yerr=[lo, hi], fmt="o-", color=col, capsize=3, label=lab)
    print(src, sub, " ".join(f"w{w}: {v:.3f}" for w, v in zip(WIDTHS, r)))
axb.set_ylabel(r"$R = \mathrm{MSE}(w)/\mathrm{MSE}(8)$")
ps.legend(axb, "lower left")
for ax in (axa, axb):
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(WIDTHS, [str(w) for w in WIDTHS]); ax.minorticks_off()
    ax.set_xlabel(r"width $w$ (heads)")
    ax.set_yticks(yt := [0.2, 0.3, 0.5, 0.7] if ax is axa else [0.4, 0.6, 1.0], [f"{v:g}" for v in yt])
    ps.make_room(ax)
for k, v in A.items():
    print(k, {r: round(v[r]["mse"], 3) for r in v})
ps.save(fig, os.path.join(ROOT, "analysis", "data_saving", "figs", "width_zero_shot"))
