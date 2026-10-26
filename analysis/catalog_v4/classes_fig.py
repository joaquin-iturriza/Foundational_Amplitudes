"""catalog_v4's tree classes by role and multiplicity against catalog_v3's (recipes/catalog_v{3,4}_roles.json, the classes
of analysis/catalog_v{3,4}/census_classes.json; recipes/select_catalog_v4.py). Bars: classes per role, one panel per
multiplicity; hatched: the catalog_v4 classes with no generated pool yet (recipes/catalog_v4_gen_stage5.yaml).
    python3 analysis/catalog_v4/classes_fig.py
"""
import json, os, sys
from collections import Counter

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

ROLES = ["train", "H1", "H2", "H3"]


def counts(cat):
    R = json.load(open(os.path.join(ROOT, "recipes", f"catalog_{cat}_roles.json")))["trees"]
    P = yaml.load(open(os.path.join(ROOT, "recipes", f"catalog_{cat}_processes.yaml")),
                  Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))["processes"]
    return Counter((P[n]["layer"], v["axis"] or "train") for n, v in R.items()), P


def main():
    c3, _ = counts("v3")
    c4, P4 = counts("v4")
    new = Counter((P4[n]["layer"], (json.load(open(os.path.join(ROOT, "recipes", "catalog_v4_roles.json")))["trees"][n]["axis"]
                                    or "train")) for n in json.load(open(os.path.join(ROOT, "recipes", "catalog_v4_roles.json")))["gen_stage5"])
    fig, axes = ps.figure(nrows=3)
    w = 0.38
    for ax, L in zip(axes, (2, 3, 4)):
        x = range(len(ROLES))
        ax.bar([i - w / 2 for i in x], [c3[(L, r)] for r in ROLES], w, color=ps.C.blue, label="catalog v3")
        ax.bar([i + w / 2 for i in x], [c4[(L, r)] for r in ROLES], w, color=ps.C.orange, label="catalog v4")
        ax.bar([i + w / 2 for i in x], [new[(L, r)] for r in ROLES], w, color="none", edgecolor="k", hatch="///",
               label="catalog v4, no pool yet")
        ax.set_xticks(list(x)); ax.set_xticklabels(ROLES)
        ax.set_xlabel(rf"role, $2\to{L}$")
    for ax in axes:
        ax.set_ylabel("tree classes")
    ps.legend(axes[0], loc="upper right")
    data = {"v3": {f"{L} {r}": c3[(L, r)] for L in (2, 3, 4) for r in ROLES},
            "v4": {f"{L} {r}": c4[(L, r)] for L in (2, 3, 4) for r in ROLES},
            "v4_no_pool": {f"{L} {r}": new[(L, r)] for L in (2, 3, 4) for r in ROLES}}
    json.dump(data, open(os.path.join(ROOT, "analysis", "catalog_v4", "data", "classes_fig.json"), "w"), indent=1)
    ps.save(fig, os.path.join(ROOT, "analysis", "catalog_v4", "figs", "classes_roles"))


if __name__ == "__main__":
    main()
