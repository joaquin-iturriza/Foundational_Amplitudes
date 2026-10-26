"""Fraction of catalog_v4's pools whose |M|^2 mixes alpha_s / alpha powers, by multiplicity, training and shared
held-out (analysis/catalog_v4/data/mixed_order_audit.json, tools/mixed_order_audit.py; plan catalog_coverage).
Pools counted per recipe entry (the heavy-quark and alpha_s variants with their base); loop-induced entries left out."""
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

d = json.load(open(os.path.join(ROOT, "analysis/catalog_v4/data/mixed_order_audit.json")))["entries"]
fig, ax = ps.figure()
cats = ["2->2", "2->2 one-loop", "2->3", "2->3 one-loop", "2->4", "2->5"]
for i, (rf, lab, col) in enumerate([("catalog_v4_train", "training", ps.C.blue),
                                    ("catalog_v4_shared_heldout", "shared held-out", ps.C.orange)]):
    tot, mix = {}, {}
    for p in yaml.safe_load(open(os.path.join(ROOT, f"recipes/{rf}.yaml")))["processes"]:
        r = d[p["name"] if p["name"] in d else p["name"].split("__")[0]]
        if r["explicit_order"]:
            continue
        c = f"2->{r['nfinal']}" + (" one-loop" if r["virt"] else "")
        tot[c] = tot.get(c, 0) + 1
        mix[c] = mix.get(c, 0) + int(bool(r["mixed"]))
    xs = [j for j, c in enumerate(cats) if c in tot]
    ax.bar([x + (i - 0.5) * 0.38 for x in xs], [mix[cats[x]] / tot[cats[x]] for x in xs], width=0.38, color=col,
           label=f"{lab}, {sum(mix.values())}/{sum(tot.values())}")
ax.set_xticks(range(len(cats)))
ax.set_xticklabels([c.replace("->", r"$\to$").replace(" one-loop", "\n1-loop") for c in cats])
ax.set_ylabel("mixed-order fraction of pools")
ps.legend(ax)
ps.save(fig, "analysis/catalog_v4/figs/mixed_order_fraction")
