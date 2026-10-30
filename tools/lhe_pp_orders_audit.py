"""Which lhe_pp pools change their coupling powers now that they read their diagram sidecars like their trees (E3058,
plan catalog_coverage). Before, an lhe_pp entry's explicit order (its tree's table order) made the order features and
the dimensionless target skip the sidecar; now both read it (mg5_pipeline_final.order_vector_from_diagrams,
target_monomial_from_diagrams). Per name prefix (pp13_, pp13v4_, ee4_, dc_, dsh_, dE100_, ...): entries, entries with a
sidecar, and those whose features or target monomial differ from the table's, by name. A table-mixed entry (q + e above
nfinal) had no table target (it was refused): it is listed as 'was_refused'.

Run where the sidecars are (data/diagrams/, CC-IN2P3): python tools/lhe_pp_orders_audit.py [--prefix pp13_ ...] --out <json>
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mg5_pipeline_final as mg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs="*", default=None, help="name prefixes to audit (default: every lhe_pp entry)")
    ap.add_argument("--diagrams", default="data/diagrams")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    by = {}
    for name, cfg in sorted(mg.PROCESSES.items()):
        if cfg.get("kind") != "lhe_pp":
            continue
        pre = re.match(r"^[A-Za-z0-9]+_", name).group(0)
        if a.prefix and pre not in a.prefix:
            continue
        r = by.setdefault(pre, {"entries": 0, "with_sidecar": 0, "features_differ": [], "target_differs": [],
                                "was_refused": [], "no_sidecar": []})
        r["entries"] += 1
        path = os.path.join(a.diagrams, f"{name}.diagrams.json")
        if not os.path.exists(path):
            r["no_sidecar"].append(name)
            continue
        r["with_sidecar"] += 1
        bo = mg.sidecar_born_orders(path)
        o = [int(x) for x in mg.order_vector(cfg)]
        if not bo:
            continue
        f = mg.order_vector_from_diagrams(cfg, bo, o)
        if f != o:
            r["features_differ"].append([name, o, f])
        t = list(mg.target_monomial_from_diagrams(cfg, bo))
        if o[2] + o[3] > int(cfg["nfinal"]):
            r["was_refused"].append([name, o, t])
        elif t != o[2:]:
            r["target_differs"].append([name, o[2:], t])
    for pre, r in by.items():
        print(f"{pre}: {r['entries']} entries, {r['with_sidecar']} with sidecar, features differ "
              f"{len(r['features_differ'])}, target differs {len(r['target_differs'])}, was refused "
              f"{len(r['was_refused'])}")
        for k in ("features_differ", "target_differs"):
            if r[k]:
                print(f"  {k}: {r[k][:12]}")
    if a.out:
        json.dump(by, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
