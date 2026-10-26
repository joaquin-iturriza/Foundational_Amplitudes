"""catalog_v4 on the 13 TeV pp pools, again on the classes whose |M|^2 is the same function in catalog_v4 and in the pp
pools (catalog_coverage; analysis/catalog_v4/pp_physics_check.py: the pp pools carry catalog_v3's physics, so a class
with a b or with the diagrams catalog_v4's electroweak orders add scores catalog_v4 on labels of another function).
The reads, groups and ratios are compare_v3.pp's (catalog_v4 against catalog_v3, Muon, two seeds each) and
compare_pp_trained's (catalog_v4 against the pp-trained AdamW arm, and each over its own optimizer's catalog_v3 arm), no
b classes (b classes all differ); per group (trained in both catalogs, held out in both, the role changed) and event set
(whole pool, inside catalog_v3's coverage, outside) the ratio on:
  all     every no-b class (today's numbers)
  agree   no electroweak diagram added (pp_physics_check classify 'none'; its eval checks that catalog_v4's standalone
          gives the label back on these)
  ew      the classes catalog_v4's electroweak orders change
    python3 analysis/catalog_v4/pp_physics_rescore.py    (needs data/pp_physics_check.json)
The last line printed is the JSON of its numbers.
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_v3 as cv
import compare_pp_trained as cp


def main():
    chk = json.load(open(os.path.join(HERE, "data", "pp_physics_check.json")))
    C = chk["classes"]
    R, p, _ = cp.rows()
    role, b = p["_role"], p["_b"]
    groups = {"train": lambda n: role[n]["v3"] == "train" and role[n]["v4"] == "train",
              "heldout": lambda n: role[n]["v3"] == "heldout" and role[n]["v4"] == "heldout",
              "role_changed": lambda n: role[n]["v3"] != role[n]["v4"] and role[n]["v4"] != "absent"}
    subsets = {"all": lambda n: True, "agree": lambda n: C[n]["cls"] == "none", "ew": lambda n: C[n]["cls"] == "ew"}
    out, claims = {}, {}
    for g, gs in groups.items():
        for s, ss in subsets.items():
            ps_ = [n for n in role if not b[n] and gs(n) and ss(n)]
            for k in cp.KEYS:
                v = cv.compare(R[k], ps_)
                d = cp.ratio(R, k, cp.V4, ("dimensionless_pp",), ps_)
                a = cp.ratio(R, k, cp.V4, cp.V3M, ps_)
                c = cp.ratio(R, k, ("dimensionless_pp",), ("dimensionless",), ps_)
                rec = float(np.log(a["ratio"]) / np.log(c["ratio"])) if a and c and c["ratio"] != 1 else None
                e = {"n": len(ps_), "v4_over_v3_muon": v and v["ratio"], "v4_lower_both_pct": v and v["v4_lower_both_pct"],
                     "gap": v and v["gap"], "exceeds_seed_spread": v and v["exceeds_seed_spread"],
                     "v4_over_pp": d and d["ratio"], "v4_lower_than_pp_pct": d and d["num_lower_pct"],
                     "pp_over_v3_adamw": c and c["ratio"], "recovered": rec}
                out.setdefault(g, {}).setdefault(s, {})[k] = e
                print(f"{g:12s} {s:5s} {k:11s} n={len(ps_):3d} v4/v3 {e['v4_over_v3_muon']:.3g} ({e['v4_lower_both_pct']:.0f}% v4 lower both) "
                      f"v4/pp {e['v4_over_pp']:.3g} ({e['v4_lower_than_pp_pct']:.0f}% lower) pp/v3adamw {e['pp_over_v3_adamw']:.3g} recovered {rec if rec is None else round(rec, 3)}")
                claims.setdefault(g, {}).setdefault(s, {})[k] = e
    json.dump(out, open(os.path.join(HERE, "data", "pp_physics_rescore.json"), "w"), indent=1)
    print(json.dumps(claims))


if __name__ == "__main__":
    main()
