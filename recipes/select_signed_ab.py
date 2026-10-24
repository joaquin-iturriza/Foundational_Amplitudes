"""The signed pools' short GPU A/B (D99; docs/results.tex, the candidate targets): catalog_v3's sign-changing one-loop
pools of stage 2a (recipes/catalog_v3_gen_stage2a.yaml) added to the short catalog run's training set, two arms on the
same data (scripts/job_catalog_norm.sh SgnR0 / SgnHead). A pool is sign-changing when its validation events hold a
negative value (analysis/catalog_daniel/data/signed_band_v3.json, neg_frac > 0; one is negative throughout); its role
is catalog_v3's (recipes/catalog_v3_roles.json, loops: train, or held out on H5). Writes
  catalog_signed_ab_indist.yaml   the signed train pools: added to catalog_daniel's 1352 training processes for training
                                  (the job writes that union, catalog_daniel_train.yaml plus these, at run time) and
                                  scored on their validation events
  catalog_signed_ab_heldout.yaml  the signed H5 pools (scored zero-shot)
The one-loop pools without a sign change and the loop-induced ones stay out: the A/B is about the signed target.
    python3 recipes/select_signed_ab.py
"""
import json, os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def flow(ps):
    """one flow mapping per line, as the other recipes"""
    return "".join("  - {" + ", ".join(f"{k}: {json.dumps(v)}" for k, v in p.items()) + "}\n" for p in ps)


def main():
    st = yaml.safe_load(open(os.path.join(HERE, "catalog_v3_gen_stage2a.yaml")))["processes"]
    roles = json.load(open(os.path.join(HERE, "catalog_v3_roles.json")))["loops"]
    sb = json.load(open(os.path.join(ROOT, "analysis", "catalog_daniel", "data", "signed_band_v3.json")))["pools"]
    signed = {k for k, v in sb.items() if v["neg_frac"] > 0}
    key = lambda p: f"{p['name']}_{p['sqrts'][0]}-{p['sqrts'][1]}GeV"
    tr = [p for p in st if key(p) in signed and roles.get(p["name"], {}).get("role") == "train"]
    ho = [p for p in st if key(p) in signed and roles.get(p["name"], {}).get("role") == "heldout"]
    left = sorted(signed - {key(p) for p in tr + ho})
    assert not left, f"signed pools with neither role: {left}"
    head = "# the signed pools' A/B (D99): written by recipes/select_signed_ab.py, do not edit\n"
    for fn, ps, what in (("catalog_signed_ab_indist.yaml", tr, f"catalog_v3's {len(tr)} signed one-loop train pools"),
                         ("catalog_signed_ab_heldout.yaml", ho, f"catalog_v3's {len(ho)} signed one-loop H5 pools")):
        with open(os.path.join(HERE, fn), "w") as f:
            f.write(head + f"# {what}\nprocesses:\n" + flow(ps))
        assert yaml.safe_load(open(os.path.join(HERE, fn)))["processes"] == ps, fn
        print(fn, len(ps))


if __name__ == "__main__":
    main()
