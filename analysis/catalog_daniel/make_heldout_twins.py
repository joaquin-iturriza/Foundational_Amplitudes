"""The training twin of each held-out process that relabels one (D74): heldout_twins.json {probe: training process},
read by tools/zero_shot_parent.py --twins, which scores the probe in log|M|^2 with the twin's statistics. The relabels
are the held-out recipe minus recipes/catalog_daniel_heldout_norelabel.yaml (96). The twin is a training process with
the probe's exact_key (classes_default.json, classify2.py: the same process up to the relabelling, so the same
|M|^2 statistics); a probe with none would fall back to its weak_key class and is listed (none does). Prints the count
per kind; heldout_twins_kind.json records it per probe.
    python analysis/catalog_daniel/make_heldout_twins.py
"""
import json, os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def names(f):
    return [p["name"] for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", f)))["processes"]]


def main():
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    train, nr = names("catalog_daniel_train.yaml"), set(names("catalog_daniel_heldout_norelabel.yaml"))
    by = {k: {} for k in ("exact_key", "weak_key")}
    for n in train:
        for k in by:
            by[k].setdefault(meta[n][k], []).append(n)
    tw, kind = {}, {}
    for n in names("catalog_daniel_heldout.yaml"):
        if n in nr:
            continue
        for k in ("exact_key", "weak_key"):
            if meta[n][k] in by[k]:
                tw[n], kind[n] = sorted(by[k][meta[n][k]])[0], k
                break
        else:
            print("no twin:", n)
    json.dump(tw, open(os.path.join(HERE, "heldout_twins.json"), "w"), indent=0, sort_keys=True)
    json.dump(kind, open(os.path.join(HERE, "heldout_twins_kind.json"), "w"), indent=0, sort_keys=True)
    print(len(tw), "twins;", {k: list(kind.values()).count(k) for k in ("exact_key", "weak_key")})
    print("dc_bbx_gvmvmx ->", tw.get("dc_bbx_gvmvmx"), "; dc_cb_cb ->", tw.get("dc_cb_cb"))


if __name__ == "__main__":
    main()
