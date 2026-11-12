#!/usr/bin/env python3
"""Figure of the CC cache census (analysis/agent_redesign/cache_census.py). Left: GB per pool family of
$SCRATCH/amp_data_cache, stacked by what reads it (a recipe of an open plan entry, another catalog_v4 recipe, an older
recipe only, no recipe), beside the training-set memmap cache and $DATA_DIR. Right (from --quick): the same cache's
pool GB by sampling tag, and the chunk work dirs.
    python analysis/agent_redesign/plot_cache_census.py <census.json> <quick.json> <out base>"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import plot_style as ps  # noqa: E402

NAMES = {"v4": "catalog-v4 pools", "dsh": "MadEvent default-share pools", "pp13v4": "MadEvent pp pools (v4)",
         "pp13": "MadEvent pp pools (v3)", "kit": "KIT benchmark pools", "other": "leftover build folders",
         "ee": "old v2/v3 pools", "memmap": "packed training sets", "datasets": "datasets folder (val, test)",
         "smix": "catalog-v4 mixture pools", "smixq": "flat-quota second set (held coverage)",
         "lhe": "MadEvent pools read by mixtures", "flat": "unshaped pools", "n900000_kit": "KIT benchmark pools",
         "chunk dirs": "leftover build folders", "other catalog_v4 recipe": "other catalog-v4 recipe"}
CLASSES = ("open plan entry", "other catalog_v4 recipe", "older recipe only", "no recipe")


def main(src, qsrc, base):
    c, q = json.load(open(src)), json.load(open(qsrc))
    fam = defaultdict(lambda: dict.fromkeys(CLASSES, 0.0))
    for k, gb in c["data_cache_by_prefix_class_gb"].items():
        pre, cls = k.split(" | ")
        fam["other" if pre.startswith("other") else pre][cls] += gb
    rows = [p for p in sorted(fam, key=lambda p: sum(fam[p].values())) if sum(fam[p].values()) >= 0.5]
    small = sum(sum(v.values()) for p, v in fam.items() if p not in rows)   # families under 0.5 GB: invisible here
    figs = ps.panels(2)   # one file per panel: the plain-word labels make a 1x2 canvas wider than the page
    ax, ax2 = figs[0][1], figs[1][1]
    left = [0.0] * len(rows)
    for cls, col in zip(CLASSES, (ps.C.blue, ps.C.sky, ps.C.orange, ps.C.vermillion)):
        w = [fam[p][cls] for p in rows]
        ax.barh(range(len(rows)), w, left=left, color=col, label=NAMES.get(cls, cls))
        left = [a + b for a, b in zip(left, w)]
    extra = [("memmap", sum(c["mmap_cache_gb"].values())), ("datasets", c["datasets_gb"]["all"])]
    for i, (lab, gb) in enumerate(extra):
        ax.barh(len(rows) + i, gb, color=ps.C.grey, label="other cache" if i == 0 else None)
    ax.set_yticks(range(len(rows) + len(extra)))
    ax.set_yticklabels([NAMES.get(p, p) for p in rows] + [NAMES[e[0]] for e in extra])
    ax.set_xlabel("GB")
    ps.legend(ax, "lower right")
    tags = {k: v for k, v in q["pool_gb_by_sampling_tag"].items() if v >= 0.5}
    tags["chunk dirs"] = q["chunk_dirs_gb"]
    names = sorted(tags, key=tags.get)
    ax2.barh(range(len(names)), [tags[n] for n in names], color=ps.C.blue)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([NAMES.get(n, n) for n in names])
    ax2.set_xlabel("GB")
    ps.save_panels(figs, base)
    print(json.dumps({"rows": {p: round(sum(fam[p].values()), 2) for p in rows}, "under_0.5GB_total": round(small, 2),
                      "extra": dict(extra), "tags": tags}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
