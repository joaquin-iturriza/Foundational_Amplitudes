#!/usr/bin/env python3
"""Figure of the CC cache census (analysis/agent_redesign/cache_census.py's JSON): GB per pool family of
$SCRATCH/amp_data_cache, stacked by what reads it (a recipe of an open plan entry, another catalog_v4 recipe, an older
recipe only, no recipe), next to the training-set memmap cache and the val/test pools in datasets.
    python analysis/agent_redesign/plot_cache_census.py <census.json> <out base>"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import plot_style as ps  # noqa: E402

CLASSES = ("open plan entry", "other catalog_v4 recipe", "older recipe only", "no recipe")


def main(src, base):
    c = json.load(open(src))
    fam = defaultdict(lambda: dict.fromkeys(CLASSES, 0.0))
    for k, gb in c["data_cache_by_prefix_class_gb"].items():
        pre, cls = k.split(" | ")
        fam["other" if pre.startswith("other") else pre][cls] += gb
    rows = sorted(fam, key=lambda p: sum(fam[p].values()))
    rows = [p for p in rows if sum(fam[p].values()) >= 0.5]   # families under 0.5 GB are not visible at this scale
    small = sum(sum(v.values()) for p, v in fam.items() if p not in rows)
    fig, ax = ps.figure()
    y = list(range(len(rows)))
    left = [0.0] * len(rows)
    cols = (ps.C.blue, ps.C.sky, ps.C.orange, ps.C.vermillion)
    for cls, col in zip(CLASSES, cols):
        w = [fam[p][cls] for p in rows]
        ax.barh(y, w, left=left, color=col, label=cls)
        left = [a + b for a, b in zip(left, w)]
    extra = [("memmap cache", sum(c["mmap_cache_gb"].values())), ("datasets val+test", c["datasets_gb"]["all"])]
    for i, (lab, gb) in enumerate(extra):
        ax.barh(len(rows) + i, gb, color=ps.C.grey,
                label="memmap / datasets" if i == 0 else None)
    ax.set_yticks(range(len(rows) + len(extra)))
    ax.set_yticklabels([f"{p}_" for p in rows] + [e[0] for e in extra])
    ax.set_xlabel("GB on /sps/lpnhe")
    ps.legend(ax, "lower right")
    ps.save(fig, base)
    print(json.dumps({"rows": {p: round(sum(fam[p].values()), 1) for p in rows}, "under_0.5GB_total": round(small, 2),
                      "extra": dict(extra)}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
