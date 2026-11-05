#!/usr/bin/env python3
"""Per-class read of the 16x-over-4x regression on the held-out 2->5 classes (H2p5; plan catalog_coverage step 55;
par:v4_length: R 2.28 catalog-sampled, 3.07 default-sampled, 38 classes, mixture_ab.py length).

The two long-best eval runs (mixture_ab.LONG_BEST: 16x FA-ccin2p3-1a46f3e0, 4x FA-ccin2p3-6f8a1eed; sigma-stage
parents, D276) saved per probe the truth t and the zero-shot prediction q (log|M|^2, tools/zero_shot_parent.py
--save-dir, $SCRATCH/zz_v4_long_best_<sweep>_<trial>/<axis>/<probe>.npz). Per probe the zero-shot error splits as
mse = level + shape (analysis/common read_errors: level = mean(q - t)^2, what one number per process removes; shape =
var(q - t)). The probes read: H2p5 (catalog-sampled) and the default-sampled probes of the H2p5 classes (PP4, EE4,
HD4, HD5 probes whose class is held out on H2p5, as mixture_ab.length's held_default). Per class, each side's
geometric mean over its probes; reported per class: the 4x mse, level and shape, the 16x/4x ratio of each, and which
part grew; summary: how many classes have the ratio above 1, the median ratio of mse, level and shape, the share of the
16x-minus-4x mse increase carried by level, and the classes ranked by their mse ratio.

    python3 analysis/catalog_v4/length_h2p5.py pull     site run on CC (read-only, CPU): -> data/length_h2p5_reads.json
    python3 analysis/catalog_v4/length_h2p5.py report   -> data/length_h2p5.json
"""
import json, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data")
RUNS = {"x16": "zz_v4_long_best_cov_v4_long_hpo8_cc_68224_trial_1262",     # FA-ccin2p3-1a46f3e0
        "x4": "zz_v4_long_best_cov_v4_long_hpo8_cc_17056_trial_1262"}      # FA-ccin2p3-6f8a1eed
AXES = ("H2p5", "PP4", "EE4", "HD4", "HD5")

REMOTE = r'''
import glob, json, os, sys
import numpy as np
out = {}
for side, d in json.loads(sys.argv[1]).items():
    for ax in sys.argv[2].split(","):
        for f in sorted(glob.glob(os.path.join(os.environ["SCRATCH"], d, ax, "*.npz"))):
            z = np.load(f)
            if "q" not in z.files:
                continue
            e = z["q"].astype(np.float64) - z["t"].astype(np.float64)
            out.setdefault(side, {}).setdefault(ax, {})[os.path.basename(f)[:-4]] = [
                float(np.mean(e ** 2)), float(np.mean(e) ** 2), float(np.var(e)), int(len(e))]
print("H2P5_READS " + json.dumps(out))
'''


def pull():
    sys.path.insert(0, ROOT)
    from analysis.common import site_python
    d = site_python(REMOTE, [f"'{json.dumps(RUNS)}'", ",".join(AXES)], "H2P5_READS")
    json.dump({"source": "q, t of the saved zero-shot npz on ccin2p3 (" + ", ".join(RUNS.values()) + "): "
                         "[mse, level, shape, n] per probe", "reads": d},
              open(os.path.join(DATA, "length_h2p5_reads.json"), "w"), indent=0)
    print({s: {a: len(v) for a, v in x.items()} for s, x in d.items()})


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def report():
    from mixture_ab import class_of, _held_axes
    cls = class_of()
    ax_of = _held_axes(cls)
    rd = json.load(open(os.path.join(DATA, "length_h2p5_reads.json")))["reads"]
    # the probes of each read: H2p5's own, and the default-sampled probes of H2p5 classes
    groups = {"catalog_sampled": ("H2p5",), "default_sampled": ("PP4", "EE4", "HD4", "HD5")}
    out = {"design": "plan catalog_coverage step 55 (per-class read of par:v4_length's H2p5 regression)",
           "runs": {"x16": "FA-ccin2p3-1a46f3e0", "x4": "FA-ccin2p3-6f8a1eed"}, "reads": {}}
    for g, axes in groups.items():
        by = {}
        for ax in axes:
            a, b = rd["x16"].get(ax, {}), rd["x4"].get(ax, {})
            for p in sorted(set(a) & set(b)):
                c = cls(p)
                if ax_of.get(c) != "H2p5":
                    continue
                if min(a[p][0], b[p][0]) <= 0:
                    continue
                by.setdefault(c, []).append((ax, p, a[p], b[p]))
        rows = {}
        for c, ps in by.items():
            # mse: the geometric mean over the class's probes (par:v4_length's R); the split: arithmetic means over
            # its probes, so level + shape = the arithmetic mean of mse (geometric means do not add)
            r = {"mse_x4": gmean([q[3][0] for q in ps]), "mse_x16": gmean([q[2][0] for q in ps])}
            r["mse_ratio"] = r["mse_x16"] / r["mse_x4"]
            for i, k in ((0, "amse"), (1, "level"), (2, "shape")):
                r[k + "_x4"] = float(np.mean([q[3][i] for q in ps])); r[k + "_x16"] = float(np.mean([q[2][i] for q in ps]))
                r[k + "_ratio"] = r[k + "_x16"] / r[k + "_x4"] if r[k + "_x4"] > 0 else None
            d_mse = r["amse_x16"] - r["amse_x4"]
            r["level_share_of_increase"] = (r["level_x16"] - r["level_x4"]) / d_mse if d_mse > 0 else None
            r["level_frac_x4"] = r["level_x4"] / r["amse_x4"]; r["level_frac_x16"] = r["level_x16"] / r["amse_x16"]
            r["probes"] = {f"{q[0]}:{q[1]}": {"x4": [round(v, 5) for v in q[3][:3]], "x16": [round(v, 5) for v in q[2][:3]]}
                           for q in ps}      # [mse, level, shape] per probe
            rows[c] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}
        rk = sorted(rows, key=lambda c: -rows[c]["mse_ratio"])
        med = {k: round(float(np.median([rows[c][k + "_ratio"] for c in rows if rows[c][k + "_ratio"] is not None])), 4)
               for k in ("mse", "level", "shape")}
        inc = [c for c in rows if rows[c]["amse_x16"] > rows[c]["amse_x4"]]
        tot = sum(rows[c]["amse_x16"] - rows[c]["amse_x4"] for c in inc)
        lev = sum(rows[c]["level_x16"] - rows[c]["level_x4"] for c in inc)
        out["reads"][g] = {"n_classes": len(rows), "n_probes": sum(len(rows[c]["probes"]) for c in rows),
                           "n_ratio_above_1": sum(rows[c]["mse_ratio"] > 1 for c in rows), "median_ratio": med,
                           "frac_classes_level_ratio_above_1": round(float(np.mean([(rows[c]["level_ratio"] or 0) > 1 for c in rows])), 4),
                           "frac_classes_shape_ratio_above_1": round(float(np.mean([(rows[c]["shape_ratio"] or 0) > 1 for c in rows])), 4),
                           "level_share_of_summed_increase": round(lev / tot, 4) if tot > 0 else None,
                           "median_level_frac": {"x4": round(float(np.median([rows[c]["level_frac_x4"] for c in rows])), 4),
                                                 "x16": round(float(np.median([rows[c]["level_frac_x16"] for c in rows])), 4)},
                           "corr_log_ratio_vs_log_mse_x4": round(float(np.corrcoef(
                               np.log([rows[c]["mse_x4"] for c in rows]), np.log([rows[c]["mse_ratio"] for c in rows]))[0, 1]), 4)
                           if len(rows) > 2 else None,
                           "ranked": rk, "classes": rows}
    json.dump(out, open(os.path.join(DATA, "length_h2p5.json"), "w"), indent=1)
    print(json.dumps({g: {k: v for k, v in x.items() if k not in ("classes", "ranked")} | {"top5": x["ranked"][:5]}
                      for g, x in out["reads"].items()}))


if __name__ == "__main__":
    {"pull": pull, "report": report}[sys.argv[1]]()
