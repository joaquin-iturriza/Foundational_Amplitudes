#!/usr/bin/env python3
"""Where the step-65 regeneration's time goes and where its short bins and uncovered held-out events sit (plan
catalog_coverage step 75), from the regeneration's own records: data/regen_g12_pull.json (per pool the flat part's
labels, t_label, t_propose and the chunk seconds, regen_read.py pull), data/regen_g12_read.json (per class the short
bins, coverage and cost ratio), data/sampling_cost_grid12.json (the old per-class core-s) and
recipes/catalog_v4_flat_range.json (the quota range).

Cost of a class (regen_read.py's): the chunk seconds of its three pools over the old per-class core-s. Split here into
  label    the flat part's labelling (t_label: the matrix element of every flat candidate)
  propose  the flat part's proposal (t_propose: drawing candidates through the channels, the cuts' rejection, the grid
           refits)
  rest     the chunk seconds minus both: the bulk and forward parts (labelled once per event) and I/O
each over the old per-class core-s. Labels per pool event: the flat labels plus one per non-flat event (0.65 of the
rows); the old's estimated as 0.97 of its core-s (coverage_check.py @27e8edd: labelling ~0.97 of the old chunk time)
over the new run's seconds per flat label (the same matrix element), an estimate. Short bins are placed against the
range: in its top or bottom 0.5 dex (the margin tools/flat_range.py adds beyond the in-support pp events of the PP4
probe's train and test pools), the next 0.5 dex, or the interior; and against the held-out in-support pp events'
extent (the PP4 val pool and the dsh_ pools, regen_read.py's coverage). Held-out events outside the kept range are
placed against the range's top.

    python3 analysis/catalog_v4/regen_profile.py    -> data/regen_g12_profile.json
"""
import collections, json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "regen_g12_profile.json")
L10 = np.log(10.0)
MARGIN = 0.5          # tools/flat_range.MARGIN_DEX


def _med(x):
    return round(float(np.median(x)), 3) if len(x) else None


def main():
    P = json.load(open(os.path.join(DATA, "regen_g12_pull.json")))["pools"]
    R = json.load(open(os.path.join(DATA, "regen_g12_read.json")))["classes"]
    old = json.load(open(os.path.join(DATA, "sampling_cost_grid12.json")))["old_per_class_s"]
    F = json.load(open(os.path.join(ROOT, "recipes", "catalog_v4_flat_range.json")))
    rows = {}
    for c, r in R.items():
        ps = [P.get(f"{c}/{x}", {}) for x in ("train", "val", "test")]
        s = sum(p.get("secs", 0.0) for p in ps); tl = sum(p.get("t_label", 0.0) for p in ps)
        tp = sum(p.get("t_propose", 0.0) for p in ps); lab = sum(p.get("labels", 0) for p in ps)
        n = sum(p.get("rows", 0) for p in ps)
        tau = tl / lab
        rows[c] = dict(nfinal=r["nfinal"], ratio=r["cost_ratio"], label=tl / old[c], propose=tp / old[c],
                       rest=(s - tl - tp) / old[c], us_label=tau * 1e6, us_propose_per_flat_label=tp / lab * 1e6,
                       flat_labels_per_flat_event=lab / (0.35 * n), labels_per_row=(lab + 0.65 * n) / n,
                       old_labels_per_row_est=old[c] * 0.97 / tau / n, budget_hit=r["budget_hit"] > 0,
                       ratio_without_propose=(s - tp) / old[c])
    out = {"cost": {}, "short": {}, "coverage": {}}
    for nm, sel in (("over_2x", lambda x: x["ratio"] > 2), ("at_most_2x", lambda x: x["ratio"] <= 2)):
        g = [x for x in rows.values() if sel(x)]
        out["cost"][nm] = {"n": len(g), **{k: _med([x[k] for x in g]) for k in
                           ("label", "propose", "rest", "us_label", "us_propose_per_flat_label", "flat_labels_per_flat_event",
                            "labels_per_row", "old_labels_per_row_est")},
                           "with_budget_hit": sum(x["budget_hit"] for x in g)}
    out["cost"]["by_nfinal"] = {str(k): {"n": len(g), "us_propose_per_flat_label": _med([x["us_propose_per_flat_label"] for x in g]),
                                        "us_label": _med([x["us_label"] for x in g])}
                                for k in (2, 3, 4) for g in [[x for x in rows.values() if x["nfinal"] == k]]}
    out["cost"]["over_2x_if_propose_zero"] = sum(x["ratio_without_propose"] > 2 for x in rows.values())
    out["cost"]["n_over_2x"] = sum(x["ratio"] > 2 for x in rows.values())
    pos, miss, tagsets, kinds = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    vs_pp = collections.Counter(); kind_of = {}
    for c, r in R.items():
        if not r["short_bins"]:
            continue
        lo, hi = F[c][0] / L10, F[c][1] / L10
        bs = r["coverage"].get("by_source", {})
        mx = [v["max_in_support"] for v in bs.values() if v.get("max_in_support") is not None]
        mn = [v["min_in_support"] for v in bs.values() if v.get("min_in_support") is not None]
        tags = set(); inside = []
        for role, a, b, f in r["short_bins"]:
            t = ("top_margin" if b > hi - MARGIN - 1e-6 else "bottom_margin" if a < lo + MARGIN - 1e-6 else
                 "top_next_0.5" if b > hi - 2 * MARGIN else "bottom_next_0.5" if a < lo + 2 * MARGIN else "interior")
            pos[t] += 1; miss[t] += 1 - f; tags.add(t)
            if mx:
                w = ("above_heldout_max" if a >= max(mx) else "below_heldout_min" if b <= min(mn) else "inside_heldout")
                vs_pp[w] += 1
                if w == "inside_heldout":
                    inside.append(f)
        tagsets["+".join(sorted(tags))] += 1
        k = ("no_heldout" if not mx else "only_beyond_heldout" if not inside else
             "inside_heldout_worst_below_0.5" if min(inside) < 0.5 else "inside_heldout_worst_0.5_0.95")
        kinds[k] += 1; kind_of[c] = k
    out["short"] = {"n_classes": sum(kinds.values()), "bins_by_position": dict(pos),
                    "missing_events_share_by_position": {k: round(v, 1) for k, v in miss.items()},
                    "classes_by_position_set": dict(tagsets.most_common()), "bins_vs_heldout_extent": dict(vs_pp),
                    "classes_by_kind": dict(kinds),
                    "budget_hit_by_kind": {k: sum(rows[c]["budget_hit"] for c in kind_of if kind_of[c] == k) for k in kinds},
                    "by_nfinal": {str(n): [sum(1 for c in kind_of if R[c]["nfinal"] == n), sum(1 for c in R if R[c]["nfinal"] == n)]
                                  for n in (2, 3, 4)},
                    "classes": kind_of}
    cov, side = collections.Counter(), collections.Counter(); gap = []
    for c, r in R.items():
        cv = r["coverage"]
        if not cv.get("outside_kept"):
            continue
        side["above"] += cv["above_kept"]; side["below"] += cv["below_kept"]
        hi = F[c][1] / L10
        gap.append(hi - cv["kept"][1])
        for s, v in cv["by_source"].items():
            if v.get("outside_kept"):
                cov[f"{s}: " + ("heldout max above range top" if v["max_in_support"] > hi else
                                "heldout max inside range, above kept max")] += 1
    out["coverage"] = {"classes": sum(1 for r in R.values() if r["coverage"].get("outside_kept")),
                       "events_by_side": dict(side), "classes_by_source_and_place": dict(cov),
                       "range_top_minus_kept_max_dex": {"median": _med(gap), "q90": round(float(np.quantile(gap, 0.9)), 3)},
                       "outside_without_short_bin": sum(1 for r in R.values() if r["coverage"].get("outside_kept") and not r["short_bins"])}
    out["per_class"] = {c: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for c, x in rows.items()}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_class"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
